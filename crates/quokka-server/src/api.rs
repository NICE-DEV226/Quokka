//! Couche HTTP : routes OpenAI-compatibles et stream SSE.
//!
//! - `GET  /health`
//! - `GET  /v1/models`
//! - `POST /v1/chat/completions` (`stream: true` → SSE, sinon JSON)
//!
//! Le handler ne touche jamais llama.cpp directement : il soumet un
//! [`Job`] à l'[`InferenceEngine`] et traduit les [`EngineEvent`] en
//! `chat.completion.chunk` au format OpenAI.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::{StreamExt, stream};
use quokka_core::pool::{LoadedModel, ModelPool, ModelRole};
use quokka_core::session::FinishReason;
use tokio::sync::mpsc as tokio_mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::engine::{EngineError, EngineEvent, InferenceEngine, Job};
use crate::types::{
    ChatCompletionRequest, ChatCompletionResponse, ChunkChoice, DeltaContent, FullChoice,
    FullMessage, ModelInfo, ModelsResponse, StreamChunk, Usage, finish_reason_name,
};

/// Modèle servi par le moteur (id API + fichier GGUF).
#[derive(Debug, Clone)]
pub struct ServedModel {
    /// Rôle moteur.
    pub role: ModelRole,
    /// Identifiant exposé dans l'API ("flash" / "expert").
    pub id: &'static str,
    /// Nom de fichier GGUF (pour résoudre `model` côté client).
    pub filename: String,
    /// Taille du fichier en octets (garde mémoire du routeur).
    pub size_bytes: u64,
}

/// Construit la liste des modèles servis depuis le pool (avant que
/// `InferenceEngine::new` ne consomme le pool).
#[must_use]
pub fn served_models(pool: &ModelPool) -> Vec<ServedModel> {
    pool.models()
        .iter()
        .map(|loaded: &LoadedModel| ServedModel {
            role: loaded.role,
            id: role_id(loaded.role),
            filename: loaded
                .path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default(),
            size_bytes: std::fs::metadata(&loaded.path)
                .map(|m| m.len())
                .unwrap_or(0),
        })
        .collect()
}

fn role_id(role: ModelRole) -> &'static str {
    match role {
        ModelRole::Flash => "flash",
        ModelRole::Expert => "expert",
    }
}

/// État partagé des handlers.
#[derive(Clone)]
pub struct AppState {
    engine: InferenceEngine,
    models: Arc<Vec<ServedModel>>,
    router: Arc<quokka_router::Router>,
    /// Taille de l'expert en Mio pour la garde mémoire du routeur.
    expert_size_mb: u32,
    next_id: Arc<AtomicU64>,
    /// Compteurs d'usage agrégés, exposés par `GET /telemetry`.
    telemetry: Arc<quokka_telemetry::Telemetry>,
}

/// `GET /telemetry` — compteurs d'usage agrégés (tokens, cache KV,
/// histogramme de routage) depuis le démarrage du processus.
async fn telemetry(State(state): State<AppState>) -> Json<quokka_telemetry::TelemetrySnapshot> {
    Json(state.telemetry.snapshot())
}

/// `POST /v1/route` — dry-run du routeur : même body que
/// `/v1/chat/completions`, retourne la décision sans inférence.
/// Utilisé par `quokka_route` (MCP) et le debugging.
async fn route_dry_run(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    let auto_requested = request
        .model
        .as_deref()
        .is_none_or(|m| matches!(m, "auto" | "quokka-auto" | "quokka"));
    let route = if auto_requested {
        route_request(&state, &request).1
    } else {
        RouteInfo {
            model_used: request.model.clone().unwrap_or_default(),
            route_ms: 0.0,
            reason: "explicit".to_owned(),
        }
    };
    Json(serde_json::json!({
        "model": route.model_used,
        "route_ms": route.route_ms,
        "reason": route.reason,
    }))
    .into_response()
}

/// Traçabilité du routage, exposée en en-têtes de réponse.
#[derive(Debug, Clone)]
pub struct RouteInfo {
    /// Id du modèle effectivement utilisé.
    pub model_used: String,
    /// Durée du routage en millisecondes (0 si choix explicite).
    pub route_ms: f32,
    /// Raison du choix (`Debug` de `RouteReason`, ou "explicit").
    pub reason: String,
}

impl AppState {
    /// Construit le state à partir du moteur et de la liste des modèles.
    #[must_use]
    pub fn new(engine: InferenceEngine, models: Vec<ServedModel>) -> Self {
        let expert_size_mb = models
            .iter()
            .filter(|m| m.role == ModelRole::Expert)
            .map(|m| m.size_bytes / (1024 * 1024))
            .max()
            .unwrap_or(0) as u32;
        Self {
            engine,
            models: Arc::new(models),
            router: Arc::new(quokka_router::Router::load(
                &quokka_router::default_config_path(),
            )),
            expert_size_mb,
            next_id: Arc::new(AtomicU64::new(1)),
            telemetry: Arc::new(quokka_telemetry::Telemetry::new()),
        }
    }

    /// Registre de télémétrie partagé (aussi accessible aux tests).
    #[must_use]
    pub fn telemetry(&self) -> Arc<quokka_telemetry::Telemetry> {
        Arc::clone(&self.telemetry)
    }

    fn completion_id(&self) -> String {
        let seq = self.next_id.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        format!("chatcmpl-{seq:08x}{nanos:08x}")
    }

    fn unix_now(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}

/// Construit le router axum.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(crate::static_files::index))
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/route", post(route_dry_run))
        .route("/telemetry", get(telemetry))
        .route("/{*path}", get(crate::static_files::asset))
        .with_state(state)
}

/// Démarre le serveur HTTP jusqu'à Ctrl-C (arrêt gracieux des
/// connexions SSE en vol).
pub async fn serve(state: AppState, addr: SocketAddr) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "quokka server listening");
    axum::serve(listener, router(state))
        .with_graceful_shutdown(async {
            if let Err(err) = tokio::signal::ctrl_c().await {
                tracing::warn!(%err, "ctrl-c handler unavailable");
            } else {
                tracing::info!("shutdown signal received");
            }
        })
        .await
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

async fn models(State(state): State<AppState>) -> Json<ModelsResponse> {
    let mut data: Vec<ModelInfo> = Vec::with_capacity(state.models.len() + 1);
    // Le routeur n'a de sens que si les deux rôles sont chargés.
    let has_flash = state.models.iter().any(|m| m.role == ModelRole::Flash);
    let has_expert = state.models.iter().any(|m| m.role == ModelRole::Expert);
    if has_flash && has_expert {
        data.push(ModelInfo {
            id: "auto",
            object: "model",
            owned_by: "quokka-router".to_owned(),
        });
    }
    data.extend(state.models.iter().map(|m| ModelInfo {
        id: m.id,
        object: "model",
        owned_by: m.filename.clone(),
    }));
    Json(ModelsResponse {
        object: "list",
        data,
    })
}

/// Erreur JSON au format OpenAI.
fn error_response(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(serde_json::json!({
            "error": { "message": message, "type": "invalid_request_error" }
        })),
    )
        .into_response()
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    if request.messages.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "messages must not be empty");
    }

    let served: Vec<(&'static str, &str)> = state
        .models
        .iter()
        .map(|m| (m.id, m.filename.as_str()))
        .collect();

    // `auto` / `quokka-auto` / absence de `model` → le routeur décide.
    // C'est le comportement par défaut : Quokka choisit Flash ou Expert.
    let auto_requested = request
        .model
        .as_deref()
        .is_none_or(|m| matches!(m, "auto" | "quokka-auto" | "quokka"));
    let (role, route) = if auto_requested {
        route_request(&state, &request)
    } else {
        let Some(role) = request.resolve_role(&served) else {
            return error_response(
                StatusCode::NOT_FOUND,
                "requested model is not served — see GET /v1/models",
            );
        };
        (
            role,
            RouteInfo {
                model_used: role_id(role).to_owned(),
                route_ms: 0.0,
                reason: "explicit".to_owned(),
            },
        )
    };

    let model_id = route.model_used.clone();
    let (events_tx, events_rx) = tokio_mpsc::channel::<EngineEvent>(64);
    let config = request.generation_config();
    let job = Job {
        messages: request
            .messages
            .into_iter()
            .map(|m| m.into_chat())
            .collect(),
        config,
        events: events_tx,
    };
    if let Err(err) = state.engine.submit(role, job) {
        let message = match err {
            EngineError::ModelNotLoaded(_) => "model is not loaded",
            EngineError::WorkerGone(_) => "inference worker unavailable",
        };
        return error_response(StatusCode::SERVICE_UNAVAILABLE, message);
    }

    if request.stream {
        sse_response(state, events_rx, model_id, &route).await
    } else {
        json_response(state, events_rx, model_id, &route).await
    }
}

/// Route une requête `auto` : classifie le dernier message porteur via
/// `quokka-router` (< 2 ms) puis rabat sur un rôle réellement servi si
/// le duo complet n'est pas chargé.
fn route_request(state: &AppState, request: &ChatCompletionRequest) -> (ModelRole, RouteInfo) {
    let text = request
        .messages
        .iter()
        .rev()
        .map(|m| m.content.as_str())
        .find(|c| !c.trim().is_empty())
        .unwrap_or_default();
    let ctx = quokka_router::RouteContext::system(state.expert_size_mb);
    let decision = state.router.route(text, &ctx);
    let mut role = match decision.target {
        quokka_router::RouteTarget::Flash => ModelRole::Flash,
        quokka_router::RouteTarget::Expert => ModelRole::Expert,
    };
    // Un seul rôle chargé → on sert ce qui existe (mieux que 503).
    let served_roles = state.engine.roles();
    if !served_roles.contains(&role) {
        role = served_roles
            .iter()
            .copied()
            .find(|r| *r == ModelRole::Flash)
            .or_else(|| served_roles.first().copied())
            .unwrap_or(ModelRole::Flash);
    }
    (
        role,
        RouteInfo {
            model_used: role_id(role).to_owned(),
            route_ms: decision.elapsed.as_secs_f32() * 1e3,
            reason: format!("{:?}", decision.reason),
        },
    )
}

/// Ajoute les en-têtes de traçabilité Quokka à une réponse.
/// `tokens_per_sec` n'est connu qu'une fois la génération terminée —
/// les réponses SSE ne le portent donc pas (en-têtes déjà partis).
fn with_route_headers(
    mut response: Response,
    route: &RouteInfo,
    tokens_per_sec: Option<f32>,
) -> Response {
    let headers = response.headers_mut();
    let route_ms = format!("{:.3}", route.route_ms);
    for (name, value) in [
        ("x-quokka-model-used", route.model_used.as_str()),
        ("x-quokka-route-time-ms", route_ms.as_str()),
        ("x-quokka-route-reason", route.reason.as_str()),
    ] {
        if let Ok(v) = axum::http::HeaderValue::from_str(value) {
            headers.insert(name, v);
        }
    }
    if let Some(tps) = tokens_per_sec
        && let Ok(v) = axum::http::HeaderValue::from_str(&format!("{tps:.1}"))
    {
        headers.insert("x-quokka-tokens-per-sec", v);
    }
    response
}

/// Agrège les événements en une réponse `chat.completion` JSON.
async fn json_response(
    state: AppState,
    mut events_rx: tokio_mpsc::Receiver<EngineEvent>,
    model_id: String,
    route: &RouteInfo,
) -> Response {
    let id = state.completion_id();
    let created = state.unix_now();
    let mut text = String::new();
    let mut finish = FinishReason::Length;
    let mut prompt_tokens = 0_usize;
    let mut cached_prompt_tokens = 0_usize;
    let mut generated_tokens = 0_usize;
    let mut elapsed = Duration::ZERO;

    while let Some(event) = events_rx.recv().await {
        match event {
            EngineEvent::Token(piece) => text.push_str(&piece),
            EngineEvent::Done {
                finish: f,
                prompt_tokens: p,
                cached_prompt_tokens: c,
                generated_tokens: g,
                elapsed: e,
            } => {
                finish = f;
                prompt_tokens = p;
                cached_prompt_tokens = c;
                generated_tokens = g;
                elapsed = e;
                state.telemetry.record_completion(
                    &model_id,
                    &route.reason,
                    p as u64,
                    c as u64,
                    g as u64,
                    e.as_secs_f64() * 1e3,
                );
            }
            EngineEvent::Error(message) => {
                return error_response(StatusCode::INTERNAL_SERVER_ERROR, &message);
            }
        }
    }

    let response = Json(ChatCompletionResponse {
        object: "chat.completion",
        id,
        created,
        model: model_id,
        choices: vec![FullChoice {
            index: 0,
            message: FullMessage {
                role: "assistant",
                content: text,
            },
            finish_reason: finish_reason_name(finish),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens: generated_tokens,
            total_tokens: prompt_tokens + generated_tokens,
            prompt_tokens_details: (cached_prompt_tokens > 0).then_some(
                crate::types::PromptTokensDetails {
                    cached_tokens: cached_prompt_tokens,
                },
            ),
        },
    })
    .into_response();
    let tokens_per_sec = if elapsed.as_secs_f32() > 0.0 {
        Some(generated_tokens as f32 / elapsed.as_secs_f32())
    } else {
        None
    };
    with_route_headers(response, route, tokens_per_sec)
}

/// Traduit le flux d'événements en SSE `chat.completion.chunk`.
async fn sse_response(
    state: AppState,
    events_rx: tokio_mpsc::Receiver<EngineEvent>,
    model_id: String,
    route: &RouteInfo,
) -> Response {
    let id = state.completion_id();
    let created = state.unix_now();
    let mut role_sent = false;
    // Le stream est 'static : la raison de routage est clonée pour la
    // télémétrie enregistrée à la fin de la génération.
    let route_reason = route.reason.clone();
    let telemetry = Arc::clone(&state.telemetry);

    let events = ReceiverStream::new(events_rx).flat_map(move |event| {
        let mut out: Vec<Event> = Vec::with_capacity(2);
        match event {
            EngineEvent::Token(piece) => {
                let delta = DeltaContent {
                    role: (!role_sent).then_some("assistant"),
                    content: Some(piece),
                };
                role_sent = true;
                out.push(chunk_event(&id, created, &model_id, Some(delta), None));
            }
            EngineEvent::Done {
                finish,
                prompt_tokens,
                cached_prompt_tokens,
                generated_tokens,
                elapsed,
            } => {
                telemetry.record_completion(
                    &model_id,
                    &route_reason,
                    prompt_tokens as u64,
                    cached_prompt_tokens as u64,
                    generated_tokens as u64,
                    elapsed.as_secs_f64() * 1e3,
                );
                out.push(chunk_event(
                    &id,
                    created,
                    &model_id,
                    None,
                    Some(finish_reason_name(finish)),
                ));
                out.push(Event::default().data("[DONE]"));
            }
            EngineEvent::Error(message) => {
                out.push(
                    Event::default()
                        .data(serde_json::json!({ "error": { "message": message } }).to_string()),
                );
                out.push(Event::default().data("[DONE]"));
            }
        }
        stream::iter(out.into_iter().map(Ok::<_, Infallible>))
    });

    let response = Sse::new(events)
        .keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("keep-alive"),
        )
        .into_response();
    // tok/s inconnu à l'envoi des en-têtes SSE — voir with_route_headers.
    with_route_headers(response, route, None)
}

fn chunk_event(
    id: &str,
    created: u64,
    model: &str,
    delta: Option<DeltaContent>,
    finish_reason: Option<&'static str>,
) -> Event {
    let chunk = StreamChunk {
        object: "chat.completion.chunk",
        id: id.to_owned(),
        created,
        model: model.to_owned(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: delta.unwrap_or(DeltaContent {
                role: None,
                content: None,
            }),
            finish_reason,
        }],
    };
    Event::default()
        .json_data(chunk)
        .unwrap_or_else(|_| Event::default().data("{}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ApiMessage;

    #[test]
    fn resolve_role_defaults_to_flash() {
        let req = ChatCompletionRequest {
            model: None,
            messages: vec![],
            stream: false,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            seed: None,
        };
        let served = [("flash", "a.gguf"), ("expert", "b.gguf")];
        assert_eq!(req.resolve_role(&served), Some(ModelRole::Flash));
    }

    #[test]
    fn resolve_role_by_filename() {
        let req = ChatCompletionRequest {
            model: Some("b.gguf".into()),
            messages: vec![],
            stream: true,
            max_tokens: Some(8),
            temperature: Some(0.0),
            top_p: None,
            top_k: None,
            seed: Some(1),
        };
        let served = [("flash", "a.gguf"), ("expert", "b.gguf")];
        assert_eq!(req.resolve_role(&served), Some(ModelRole::Expert));
        let config = req.generation_config();
        assert_eq!(config.max_tokens, 8);
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.seed, 1);
    }

    #[test]
    fn resolve_role_unknown_is_none() {
        let req = ChatCompletionRequest {
            model: Some("nope".into()),
            messages: vec![],
            stream: false,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            seed: None,
        };
        let served = [("flash", "a.gguf")];
        assert_eq!(req.resolve_role(&served), None);
    }

    #[test]
    fn api_message_role_mapping() {
        use quokka_core::session::ChatRole;
        let msg = ApiMessage {
            role: "system".into(),
            content: "be nice".into(),
        };
        assert_eq!(msg.into_chat().role, ChatRole::System);
        let msg = ApiMessage {
            role: "wizard".into(),
            content: "x".into(),
        };
        assert_eq!(msg.into_chat().role, ChatRole::User);
    }
}
