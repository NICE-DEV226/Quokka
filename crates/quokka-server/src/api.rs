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
    next_id: Arc<AtomicU64>,
}

impl AppState {
    /// Construit le state à partir du moteur et de la liste des modèles.
    #[must_use]
    pub fn new(engine: InferenceEngine, models: Vec<ServedModel>) -> Self {
        Self {
            engine,
            models: Arc::new(models),
            next_id: Arc::new(AtomicU64::new(1)),
        }
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
    Json(ModelsResponse {
        object: "list",
        data: state
            .models
            .iter()
            .map(|m| ModelInfo {
                id: m.id,
                object: "model",
                owned_by: m.filename.clone(),
            })
            .collect(),
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
    let Some(role) = request.resolve_role(&served) else {
        return error_response(
            StatusCode::NOT_FOUND,
            "requested model is not served — see GET /v1/models",
        );
    };

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

    let model_id = role_id(role).to_owned();
    if request.stream {
        sse_response(state, events_rx, model_id).await
    } else {
        json_response(state, events_rx, model_id).await
    }
}

/// Agrège les événements en une réponse `chat.completion` JSON.
async fn json_response(
    state: AppState,
    mut events_rx: tokio_mpsc::Receiver<EngineEvent>,
    model_id: String,
) -> Response {
    let id = state.completion_id();
    let created = state.unix_now();
    let mut text = String::new();
    let mut finish = FinishReason::Length;
    let mut prompt_tokens = 0_usize;
    let mut generated_tokens = 0_usize;

    while let Some(event) = events_rx.recv().await {
        match event {
            EngineEvent::Token(piece) => text.push_str(&piece),
            EngineEvent::Done {
                finish: f,
                prompt_tokens: p,
                generated_tokens: g,
                ..
            } => {
                finish = f;
                prompt_tokens = p;
                generated_tokens = g;
            }
            EngineEvent::Error(message) => {
                return error_response(StatusCode::INTERNAL_SERVER_ERROR, &message);
            }
        }
    }

    Json(ChatCompletionResponse {
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
        },
    })
    .into_response()
}

/// Traduit le flux d'événements en SSE `chat.completion.chunk`.
async fn sse_response(
    state: AppState,
    events_rx: tokio_mpsc::Receiver<EngineEvent>,
    model_id: String,
) -> Response {
    let id = state.completion_id();
    let created = state.unix_now();
    let mut role_sent = false;

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
            EngineEvent::Done { finish, .. } => {
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

    Sse::new(events)
        .keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("keep-alive"),
        )
        .into_response()
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
