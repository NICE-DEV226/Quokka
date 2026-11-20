//! Moteur d'inférence : un thread worker par rôle de modèle.
//!
//! `LlamaContext` n'est pas `Sync` et `ModelPool` n'est pas `Clone` : le
//! pool est partagé en `Arc` entre les workers, et chaque worker exécute
//! les générations de **son** rôle en série sur son propre thread std.
//! Les résultats sont streamés vers le handler axum via un
//! `tokio::sync::mpsc` émetteur glissé dans le job — `blocking_send`
//! fonctionne depuis un thread non-async.
//!
//! Une **session persistante par worker** : le `LlamaContext` (et son
//! KV-cache) survit entre les requêtes et le préfixe de tokens commun est
//! réutilisé — les tours suivants d'une conversation ne re-paient que le
//! delta de prompt au lieu de tout le contexte.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc as std_mpsc;
use std::thread;
use std::time::Duration;

use quokka_core::pool::ModelPool;
use quokka_core::pool::ModelRole;
use quokka_core::session::SessionConfig;
use quokka_core::session::{ChatMessage, FinishReason, GenerationConfig};
use tokio::sync::mpsc as tokio_mpsc;

/// Événement de génération envoyé au handler HTTP.
#[derive(Debug)]
pub enum EngineEvent {
    /// Pièce de texte UTF-8 valide.
    Token(String),
    /// Génération terminée.
    Done {
        finish: FinishReason,
        prompt_tokens: usize,
        cached_prompt_tokens: usize,
        generated_tokens: usize,
        elapsed: Duration,
    },
    /// Erreur pendant la génération.
    Error(String),
}

/// Requête de génération pour un worker.
pub struct Job {
    /// Conversation à compléter.
    pub messages: Vec<ChatMessage>,
    /// Paramètres d'échantillonnage.
    pub config: GenerationConfig,
    /// Canal de retour vers le handler.
    pub events: tokio_mpsc::Sender<EngineEvent>,
}

/// Moteur : dispatch des jobs vers le worker du rôle demandé.
///
/// `Clone` à faible coût (canaux `Send` clonables) pour le state axum.
#[derive(Clone)]
pub struct InferenceEngine {
    workers: Arc<HashMap<ModelRole, std_mpsc::Sender<Job>>>,
}

impl InferenceEngine {
    /// Démarre un worker par modèle chargé dans `pool`.
    ///
    /// `session` configure la fenêtre de contexte et le threading CPU de
    /// chaque session. Les workers tournent jusqu'à drop de l'engine
    /// (canaux fermés → `recv` retourne `Err` → thread terminé).
    pub fn new(pool: ModelPool, session: SessionConfig) -> Self {
        let pool = Arc::new(pool);
        let mut workers = HashMap::new();

        for loaded in pool.models() {
            let role = loaded.role;
            let (tx, rx) = std_mpsc::channel::<Job>();
            let pool = Arc::clone(&pool);
            let session = session.clone();

            let spawned = thread::Builder::new()
                .name(format!("quokka-infer-{role:?}").to_lowercase())
                .spawn(move || worker_loop(pool, role, session, rx));
            match spawned {
                Ok(_handle) => {
                    workers.insert(role, tx);
                }
                Err(err) => {
                    tracing::error!(role = ?role, %err, "failed to spawn inference worker");
                }
            }
        }

        Self {
            workers: Arc::new(workers),
        }
    }

    /// Rôles de modèles effectivement servis.
    #[must_use]
    pub fn roles(&self) -> Vec<ModelRole> {
        let mut roles: Vec<ModelRole> = self.workers.keys().copied().collect();
        roles.sort_by_key(|role| *role as u8);
        roles
    }

    /// Soumet un job au worker du rôle. Erreur si le rôle n'est pas
    /// chargé ou si le worker est mort.
    pub fn submit(&self, role: ModelRole, job: Job) -> Result<(), EngineError> {
        let tx = self
            .workers
            .get(&role)
            .ok_or(EngineError::ModelNotLoaded(role))?;
        tx.send(job).map_err(|_| EngineError::WorkerGone(role))
    }
}

/// Erreurs du moteur (canal vers les handlers HTTP).
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// Aucun modèle chargé pour ce rôle.
    #[error("model role {0:?} is not loaded")]
    ModelNotLoaded(ModelRole),
    /// Le thread worker du rôle est mort.
    #[error("inference worker for {0:?} is gone")]
    WorkerGone(ModelRole),
}

/// Boucle du worker : une session persistante, jobs exécutés en série.
/// La session est créée une fois et réutilisée — le KV-cache garde le
/// préfixe de la requête précédente pour éviter le re-prefill complet.
fn worker_loop(
    pool: Arc<ModelPool>,
    role: ModelRole,
    session_config: SessionConfig,
    rx: std_mpsc::Receiver<Job>,
) {
    tracing::info!(role = ?role, "inference worker started");
    let mut session = match pool.session(role, &session_config) {
        Ok(session) => Some(session),
        Err(err) => {
            tracing::error!(role = ?role, %err, "failed to create inference session");
            None
        }
    };
    while let Ok(job) = rx.recv() {
        run_job(&pool, role, &session_config, &mut session, job);
    }
    tracing::info!(role = ?role, "inference worker stopped");
}

fn run_job<'p>(
    pool: &'p ModelPool,
    role: ModelRole,
    session_config: &SessionConfig,
    session_slot: &mut Option<quokka_core::session::InferenceSession<'p>>,
    job: Job,
) {
    let Job {
        messages,
        config,
        events,
    } = job;

    // Session absente (échec à la création initiale) → on retente par job.
    if session_slot.is_none() {
        match pool.session(role, session_config) {
            Ok(session) => *session_slot = Some(session),
            Err(err) => {
                let _ = events.blocking_send(EngineEvent::Error(err.to_string()));
                return;
            }
        }
    }
    let Some(session) = session_slot.as_mut() else {
        let _ = events.blocking_send(EngineEvent::Error("no inference session".to_owned()));
        return;
    };

    let result = session.chat(&messages, &config, &mut |piece| {
        // Récepteur fermé (client SSE parti) → on ignore, la génération
        // continue pour ne pas gaspiller le KV-cache déjà calculé.
        let _ = events.blocking_send(EngineEvent::Token(piece.to_owned()));
    });

    match result {
        Ok(result) => {
            let _ = events.blocking_send(EngineEvent::Done {
                finish: result.finish,
                prompt_tokens: result.prompt_tokens,
                cached_prompt_tokens: result.cached_prompt_tokens,
                generated_tokens: result.generated_tokens,
                elapsed: result.elapsed,
            });
        }
        Err(err) => {
            // État KV potentiellement incohérent → reset pour la prochaine
            // requête plutôt que de propager la corruption.
            session.reset_kv();
            let _ = events.blocking_send(EngineEvent::Error(err.to_string()));
        }
    }
}
