//! Moteur d'inférence : un thread worker par rôle de modèle.
//!
//! `LlamaContext` n'est pas `Sync` et `ModelPool` n'est pas `Clone` : le
//! pool est partagé en `Arc` entre les workers, et chaque worker exécute
//! les générations de **son** rôle en série sur son propre thread std.
//! Les résultats sont streamés vers le handler axum via un
//! `tokio::sync::mpsc` émetteur glissé dans le job — `blocking_send`
//! fonctionne depuis un thread non-async.
//!
//! Une **session est recréée par requête** : réutiliser un contexte avec
//! un prompt plus court laisserait des cellules KV stale aux positions
//! intermédiaires, que llama.cpp prendrait en compte à la génération
//! suivante. Le coût d'allocation du KV-cache est assumé en v0.1 ;
//! le cache de sessions (LRU + sérialisation) est un chantier Phase 3.

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

/// Boucle du worker : une session par job, events streamés.
fn worker_loop(
    pool: Arc<ModelPool>,
    role: ModelRole,
    session_config: SessionConfig,
    rx: std_mpsc::Receiver<Job>,
) {
    tracing::info!(role = ?role, "inference worker started");
    while let Ok(job) = rx.recv() {
        run_job(&pool, role, &session_config, job);
    }
    tracing::info!(role = ?role, "inference worker stopped");
}

fn run_job(pool: &ModelPool, role: ModelRole, session_config: &SessionConfig, job: Job) {
    let Job {
        messages,
        config,
        events,
    } = job;

    let mut session = match pool.session(role, session_config) {
        Ok(session) => session,
        Err(err) => {
            let _ = events.blocking_send(EngineEvent::Error(err.to_string()));
            return;
        }
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
                generated_tokens: result.generated_tokens,
                elapsed: result.elapsed,
            });
        }
        Err(err) => {
            let _ = events.blocking_send(EngineEvent::Error(err.to_string()));
        }
    }
}
