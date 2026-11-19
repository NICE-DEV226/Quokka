//! Erreurs de `quokka-core` (`thiserror`, DoD §8.1 : zéro panic).

use std::path::PathBuf;

use thiserror::Error;

/// Surface d'erreur du moteur d'inférence.
#[derive(Debug, Error)]
pub enum CoreError {
    /// Le backend llama.cpp n'a pas pu être initialisé.
    #[error("llama.cpp backend init failed: {0}")]
    BackendInit(String),

    /// Chargement d'un fichier GGUF impossible (absent, corrompu, version
    /// de format non supportée).
    #[error("cannot load model `{path}`: {reason}")]
    ModelLoad {
        /// Chemin du GGUF.
        path: PathBuf,
        /// Cause lisible.
        reason: String,
    },

    /// La création du contexte d'inférence a échoué (mémoire insuffisante,
    /// contexte trop grand pour la VRAM/RAM disponible).
    #[error("cannot create inference context for `{model}`: {reason}")]
    ContextCreate {
        /// Nom du modèle.
        model: String,
        /// Cause lisible.
        reason: String,
    },

    /// Erreur pendant la génération (tokenisation, sampling, annulation).
    #[error("inference failed: {0}")]
    Inference(String),

    /// Une session demandée n'existe pas (ou a été évincée du KV-cache).
    #[error("unknown session `{0}`")]
    UnknownSession(String),
}

/// Alias de commodité utilisé dans tout le crate.
pub type Result<T> = std::result::Result<T, CoreError>;
