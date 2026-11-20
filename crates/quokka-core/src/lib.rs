//! Moteur d'inférence Quokka : bindings `llama.cpp`, chargement `mmap`,
//! orchestration du KV-cache et offload CPU/GPU.
//!
//! Ce crate est la **seule** frontière `unsafe` du workspace (DoD §8.4) :
//! tous les blocs `unsafe` vivent dans les wrappers FFI et portent un
//! commentaire `// SAFETY:` prouvant leur invariant. L'API publique reste
//! sûre — les appelants (`quokka-server`, le CLI) ne voient que des
//! `Result`.
//!
//! # Phase 2 (en cours)
//!
//! Squelette posé : surface d'erreur [`CoreError`]. La `ModelPool`, les
//! sessions KV-cache et le stream de tokens arrivent dans les work
//! packages WP2.2–WP2.4 de `docs/ROADMAP.md`.
//!
//! # Exemple (cible, pas encore implémenté)
//! ```ignore
//! use quokka_core::{ModelPool, InferenceRequest};
//!
//! # fn demo() -> quokka_core::Result<()> {
//! let pool = ModelPool::load_recommended()?;
//! let mut session = pool.session("chat-1")?;
//! let tokens = session.generate(InferenceRequest::chat("Hello")).await?;
//! # Ok(())
//! # }
//! ```

pub mod error;
pub mod pool;
pub mod session;

pub use error::{CoreError, Result};
pub use pool::{LoadedModel, ModelPool, ModelRole, compute_gpu_layers};
pub use session::{
    ChatMessage, ChatResult, ChatRole, FinishReason, GenerationConfig, InferenceSession,
    SessionConfig,
};
