//! `quokka-server` — API HTTP OpenAI-compatible au-dessus de
//! `quokka-core` : `/v1/chat/completions` (JSON + SSE), `/v1/models`,
//! `/health`.
//!
//! L'inférence tourne sur des threads dédiés ([`engine::InferenceEngine`])
//! car `LlamaContext` n'est pas `Sync` ; ce crate ne fait que traduire
//! HTTP ↔ événements de génération.

pub mod api;
pub mod engine;
pub mod static_files;
pub mod types;

pub use api::{AppState, ServedModel, router, serve, served_models};
pub use engine::{EngineError, EngineEvent, InferenceEngine, Job};
