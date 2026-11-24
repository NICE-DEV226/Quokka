//! Types du protocole OpenAI `/v1/chat/completions` et `/v1/models`.
//!
//! Conversion vers les types `quokka-core` : le rôle `model` de la
//! requête est résolu en [`ModelRole`] par nom (`"flash"`, `"expert"`)
//! ou par nom de fichier GGUF.

use quokka_core::pool::ModelRole;
use quokka_core::session::{ChatMessage, ChatRole, GenerationConfig};
use serde::{Deserialize, Serialize};

/// Requête `POST /v1/chat/completions`.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    /// Modèle demandé : `"flash"`, `"expert"` ou nom de fichier GGUF.
    /// Défaut : `flash`.
    #[serde(default)]
    pub model: Option<String>,
    /// Conversation.
    pub messages: Vec<ApiMessage>,
    /// `true` → SSE ; `false` → JSON complet.
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub seed: Option<u32>,
    /// Speculative decoding par n-grammes. Défaut `true` ; sans effet
    /// hors greedy (`temperature == 0`). Champ non-standard OpenAI.
    #[serde(default)]
    pub speculative: Option<bool>,
}

/// Message au format OpenAI. `content` est une string simple — les
/// contenus multi-modaux (tableaux de parts) sont rejetés en v0.1.
#[derive(Debug, Deserialize)]
pub struct ApiMessage {
    /// `"system"`, `"user"`, `"assistant"` (ou autre → user).
    pub role: String,
    /// Contenu textuel.
    pub content: String,
}

impl ApiMessage {
    /// Convertit en `ChatMessage` du moteur.
    pub fn into_chat(self) -> ChatMessage {
        let role = match self.role.as_str() {
            "system" => ChatRole::System,
            "assistant" => ChatRole::Assistant,
            _ => ChatRole::User,
        };
        ChatMessage {
            role,
            content: self.content,
        }
    }
}

impl ChatCompletionRequest {
    /// Résout le rôle de modèle demandé.
    ///
    /// `None` ou `"flash"` → Flash ; `"expert"` → Expert ; sinon on
    /// compare au nom de fichier GGUF servi (pour les clients qui
    /// énumèrent `/v1/models`).
    pub fn resolve_role(&self, served: &[(&'static str, &str)]) -> Option<ModelRole> {
        let model = self.model.as_deref().unwrap_or("flash");
        match model {
            "flash" => Some(ModelRole::Flash),
            "expert" => Some(ModelRole::Expert),
            other => served
                .iter()
                .find(|(_, filename)| *filename == other)
                .map(|(role, _)| match *role {
                    "expert" => ModelRole::Expert,
                    _ => ModelRole::Flash,
                }),
        }
    }

    /// `GenerationConfig` avec les overrides de la requête.
    pub fn generation_config(&self) -> GenerationConfig {
        let mut config = GenerationConfig::default();
        if let Some(v) = self.max_tokens {
            config.max_tokens = v;
        }
        if let Some(v) = self.temperature {
            config.temperature = v;
        }
        if let Some(v) = self.top_p {
            config.top_p = v;
        }
        if let Some(v) = self.top_k {
            config.top_k = v;
        }
        if let Some(v) = self.seed {
            config.seed = v;
        }
        if let Some(v) = self.speculative {
            config.speculative = v;
        }
        config
    }
}

/// Entrée de `GET /v1/models`.
#[derive(Debug, Serialize)]
pub struct ModelInfo {
    /// Identifiant du modèle ("flash" / "expert").
    pub id: &'static str,
    /// Toujours `"model"` (convention OpenAI).
    pub object: &'static str,
    /// Nom de fichier GGUF sous-jacent.
    pub owned_by: String,
}

/// Réponse `GET /v1/models`.
#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    /// Toujours `"list"`.
    pub object: &'static str,
    /// Modèles servis.
    pub data: Vec<ModelInfo>,
}

/// Chunk SSE `chat.completion.chunk`.
#[derive(Debug, Serialize)]
pub struct StreamChunk {
    /// `"chat.completion.chunk"`.
    pub object: &'static str,
    /// Identifiant unique de la complétion.
    pub id: String,
    /// Timestamp Unix de création.
    pub created: u64,
    /// Modèle ayant répondu.
    pub model: String,
    /// Choix de complétion.
    pub choices: Vec<ChunkChoice>,
}

/// Choix dans un chunk.
#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    /// Toujours 0 (mono-séquence).
    pub index: u32,
    /// Delta partiel.
    pub delta: DeltaContent,
    /// Raison de fin (`null` pendant le stream).
    pub finish_reason: Option<&'static str>,
}

/// Contenu delta.
#[derive(Debug, Serialize)]
pub struct DeltaContent {
    /// Rôle émis une seule fois en premier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    /// Morceau de texte.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// Réponse JSON `chat.completion` (non-stream).
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    /// `"chat.completion"`.
    pub object: &'static str,
    /// Identifiant unique.
    pub id: String,
    /// Timestamp Unix.
    pub created: u64,
    /// Modèle ayant répondu.
    pub model: String,
    /// Choix.
    pub choices: Vec<FullChoice>,
    /// Usage tokens.
    pub usage: Usage,
}

/// Choix dans la réponse complète.
#[derive(Debug, Serialize)]
pub struct FullChoice {
    /// Toujours 0.
    pub index: u32,
    /// Message final.
    pub message: FullMessage,
    /// Raison de fin.
    pub finish_reason: &'static str,
}

/// Message final.
#[derive(Debug, Serialize)]
pub struct FullMessage {
    /// `"assistant"`.
    pub role: &'static str,
    /// Texte complet.
    pub content: String,
}

/// Usage au format OpenAI.
#[derive(Debug, Serialize)]
pub struct Usage {
    /// Tokens du prompt.
    pub prompt_tokens: usize,
    /// Tokens générés.
    pub completion_tokens: usize,
    /// Total.
    pub total_tokens: usize,
    /// Détail OpenAI : tokens du prompt servis par le KV-cache
    /// (préfixe réutilisé entre requêtes).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

/// Détail `prompt_tokens` (sous-ensemble de la spec OpenAI).
#[derive(Debug, Serialize)]
pub struct PromptTokensDetails {
    /// Tokens du prompt dont le KV-cache a été réutilisé.
    pub cached_tokens: usize,
}

/// Mappe `FinishReason` vers le vocabulaire OpenAI.
#[must_use]
pub fn finish_reason_name(finish: quokka_core::session::FinishReason) -> &'static str {
    match finish {
        quokka_core::session::FinishReason::Stop => "stop",
        quokka_core::session::FinishReason::Length
        | quokka_core::session::FinishReason::ContextFull => "length",
    }
}
