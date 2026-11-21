//! `quokka mcp` — serveur MCP (stdio) proxyfiant l'API HTTP d'une
//! instance `quokka run` en cours. Les outils exposés :
//!
//! - `quokka_chat` : complétion chat (modèle `auto` par défaut →
//!   routage Flash/Expert côté serveur) ;
//! - `quokka_models` : liste des modèles servis ;
//! - `quokka_route` : dry-run du routeur (`POST /v1/route`) — quelle
//!   décision prendrait Quokka pour ce prompt, sans inférence.
//!
//! Le proxy évite de charger les poids GGUF une seconde fois : le
//! processus MCP reste léger et partage le KV-cache du serveur.

use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock, ErrorData, ServerCapabilities, ServerConfig};
use rmcp::{ServerHandler, ServiceExt, schemars, tool, tool_handler, tool_router};
use serde::Deserialize;

/// Outils MCP exposés par `quokka mcp`.
#[derive(Clone)]
pub struct QuokkaMcp {
    /// Base URL de l'API (`http://127.0.0.1:11435` par défaut).
    base: String,
    client: reqwest::Client,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ChatParams {
    /// Message utilisateur à envoyer au modèle.
    prompt: String,
    /// `"auto"` (défaut, le routeur décide), `"flash"`, `"expert"` ou
    /// nom de fichier GGUF.
    #[serde(default)]
    model: Option<String>,
    /// Nombre max de tokens générés. Défaut : celui du serveur.
    #[serde(default)]
    max_tokens: Option<u32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct PromptParams {
    /// Prompt à classifier.
    prompt: String,
}

impl QuokkaMcp {
    /// Construit le service MCP pointant `base`.
    #[must_use]
    pub fn new(base: String) -> Self {
        Self {
            base: base.trim_end_matches('/').to_owned(),
            client: reqwest::Client::new(),
        }
    }

    /// Lance le serveur MCP sur stdio. Retourne quand le client coupe.
    pub async fn serve_stdio(self) -> anyhow::Result<()> {
        let service = self
            .serve(rmcp::transport::stdio())
            .await
            .map_err(|err| anyhow::anyhow!("mcp serve: {err}"))?;
        service
            .waiting()
            .await
            .map_err(|err| anyhow::anyhow!("mcp shutdown: {err}"))?;
        Ok(())
    }

    async fn post_json(
        &self,
        path: &str,
        body: serde_json::Value,
    ) -> Result<serde_json::Value, ErrorData> {
        let resp = self
            .client
            .post(format!("{}{path}", self.base))
            .json(&body)
            .send()
            .await
            .map_err(|err| ErrorData::internal_error(err.to_string(), None))?;
        let status = resp.status();
        let value: serde_json::Value = resp
            .json()
            .await
            .map_err(|err| ErrorData::internal_error(err.to_string(), None))?;
        if !status.is_success() {
            let message = value
                .pointer("/error/message")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("quokka request failed")
                .to_owned();
            return Err(ErrorData::internal_error(
                format!("{status}: {message}"),
                None,
            ));
        }
        Ok(value)
    }
}

#[tool_router]
impl QuokkaMcp {
    #[tool(
        name = "quokka_chat",
        description = "Send a prompt to the local Quokka model (auto-routes between Flash and Expert unless a model is specified). Runs 100% locally via the quokka server."
    )]
    async fn chat(
        &self,
        Parameters(params): Parameters<ChatParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let mut body = serde_json::json!({
            "model": params.model.unwrap_or_else(|| "auto".to_owned()),
            "messages": [{ "role": "user", "content": params.prompt }],
            "stream": false,
        });
        if let Some(max) = params.max_tokens {
            body["max_tokens"] = serde_json::json!(max);
        }
        let value = self.post_json("/v1/chat/completions", body).await?;
        let content = value
            .pointer("/choices/0/message/content")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let usage = value.get("usage").cloned().unwrap_or_default();
        let model = value
            .pointer("/x-quokka-model-used")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned);
        let text = match model {
            Some(m) => format!("[{m}]\n{content}\n\nusage: {usage}"),
            None => format!("{content}\n\nusage: {usage}"),
        };
        Ok(CallToolResult::success(vec![ContentBlock::text(text)]))
    }

    #[tool(
        name = "quokka_models",
        description = "List the models served by the local Quokka instance."
    )]
    async fn models(&self) -> Result<CallToolResult, ErrorData> {
        let resp = self
            .client
            .get(format!("{}/v1/models", self.base))
            .send()
            .await
            .map_err(|err| ErrorData::internal_error(err.to_string(), None))?;
        let value: serde_json::Value = resp
            .json()
            .await
            .map_err(|err| ErrorData::internal_error(err.to_string(), None))?;
        let ids: Vec<&str> = value
            .get("data")
            .and_then(serde_json::Value::as_array)
            .map(|arr| arr.iter().filter_map(|m| m.get("id")?.as_str()).collect())
            .unwrap_or_default();
        Ok(CallToolResult::success(vec![ContentBlock::text(
            ids.join("\n"),
        )]))
    }

    #[tool(
        name = "quokka_route",
        description = "Dry-run the Quokka router: returns which model (flash/expert) would handle a prompt and why, without running inference. Useful to understand routing decisions."
    )]
    async fn route(
        &self,
        Parameters(params): Parameters<PromptParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let value = self
            .post_json(
                "/v1/route",
                serde_json::json!({
                    "model": "auto",
                    "messages": [{ "role": "user", "content": params.prompt }],
                }),
            )
            .await?;
        Ok(CallToolResult::success(vec![ContentBlock::text(
            serde_json::to_string_pretty(&value)
                .map_err(|err| ErrorData::internal_error(err.to_string(), None))?,
        )]))
    }
}

#[tool_handler]
impl ServerHandler for QuokkaMcp {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(rmcp::model::Implementation::new(
                "quokka-mcp",
                env!("CARGO_PKG_VERSION"),
            ))
            .with_instructions(
                "Local AI tools backed by a running `quokka run` server. \
                 quokka_chat runs prompts fully offline (auto Flash/Expert routing); \
                 quokka_route explains routing decisions; quokka_models lists models.",
            )
    }
}

/// Point d'entrée `quokka mcp [--url ...]`.
pub async fn run(base: String) -> anyhow::Result<()> {
    QuokkaMcp::new(base).serve_stdio().await
}
