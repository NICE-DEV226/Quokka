//! Session d'inférence : contexte llama.cpp, chat template, KV-cache par
//! session et boucle de génération échantillonnée.
//!
//! Une [`InferenceSession`] emprunte un [`LlamaModel`] de la
//! [`ModelPool`](crate::ModelPool) et possède son [`LlamaContext`]. Le
//! contexte llama.cpp n'est **pas** `Sync` — une session = une requête,
//! exécutée sur un thread dédié (le serveur WP2.4 serialisera l'accès via
//! un sémaphore sur le slot de calcul).
//!
//! Le prompt est encodé via le **chat template embarqué du GGUF**
//! ([`LlamaModel::chat_template`]) — jamais de formatage à la main, qui
//! divergerait par modèle. La préparation du prompt se fait par chunks de
//! `n_batch` avec positions absolues (`LlamaBatch::add`), puis chaque
//! token généré est re-décodé seul avec `logits = true`.

use std::num::NonZeroU32;
use std::time::{Duration, Instant};

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

use crate::error::{CoreError, Result};
use crate::pool::{LoadedModel, ModelPool, ModelRole};

/// Rôle d'un message de conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

/// Message du protocole chat (côté API, avant application du template).
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// Rôle du message.
    pub role: ChatRole,
    /// Contenu textuel.
    pub content: String,
}

/// Paramètres de session (mémoire/contexte).
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Fenêtre de contexte en tokens. `None` → `min(n_ctx_train, 8192)`.
    pub n_ctx: Option<u32>,
    /// Taille max d'un batch de décodage. Défaut 512.
    pub n_batch: u32,
    /// Threads CPU pour le décodage. `None` → parallélisme disponible.
    pub n_threads: Option<i32>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            n_ctx: None,
            n_batch: 512,
            n_threads: None,
        }
    }
}

/// Paramètres d'échantillonnage par génération.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Nombre max de tokens générés.
    pub max_tokens: u32,
    /// Température (0 = greedy).
    pub temperature: f32,
    /// Noyau top-p.
    pub top_p: f32,
    /// Noyau top-k (`<= 0` désactive).
    pub top_k: i32,
    /// Graine du sampler `dist`.
    pub seed: u32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            seed: 0x5EED,
        }
    }
}

/// Raison d'arrêt de la génération.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// Token EOG/EOF rencontré.
    Stop,
    /// `max_tokens` atteint.
    Length,
    /// Fenêtre de contexte pleine en cours de génération.
    ContextFull,
}

/// Résultat d'un appel [`InferenceSession::chat`].
#[derive(Debug)]
pub struct ChatResult {
    /// Texte complet généré.
    pub text: String,
    /// Tokens du prompt encodé.
    pub prompt_tokens: usize,
    /// Tokens générés.
    pub generated_tokens: usize,
    /// Raison d'arrêt.
    pub finish: FinishReason,
    /// Durée totale (prompt + génération).
    pub elapsed: Duration,
}

/// Session d'inférence sur un modèle du pool.
pub struct InferenceSession<'a> {
    model: &'a LlamaModel,
    ctx: LlamaContext<'a>,
    template: LlamaChatTemplate,
    n_ctx: u32,
    n_batch: u32,
}

impl ModelPool {
    /// Ouvre une session d'inférence sur le modèle `role`.
    ///
    /// Le KV-cache est alloué à la création du contexte et vit le temps de
    /// la session. Le contexte n'est pas `Sync` : la session doit être
    /// utilisée depuis un seul thread à la fois.
    pub fn session(&self, role: ModelRole, config: &SessionConfig) -> Result<InferenceSession<'_>> {
        let loaded = self
            .get(role)
            .ok_or_else(|| CoreError::Inference(format!("model role {role:?} not loaded")))?;
        InferenceSession::open(loaded, self.backend(), config)
    }
}

impl<'a> InferenceSession<'a> {
    fn open(
        loaded: &'a LoadedModel,
        backend: &'a LlamaBackend,
        config: &SessionConfig,
    ) -> Result<Self> {
        let model = &loaded.model;
        let n_ctx = config
            .n_ctx
            .unwrap_or_else(|| model.n_ctx_train().min(8192))
            .max(1);
        let n_threads = config.n_threads.unwrap_or_else(|| {
            i32::try_from(
                std::thread::available_parallelism()
                    .map(std::num::NonZero::get)
                    .unwrap_or(4),
            )
            .unwrap_or(i32::MAX)
        });

        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_batch(config.n_batch.max(1))
            .with_n_ubatch(config.n_batch.max(1))
            .with_n_seq_max(1)
            .with_n_threads(n_threads)
            .with_n_threads_batch(n_threads);

        let ctx = model
            .new_context(backend, params)
            .map_err(|err| CoreError::ContextCreate {
                model: loaded.path.display().to_string(),
                reason: err.to_string(),
            })?;

        let template = model
            .chat_template(None)
            .map_err(|err| CoreError::Inference(format!("chat template: {err}")))?;

        Ok(Self {
            model,
            ctx,
            template,
            n_ctx,
            n_batch: config.n_batch.max(1),
        })
    }

    /// Génère une réponse chat. `emit` reçoit chaque pièce de texte UTF-8
    /// valide au fil de la génération (prête pour SSE).
    pub fn chat(
        &mut self,
        messages: &[ChatMessage],
        config: &GenerationConfig,
        emit: &mut dyn FnMut(&str),
    ) -> Result<ChatResult> {
        let started = Instant::now();
        let prompt = self.render_prompt(messages)?;
        let tokens = self.tokenize_prompt(&prompt);
        let prompt_tokens = tokens.len();

        self.decode_prompt(&tokens)?;

        let mut sampler = self.build_sampler(config);
        let mut batch = LlamaBatch::new(self.n_batch as usize, 1);
        let mut out = Utf8PieceBuffer::new();
        let mut text = String::new();
        let mut pos = prompt_tokens as i32;
        let mut generated = 0_u32;
        let mut finish = FinishReason::Length;

        let mut emit_piece = |piece: &str| {
            text.push_str(piece);
            emit(piece);
        };

        loop {
            if generated >= config.max_tokens {
                break;
            }
            if pos >= self.n_ctx as i32 {
                finish = FinishReason::ContextFull;
                break;
            }

            let token = sampler.sample(&self.ctx, -1);
            if self.model.vocab().is_eog(token) {
                finish = FinishReason::Stop;
                break;
            }

            let piece = self.model.vocab().token_to_piece(token, false, None);
            out.push(&piece, &mut emit_piece);
            generated += 1;

            batch.clear();
            batch
                .add(token, pos, &[0], true)
                .map_err(|err| CoreError::Inference(format!("batch add: {err}")))?;
            self.decode(&mut batch)?;
            sampler.accept(token);
            pos += 1;
        }

        out.finish(&mut emit_piece);

        Ok(ChatResult {
            text,
            prompt_tokens,
            generated_tokens: generated as usize,
            finish,
            elapsed: started.elapsed(),
        })
    }

    /// Prompt complet via le chat template GGUF (`add_ass = true` ajoute
    /// l'en-tête de tour assistant).
    fn render_prompt(&self, messages: &[ChatMessage]) -> Result<String> {
        let chat: Vec<LlamaChatMessage> = messages
            .iter()
            .map(|message| {
                let role = match message.role {
                    ChatRole::System => "system",
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                };
                LlamaChatMessage::new(role.to_owned(), message.content.clone())
            })
            .collect::<std::result::Result<_, _>>()
            .map_err(|err| CoreError::Inference(format!("chat message: {err}")))?;

        self.model
            .apply_chat_template(&self.template, &chat, true)
            .map_err(|err| CoreError::Inference(format!("chat template: {err}")))
    }

    fn tokenize_prompt(&self, prompt: &str) -> Vec<LlamaToken> {
        let vocab = self.model.vocab();
        vocab.tokenize(prompt.as_bytes(), vocab.should_add_bos(), false)
    }

    /// Décode le prompt par chunks de `n_batch`, positions absolues.
    /// Seul le **dernier** token du prompt porte `logits = true`.
    fn decode_prompt(&mut self, tokens: &[LlamaToken]) -> Result<()> {
        let mut batch = LlamaBatch::new(self.n_batch as usize, 1);
        let mut pos = 0_i32;
        let last_global = tokens.len() as i32 - 1;
        for chunk in tokens.chunks(self.n_batch as usize) {
            batch.clear();
            for (i, &token) in chunk.iter().enumerate() {
                let global = pos + i as i32;
                batch
                    .add(token, global, &[0], global == last_global)
                    .map_err(|err| CoreError::Inference(format!("prompt batch: {err}")))?;
            }
            self.decode(&mut batch)?;
            pos += chunk.len() as i32;
        }
        Ok(())
    }

    fn decode(&mut self, batch: &mut LlamaBatch<'_>) -> Result<()> {
        self.ctx
            .decode(batch)
            .map_err(|err| CoreError::Inference(format!("decode: {err}")))
    }

    fn build_sampler(&self, config: &GenerationConfig) -> LlamaSampler {
        let mut samplers = Vec::with_capacity(4);
        if config.top_k > 0 {
            samplers.push(LlamaSampler::top_k(config.top_k));
        }
        samplers.push(LlamaSampler::top_p(config.top_p, 1));
        if config.temperature > 0.0 {
            samplers.push(LlamaSampler::temp(config.temperature));
        }
        samplers.push(LlamaSampler::dist(config.seed));
        LlamaSampler::chain_simple(samplers)
    }
}

/// Accumulate les octets des pièces de tokens et n'émet que des préfixes
/// UTF-8 complets — un token peut couper un caractère multi-octets.
/// `emit` est appelé tant que `self` est mutement emprunté, ce qui permet
/// au caller de cumuler le texte sans copie intermédiaire.
struct Utf8PieceBuffer {
    pending: Vec<u8>,
}

impl Utf8PieceBuffer {
    fn new() -> Self {
        Self {
            pending: Vec::with_capacity(64),
        }
    }

    fn push(&mut self, piece: &[u8], emit: &mut dyn FnMut(&str)) {
        self.pending.extend_from_slice(piece);
        let mut valid = utf8_valid_up_to(&self.pending);
        // Un octet invalide en tête (pas juste une fin de séquence
        // incomplète) bloquerait le buffer à jamais : on le saute, la
        // reconstruction se fait en `finish` en lossy.
        if valid == 0
            && !self.pending.is_empty()
            && let Err(err) = std::str::from_utf8(&self.pending)
            && err.error_len().is_some()
        {
            self.pending.remove(0);
            valid = utf8_valid_up_to(&self.pending);
        }
        if valid == 0 {
            return;
        }
        let piece_str = std::str::from_utf8(&self.pending[..valid]).unwrap_or("");
        emit(piece_str);
        self.pending.drain(..valid);
    }

    /// Vide le reliquat en fin de génération (séquence incomplète ou
    /// invalide → lossy assumé, borné à la fin du flux).
    fn finish(&mut self, emit: &mut dyn FnMut(&str)) {
        if self.pending.is_empty() {
            return;
        }
        let rest = String::from_utf8_lossy(&self.pending);
        emit(&rest);
        self.pending.clear();
    }
}

/// Longueur du plus long préfixe UTF-8 valide de `bytes`.
fn utf8_valid_up_to(bytes: &[u8]) -> usize {
    match std::str::from_utf8(bytes) {
        Ok(_) => bytes.len(),
        Err(err) => err.valid_up_to(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_pieces(pieces: &[&[u8]]) -> (Vec<String>, Vec<String>) {
        let mut buf = Utf8PieceBuffer::new();
        let mut emitted = Vec::new();
        for piece in pieces {
            buf.push(piece, &mut |s| emitted.push(s.to_owned()));
        }
        buf.finish(&mut |s| emitted.push(s.to_owned()));
        (emitted, vec![])
    }

    #[test]
    fn utf8_buffer_emits_only_complete_sequences() {
        // "héllo" en bytes : 'é' = 0xC3 0xA9 peut être coupé entre tokens.
        let h = b"h";
        let e1 = [0xC3];
        let e2 = [0xA9];
        let rest = "llo".as_bytes();
        let (emitted, _) = run_pieces(&[h, &e1, &e2, rest]);
        assert_eq!(emitted.concat(), "héllo");
    }

    #[test]
    fn utf8_buffer_handles_empty_and_ascii() {
        let (emitted, _) = run_pieces(&[b"Hello ", b"world"]);
        assert_eq!(emitted.concat(), "Hello world");
        let (emitted, _) = run_pieces(&[]);
        assert!(emitted.is_empty());
    }

    #[test]
    fn utf8_buffer_drops_invalid_leading_byte() {
        let (emitted, _) = run_pieces(&[&[0xFF], b"ok"]);
        assert_eq!(emitted.concat(), "ok");
    }

    #[test]
    fn generation_config_defaults_are_sane() {
        let config = GenerationConfig::default();
        assert!(config.max_tokens > 0);
        assert!(config.temperature > 0.0 && config.temperature <= 2.0);
        assert!(config.top_p > 0.0 && config.top_p <= 1.0);
    }
}
