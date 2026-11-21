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
use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams};
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
    /// Répertoire de persistance KV (`~/.quokka/cache/kv`). `None` →
    /// pas d'offload : le KV vit uniquement en RAM le temps de la
    /// session. Quand défini, un changement de conversation sauvegarde
    /// l'état KV compressé LZ4 et tente de restaurer celui de la
    /// conversation cible.
    pub kv_cache_dir: Option<std::path::PathBuf>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            n_ctx: None,
            n_batch: 512,
            n_threads: None,
            kv_cache_dir: None,
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
    /// Tokens du prompt dont le KV-cache a été réutilisé (préfixe commun
    /// avec la requête précédente sur ce contexte).
    pub cached_prompt_tokens: usize,
    /// Tokens générés.
    pub generated_tokens: usize,
    /// Raison d'arrêt.
    pub finish: FinishReason,
    /// Durée totale (prompt + génération).
    pub elapsed: Duration,
}

/// Session d'inférence sur un modèle du pool.
///
/// Le contexte est **persistant** : le KV-cache survit entre les appels
/// [`chat`](Self::chat) et la séquence de tokens précédente est conservée
/// pour réutiliser le préfixe commun (comme le prompt cache d'llama-server).
/// Une session = un thread à la fois (contexte non `Sync`).
pub struct InferenceSession<'a> {
    model: &'a LlamaModel,
    ctx: LlamaContext<'a>,
    template: LlamaChatTemplate,
    n_ctx: u32,
    n_batch: u32,
    /// Séquence complète (prompt + tokens générés) de la dernière requête
    /// successfully décodée dans ce contexte — base du préfixe commun.
    prev_tokens: Vec<LlamaToken>,
    /// Clé de la conversation courante (hash du premier message user).
    session_key: Option<u64>,
    /// Tag modèle pour nommer les fichiers KV (`<tag>_<key>.kv.lz4`).
    model_tag: String,
    /// Répertoire d'offload KV, `None` si désactivé.
    cache_dir: Option<std::path::PathBuf>,
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
            .with_n_threads_batch(n_threads)
            // Flash attention : moins de bande passante sur le KV-cache,
            // gain net sur les contextes longs — sans perte de qualité.
            .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED)
            // KV-cache quantifié q8_0 : ~2× moins de RAM qu'en f16 et
            // souvent plus rapide (decode est memory-bound).
            .with_type_k(KvCacheType::Q8_0)
            .with_type_v(KvCacheType::Q8_0);

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
            prev_tokens: Vec::new(),
            session_key: None,
            model_tag: model_tag(&loaded.path),
            cache_dir: config.kv_cache_dir.clone(),
        })
    }

    /// Vide le KV-cache et la mémoire de préfixe — à appeler après une
    /// erreur de décodage pour repartir d'un état sain.
    pub fn reset_kv(&mut self) {
        self.ctx.clear_kv_cache();
        self.prev_tokens.clear();
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

        if prompt_tokens as u32 > self.n_ctx {
            return Err(CoreError::Inference(format!(
                "prompt ({prompt_tokens} tokens) exceeds context window ({})",
                self.n_ctx
            )));
        }

        // Changement de conversation : sauvegarde le KV de l'ancienne
        // session sur disque (LZ4) puis tente de restaurer celui de la
        // nouvelle. Best-effort — un échec laisse juste un cache froid.
        if self.cache_dir.is_some() {
            let key = session_key(messages);
            if self.session_key != Some(key) {
                self.switch_session(key, &tokens);
            }
        }

        // Préfixe commun avec la séquence précédente : on ne garde dans le
        // KV-cache que `keep` tokens et on ne décode que le suffixe.
        // `keep` est borné à prompt_tokens-1 pour garantir au moins un
        // token fraîchement décodé → logits frais pour le premier sample.
        let keep =
            common_prefix_len(&self.prev_tokens, &tokens).min(prompt_tokens.saturating_sub(1));
        self.truncate_kv(keep as u32)?;
        self.decode_prompt(&tokens[keep..], keep as i32)?;

        let mut sampler = self.build_sampler(config);
        let mut batch = LlamaBatch::new(self.n_batch as usize, 1);
        let mut out = Utf8PieceBuffer::new();
        let mut text = String::new();
        let mut pos = prompt_tokens as i32;
        let mut generated = 0_u32;
        let mut finish = FinishReason::Length;
        let mut generated_ids = Vec::with_capacity(config.max_tokens as usize);

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
            generated_ids.push(token);

            batch.clear();
            batch
                .add(token, pos, &[0], true)
                .map_err(|err| CoreError::Inference(format!("batch add: {err}")))?;
            self.decode(&mut batch)?;
            sampler.accept(token);
            pos += 1;
        }

        out.finish(&mut emit_piece);

        // Mémoire de préfixe pour la prochaine requête : prompt complet +
        // tokens générés (l'EOG n'est jamais décodé dans le KV, on
        // l'exclut donc aussi ici).
        self.prev_tokens.clear();
        self.prev_tokens.extend_from_slice(&tokens);
        self.prev_tokens.extend_from_slice(&generated_ids);

        Ok(ChatResult {
            text,
            prompt_tokens,
            cached_prompt_tokens: keep,
            generated_tokens: generated as usize,
            finish,
            elapsed: started.elapsed(),
        })
    }

    /// Supprime du KV-cache toutes les cellules à partir de la position
    /// `keep` (séquence 0). `keep = 0` repart de zéro.
    fn truncate_kv(&mut self, keep: u32) -> Result<()> {
        self.ctx
            .kv_cache_seq_rm(0, Some(keep), None)
            .map_err(|err| CoreError::Inference(format!("kv truncate: {err}")))?;
        Ok(())
    }

    /// Change de conversation : persiste le KV de l'ancienne session,
    /// puis tente de restaurer celui de la nouvelle depuis le disque.
    /// Toute erreur est dégradée en cache froid — jamais fatale au chat.
    fn switch_session(&mut self, key: u64, tokens: &[LlamaToken]) {
        if let Some(old) = self.session_key.take() {
            self.save_state(old);
        }
        self.ctx.clear_kv_cache();
        self.prev_tokens.clear();
        self.session_key = Some(key);
        if let Some(loaded) = self.load_state(key, tokens) {
            self.prev_tokens = loaded;
        }
    }

    /// Écrit l'état KV courant (séquence 0, `prev_tokens`) dans
    /// `<dir>/<tag>_<key:016x>.kv.lz4` via `state_save_file` + LZ4.
    fn save_state(&self, key: u64) {
        let Some(dir) = &self.cache_dir else { return };
        // En dessous de ~16 tokens le gain d'une restauration est nul.
        if self.prev_tokens.len() < 16 {
            return;
        }
        if let Err(err) = std::fs::create_dir_all(dir) {
            tracing::warn!("kv cache dir: {err}");
            return;
        }
        let name = format!("{}_{}.kv", self.model_tag, key);
        let raw = dir.join(format!(".{name}.tmp"));
        let final_path = dir.join(format!("{name}.lz4"));
        let saved: std::result::Result<(), String> = (|| {
            self.ctx
                .state_save_file(&raw, &self.prev_tokens)
                .map_err(|err| err.to_string())?;
            let bytes = std::fs::read(&raw).map_err(|err| err.to_string())?;
            std::fs::write(&final_path, lz4_flex::compress_prepend_size(&bytes))
                .map_err(|err| err.to_string())?;
            std::fs::remove_file(&raw).map_err(|err| err.to_string())
        })();
        match saved {
            Ok(()) => self.prune_cache(),
            Err(err) => {
                tracing::warn!("kv state save failed: {err}");
                let _ = std::fs::remove_file(&raw);
            }
        }
    }

    /// Charge `<tag>_<key>.kv.lz4`, restaure le KV et retourne les tokens
    /// **uniquement si** ils forment un préfixe exact du prompt courant —
    /// sinon le KV est vidé et `None` est retourné (cache froid sain).
    fn load_state(&mut self, key: u64, tokens: &[LlamaToken]) -> Option<Vec<LlamaToken>> {
        let dir = self.cache_dir.as_ref()?;
        let path = dir.join(format!("{}_{}.kv.lz4", self.model_tag, key));
        let compressed = std::fs::read(&path).ok()?;
        let raw_bytes = lz4_flex::decompress_size_prepended(&compressed).ok()?;
        let raw = dir.join(format!(".{}.tmp", path.file_name()?.to_str()?));
        if std::fs::write(&raw, &raw_bytes).is_err() {
            return None;
        }
        let loaded = self
            .ctx
            .state_load_file(&raw, self.n_ctx as usize)
            .map_err(|err| {
                tracing::warn!("kv state load failed: {err}");
                err
            })
            .ok();
        let _ = std::fs::remove_file(&raw);
        let loaded = loaded?;
        if loaded.len() <= tokens.len() && tokens.starts_with(&loaded) {
            Some(loaded)
        } else {
            tracing::debug!("kv state rejected: token prefix mismatch");
            self.ctx.clear_kv_cache();
            None
        }
    }

    /// Garde au plus `MAX_KV_CACHE` fichiers par modèle, en supprimant
    /// les plus anciens par mtime. `shortcut: récence par mtime de
    /// sauvegarde seulement — un fichier chargé mais pas re-sauvegardé
    /// peut être évincté à chaud ; passe au `filetime` si ça arrive.`
    fn prune_cache(&self) {
        const MAX_KV_CACHE: usize = 8;
        let Some(dir) = &self.cache_dir else { return };
        let prefix = format!("{}_.kv.lz4", self.model_tag);
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let mut files: Vec<(std::time::SystemTime, std::path::PathBuf)> = entries
            .filter_map(std::result::Result::ok)
            .filter(|e| {
                e.file_name()
                    .to_str()
                    .is_some_and(|n| n.starts_with(&prefix))
            })
            .filter_map(|e| {
                let meta = e.metadata().ok()?;
                Some((meta.modified().ok()?, e.path()))
            })
            .collect();
        if files.len() <= MAX_KV_CACHE {
            return;
        }
        let excess = files.len().saturating_sub(MAX_KV_CACHE);
        files.sort_by_key(|(mtime, _)| *mtime);
        for (_, path) in files.into_iter().take(excess) {
            let _ = std::fs::remove_file(path);
        }
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
        // `parse_special = true` est obligatoire : le chat template produit
        // des marqueurs `<|start_header_id|>`… qui doivent devenir de
        // vrais tokens spéciaux, pas du texte littéral (sinon le modèle
        // "répond" en régénérant le template — observé en e2e).
        vocab.tokenize(prompt.as_bytes(), vocab.should_add_bos(), true)
    }

    /// Décode `tokens` (sous-séquence du prompt) en partant de la position
    /// absolue `start_pos`, par chunks de `n_batch`. Seul le **dernier**
    /// token du prompt porte `logits = true`.
    fn decode_prompt(&mut self, tokens: &[LlamaToken], start_pos: i32) -> Result<()> {
        let mut batch = LlamaBatch::new(self.n_batch as usize, 1);
        let last_global = start_pos + tokens.len() as i32 - 1;
        for (offset, chunk) in tokens.chunks(self.n_batch as usize).enumerate() {
            batch.clear();
            for (i, &token) in chunk.iter().enumerate() {
                let global = start_pos + (offset * self.n_batch as usize + i) as i32;
                batch
                    .add(token, global, &[0], global == last_global)
                    .map_err(|err| CoreError::Inference(format!("prompt batch: {err}")))?;
            }
            self.decode(&mut batch)?;
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

/// Identité d'une conversation : FNV-1a du contenu du premier message
/// user (ou du premier message à défaut). Deux conversations avec le
/// même premier message partagent un slot — la vérification par préfixe
/// de tokens à la restauration rend la collision inoffensive.
fn session_key(messages: &[ChatMessage]) -> u64 {
    let first = messages
        .iter()
        .find(|m| m.role == ChatRole::User && !m.content.is_empty())
        .or_else(|| messages.first());
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in first.map_or("".as_bytes(), |m| m.content.as_bytes()) {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Nom de fichier sûr pour le tag modèle (stem du GGUF, alnum + `-_.`).
fn model_tag(path: &std::path::Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Longueur du plus long préfixe commun entre deux séquences de tokens.
fn common_prefix_len(a: &[LlamaToken], b: &[LlamaToken]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
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
    fn session_key_uses_first_user_message() {
        let msgs = |content: &str| {
            vec![ChatMessage {
                role: ChatRole::User,
                content: content.to_owned(),
            }]
        };
        assert_eq!(session_key(&msgs("salut")), session_key(&msgs("salut")));
        assert_ne!(session_key(&msgs("salut")), session_key(&msgs("bye")));
        // Message system seul → fallback sur le premier message.
        let sys = vec![ChatMessage {
            role: ChatRole::System,
            content: "sys".to_owned(),
        }];
        assert_ne!(session_key(&sys), session_key(&msgs("salut")));
    }

    #[test]
    fn model_tag_is_filename_safe() {
        assert_eq!(
            model_tag(std::path::Path::new(
                "/m/Meta-Llama-3.2-3B-Instruct.Q4_K_M.gguf"
            )),
            "Meta-Llama-3.2-3B-Instruct.Q4_K_M"
        );
        assert_eq!(model_tag(std::path::Path::new("/m/a b/café.gguf")), "caf_");
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
