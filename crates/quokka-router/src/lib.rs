//! Routeur sémantique Flash/Expert — pipeline à deux étages, budget < 2 ms.
//!
//! Voir `docs/adr/0002-router-classifier-stack.md` et `docs/ARCHITECTURE.md`
//! §4.2. Le classifieur est **lexical et déterministe** : sac de mots +
//! bigrammes hashés vers des centroïdes d'ancres, similarité cosinus. Aucun
//! modèle, aucun réseau, aucune allocation lourde — la latence est bornée
//! par construction et vérifiable en test.
//!
//! Écart assumé avec l'ADR : le Stage 2 y prévoyait `all-MiniLM-L6-v2` via
//! `ort` (ONNX Runtime). Le classifieur centroïde est livré d'abord parce
//! qu'il est sans dépendance native et mesurable dès maintenant ; le
//! backend ONNX pourra remplacer [`Router::classify`] sans toucher au
//! pipeline (Stage 1, garde mémoire, API de décision restent identiques).

use std::time::Instant;

/// Cible de routage — miroir de `ModelRole` côté serveur (le crate reste
/// sans dépendance vers `quokka-core` pour rester léger).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteTarget {
    /// Modèle rapide (1B–3B), résident.
    Flash,
    /// Modèle costaud (7B+), raisonnement/code.
    Expert,
}

/// Raison du routage — exposée en log et dans les en-têtes de réponse.
#[derive(Debug, Clone, PartialEq)]
pub enum RouteReason {
    /// Override explicite `@flash` / `@expert` dans le prompt.
    Override,
    /// Prompt court sans marqueur de complexité (< ~25 tokens).
    ShortPrompt,
    /// Fence de code ou mot-clé de programmation détecté.
    CodeDetected,
    /// Terme de domaine expert (systèmes distribués, crypto, maths
    /// formelles…) détecté au Stage 1.
    DomainDetected,
    /// Décision du classifieur sémantique (scores cosinus).
    Semantic { expert_score: f32, flash_score: f32 },
    /// L'expert a été choisi mais la RAM disponible est insuffisante —
    /// repli sur Flash plutôt que risquer un freeze (Memory Pressure Guard).
    MemoryPressureFallback { needed_mb: u32, available_mb: u32 },
    /// Règle `[[rules]]` de `~/.quokka/routing.toml`.
    Rule { name: String },
}

/// Décision de routage complète.
#[derive(Debug, Clone)]
pub struct RouteDecision {
    /// Modèle cible.
    pub target: RouteTarget,
    /// Pourquoi ce choix.
    pub reason: RouteReason,
    /// Durée totale du routage (doit rester < 2 ms).
    pub elapsed: std::time::Duration,
}

/// Contexte système pour la garde mémoire.
#[derive(Debug, Clone, Copy)]
pub struct RouteContext {
    /// RAM disponible en Mio (sonde live, pas le profil de boot).
    pub available_ram_mb: Option<u32>,
    /// Taille du modèle Expert en Mio (poids mmap + marge de travail).
    pub expert_size_mb: u32,
}

impl RouteContext {
    /// Sonde la RAM disponible via sysinfo (~100 µs, pas de scan processus).
    #[must_use]
    pub fn system(expert_size_mb: u32) -> Self {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        Self {
            available_ram_mb: u32::try_from(sys.available_memory() / (1024 * 1024)).ok(),
            expert_size_mb,
        }
    }
}

/// Marge de sécurité au-delà de la taille de l'expert (spec §4.2 :
/// `expert_model_size + 1500 MB`).
const RAM_SAFETY_BUFFER_MB: u32 = 1536;

/// Budget de latence du routeur (spec : < 2 ms par requête).
pub const ROUTE_BUDGET: std::time::Duration = std::time::Duration::from_millis(2);

/// Dimension du vecteur de features (unigrammes + bigrammes hashés).
const DIM: usize = 512;

/// Mots-clés forçant le court-circuit Expert au Stage 1.
const CODE_MARKERS: &[&str] = &[
    "```", "fn ", "impl ", "struct ", "enum ", "trait ", "def ", "class ", "SELECT ",
    "INSERT ", "UPDATE ", "DELETE FROM", "JOIN ", "async ", "await ", "=>", "nullptr",
    "segmentation fault", "stack overflow", "compile error", "borrow checker",
    "type error", "undefined reference", "regex", "sql query", "api endpoint",
    "git rebase", "docker compose", "kubernetes", "http status",
];

/// Termes techniques non ambigus matchés sur frontières de mot
/// (`has_word`) — évite « improve »→« prove », « draft »→« raft ».
const CODE_WORDS: &[&str] = &[
    "binary search", "search tree", "lru cache", "memory leak", "unit test",
    "unit tests", "integration test", "stack trace", "race condition",
    "deadlock", "mutex", "rwlock", "null pointer", "segfault", "backpressure",
    "nested loop", "producer consumer", "git hook", "cli tool", "yaml",
    "sql", "nosql", "b-tree", "b tree", "b trees", "b-tree indexing",
    "database indexing", "typescript", "javascript", "golang", "kotlin",
    "swift", "c++", "c#", "objective-c", "php", "ruby", "scala", "haskell",
    "lua", "perl", "bash", "powershell", "docker", "regex",
];

/// Langages ambigus seuls (« rust » oxydation, « python » serpent,
/// « java » île) : ne déclenchent Expert qu'avec un contexte code.
const LANGUAGES: &[&str] = &[
    "rust", "python", "javascript", "typescript", "golang", "kotlin",
    "swift", "c++", "c#", "php", "ruby", "scala", "haskell", "lua",
    "perl", "bash", "powershell", "objective-c", "sql",
];

/// Contexte qui lève l'ambiguïté d'un nom de langage.
const CODE_CONTEXT: &[&str] = &[
    "function", "code", "program", "script", "error", "debug", "implement",
    "refactor", "compile", "exception", "variable", "import ", "query",
    "bug", "leak", "syntax", "runtime", "api", "endpoint", "test",
];

/// Marqueurs de domaine expert : termes qui signalent une charge de
/// conception, de preuve ou de analyse profonde — au-delà de la simple
/// syntaxe de code. Matchés avec frontière **début** (`has_word_start`)
/// pour conserver le stemming (« postgres »→« postgresql ») sans
/// sous-chaînes parasites.
const EXPERT_DOMAIN_MARKERS: &[&str] = &[
    // Systèmes distribués & infra
    "distributed", "microservice", "shard", "consistent hashing", "consensus",
    "raft", "paxos", "kafka", "rabbitmq", "event streaming", "cdn", "failover",
    "disaster recovery", "saga", "exactly once", "eventual consistency",
    "strong consistency", "rate limiter", "load balanc", "service mesh",
    "kubernetes", "terraform", "postgres", "database schema", "migration",
    "observability", "prometheus", "grafana", "feature store", "url shortener",
    // Crypto & sécurité
    "oauth", "tls ", "ssl ", "encryption", "cryptograph", "zero knowledge",
    "vulnerabilit", "injection", "intrusion", "exfiltration", "penetration",
    "authentication flow", "authorization", "certificate", "xss", "csrf",
    // Algo, maths & ML
    "proof", "prove", "theorem", "induction", "recurrence", "combinatoric",
    "big o", "amortized", "asymptot", "np-complete", "eigenvalue", "matrix",
    "integral", "derivative", "closed form", "bayes", "probability distribution",
    "expected value", "variance", "overfitting", "bias variance", "gradient descent",
    "convergence", "transformer", "attention mechanism", "positional encoding",
    "complexity", "algorithm", "sorting", "recursion", "dynamic programming",
    // Raisonnement & analyse
    "analyze ", "analyse ", "tradeoff", "trade-off", "architecture",
    "system design", "bottleneck", "scalability", "throughput", "latency",
];

/// Marqueurs de raisonnement qui empêchent le court-circuit « prompt
/// court » — le Stage 2 décide ensuite.
const REASONING_MARKERS: &[&str] = &[
    "step by step", "prove ", "proof", "derive ", "calculate ", "theorem",
    "logic puzzle", "pigeonhole", "induction", "recurrence", "complexity",
    "analyze ", "evaluate ", "compare and contrast", "trade-off", "tradeoff",
];

/// Overrides explicites reconnus dans le texte utilisateur.
const OVERRIDE_FLASH: &[&str] = &["@flash", "@fast"];
const OVERRIDE_EXPERT: &[&str] = &["@expert", "@slow", "@deep"];

/// Règle déclarative lue depuis `~/.quokka/routing.toml` (spec §4.2).
/// La première règle qui matche gagne, avant la porte lexicale.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RouteRule {
    /// Nom de la règle (log + `RouteReason::Rule`).
    pub name: String,
    /// Mots-clés déclencheurs (sous-chaînes, insensibles à la casse).
    #[serde(default)]
    pub match_keywords: Vec<String>,
    /// Court-circuite vers `target` si le prompt fait moins de N tokens
    /// estimés — prioritaire sur `match_keywords` quand les deux sont vides ?
    /// Non : les deux conditions sont ET.
    #[serde(default)]
    pub max_input_tokens: Option<usize>,
    /// Cible forcée par la règle.
    pub target_tier: String,
}

/// Politique de routage (`[policy]` du fichier TOML).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RoutePolicy {
    /// `eco` | `balanced` | `performance` — module le seuil sémantique.
    #[serde(default = "default_mode")]
    pub mode: String,
    /// Marge RAM au-delà de la taille expert (défaut spec : 1536 Mio).
    #[serde(default = "default_buffer")]
    pub ram_safety_buffer_mb: u32,
}

fn default_mode() -> String {
    "balanced".to_owned()
}

fn default_buffer() -> u32 {
    RAM_SAFETY_BUFFER_MB
}

/// Fichier de configuration optionnel.
#[derive(Debug, Default, serde::Deserialize)]
pub struct RouteConfig {
    /// `[policy]`.
    #[serde(default)]
    pub policy: Option<RoutePolicy>,
    /// `[[rules]]`.
    #[serde(default)]
    pub rules: Vec<RouteRule>,
}

/// Chemin canonique `~/.quokka/routing.toml`.
#[must_use]
pub fn default_config_path() -> std::path::PathBuf {
    dirs_home().join(".quokka").join("routing.toml")
}

fn dirs_home() -> std::path::PathBuf {
    std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("."))
}

/// Ancres Expert : extraits représentatifs des charges que seul le
/// grand modèle gère bien. Le centroïde est la moyenne de leurs vecteurs.
const EXPERT_ANCHORS: &[&str] = &[
    "write a rust function that parses json and propagates errors with thiserror",
    "debug this stack trace: borrow checker lifetime error in async task",
    "refactor this module architecture to reduce coupling and improve testability",
    "prove by induction that this recursive algorithm always terminates",
    "write a sql query joining three tables with window functions and ctes",
    "implement a concurrent cache with eviction policy in tokio",
    "analyze the time and space complexity of this divide and conquer algorithm",
    "design a distributed system that handles millions of concurrent requests",
    "translate this python code to idiomatic rust with proper error handling",
    "explain the mathematical derivation behind gradient descent optimizer",
    "fix the race condition in this multithreaded producer consumer pipeline",
    "review this pull request for security vulnerabilities and edge cases",
    "optimize this database schema and write the migration queries",
    "implement a consistent hashing ring for a distributed cache cluster",
    "derive the closed form solution for this recurrence relation",
    "write unit tests covering edge cases for this parsing function",
    "compare these two sorting algorithms asymptotically and pick one",
    "debug why this container keeps getting oom killed under load",
    "explain transformer attention heads and rotary positional embeddings",
    "build a regex that validates international phone numbers",
    // Architecture & systèmes distribués
    "design a distributed cache with consistent hashing and replication",
    "review this microservices architecture for bottlenecks and single points of failure",
    "explain how database sharding and consistent hashing distribute load",
    "design a rate limiter that handles burst traffic fairly across tenants",
    "compare event streaming platforms kafka vs rabbitmq for our use case",
    "explain how cdn edge caching and cache invalidation work at scale",
    "design a notification system with exactly once delivery guarantees",
    "explain saga pattern and distributed transactions across microservices",
    "analyze tradeoffs of monolith vs microservices architecture for a startup",
    "design multi region failover and disaster recovery for a database",
    "explain consensus algorithms like raft and paxos in detail",
    "explain how b tree and lsm tree indexing work inside databases",
    // Maths, algo & ML
    "prove by mathematical induction that this statement holds for all n",
    "derive the closed form solution of this linear recurrence relation",
    "analyze time complexity using big o and amortized analysis",
    "explain bias variance tradeoff and overfitting in machine learning",
    "derive bayes theorem and apply it to a medical diagnosis example",
    "solve this combinatorics counting problem step by step",
    "explain gradient descent convergence and learning rate scheduling",
    // Sécurité & protocoles
    "explain tls handshake and certificate validation at cryptographic level",
    "review this authentication flow for oauth2 security vulnerabilities",
    "explain symmetric vs asymmetric encryption and key exchange protocols",
    "analyze this network traffic for signs of intrusion or exfiltration",
    "explain how zero knowledge proofs work conceptually",
];

/// Ancres Flash : charges triviales où le petit modèle suffit — les
/// domaines couvrent large (conversation, voyage, cuisine, culture,
/// définitions, rédaction simple) pour que le centroïde représente
/// « simple » et pas seulement « bavardage ».
const FLASH_ANCHORS: &[&str] = &[
    "hello how are you today",
    "what is the capital of france",
    "summarize this article in three sentences",
    "translate hello world to spanish",
    "write a short poem about the sea",
    "what does this word mean in english",
    "tell me a joke about programmers",
    "how do I make pancakes from scratch",
    "rewrite this sentence to be more polite",
    "give me a recipe for pasta carbonara",
    "what time is it in tokyo",
    "recommend a good science fiction book",
    "plan a weekend trip to the mountains with hiking trails",
    "suggest a cozy cabin rental for a family vacation",
    "what should I pack for a beach holiday in thailand",
    "write a friendly email asking for a day off work",
    "give me ideas for a birthday gift for my mom",
    "how do I train a puppy to sit and stay",
    "what are some easy home workouts without equipment",
    "recommend a good movie for a rainy sunday night",
    "help me write a wedding toast for my best friend",
    "what is a good name for a new puppy",
    "explain what a black hole is in simple terms",
    "give me tips for better sleep at night",
    "how do I remove a stain from a white shirt",
    "what plants are easy to grow indoors",
    "suggest a menu for a dinner party for six people",
    "how do I learn basic photography",
    "write a short story about a dragon and a knight",
    "what are the best beaches in portugal",
    "improve this paragraph so it sounds more professional and fluent",
    "edit this text for clarity and fix the grammar mistakes",
    "make this email sound warmer and more friendly",
];

/// Le routeur : centroïdes pré-calculés à la construction (~40 µs une
/// fois, jamais à la requête), avec poids IDF par bucket pour que les
/// mots-outils (« the », « and ») ne diluent pas le signal sémantique.
pub struct Router {
    idf: Vec<f32>,
    expert_centroid: Vec<f32>,
    flash_centroid: Vec<f32>,
    /// Règles utilisateur (`routing.toml`), évaluées après les overrides.
    rules: Vec<RouteRule>,
    /// Marge ajoutée au score expert requis selon `policy.mode` :
    /// `eco` > 0 (favorise Flash), `performance` < 0 (favorise Expert).
    semantic_margin: f32,
    /// Marge RAM de la garde mémoire (configurable via `[policy]`).
    ram_buffer_mb: u32,
}

impl Router {
    /// Pré-calcule IDF + centroïdes d'ancres, sans fichier de config.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(RouteConfig::default())
    }

    /// Charge `~/.quokka/routing.toml` si présent ; tout autre problème
    /// (fichier absent, TOML invalide) retombe sur la config par défaut —
    /// un fichier de config cassé ne doit jamais empêcher le serveur de
    /// démarrer, mais l'erreur est loguée pour diagnostic.
    #[must_use]
    pub fn load(path: &std::path::Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(text) => match toml::from_str::<RouteConfig>(&text) {
                Ok(config) => Self::with_config(config),
                Err(err) => {
                    tracing::warn!("invalid routing.toml at {}: {err} — using defaults", path.display());
                    Self::new()
                }
            },
            Err(_) => Self::new(),
        }
    }

    /// Routeur avec configuration explicite (règles + politique).
    #[must_use]
    pub fn with_config(config: RouteConfig) -> Self {
        let idf = idf_weights(EXPERT_ANCHORS, FLASH_ANCHORS);
        let (semantic_margin, ram_buffer_mb) = config
            .policy
            .map(|p| {
                let margin = match p.mode.as_str() {
                    "eco" => 0.05,
                    "performance" => -0.05,
                    _ => 0.0,
                };
                (margin, p.ram_safety_buffer_mb)
            })
            .unwrap_or((0.0, RAM_SAFETY_BUFFER_MB));
        Self {
            expert_centroid: centroid(EXPERT_ANCHORS, &idf),
            flash_centroid: centroid(FLASH_ANCHORS, &idf),
            idf,
            rules: config.rules,
            semantic_margin,
            ram_buffer_mb,
        }
    }

    /// Route une conversation. `text` = concaténation des messages
    /// (l'appelant décide de la pondération — le dernier message user
    /// domine en pratique via les bigrammes de frontière).
    ///
    /// `ctx` active la garde mémoire : un choix Expert peut être rétrogradé
    /// en Flash si la RAM disponible ne couvre pas le modèle + 1,5 Gio.
    #[must_use]
    pub fn route(&self, text: &str, ctx: &RouteContext) -> RouteDecision {
        let started = Instant::now();

        // Stage 1 — porte lexicale (< 0,1 ms).
        let lowered = text.to_lowercase();
        let est_tokens = lowered.len() / 4;
        let decision = if OVERRIDE_EXPERT.iter().any(|m| has_token(&lowered, m)) {
            Some((RouteTarget::Expert, RouteReason::Override))
        } else if OVERRIDE_FLASH.iter().any(|m| has_token(&lowered, m)) {
            Some((RouteTarget::Flash, RouteReason::Override))
        } else if let Some(rule) = self.match_rule(&lowered, est_tokens) {
            // Règles utilisateur : premier match gagne.
            let target = match rule.target_tier.as_str() {
                "expert" => RouteTarget::Expert,
                _ => RouteTarget::Flash,
            };
            Some((target, RouteReason::Rule { name: rule.name.clone() }))
        } else if CODE_MARKERS.iter().any(|m| lowered.contains(m))
            || CODE_WORDS.iter().any(|m| has_word(&lowered, m))
            || (LANGUAGES.iter().any(|m| has_word(&lowered, m))
                && CODE_CONTEXT.iter().any(|m| lowered.contains(m)))
        {
            Some((RouteTarget::Expert, RouteReason::CodeDetected))
        } else if EXPERT_DOMAIN_MARKERS
            .iter()
            .any(|m| has_word_start(&lowered, m))
        {
            Some((RouteTarget::Expert, RouteReason::DomainDetected))
        } else if is_short_simple(&lowered) {
            Some((RouteTarget::Flash, RouteReason::ShortPrompt))
        } else {
            None
        };

        let (mut target, reason) = match decision {
            Some((target, reason)) => (target, reason),
            // Stage 2 — centroïdes cosinus, marge ajustée par policy.mode.
            None => {
                let v = weighted_vector(&lowered, &self.idf);
                let expert_score = cosine(&v, &self.expert_centroid);
                let flash_score = cosine(&v, &self.flash_centroid);
                (
                    if expert_score - self.semantic_margin > flash_score {
                        RouteTarget::Expert
                    } else {
                        RouteTarget::Flash
                    },
                    RouteReason::Semantic {
                        expert_score,
                        flash_score,
                    },
                )
            }
        };

        // Memory Pressure Guard : l'expert ne doit jamais compromettre le
        // système — repli Flash si la marge n'est pas là.
        if target == RouteTarget::Expert
            && let Some(available) = ctx.available_ram_mb
        {
            let needed = ctx.expert_size_mb.saturating_add(self.ram_buffer_mb);
            if available < needed {
                target = RouteTarget::Flash;
                return RouteDecision {
                    target,
                    reason: RouteReason::MemoryPressureFallback {
                        needed_mb: needed,
                        available_mb: available,
                    },
                    elapsed: started.elapsed(),
                };
            }
        }

        RouteDecision {
            target,
            reason,
            elapsed: started.elapsed(),
        }
    }

    /// Première règle `[[rules]]` dont toutes les conditions remplies
    /// matchent. Une règle sans aucune condition est ignorée (piège de
    /// config qui court-circuiterait tout le trafic).
    fn match_rule<'r>(&'r self, lowered: &str, est_tokens: usize) -> Option<&'r RouteRule> {
        self.rules.iter().find(|rule| {
            if rule.match_keywords.is_empty() && rule.max_input_tokens.is_none() {
                return false;
            }
            let kw_ok = rule.match_keywords.is_empty()
                || rule
                    .match_keywords
                    .iter()
                    .any(|k| lowered.contains(&k.to_lowercase()));
            let tok_ok = rule
                .max_input_tokens
                .is_none_or(|n| est_tokens <= n);
            kw_ok && tok_ok
        })
    }
}

impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}

/// Prompt « court et simple » : ~< 100 octets de tokens effectifs (≈ 25
/// tokens) sans marqueur de raisonnement ni de complexité.
fn is_short_simple(lowered: &str) -> bool {
    // Estimation tokens ≈ chars/4 (BPE ~3,5-4 chars/token en moyenne).
    let est_tokens = lowered.len() / 4;
    est_tokens < 25 && !REASONING_MARKERS.iter().any(|m| lowered.contains(m))
}

/// Mot-clé présent avec frontières non-alphanumériques **des deux côtés**
/// (« rust » dans « rustacean » non, dans « (rust) » oui).
fn has_word(text: &str, needle: &str) -> bool {
    text.match_indices(needle).any(|(i, _)| {
        let before = text[..i].chars().next_back();
        let after = text[i + needle.len()..].chars().next();
        before.is_none_or(|c| !c.is_alphanumeric())
            && after.is_none_or(|c| !c.is_alphanumeric())
    })
}

/// Mot-clé avec frontière non-alphanumérique **au début seulement** :
/// « postgres » matche « postgresql » mais pas « impostor ».
fn has_word_start(text: &str, needle: &str) -> bool {
    let needle = needle.trim();
    text.match_indices(needle).any(|(i, _)| {
        text[..i]
            .chars()
            .next_back()
            .is_none_or(|c| !c.is_alphanumeric())
    })
}

/// `@flash` doit matcher comme token entier, pas dans `@flashcards`.
fn has_token(text: &str, needle: &str) -> bool {
    text.match_indices(needle).any(|(i, _)| {
        let before = text[..i].chars().next_back();
        let after = text[i + needle.len()..].chars().next();
        before.is_none_or(|c| c.is_whitespace() || c == '@')
            && after.is_none_or(|c| c.is_whitespace() || c.is_ascii_punctuation())
    })
}

/// Comptes bruts (sans normalisation) — base du calcul IDF.
fn raw_counts(text: &str) -> Vec<f32> {
    let mut counts = vec![0.0_f32; DIM];
    let mut prev: Option<u64> = None;
    for word in text.split(|c: char| !c.is_alphanumeric()) {
        if word.len() < 2 {
            continue;
        }
        let h = fnv(word.as_bytes());
        counts[(h as usize) % DIM] += 1.0;
        if let Some(p) = prev {
            // Bigramme : hash du couple prev→word, ancré sur le début.
            let mut buf = p.to_le_bytes().to_vec();
            buf.extend_from_slice(word.as_bytes());
            counts[(fnv(&buf) as usize) % DIM] += 0.6;
        }
        prev = Some(h);
    }
    counts
}

/// Applique les poids IDF puis normalise L2.
fn weighted_vector(text: &str, idf: &[f32]) -> Vec<f32> {
    let mut v = raw_counts(text);
    for (x, w) in v.iter_mut().zip(idf) {
        *x *= w;
    }
    normalize(&mut v);
    v
}

/// IDF par bucket : `ln((N+1)/(df+1))` où df = nombre d'ancres (des deux
/// classes) contenant le bucket. Les features présentes partout (mots
/// outils) tendent vers 0 ; les features rares gardent leur poids.
fn idf_weights(expert: &[&str], flash: &[&str]) -> Vec<f32> {
    let mut df = vec![0_u32; DIM];
    for anchor in expert.iter().chain(flash) {
        let v = raw_counts(anchor);
        for (i, &c) in v.iter().enumerate() {
            if c > 0.0 {
                df[i] += 1;
            }
        }
    }
    let n = (expert.len() + flash.len()) as f32;
    df.iter()
        .map(|&d| ((n + 1.0) / (d as f32 + 1.0)).ln())
        .collect()
}

/// Centroïde = moyenne des vecteurs d'ancres pondérés IDF, renormalisée.
fn centroid(anchors: &[&str], idf: &[f32]) -> Vec<f32> {
    let mut acc = vec![0.0_f32; DIM];
    for anchor in anchors {
        let v = weighted_vector(anchor, idf);
        for (a, x) in acc.iter_mut().zip(&v) {
            *a += x;
        }
    }
    normalize(&mut acc);
    acc
}

fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// FNV-1a 64 bits — hash stable, pas de DoS par collision triviale.
fn fnv(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> RouteContext {
        RouteContext {
            available_ram_mb: Some(8192),
            expert_size_mb: 2048,
        }
    }

    #[test]
    fn overrides_win_over_everything() {
        let r = Router::new();
        assert_eq!(
            r.route("@expert write me a poem", &ctx()).target,
            RouteTarget::Expert
        );
        assert_eq!(
            r.route("@flash debug this rust borrow error", &ctx()).target,
            RouteTarget::Flash
        );
        // @flashcards ne doit PAS matcher @flash.
        let d = r.route("explain @flashcards please", &ctx());
        assert!(matches!(d.reason, RouteReason::Semantic { .. } | RouteReason::ShortPrompt));
    }

    #[test]
    fn code_markers_route_to_expert() {
        let r = Router::new();
        for prompt in [
            "fix this ```rust fn main() {} ```",
            "my program gives a segmentation fault",
            "write a sql query that joins users and orders",
            "the borrow checker rejects this async code",
        ] {
            let d = r.route(prompt, &ctx());
            assert_eq!(d.target, RouteTarget::Expert, "prompt: {prompt}");
            assert!(matches!(d.reason, RouteReason::CodeDetected));
        }
    }

    #[test]
    fn short_simple_prompts_route_to_flash() {
        let r = Router::new();
        for prompt in ["hi", "what is 2+2", "tell me a joke", "thanks!"] {
            let d = r.route(prompt, &ctx());
            assert_eq!(d.target, RouteTarget::Flash, "prompt: {prompt}");
            assert_eq!(d.reason, RouteReason::ShortPrompt);
        }
    }

    #[test]
    fn semantic_stage_disambiguates() {
        let r = Router::new();
        // Ambigus : ni override, ni fence, ni assez court pour le
        // court-circuit si > 25 tokens estimés.
        let expertish = "I need a careful architectural review of this concurrent rust service: \
                         it uses tokio tasks, a shared cache and a postgres pool, and under \
                         load the latency spikes. Where should I start debugging?";
        let flashish = "I am planning a weekend trip to the mountains with my family, can you \
                        suggest some easy hiking trails and a cozy mountain cabin to stay in?";
        assert_eq!(r.route(expertish, &ctx()).target, RouteTarget::Expert);
        assert_eq!(r.route(flashish, &ctx()).target, RouteTarget::Flash);
    }

    #[test]
    fn routing_toml_rules_apply_before_classifier() {
        let config = RouteConfig {
            policy: None,
            rules: vec![
                RouteRule {
                    name: "legal-review".to_owned(),
                    match_keywords: vec!["contract".to_owned()],
                    max_input_tokens: None,
                    target_tier: "expert".to_owned(),
                },
                RouteRule {
                    name: "tiny-questions".to_owned(),
                    match_keywords: vec![],
                    max_input_tokens: Some(10),
                    target_tier: "flash".to_owned(),
                },
            ],
        };
        let r = Router::with_config(config);
        // "contract" déclenche la règle expert même si le classifieur
        // sémantique aurait peut-être choisi Flash.
        let d = r.route(
            "please review this contract clause and its implications in detail",
            &ctx(),
        );
        assert_eq!(d.target, RouteTarget::Expert);
        assert!(matches!(d.reason, RouteReason::Rule { ref name } if name == "legal-review"));
        // max_input_tokens=10 → flash pour une question minuscule.
        let d = r.route("hey there", &ctx());
        assert!(matches!(d.reason, RouteReason::Rule { ref name } if name == "tiny-questions"));
        // Override @expert bat les règles.
        let d = r.route("@flash contract review", &ctx());
        assert_eq!(d.target, RouteTarget::Flash);
    }

    #[test]
    fn memory_pressure_guard_downgrades_expert() {
        let r = Router::new();
        let tight = RouteContext {
            available_ram_mb: Some(1024),
            expert_size_mb: 8192,
        };
        let d = r.route("@expert refactor this codebase", &tight);
        assert_eq!(d.target, RouteTarget::Flash);
        assert!(matches!(
            d.reason,
            RouteReason::MemoryPressureFallback { .. }
        ));
        // Sans contexte (sonde indisponible) → pas de garde, choix nominal.
        let no_ctx = RouteContext {
            available_ram_mb: None,
            expert_size_mb: 8192,
        };
        assert_eq!(
            r.route("@expert refactor this codebase", &no_ctx).target,
            RouteTarget::Expert
        );
    }

    #[test]
    fn routing_stays_under_budget() {
        let r = Router::new();
        let ctx = ctx();
        let prompts = [
            "hi",
            "write a rust async tcp server with graceful shutdown",
            "explain the theory of relativity simply",
            "debug this segfault in my c program",
            "what is the weather like",
            "implement a red black tree in python",
            "translate this paragraph into french",
            "how do neural networks learn",
            "fix the merge conflict in git",
            "tell me about quokkas",
        ];
        for prompt in prompts {
            let d = r.route(prompt, &ctx);
            assert!(
                d.elapsed < ROUTE_BUDGET,
                "route exceeded 2ms budget: {:?} took {:?}",
                prompt,
                d.elapsed
            );
        }
        // Et en moyenne on est à des ordres de grandeur en dessous.
        let d = r.route(
            "analyze and refactor this complex multi crate rust workspace architecture",
            &ctx,
        );
        assert!(d.elapsed < ROUTE_BUDGET);
    }
}
