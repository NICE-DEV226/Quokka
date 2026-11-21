//! `quokka` — point d'entrée CLI. Surface : profilage matériel,
//! téléchargement de modèles, serveur d'inférence OpenAI-compatible.

use std::net::{SocketAddr, ToSocketAddrs};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use quokka_core::{ModelPool, SessionConfig, resolve_model_path};
use quokka_hardware::registry::ModelSpec;
use quokka_hardware::{GpuBackend, HardwareProfile, Tier, discovery, download, registry};
use quokka_server::{AppState, InferenceEngine};

mod mcp;

#[derive(Debug, Parser)]
#[command(
    name = "quokka",
    version,
    about = "Quokka — smart local AI router & inference engine"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Probe le matériel hôte et affiche le profil, le tier et le duo
    /// Flash/Expert recommandé.
    Hardware {
        /// Sortie JSON machine-readable plutôt que le rapport lisible.
        #[arg(long)]
        json: bool,
    },
    /// Télécharge un modèle GGUF depuis Hugging Face vers `~/.quokka/models`.
    ///
    /// Cibles : `auto` (paire recommandée pour cette machine), un tier
    /// (`lightweight`, `standard`, `workstation` — tire Flash + Expert),
    /// ou `flash` / `expert` (modèle correspondant de la paire recommandée).
    Pull {
        /// Cible de téléchargement.
        target: String,
    },
    /// Démarre le moteur d'inférence : charge la paire Flash/Expert
    /// recommandée et sert l'API OpenAI-compatible jusqu'à Ctrl-C.
    Run {
        /// Adresse d'écoute du serveur HTTP.
        #[arg(long, default_value = "127.0.0.1:11435")]
        listen: String,
    },
    /// État de la machine et des modèles : tier, paire recommandée,
    /// présence effective des GGUF et modèles locaux découverts.
    Status {
        /// Sortie JSON machine-readable.
        #[arg(long)]
        json: bool,
    },
    /// Mesure les performances via le serveur en cours d'exécution :
    /// latence préfill, débit de décodage et gain du KV-cache sur un
    /// second tour de conversation. Nécessite `quokka run` actif.
    Bench {
        /// Adresse du serveur à tester.
        #[arg(long, default_value = "127.0.0.1:11435")]
        listen: String,
        /// Modèle à benchmarker (`flash` ou `expert`).
        #[arg(long, default_value = "flash")]
        model: String,
        /// Tokens générés pour la mesure de décodage.
        #[arg(long, default_value_t = 64)]
        tokens: u32,
        /// Sortie JSON machine-readable.
        #[arg(long)]
        json: bool,
    },
    /// Démarre un serveur MCP (stdio) proxyfiant l'API d'un `quokka run`
    /// en cours — pour Cursor, Claude Code et autres clients MCP.
    /// Outils : `quokka_chat`, `quokka_models`, `quokka_route`.
    Mcp {
        /// Adresse du serveur Quokka à proxyfier.
        #[arg(long, default_value = "http://127.0.0.1:11435")]
        url: String,
    },
}

/// Charge utile de `quokka hardware --json`, consommable par les outils
/// d'audit et la future télémétrie.
#[derive(serde::Serialize)]
struct HardwareReport {
    profile: HardwareProfile,
    tier: Tier,
    recommended: registry::ModelPair,
    local_model_count: usize,
    local_model_bytes: u64,
}

fn main() -> Result<()> {
    // Silencieux par défaut ; RUST_LOG=debug pour la traçabilité des probes.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Hardware { json } => print_hardware(json),
        Command::Pull { target } => pull_models(&target),
        // Runtime dédié : `main` reste sync car le downloader crée son
        // propre runtime en interne — le dropper depuis un contexte async
        // panique (tokio blocking/shutdown.rs).
        Command::Run { listen } => {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .context("tokio runtime")?;
            runtime.block_on(run_server(&listen))
        }
        Command::Status { json } => print_status(json),
        Command::Bench {
            listen,
            model,
            tokens,
            json,
        } => bench(&listen, &model, tokens, json),
        Command::Mcp { url } => {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .context("tokio runtime")?;
            runtime.block_on(mcp::run(url))
        }
    }
}

fn print_hardware(json: bool) -> Result<()> {
    let profile = HardwareProfile::probe().context("hardware probe failed")?;
    let tier = profile.classify_tier();
    let pair = profile.recommended_pair().context("model registry")?;
    let local = discovery::discover();
    let local_model_bytes: u64 = local.iter().map(|model| model.size_bytes).sum();

    if json {
        let report = HardwareReport {
            profile: profile.clone(),
            tier,
            recommended: pair.clone(),
            local_model_count: local.len(),
            local_model_bytes,
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&report).context("serialize hardware report")?
        );
        return Ok(());
    }

    let gpu = match &profile.gpu_backend {
        GpuBackend::Cuda {
            name,
            vram_mb,
            device_count,
        } => format!(
            "CUDA — {name} ({} VRAM · {device_count} device(s))",
            fmt_mib(*vram_mb)
        ),
        GpuBackend::Metal { unified_ram_mb } => {
            format!("Metal — unified memory ({})", fmt_mib(*unified_ram_mb))
        }
        GpuBackend::Vulkan {
            name,
            vram_mb,
            device_count,
        } => format!(
            "Vulkan — {name} ({} VRAM · {device_count} device(s))",
            fmt_mib(*vram_mb)
        ),
        GpuBackend::None => "none (CPU-only inference)".to_owned(),
    };

    let cores = match profile.physical_cores {
        Some(physical) => format!("{} logical / {physical} physical", profile.cpu_cores),
        None => format!("{} logical", profile.cpu_cores),
    };

    let brand = if profile.cpu_brand.is_empty() {
        "<unknown>"
    } else {
        &profile.cpu_brand
    };

    println!("Quokka · Hardware Profile");
    println!("─────────────────────────");
    println!("OS/Arch : {} · {}", profile.os, profile.arch);
    println!("CPU     : {brand} ({cores})");
    println!(
        "SIMD    : AVX2={} · AVX-512={} · NEON={}",
        yes(profile.has_avx2),
        yes(profile.has_avx512f),
        yes(profile.has_neon)
    );
    println!(
        "RAM     : {} total · {} available",
        fmt_mib(profile.total_ram_mb),
        fmt_mib(profile.available_ram_mb)
    );
    println!("GPU     : {gpu}");
    println!("Tier    : {tier}");
    println!();
    println!("Recommended pair (registry/profiles.json):");
    println!(
        "  Flash  : {} · {} (~{})",
        pair.flash.model.filename,
        pair.flash.quantization,
        fmt_mib(pair.flash.approx_size_mb)
    );
    println!("           {}", pair.flash.model.hf_url());
    println!(
        "  Expert : {} · {} (~{})",
        pair.expert.model.filename,
        pair.expert.quantization,
        fmt_mib(pair.expert.approx_size_mb)
    );
    println!("           {}", pair.expert.model.hf_url());
    println!();
    println!(
        "Local GGUF models: {} found ({}) across Ollama/LM Studio/HF cache",
        local.len(),
        fmt_bytes(local_model_bytes)
    );

    Ok(())
}

fn yes(flag: bool) -> &'static str {
    if flag { "yes" } else { "no" }
}

/// MiB → "12.3 Gio" (arrondi 0,1).
fn fmt_mib(mb: u64) -> String {
    format!("{:.1} GiB", mb as f64 / 1024.0)
}

/// Octets → Gio.
fn fmt_bytes(bytes: u64) -> String {
    fmt_mib(bytes / quokka_hardware::MIB)
}

/// Résout la cible `pull` en une liste de [`ModelSpec`] du registre.
fn resolve_pull_targets(target: &str) -> Result<Vec<ModelSpec>> {
    let registry = registry::load().context("model registry")?;
    let normalized = target.to_ascii_lowercase();

    match normalized.as_str() {
        "auto" => {
            let profile = HardwareProfile::probe().context("hardware probe failed")?;
            let pair = profile.recommended_pair().context("model registry")?;
            println!("auto → tier {}", profile.classify_tier());
            Ok(vec![pair.flash, pair.expert])
        }
        "flash" | "expert" => {
            let profile = HardwareProfile::probe().context("hardware probe failed")?;
            let pair = profile.recommended_pair().context("model registry")?;
            let spec = if normalized == "flash" {
                pair.flash
            } else {
                pair.expert
            };
            Ok(vec![spec])
        }
        name => {
            let tier = match name {
                "lightweight" => Tier::Lightweight,
                "standard" => Tier::Standard,
                "workstation" => Tier::Workstation,
                _ => anyhow::bail!(
                    "unknown pull target `{target}` — expected auto, flash, expert, \
                     lightweight, standard or workstation"
                ),
            };
            let pair = registry.tiers.get(tier);
            Ok(vec![pair.flash.clone(), pair.expert.clone()])
        }
    }
}

fn pull_models(target: &str) -> Result<()> {
    let specs = resolve_pull_targets(target)?;
    let dest = download::models_dir().context("cannot resolve ~/.quokka/models")?;

    for spec in &specs {
        let model = &spec.model;
        println!(
            "⬇ {} · {} (~{} Mio)",
            model.filename, spec.quantization, spec.approx_size_mb
        );

        let bar = ProgressBar::new(spec.approx_size_mb * quokka_hardware::MIB);
        bar.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} {msg:<48} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, ETA {eta})",
            )
            .context("progress style")?
            .progress_chars("=>-"),
        );
        bar.set_message(model.filename.clone());

        let bar_ref = bar.clone();
        let path = download::pull(model, &dest, move |downloaded, total| {
            if total > 0 {
                bar_ref.set_length(total);
            }
            bar_ref.set_position(downloaded);
        })
        .with_context(|| format!("download {}", model.filename))?;

        bar.finish_and_clear();
        println!("  ✓ {}", path.display());
    }

    Ok(())
}

/// `quokka run` — charge la paire Flash/Expert recommandée et sert
/// l'API OpenAI-compatible sur `--listen` jusqu'à Ctrl-C.
async fn run_server(listen: &str) -> Result<()> {
    let profile = HardwareProfile::probe().context("hardware probe failed")?;
    let pair = profile.recommended_pair().context("model registry")?;
    let tier = profile.classify_tier();

    println!("Quokka · tier {tier}");
    println!(
        "Flash  : {} ({} Mio)",
        pair.flash.model.filename, pair.flash.approx_size_mb
    );
    println!(
        "Expert : {} ({} Mio)",
        pair.expert.model.filename, pair.expert.approx_size_mb
    );

    let bar = ProgressBar::new_spinner();
    bar.set_style(
        ProgressStyle::with_template("{spinner:.green} {msg}").context("progress style")?,
    );
    bar.set_message("loading models (mmap)…");
    let bar_ref = bar.clone();
    let progress: Arc<dyn Fn(f32) + Send + Sync> = Arc::new(move |_| bar_ref.tick());

    let pool = ModelPool::load_with_progress(&profile, &pair, Some(progress))
        .context("model load failed — run `quokka pull auto` first")?;
    bar.finish_and_clear();

    let served = quokka_server::served_models(&pool);
    for model in &served {
        println!("loaded {} → {}", model.id, model.filename);
    }

    // KV-cache persisté sur disque (LZ4, ~8 conversations par modèle) :
    // revenir sur une ancienne conversation restaure son préfixe au lieu
    // de re-payer tout le préfill. `None` si pas de home résolvable.
    let session_config = SessionConfig {
        kv_cache_dir: std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .map(|home| PathBuf::from(home).join(".quokka").join("cache").join("kv")),
        ..SessionConfig::default()
    };
    let engine = InferenceEngine::new(pool, session_config);
    let state = AppState::new(engine, served);

    let addr: SocketAddr = listen
        .to_socket_addrs()
        .context("invalid --listen address")?
        .next()
        .context("--listen resolved to no address")?;

    println!("serving OpenAI-compatible API on http://{addr}");
    println!("  GET  /v1/models");
    println!("  POST /v1/chat/completions  (stream: true pour SSE)");

    quokka_server::serve(state, addr)
        .await
        .context("server failed")
}

/// `quokka status` — présence effective des modèles recommandés.
fn print_status(json: bool) -> Result<()> {
    let profile = HardwareProfile::probe().context("hardware probe failed")?;
    let tier = profile.classify_tier();
    let pair = profile.recommended_pair().context("model registry")?;
    let local = discovery::discover();

    let mut model_states = Vec::with_capacity(2);
    for (role, spec) in [("flash", &pair.flash), ("expert", &pair.expert)] {
        let state = match resolve_model_path(spec) {
            Ok(path) => {
                let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                ModelState {
                    role,
                    filename: spec.model.filename.clone(),
                    present: true,
                    size_bytes: Some(size),
                    path: Some(path.display().to_string()),
                }
            }
            Err(_) => ModelState {
                role,
                filename: spec.model.filename.clone(),
                present: false,
                size_bytes: None,
                path: None,
            },
        };
        model_states.push(state);
    }

    if json {
        #[derive(serde::Serialize)]
        struct StatusReport<'a> {
            tier: Tier,
            models: &'a [ModelState<'a>],
            local_model_count: usize,
            local_model_bytes: u64,
        }
        let report = StatusReport {
            tier,
            models: &model_states,
            local_model_count: local.len(),
            local_model_bytes: local.iter().map(|m| m.size_bytes).sum(),
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&report).context("serialize status")?
        );
        return Ok(());
    }

    println!("Quokka · Status");
    println!("────────────────");
    println!("Tier    : {tier}");
    for model in &model_states {
        if model.present {
            println!(
                "{} : {} — present ({})",
                model.role,
                model.filename,
                fmt_bytes(model.size_bytes.unwrap_or(0))
            );
        } else {
            println!(
                "{} : {} — MISSING (`quokka pull auto`)",
                model.role, model.filename
            );
        }
    }
    println!(
        "Local   : {} GGUF discovered ({})",
        local.len(),
        fmt_bytes(local.iter().map(|m| m.size_bytes).sum())
    );
    Ok(())
}

/// Réponse minimale de `/v1/chat/completions` pour le bench.
#[derive(serde::Deserialize)]
struct BenchReply {
    choices: Vec<BenchChoice>,
    usage: BenchUsage,
}

#[derive(serde::Deserialize)]
struct BenchChoice {
    message: BenchMessage,
}

#[derive(serde::Deserialize)]
struct BenchMessage {
    content: String,
}

#[derive(serde::Deserialize)]
struct BenchUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    #[serde(default)]
    prompt_tokens_details: Option<BenchPromptDetails>,
}

#[derive(serde::Deserialize)]
struct BenchPromptDetails {
    cached_tokens: usize,
}

struct BenchSample {
    wall: std::time::Duration,
    prompt_tokens: usize,
    completion_tokens: usize,
    cached_tokens: usize,
    text: String,
}

/// `quokka bench` — mesure via le serveur live : préfill (max_tokens=1),
/// débit de décodage, et gain du KV-cache au second tour. Le nonce rend
/// chaque prompt unique pour éviter les hits de cache entre runs.
fn bench(listen: &str, model: &str, tokens: u32, json: bool) -> Result<()> {
    let base = format!("http://{listen}");
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .context("http client")?;

    client
        .get(format!("{base}/health"))
        .send()
        .and_then(reqwest::blocking::Response::error_for_status)
        .with_context(|| format!("server unreachable at {base} — start `quokka run` first"))?;

    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    // ~100 tokens de prompt : assez long pour que le préfill soit mesurable.
    // Deux nonces distincts : R1 et R2 ne doivent PAS partager de préfixe,
    // sinon R2 hériterait du KV-cache de R1 et la mesure de décodage serait
    // faussée (le préfill serait sauté).
    let prompt_a = format!(
        "Benchmark context {nonce}-a. The quick brown fox jumps over the lazy dog. \
         Pack my box with five dozen liquor jugs. How vexingly quick daft zebras jump! \
         Sphinx of black quartz, judge my vow. Write a detailed explanation of how \
         a transformer neural network processes this text, covering attention, \
         feed-forward layers and residual connections."
    );
    let prompt_b = prompt_a.replacen("-a.", "-b.", 1);

    let post = |messages: serde_json::Value, max_tokens: u32| -> Result<BenchSample> {
        let body = serde_json::json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        });
        let started = std::time::Instant::now();
        let reply: BenchReply = client
            .post(format!("{base}/v1/chat/completions"))
            .json(&body)
            .send()
            .and_then(reqwest::blocking::Response::error_for_status)
            .context("chat completion request")?
            .json()
            .context("parse chat completion")?;
        let wall = started.elapsed();
        let usage = reply.usage;
        Ok(BenchSample {
            wall,
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            cached_tokens: usage.prompt_tokens_details.map_or(0, |d| d.cached_tokens),
            text: reply
                .choices
                .into_iter()
                .next()
                .map_or(String::new(), |c| c.message.content),
        })
    };

    // R1 : max_tokens=1 → quasi-pur préfill (overhead HTTP inclus).
    let r1 = post(serde_json::json!([{"role":"user","content":prompt_a}]), 1)?;
    // R2 : prompt différent → pas de hit de cache ;
    // la différence wall2-wall1 isole le décodage de `tokens` tokens.
    let r2 = post(
        serde_json::json!([{"role":"user","content":prompt_b}]),
        tokens,
    )?;
    // R3 : second tour de la conversation de R2 → doit frapper le KV-cache.
    let r3 = post(
        serde_json::json!([
            {"role":"user","content":prompt_b},
            {"role":"assistant","content":r2.text},
            {"role":"user","content":"Summarize the previous answer in one sentence."},
        ]),
        tokens.min(32),
    )?;

    let decode_secs = r2.wall.as_secs_f64() - r1.wall.as_secs_f64();
    let decode_tps = if decode_secs > 0.0 && r2.completion_tokens > 1 {
        (r2.completion_tokens - 1) as f64 / decode_secs
    } else {
        0.0
    };

    if json {
        println!(
            "{}",
            serde_json::json!({
                "model": model,
                "prefill": {
                    "wall_ms": r1.wall.as_millis(),
                    "prompt_tokens": r1.prompt_tokens,
                },
                "decode": {
                    "wall_ms": r2.wall.as_millis(),
                    "completion_tokens": r2.completion_tokens,
                    "est_tokens_per_sec": decode_tps,
                },
                "cached_turn": {
                    "wall_ms": r3.wall.as_millis(),
                    "prompt_tokens": r3.prompt_tokens,
                    "cached_tokens": r3.cached_tokens,
                    "completion_tokens": r3.completion_tokens,
                },
            })
        );
        return Ok(());
    }

    println!("Quokka · Bench ({model} @ {base})");
    println!("─────────────────────────────────────────");
    println!(
        "Prefill   : {:>7.0} ms  ({} prompt tokens)",
        r1.wall.as_secs_f64() * 1e3,
        r1.prompt_tokens
    );
    println!(
        "Decode    : {:>7.0} ms  → ~{decode_tps:.1} tok/s  ({} tokens)",
        r2.wall.as_secs_f64() * 1e3,
        r2.completion_tokens
    );
    println!(
        "Cached    : {:>7.0} ms  ({} / {} prompt tokens réutilisés)",
        r3.wall.as_secs_f64() * 1e3,
        r3.cached_tokens,
        r3.prompt_tokens
    );
    Ok(())
}

/// État d'un modèle recommandé pour `quokka status`.
#[derive(serde::Serialize)]
struct ModelState<'a> {
    role: &'a str,
    filename: String,
    present: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    size_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<String>,
}
