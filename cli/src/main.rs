//! `quokka` — point d'entrée CLI. Surface : profilage matériel,
//! téléchargement de modèles, serveur d'inférence OpenAI-compatible.

use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use quokka_core::{ModelPool, SessionConfig, resolve_model_path};
use quokka_hardware::registry::ModelSpec;
use quokka_hardware::{GpuBackend, HardwareProfile, Tier, discovery, download, registry};
use quokka_server::{AppState, InferenceEngine};

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

    let engine = InferenceEngine::new(pool, SessionConfig::default());
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
