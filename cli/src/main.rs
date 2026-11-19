//! `quokka` — point d'entrée CLI. Surface Phase 1 : profilage matériel
//! (prémisse de `quokka run auto` / `quokka status`).

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use quokka_hardware::registry::ModelSpec;
use quokka_hardware::{discovery, download, registry, GpuBackend, HardwareProfile, Tier};

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
        } => format!("CUDA — {name} ({} VRAM · {device_count} device(s))", fmt_mib(*vram_mb)),
        GpuBackend::Metal { unified_ram_mb } => {
            format!("Metal — unified memory ({})", fmt_mib(*unified_ram_mb))
        }
        GpuBackend::Vulkan {
            name,
            vram_mb,
            device_count,
        } => format!("Vulkan — {name} ({} VRAM · {device_count} device(s))", fmt_mib(*vram_mb)),
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
            let pair = profile
                .recommended_pair()
                .context("model registry")?;
            println!("auto → tier {}", profile.classify_tier());
            Ok(vec![pair.flash, pair.expert])
        }
        "flash" | "expert" => {
            let profile = HardwareProfile::probe().context("hardware probe failed")?;
            let pair = profile
                .recommended_pair()
                .context("model registry")?;
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
