//! `ModelPool` — chargement des poids GGUF via `llama.cpp`.
//!
//! Le pool possède le [`LlamaBackend`] (initialisé une fois) et les
//! [`LlamaModel`] Flash/Expert issus du [`ModelPair`] recommandé. Chaque
//! modèle est chargé en `mmap` (sauf `RamPinned` qui ajoute `mlock`), avec
//! `n_gpu_layers` calculé depuis la VRAM dédiée et les métadonnées GGUF
//! (`block_count` lu via [`GgufContext`] — aucune allocation du modèle
//! complet nécessaire pour cette estimation).
//!
//! La résolution du fichier suit la politique zéro-doublon :
//! `~/.quokka/models/<filename>` d'abord, puis n'importe quel GGUF
//! découvert par `quokka-hardware` portant le même nom de fichier.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use llama_cpp_2::gguf::GgufContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use quokka_hardware::gpu::GpuBackend;
use quokka_hardware::registry::{ModelPair, ModelSpec, Placement};
use quokka_hardware::{HardwareProfile, discovery};

use crate::error::{CoreError, Result};

/// Rôle d'un modèle dans le duo Flash/Expert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelRole {
    /// Modèle 1B–3B résident pour les prompts simples.
    Flash,
    /// Modèle 7B–32B pour raisonnement et code.
    Expert,
}

/// Un modèle GGUF chargé en mémoire (ou mappé).
pub struct LoadedModel {
    /// Rôle dans le duo.
    pub role: ModelRole,
    /// Chemin du fichier GGUF résolu.
    pub path: PathBuf,
    /// Backend llama.cpp sous-jacent.
    pub model: LlamaModel,
    /// Couches effectivement déchargées sur GPU.
    pub n_gpu_layers: u32,
    /// Stratégie de placement demandée par le registre.
    pub placement: Placement,
}

/// Pool de modèles chargés + backend llama.cpp partagé.
pub struct ModelPool {
    backend: LlamaBackend,
    models: Vec<LoadedModel>,
}

impl ModelPool {
    /// Charge le duo Flash/Expert recommandé pour `profile`.
    ///
    /// Échoue si un modèle n'est pas présent localement — le message
    /// d'erreur indique la commande `quokka pull` à lancer.
    pub fn load(profile: &HardwareProfile, pair: &ModelPair) -> Result<Self> {
        Self::load_with_progress(profile, pair, None)
    }

    /// Variante avec callback de progression de chargement
    /// (`0.0..=1.0` par modèle, appelé depuis le thread de chargement).
    pub fn load_with_progress(
        profile: &HardwareProfile,
        pair: &ModelPair,
        progress: Option<Arc<dyn Fn(f32) + Send + Sync>>,
    ) -> Result<Self> {
        let backend =
            LlamaBackend::init().map_err(|err| CoreError::BackendInit(err.to_string()))?;

        let mut models = Vec::with_capacity(2);
        for (role, spec) in [
            (ModelRole::Flash, &pair.flash),
            (ModelRole::Expert, &pair.expert),
        ] {
            models.push(load_one(&backend, profile, role, spec, progress.clone())?);
        }

        Ok(Self { backend, models })
    }

    /// Backend llama.cpp partagé (nécessaire pour créer des contextes).
    #[must_use]
    pub fn backend(&self) -> &LlamaBackend {
        &self.backend
    }

    /// Modèle chargé pour un rôle donné.
    #[must_use]
    pub fn get(&self, role: ModelRole) -> Option<&LoadedModel> {
        self.models.iter().find(|model| model.role == role)
    }

    /// Tous les modèles chargés (Flash puis Expert).
    #[must_use]
    pub fn models(&self) -> &[LoadedModel] {
        &self.models
    }
}

/// Charge un `ModelSpec` : résolution du chemin → métadonnées GGUF →
/// calcul `n_gpu_layers` → `llama_load_model_from_file`.
fn load_one(
    backend: &LlamaBackend,
    profile: &HardwareProfile,
    role: ModelRole,
    spec: &ModelSpec,
    progress: Option<Arc<dyn Fn(f32) + Send + Sync>>,
) -> Result<LoadedModel> {
    let path = resolve_model_path(spec)?;
    let size_bytes = std::fs::metadata(&path)
        .map_err(|source| CoreError::ModelLoad {
            path: path.clone(),
            reason: format!("cannot stat file: {source}"),
        })?
        .len();

    let n_layer = gguf_block_count(&path).unwrap_or(0);
    let n_gpu_layers = compute_gpu_layers(&profile.gpu_backend, size_bytes, n_layer);

    let mut params = LlamaModelParams::default()
        .with_n_gpu_layers(n_gpu_layers)
        .with_use_mmap(true);
    // `RamPinned` = Flash des tiers à faible RAM : mlock si le registry
    // le demande, pour éviter la pagination à la demande sur le chemin
    // critique. Les autres placements restent en mmap pur.
    if spec.placement == Placement::RamPinned {
        params = params.with_use_mlock(true);
    }
    if let Some(progress) = progress {
        params = params.with_progress_callback(move |value| {
            progress(value);
            true
        });
    }

    let model = LlamaModel::load_from_file(backend, &path, &params).map_err(|err| {
        CoreError::ModelLoad {
            path: path.clone(),
            reason: err.to_string(),
        }
    })?;

    tracing::info!(
        role = ?role,
        path = %path.display(),
        size_mb = size_bytes / quokka_hardware::MIB,
        n_gpu_layers,
        "model loaded"
    );

    Ok(LoadedModel {
        role,
        path,
        model,
        n_gpu_layers,
        placement: spec.placement,
    })
}

/// Résout le GGUF d'un `ModelSpec` : `~/.quokka/models/<filename>` puis
/// n'importe quel blob découvert (Ollama, LM Studio, cache HF).
fn resolve_model_path(spec: &ModelSpec) -> Result<PathBuf> {
    let filename = &spec.model.filename;

    if let Some(dir) = quokka_hardware::download::models_dir() {
        let candidate = dir.join(filename);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }

    let found = discovery::discover()
        .into_iter()
        .find(|model| model.path.file_name().and_then(|n| n.to_str()) == Some(filename.as_str()));
    if let Some(model) = found {
        return Ok(model.path);
    }

    Err(CoreError::ModelLoad {
        path: PathBuf::from(filename),
        reason: format!(
            "model not found locally — run `quokka pull` to download {}",
            spec.model.hf_url()
        ),
    })
}

/// Lit `block_count` depuis les métadonnées GGUF (`<arch>.block_count`).
/// `None` si le fichier n'est pas un GGUF lisible ou la clé absente.
fn gguf_block_count(path: &Path) -> Option<u32> {
    let gguf = GgufContext::from_file(path)?;
    let arch_idx = gguf.find_key("general.architecture");
    if arch_idx < 0 {
        return None;
    }
    let arch = gguf.val_str(arch_idx)?;
    let key = format!("{arch}.block_count");
    let idx = gguf.find_key(&key);
    (idx >= 0).then(|| gguf.val_u32(idx))
}

/// Calcule `n_gpu_layers` pour un modèle de `size_bytes` et `n_layer`
/// couches, étant donnée la VRAM dédiée déjà corrigée des marges.
///
/// Règle :
/// - pas de VRAM dédiée ou métadonnées illisibles → 0 (CPU pur) ou
///   offload total si le modèle tient entièrement (`u32::MAX` = toutes
///   les couches côté llama.cpp) ;
/// - sinon `floor(vram / taille_par_couche)` borné à `n_layer`, à
///   condition de couvrir ≥ 10 % des couches (en dessous, le surcoût
///   PCIe dépasse le gain).
///
/// `usable_vram_mb` doit être **net de réserve** : l'appelant applique
/// ~15 % + une marge pour le KV-cache GPU (estimation affinée en WP2.3
/// quand la taille de contexte sera connue).
#[must_use]
pub fn compute_gpu_layers(gpu_backend: &GpuBackend, size_bytes: u64, n_layer: u32) -> u32 {
    const VRAM_RESERVE_PERCENT: u64 = 85; // 15 % de réserve driver/KV

    let Some(vram_mb) = gpu_backend.dedicated_vram_mb() else {
        // Metal : mémoire unifiée — le modèle est déjà en RAM, offloader
        // reste possible via llama.cpp mais la VRD "dédiée" n'existe pas.
        // Politique : pas d'offload automatique sur Metal en v0.1.
        return 0;
    };
    let usable_mb = vram_mb * VRAM_RESERVE_PERCENT / 100;
    let size_mb = size_bytes / quokka_hardware::MIB;

    if n_layer == 0 || size_mb == 0 {
        // Sans métadonnées : tout ou rien — jamais de fraction aveugle.
        return if usable_mb >= size_mb { u32::MAX } else { 0 };
    }
    if usable_mb >= size_mb {
        return u32::MAX; // tient entièrement : toutes les couches
    }

    let per_layer_mb = size_mb / u64::from(n_layer);
    if per_layer_mb == 0 {
        return 0;
    }
    let layers = usable_mb / per_layer_mb;
    let layers = u32::try_from(layers).unwrap_or(u32::MAX).min(n_layer);
    // Sous ~10 % des couches, l'offload coûte plus cher en transferts
    // PCIe qu'il ne rapporte : on reste CPU pur.
    if layers * 10 < n_layer {
        return 0;
    }
    layers
}

#[cfg(test)]
mod tests {
    use super::*;
    use quokka_hardware::gpu::GpuBackend;

    fn cuda(vram_mb: u64) -> GpuBackend {
        GpuBackend::Cuda {
            name: "test".into(),
            vram_mb,
            device_count: 1,
        }
    }

    #[test]
    fn cpu_only_backend_never_offloads() {
        assert_eq!(compute_gpu_layers(&GpuBackend::None, 4 << 30, 32), 0);
        assert_eq!(
            compute_gpu_layers(
                &GpuBackend::Metal {
                    unified_ram_mb: 16_000
                },
                4 << 30,
                32
            ),
            0
        );
    }

    #[test]
    fn model_fitting_vram_offloads_fully() {
        // 4 Gio de poids, 8 Gio VRAM → 85 % = 6,8 Gio utiles > 4 Gio.
        assert_eq!(compute_gpu_layers(&cuda(8 * 1024), 4 << 30, 32), u32::MAX);
    }

    #[test]
    fn partial_offload_scales_with_layers() {
        // 9 Gio de poids / 36 couches = 0,25 Gio/couche ; VRAM 8 Gio
        // → 6,8 Gio utiles → ~27 couches.
        let layers = compute_gpu_layers(&cuda(8 * 1024), 9 << 30, 36);
        assert!(layers > 0 && layers < 36, "got {layers}");
    }

    #[test]
    fn tiny_vram_offloads_nothing() {
        assert_eq!(compute_gpu_layers(&cuda(1024), 9 << 30, 36), 0);
    }

    #[test]
    fn missing_metadata_is_all_or_nothing() {
        assert_eq!(compute_gpu_layers(&cuda(8 * 1024), 4 << 30, 0), u32::MAX);
        assert_eq!(compute_gpu_layers(&cuda(1024), 9 << 30, 0), 0);
    }
}
