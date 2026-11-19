//! Registre matériel → GGUF (« auto-tuner »).
//!
//! Le registre vit dans `registry/profiles.json` à la racine du dépôt et est
//! **embarqué à la compilation** (binaire unique zéro-config). L'utilisateur
//! peut le surcharger via `~/.quokka/profiles.json` (même schéma) — utile
//! pour les machines air-gap ou une curation de modèles personnalisée.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{HardwareError, Result};
use crate::profile::Tier;

/// Registre livré dans le dépôt, compilé dans le binaire.
pub const EMBEDDED_REGISTRY: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../registry/profiles.json"
));

/// Où les poids doivent résider une fois chargés (pilote `n_gpu_layers`,
/// `use_mlock` dans `quokka-core`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Placement {
    /// Toujours résident en RAM (`use_mlock = true`) — le modèle Flash des
    /// tiers à faible RAM.
    RamPinned,
    /// Backé `mmap`, paginé à la demande — le modèle Expert des tiers à
    /// faible RAM.
    MmapOnDemand,
    /// Offload GPU partiel + `mmap` pour le reste.
    HybridOffload,
    /// Entièrement déchargé en VRAM.
    GpuFull,
}

/// Pointeur minimal vers un fichier GGUF sur Hugging Face.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRef {
    /// Dépôt HF, ex. "bartowski/Qwen2.5-3B-Instruct-GGUF".
    pub repo_id: String,
    /// Nom du fichier GGUF dans le dépôt.
    pub filename: String,
}

impl ModelRef {
    /// URL de téléchargement direct (redirige vers l'objet LFS du CDN).
    #[must_use]
    pub fn hf_url(&self) -> String {
        format!(
            "https://huggingface.co/{}/resolve/main/{}",
            self.repo_id, self.filename
        )
    }
}

/// Recommandation de modèle complète.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelSpec {
    #[serde(flatten)]
    pub model: ModelRef,
    /// Étiquette de quantification, ex. "Q4_K_M".
    pub quantization: String,
    /// Taille approximative de téléchargement/emprise, en MiB.
    pub approx_size_mb: u64,
    /// Stratégie de placement en mémoire/VRAM.
    pub placement: Placement,
    /// Builds alternatifs équivalents (même créneau du tier).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub alternates: Vec<ModelRef>,
}

/// Le duo Flash/Expert d'un tier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelPair {
    /// Modèle 1B–3B toujours résident pour les prompts simples.
    pub flash: ModelSpec,
    /// Modèle 7B–32B pour le raisonnement/code complexe.
    pub expert: ModelSpec,
}

/// Duos de modèles par tier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TierMap {
    pub lightweight: ModelPair,
    pub standard: ModelPair,
    pub workstation: ModelPair,
}

impl TierMap {
    /// Duo du tier demandé.
    #[must_use]
    pub fn get(&self, tier: Tier) -> &ModelPair {
        match tier {
            Tier::Lightweight => &self.lightweight,
            Tier::Standard => &self.standard,
            Tier::Workstation => &self.workstation,
        }
    }
}

/// Document racine de `profiles.json`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelRegistry {
    /// Version de schéma (à incrémenter sur changement cassant).
    pub version: u32,
    pub tiers: TierMap,
}

/// `~/.quokka/profiles.json`, si l'utilisateur en a créé un.
#[must_use]
pub fn user_override_path() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(|home| PathBuf::from(home).join(".quokka").join("profiles.json"))
}

/// Charge le registre actif : override utilisateur s'il existe (erreur forte
/// et explicite sur fichier malformé — une mauvaise config ne doit pas
/// silencieusement dégrader), sinon l'embarqué.
pub fn load() -> Result<ModelRegistry> {
    load_from(user_override_path().as_deref())
}

/// Variante injectable de [`load`] pour les tests (pas de mutation d'env).
pub fn load_from(override_path: Option<&Path>) -> Result<ModelRegistry> {
    match override_path.filter(|path| path.is_file()) {
        Some(path) => {
            let raw =
                std::fs::read_to_string(path).map_err(|source| HardwareError::RegistryIo {
                    path: path.display().to_string(),
                    source,
                })?;
            serde_json::from_str(&raw).map_err(|source| HardwareError::RegistryParse {
                path: path.display().to_string(),
                source,
            })
        }
        None => embedded(),
    }
}

/// Parse le registre embarqué.
pub fn embedded() -> Result<ModelRegistry> {
    serde_json::from_str(EMBEDDED_REGISTRY).map_err(|source| HardwareError::RegistryParse {
        path: "<embedded registry/profiles.json>".to_owned(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_registry_is_valid_and_complete() {
        let registry = embedded().expect("embedded registry must always parse");
        assert_eq!(registry.version, 1);
        for tier in [Tier::Lightweight, Tier::Standard, Tier::Workstation] {
            let pair = registry.tiers.get(tier);
            assert!(!pair.flash.model.repo_id.is_empty());
            assert!(!pair.expert.model.repo_id.is_empty());
            // Flash strictement plus petit que Expert au sein d'un tier.
            assert!(
                pair.flash.approx_size_mb < pair.expert.approx_size_mb,
                "tier {tier}"
            );
        }
    }

    #[test]
    fn hf_url_points_at_resolve_endpoint() {
        let reference = ModelRef {
            repo_id: "bartowski/Qwen2.5-3B-Instruct-GGUF".into(),
            filename: "Qwen2.5-3B-Instruct-Q8_0.gguf".into(),
        };
        assert_eq!(
            reference.hf_url(),
            "https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q8_0.gguf"
        );
    }

    #[test]
    fn malformed_override_fails_loudly() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let bad = tmp.path().join("profiles.json");
        std::fs::write(&bad, "{ not json").expect("write malformed override");

        let error = load_from(Some(&bad)).expect_err("malformed override must fail");
        assert!(matches!(error, HardwareError::RegistryParse { .. }));
    }

    #[test]
    fn missing_override_falls_back_to_embedded() {
        let missing = PathBuf::from("/nonexistent/quokka-test-profiles.json");
        assert!(load_from(Some(&missing)).is_ok());
    }
}
