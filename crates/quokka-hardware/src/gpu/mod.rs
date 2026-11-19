//! Détection du backend GPU.
//!
//! Ordre de probe (premier trouvé gagne) : **CUDA** (NVML) → **Metal**
//! (macOS, mémoire unifiée) → **Vulkan** (GPU discrets uniquement) → `None`.
//! Tous les détecteurs chargent leur bibliothèque pilote dynamiquement à
//! l'exécution : une machine sans pilote passe simplement au backend
//! suivant — le probe ne échoue jamais, il dégrade.

mod cuda;
#[cfg(target_os = "macos")]
pub mod metal;
mod vulkan;

use serde::{Deserialize, Serialize};

use crate::memory::MIB;

/// Backend de calcul pilotant les décisions d'offload `n_gpu_layers`.
///
/// Tous les champs `*_mb` sont en MiB (`bytes >> 20`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuBackend {
    /// GPU NVIDIA exposé via NVML.
    Cuda {
        /// Nom marketing du périphérique, ex. "NVIDIA GeForce RTX 4070".
        name: String,
        /// VRAM du plus gros périphérique, en MiB.
        vram_mb: u64,
        /// Nombre de périphériques CUDA détectés.
        device_count: u32,
    },
    /// Mémoire unifiée Apple Silicon (budget VRAM == RAM).
    Metal {
        /// RAM unifiée totale, en MiB.
        unified_ram_mb: u64,
    },
    /// GPU discret exposé via un ICD Vulkan.
    Vulkan {
        /// Nom marketing issu de `VkPhysicalDeviceProperties`.
        name: String,
        /// Plus grand heap DEVICE_LOCAL du meilleur périphérique, en MiB.
        vram_mb: u64,
        /// Nombre de GPU discrets détectés.
        device_count: u32,
    },
    /// Aucun accélérateur exploitable : inférence CPU-only.
    None,
}

impl GpuBackend {
    /// Identifiant stable pour logs/télémétrie.
    #[must_use]
    pub fn backend_name(&self) -> &'static str {
        match self {
            Self::Cuda { .. } => "cuda",
            Self::Metal { .. } => "metal",
            Self::Vulkan { .. } => "vulkan",
            Self::None => "none",
        }
    }

    /// VRAM dédiée en MiB. Renvoie `None` pour CPU-only et pour Metal :
    /// la mémoire unifiée n'est *pas* de la VRAM dédiée et passe par les
    /// seuils RAM de la matrice de tiers (un Mac 16 Go ne doit pas être
    /// promu Workstation comme s'il avait un dGPU 16 Go).
    #[must_use]
    pub fn dedicated_vram_mb(&self) -> Option<u64> {
        match self {
            Self::Cuda { vram_mb, .. } | Self::Vulkan { vram_mb, .. } => Some(*vram_mb),
            Self::Metal { .. } | Self::None => None,
        }
    }

    /// Nom du périphérique, si un GPU a été trouvé.
    #[must_use]
    pub fn device_name(&self) -> Option<&str> {
        match self {
            Self::Cuda { name, .. } | Self::Vulkan { name, .. } => Some(name.as_str()),
            Self::Metal { .. } | Self::None => None,
        }
    }
}

/// Détecte le meilleur backend GPU disponible.
pub fn detect_backend() -> GpuBackend {
    if let Some(gpu) = cuda::detect() {
        return GpuBackend::Cuda {
            name: gpu.name,
            vram_mb: gpu.vram_bytes / MIB,
            device_count: gpu.device_count,
        };
    }

    #[cfg(target_os = "macos")]
    {
        if let Some(unified_ram_bytes) = metal::detect() {
            return GpuBackend::Metal {
                unified_ram_mb: unified_ram_bytes / MIB,
            };
        }
    }

    if let Some(gpu) = vulkan::detect() {
        return GpuBackend::Vulkan {
            name: gpu.name,
            vram_mb: gpu.vram_bytes / MIB,
            device_count: gpu.device_count,
        };
    }

    tracing::debug!("no CUDA/Metal/Vulkan backend available, falling back to CPU-only");
    GpuBackend::None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedicated_vram_excludes_metal_and_none() {
        let cuda = GpuBackend::Cuda {
            name: "RTX 4070".into(),
            vram_mb: 11_996,
            device_count: 1,
        };
        assert_eq!(cuda.dedicated_vram_mb(), Some(11_996));
        assert_eq!(cuda.backend_name(), "cuda");
        assert_eq!(cuda.device_name(), Some("RTX 4070"));

        let metal = GpuBackend::Metal {
            unified_ram_mb: 15_667,
        };
        assert_eq!(metal.dedicated_vram_mb(), None);

        assert_eq!(GpuBackend::None.dedicated_vram_mb(), None);
        assert_eq!(GpuBackend::None.backend_name(), "none");
    }
}
