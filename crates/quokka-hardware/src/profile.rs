//! Assemblage du [`HardwareProfile`] et matrice de décision des tiers
//! matériels (`docs/ARCHITECTURE.md §4.1`).

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::gpu::GpuBackend;
use crate::memory::MIB;
use crate::registry::ModelPair;

/// Instantané complet de l'hôte, tel que défini par `docs/ARCHITECTURE.md §4.1`.
///
/// Tous les champs `*_mb` sont en MiB (`bytes >> 20`) malgré le suffixe
/// historique du spec.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HardwareProfile {
    /// "windows" | "linux" | "macos".
    pub os: String,
    /// "x86_64" | "aarch64" | …
    pub arch: String,
    /// RAM physique totale.
    pub total_ram_mb: u64,
    /// RAM utilisable sans swapper *maintenant* (Linux : `MemAvailable`).
    pub available_ram_mb: u64,
    /// Cœurs logiques (threads SMT).
    pub cpu_cores: usize,
    pub has_avx2: bool,
    pub has_neon: bool,
    /// Backend GPU + VRAM dédiée.
    pub gpu_backend: GpuBackend,
    // --- Contexte étendu (télémétrie, Flex Card) -------------------------
    /// Nom marketing du CPU (peut être vide sous hyperviseur).
    pub cpu_brand: String,
    /// Cœurs physiques si l'OS les expose.
    pub physical_cores: Option<usize>,
    pub has_avx512f: bool,
}

impl HardwareProfile {
    /// Probe l'hôte complet : RAM → topologie CPU + SIMD → backend GPU.
    ///
    /// One-shot, mono-thread, sans verrou ; coût de quelques ms au démarrage
    /// (dominé par l'aller-retour d'instance Vulkan quand aucun CUDA n'est
    /// présent).
    pub fn probe() -> Result<Self> {
        let memory = crate::memory::probe()?;
        let cpu = crate::cpu::probe()?;
        let gpu_backend = crate::gpu::detect_backend();

        Ok(Self {
            os: std::env::consts::OS.into(),
            arch: cpu.arch,
            total_ram_mb: memory.total_bytes / MIB,
            available_ram_mb: memory.available_bytes / MIB,
            cpu_cores: cpu.logical_cores,
            has_avx2: cpu.x86_simd.avx2,
            has_neon: cpu.has_neon,
            gpu_backend,
            cpu_brand: cpu.brand,
            physical_cores: cpu.physical_cores,
            has_avx512f: cpu.x86_simd.avx512f,
        })
    }

    /// Classe l'hôte en [`Tier`] (matrice de décision §4.1).
    #[must_use]
    pub fn classify_tier(&self) -> Tier {
        Tier::from_hardware(self.total_ram_mb, self.gpu_backend.dedicated_vram_mb())
    }

    /// Duo de modèles recommandé pour cet hôte, issu du registre actif
    /// (override utilisateur `~/.quokka/profiles.json`, sinon l'embarqué).
    pub fn recommended_pair(&self) -> Result<ModelPair> {
        Ok(crate::registry::load()?
            .tiers
            .get(self.classify_tier())
            .clone())
    }
}

/// Tiers matériel pilotant la sélection du duo de modèles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    /// ≤ 8 Go RAM, pas de dGPU — machine portable.
    Lightweight,
    /// 16 Go RAM ou 4–6 Go de VRAM — PC / Mac standard.
    Standard,
    /// ≥ 32 Go RAM ou ≥ 12 Go de VRAM — station de travail.
    Workstation,
}

impl Tier {
    /// Matrice de décision. Les seuils sont en vrais MiB et gardent une marge
    /// sous les capacités marketing (une machine « 16 Go » rapporte
    /// ~15,3 Gio ; une carte « 12 Go » ~11,9 Gio de VRAM exploitable) :
    ///
    /// - **Workstation** : RAM ≥ 28 Gio **ou** VRAM dédiée ≥ 11 Gio ;
    /// - **Standard** : RAM ≥ 14 Gio **ou** VRAM dédiée ≥ 3,5 Gio ;
    /// - **Lightweight** : tout le reste.
    ///
    /// La mémoire unifiée (Metal) est volontairement exclue de la condition
    /// VRAM : un Mac 16 Go est une machine Standard, pas une Workstation.
    #[must_use]
    pub fn from_hardware(total_ram_mb: u64, dedicated_vram_mb: Option<u64>) -> Self {
        const WORKSTATION_RAM_MB: u64 = 28 * 1024;
        const WORKSTATION_VRAM_MB: u64 = 11 * 1024;
        const STANDARD_RAM_MB: u64 = 14 * 1024;
        const STANDARD_VRAM_MB: u64 = 3584;

        let vram_mb = dedicated_vram_mb.unwrap_or(0);
        if total_ram_mb >= WORKSTATION_RAM_MB || vram_mb >= WORKSTATION_VRAM_MB {
            Self::Workstation
        } else if total_ram_mb >= STANDARD_RAM_MB || vram_mb >= STANDARD_VRAM_MB {
            Self::Standard
        } else {
            Self::Lightweight
        }
    }
}

impl std::fmt::Display for Tier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Lightweight => "Lightweight",
            Self::Standard => "Standard",
            Self::Workstation => "Workstation",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(total_ram_mb: u64, gpu_backend: GpuBackend) -> HardwareProfile {
        HardwareProfile {
            os: "linux".into(),
            arch: "x86_64".into(),
            total_ram_mb,
            available_ram_mb: total_ram_mb / 2,
            cpu_cores: 8,
            has_avx2: true,
            has_neon: false,
            gpu_backend,
            cpu_brand: "test fixture".into(),
            physical_cores: Some(4),
            has_avx512f: false,
        }
    }

    #[test]
    fn tier_matrix_follows_architecture_spec() {
        use GpuBackend::{Cuda, Metal, None as NoGpu, Vulkan};

        // Laptop « 8 Go » (7,4 Gio rapportés), iGPU seul → Lightweight.
        assert_eq!(profile(7600, NoGpu).classify_tier(), Tier::Lightweight);

        // Laptop « 16 Go » (15,3 Gio rapportés), sans dGPU → Standard.
        assert_eq!(profile(15_667, NoGpu).classify_tier(), Tier::Standard);

        // Station « 32 Go » (31,2 Gio rapportés) → Workstation.
        assert_eq!(profile(31_949, NoGpu).classify_tier(), Tier::Workstation);

        // dGPU 12 Go : promeut une machine 8 Go en Workstation.
        assert_eq!(
            profile(
                7600,
                Cuda {
                    name: "RTX 4070 Ti".into(),
                    vram_mb: 12_238,
                    device_count: 1
                }
            )
            .classify_tier(),
            Tier::Workstation
        );

        // dGPU 4 Go : monte une machine 8 Go en Standard.
        assert_eq!(
            profile(
                7600,
                Vulkan {
                    name: "Radeon RX 580".into(),
                    vram_mb: 4054,
                    device_count: 1
                }
            )
            .classify_tier(),
            Tier::Standard
        );

        // Mac 16 Go : la mémoire unifiée ne compte PAS comme VRAM dédiée.
        assert_eq!(
            profile(15_667, Metal { unified_ram_mb: 15_667 }).classify_tier(),
            Tier::Standard
        );

        // Mac 32 Go → Workstation via la RAM, pas via l'unifié.
        assert_eq!(
            profile(31_949, Metal { unified_ram_mb: 31_949 }).classify_tier(),
            Tier::Workstation
        );
    }

    #[test]
    fn tier_boundaries_are_exact() {
        assert_eq!(Tier::from_hardware(14_335, None), Tier::Lightweight);
        assert_eq!(Tier::from_hardware(14_336, None), Tier::Standard);
        assert_eq!(Tier::from_hardware(28_671, None), Tier::Standard);
        assert_eq!(Tier::from_hardware(28_672, None), Tier::Workstation);
        assert_eq!(Tier::from_hardware(0, Some(11_264)), Tier::Workstation);
        assert_eq!(Tier::from_hardware(0, Some(3584)), Tier::Standard);
        assert_eq!(Tier::from_hardware(0, Some(3583)), Tier::Lightweight);
    }
}
