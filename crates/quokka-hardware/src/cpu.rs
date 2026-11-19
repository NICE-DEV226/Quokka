//! Probe CPU : architecture, topologie (cœurs logiques/physiques), nom
//! modèle et bits SIMD (`AVX2`/`AVX-512`/`FMA` sur x86_64, `NEON` sur
//! aarch64) qui pilotent la sélection des kernels `llama.cpp`.

use serde::{Deserialize, Serialize};
use sysinfo::{CpuRefreshKind, RefreshKind, System};

use crate::error::{HardwareError, Result};

/// Bits SIMD x86_64 pertinents pour les kernels GGML.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct X86Simd {
    /// Vecteurs 256 bits entiers/flottants — socle d'une inférence CPU correcte.
    pub avx2: bool,
    /// Fondations 512 bits — débloque les kernels GGML larges si présent.
    pub avx512f: bool,
    /// Multiply-add fusionné, systématiquement couplé à AVX2 en pratique.
    pub fma: bool,
}

/// Instantané du CPU hôte.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    /// `std::env::consts::ARCH` à la compilation ("x86_64", "aarch64", …).
    pub arch: String,
    /// Nom marketing lu dans /proc/cpuinfo (ou équivalent), ex.
    /// "AMD Ryzen 9 7950X 16-Core Processor". Peut être vide sous certains
    /// hyperviseurs.
    pub brand: String,
    /// Cœurs logiques (threads SMT).
    pub logical_cores: usize,
    /// Cœurs physiques si l'OS les expose.
    pub physical_cores: Option<usize>,
    /// Drapeaux SIMD ; tous `false` sur hôte non-x86_64.
    pub x86_simd: X86Simd,
    /// NEON (toujours `true` sur aarch64, où NEON est obligatoire).
    pub has_neon: bool,
}

/// Probe la topologie CPU et les capacités SIMD.
pub fn probe() -> Result<CpuInfo> {
    // Rafraîchissement CPU seul : évite le coût du scan disques/processus.
    let system =
        System::new_with_specifics(RefreshKind::nothing().with_cpu(CpuRefreshKind::everything()));

    let logical_cores = system.cpus().len();
    if logical_cores == 0 {
        return Err(HardwareError::Probe {
            reason: "CPU probe returned zero logical cores".into(),
        });
    }

    let brand = system
        .cpus()
        .first()
        .map(|cpu| cpu.brand().trim().to_owned())
        .unwrap_or_default();
    if brand.is_empty() {
        tracing::debug!("CPU brand string empty (common under some hypervisors)");
    }

    Ok(CpuInfo {
        arch: std::env::consts::ARCH.into(),
        brand,
        logical_cores,
        physical_cores: system.physical_core_count(),
        x86_simd: probe_x86_simd(),
        has_neon: probe_neon(),
    })
}

/// Détection SIMD à l'exécution. `is_x86_feature_detected!` gère
/// correctement les vérifications CPUID multi-fournisseurs.
fn probe_x86_simd() -> X86Simd {
    #[cfg(target_arch = "x86_64")]
    {
        X86Simd {
            avx2: std::arch::is_x86_feature_detected!("avx2"),
            avx512f: std::arch::is_x86_feature_detected!("avx512f"),
            fma: std::arch::is_x86_feature_detected!("fma"),
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        X86Simd::default()
    }
}

/// NEON est architecturalement garanti sur aarch64 ; on vérifie quand même
/// par symétrie avec le chemin x86 (et pour les cibles aarch32 éventuelles).
fn probe_neon() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_reports_sane_cpu() {
        let cpu = probe().expect("cpu probe cannot fail on a booted host");
        assert!(cpu.logical_cores >= 1);
        assert_eq!(cpu.arch, std::env::consts::ARCH);

        #[cfg(target_arch = "x86_64")]
        assert!(!cpu.has_neon);
        #[cfg(target_arch = "aarch64")]
        assert!(cpu.has_neon);
        #[cfg(not(target_arch = "x86_64"))]
        assert!(!cpu.x86_simd.avx2);
    }
}
