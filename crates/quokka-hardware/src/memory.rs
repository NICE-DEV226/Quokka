//! Probe RAM : mémoire totale vs *disponible*.
//!
//! « Disponible » est le chiffre qui compte pour le placement des modèles :
//! sur Linux, `sysinfo` lit `/proc/meminfo::MemAvailable` (mémoire utilisable
//! sans toucher au swap), et *pas* le compteur « free » qui exclut à tort le
//! page cache reclaimable.

use serde::{Deserialize, Serialize};
use sysinfo::System;

use crate::error::{HardwareError, Result};

/// Octets par MiB — toutes les tailles publiques de ce crate sont en MiB.
pub const MIB: u64 = 1024 * 1024;

/// Mémoire totale vs disponible, en octets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryInfo {
    /// RAM physique installée.
    pub total_bytes: u64,
    /// Mémoire utilisable immédiatement sans swapper (Linux : `MemAvailable`).
    pub available_bytes: u64,
}

impl MemoryInfo {
    /// Total en MiB.
    #[must_use]
    pub fn total_mib(&self) -> u64 {
        self.total_bytes / MIB
    }

    /// Disponible en MiB.
    #[must_use]
    pub fn available_mib(&self) -> u64 {
        self.available_bytes / MIB
    }
}

/// Probe la mémoire système par le chemin le moins cher et le plus précis :
/// un rafraîchissement *memory-only* (pas de scan CPU/disques/processus).
pub fn probe() -> Result<MemoryInfo> {
    let mut system = System::new();
    system.refresh_memory();

    let total_bytes = system.total_memory();
    if total_bytes == 0 {
        return Err(HardwareError::Probe {
            reason: "memory probe returned 0 bytes of total RAM".into(),
        });
    }

    // Certains runtimes de conteneurs remontent des valeurs cgroup absurdes :
    // on clampe défensivement plutôt que de propager un invariant cassé.
    let available_bytes = system.available_memory().min(total_bytes);

    Ok(MemoryInfo {
        total_bytes,
        available_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_reports_sane_memory() {
        let mem = probe().expect("memory probe cannot fail on a booted host");
        assert!(mem.total_bytes > 0);
        assert!(mem.available_bytes > 0);
        assert!(mem.available_bytes <= mem.total_bytes);
        assert_eq!(mem.total_mib(), mem.total_bytes / MIB);
    }
}
