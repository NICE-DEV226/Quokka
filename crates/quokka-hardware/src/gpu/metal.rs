//! Détection Apple Silicon (macOS uniquement).
//!
//! Les GPU Apple utilisent une **mémoire unifiée** : il n'existe pas de VRAM
//! dédiée. On rapporte la RAM totale comme budget unifié ; la couche de
//! placement runtime soustraira la réserve macOS (~4 Gio) au moment de
//! calculer la taille réelle d'offload.

use crate::memory;

/// Mémoire unifiée totale en octets, si le probe a réussi.
pub fn detect() -> Option<u64> {
    memory::probe().ok().map(|mem| mem.total_bytes)
}
