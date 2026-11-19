//! Détection NVIDIA via NVML.
//!
//! `nvml-wrapper` charge `libnvidia-ml.so.1` dynamiquement : sur une machine
//! sans pile pilote NVIDIA, `Nvml::init()` échoue proprement et on renvoie
//! `None` — jamais d'erreur, jamais de lien compile-time.

use nvml_wrapper::Nvml;

/// Meilleur périphérique NVIDIA trouvé (VRAM max) + nombre total.
#[derive(Debug, Clone)]
pub struct CudaGpu {
    pub name: String,
    pub vram_bytes: u64,
    pub device_count: u32,
}

/// Détecte les GPU NVIDIA. Renvoie `None` si NVML ou le pilote est absent.
pub fn detect() -> Option<CudaGpu> {
    let nvml = match Nvml::init() {
        Ok(nvml) => nvml,
        Err(err) => {
            tracing::debug!("NVML unavailable ({err}), skipping CUDA detection");
            return None;
        }
    };

    let count = nvml.device_count().unwrap_or(0);
    if count == 0 {
        return None;
    }

    let mut best: Option<CudaGpu> = None;
    for index in 0..count {
        let Ok(device) = nvml.device_by_index(index) else {
            continue;
        };
        // `memory_info()` expose la capacité *physique* du framebuffer,
        // pas la tranche visible par l'OS.
        let (Ok(memory), Ok(name)) = (device.memory_info(), device.name()) else {
            continue;
        };

        if best.as_ref().is_none_or(|gpu| memory.total > gpu.vram_bytes) {
            best = Some(CudaGpu {
                name,
                vram_bytes: memory.total,
                device_count: count,
            });
        }
    }
    best
}
