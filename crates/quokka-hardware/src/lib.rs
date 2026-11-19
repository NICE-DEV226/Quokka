//! Probe matériel, auto-tuner et découverte GGUF locale pour le moteur
//! d'inférence Quokka.
//!
//! Ce crate répond à une seule question, aussi précisément et aussi
//! bon marché que possible : *« Que peut faire tourner cette machine
//! maintenant ? »* Il produit un [`HardwareProfile`] (RAM, topologie CPU,
//! capacités SIMD, backend GPU + VRAM), classe l'hôte en [`Tier`] suivant la
//! matrice de décision de `docs/ARCHITECTURE.md §4.1`, et recommande le duo
//! de modèles (*Flash* + *Expert*) issu du `registry/profiles.json` embarqué.
//!
//! Règles de conception :
//! - Le probe est **one-shot, mono-thread et sans verrou** : il ne tourne
//!   qu'au démarrage, aucun besoin de primitive de synchronisation.
//! - Chaque probe dégrade proprement : une bibliothèque NVML ou un ICD
//!   Vulkan absent donne `GpuBackend::None`, jamais une erreur. Les échecs
//!   durs (registre imparsable) remontent en [`HardwareError`].
//! - Toutes les tailles sont en MiB (`bytes >> 20`) pour garder la matrice
//!   de décision en arithmétique entière.
//!
//! # Exemple
//! ```no_run
//! use quokka_hardware::HardwareProfile;
//!
//! let profile = HardwareProfile::probe().expect("hardware probe");
//! println!("tier: {}", profile.classify_tier());
//! let pair = profile.recommended_pair().expect("registry");
//! println!("flash: {}", pair.flash.model.hf_url());
//! ```

pub mod cpu;
pub mod discovery;
pub mod download;
pub mod error;
pub mod gpu;
pub mod memory;
pub mod profile;
pub mod registry;

pub use error::{HardwareError, Result};
pub use gpu::GpuBackend;
pub use memory::MIB;
pub use profile::{HardwareProfile, Tier};
pub use registry::{ModelPair, ModelRef, ModelRegistry, ModelSpec, Placement};
