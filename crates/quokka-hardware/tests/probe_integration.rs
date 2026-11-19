//! Probes de bout en bout contre l'hôte réel — tolérants à l'environnement
//! pour que la matrice CI (x86_64 + aarch64, avec ou sans GPU) reste verte.

use quokka_hardware::{discovery, registry, HardwareProfile, Tier};

#[test]
fn probe_returns_a_consistent_profile() {
    let profile = HardwareProfile::probe().expect("probing a booted host cannot fail");

    assert_eq!(profile.os, std::env::consts::OS);
    assert_eq!(profile.arch, std::env::consts::ARCH);
    assert!(profile.total_ram_mb > 0);
    assert!(profile.available_ram_mb > 0);
    assert!(profile.available_ram_mb <= profile.total_ram_mb);
    assert!(profile.cpu_cores >= 1);

    // Les drapeaux SIMD doivent être cohérents avec l'architecture d'exécution.
    #[cfg(target_arch = "x86_64")]
    assert!(!profile.has_neon);
    #[cfg(target_arch = "aarch64")]
    assert!(profile.has_neon);

    // classify_tier doit répondre sur n'importe quel hôte, GPU ou pas.
    let _tier = profile.classify_tier();
}

#[test]
fn recommended_pair_exists_for_the_probed_host() {
    let profile = HardwareProfile::probe().expect("probe");
    let pair = profile.recommended_pair().expect("registry");
    assert!(!pair.flash.model.repo_id.is_empty());
    assert!(!pair.expert.model.repo_id.is_empty());
    assert!(pair.flash.approx_size_mb < pair.expert.approx_size_mb);
}

#[test]
fn registry_load_uses_embedded_default_without_override() {
    // Sans ~/.quokka/profiles.json (le cas en CI), on retombe sur l'embarqué.
    let registry = registry::load().expect("registry");
    let pair = registry.tiers.get(Tier::Standard);
    assert!(pair.expert.approx_size_mb > pair.flash.approx_size_mb);
}

#[test]
fn discovery_never_panics_on_default_roots() {
    let _ = discovery::discover();
}
