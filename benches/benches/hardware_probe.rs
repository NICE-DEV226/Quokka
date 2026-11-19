//! Benchmarks du coût de démarrage du probe matériel. Le probe ne tourne
//! qu'une fois par lancement de process ; ces nombres protègent contre les
//! régressions accidentelles (ex. quelqu'un basculant vers
//! `System::new_all()`, qui scanne aussi processus et disques).

use criterion::{criterion_group, criterion_main, Criterion};
use quokka_hardware::{memory, registry, GpuBackend, HardwareProfile};

fn bench_probe(c: &mut Criterion) {
    c.bench_function("hardware/memory_refresh", |b| {
        b.iter(|| memory::probe().expect("memory probe"));
    });

    c.bench_function("hardware/full_profile_probe", |b| {
        b.iter(|| HardwareProfile::probe().expect("full probe"));
    });

    let profile = synthetic_workstation_profile();
    c.bench_function("hardware/tier_classification", |b| {
        b.iter(|| profile.classify_tier());
    });

    c.bench_function("hardware/registry_parse_embedded", |b| {
        b.iter(|| registry::embedded().expect("embedded registry"));
    });
}

/// Fixture statique : évite de coupler le bench de la matrice de décision à
/// l'hôte qui l'exécute.
fn synthetic_workstation_profile() -> HardwareProfile {
    HardwareProfile {
        os: "linux".into(),
        arch: "x86_64".into(),
        total_ram_mb: 31_949,
        available_ram_mb: 16_384,
        cpu_cores: 32,
        has_avx2: true,
        has_neon: false,
        gpu_backend: GpuBackend::Cuda {
            name: "NVIDIA GeForce RTX 4090".into(),
            vram_mb: 24_551,
            device_count: 1,
        },
        cpu_brand: "bench fixture".into(),
        physical_cores: Some(16),
        has_avx512f: true,
    }
}

criterion_group!(benches, bench_probe);
criterion_main!(benches);
