<div align="center">
  <img src="brand/mascot.svg" alt="Quokka" width="220" />

  # QUOKKA

  **Routeur IA local et moteur d'inférence haute performance — un seul binaire, zéro configuration.**

  [![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
  [![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](https://www.rust-lang.org)
</div>

---

## Le problème

Les runners d'inférence locaux (Ollama, LM Studio) vous obligent à choisir vos modèles à la main et saturent la RAM/VRAM de votre PC. Quokka scanne le matériel au démarrage, installe le duo de modèles optimal (*Flash* + *Expert*) et aiguille chaque requête en **< 2 ms** vers le bon modèle — le petit pour le trivial, le gros pour le raisonnement et le code.

- **Hardware-aware** : sonde RAM, SIMD (AVX2/AVX-512/NEON) et VRAM (CUDA/Metal/Vulkan), puis calibre la quantization et l'offload GPU (`n_gpu_layers`) tout seul.
- **Mémoire sobre** : poids en `mmap`, KV-cache évincé vers le disque (LZ4), swap Flash ↔ Expert sans OOM.
- **Compatible partout** : API OpenAI (`/v1/chat/completions`, SSE, tool calling) et serveur MCP natif.

## État actuel — Phase 1

Le workspace est en place ; le moteur d'inférence arrive en Phase 2. Aujourd'hui le binaire sait **profiler votre machine** et **télécharger les modèles recommandés** :

```bash
# Build depuis les sources (Rust ≥ 1.85)
cargo build --release -p quokka

# 1. Sonde CPU/RAM/SIMD/GPU, affiche le tier et le duo Flash/Expert recommandé
./target/release/quokka hardware

# 2. Télécharge le duo recommandé (ou une cible précise) vers ~/.quokka/models
./target/release/quokka pull auto        # paire du tier de cette machine
./target/release/quokka pull flash       # ou : expert, lightweight, standard, workstation
```

Sortie JSON disponible : `quokka hardware --json`.

## Roadmap

| Phase | Contenu | Statut |
| --- | --- | --- |
| 1 | Probe matériel, registre de tiers, découverte GGUF, downloader HF | ✅ fait |
| 2 | `quokka-core` (`llama.cpp` FFI, `mmap`), API OpenAI SSE | 🚧 en cours |
| 3 | Routeur < 2 ms, KV-cache LZ4, `MemoryPressureFallback` | à venir |
| 4 | Serveur MCP, Web UI embarquée, releases multi-plateformes | à venir |

## Architecture

Voir [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) pour la spécification complète et [`docs/adr/`](docs/adr/) pour les décisions structurantes.

```
crates/
  quokka-hardware/   probe matériel, auto-tuner, registre, downloader HF, découverte GGUF
  quokka-core/       (Phase 2) FFI llama.cpp, mmap, KV-cache
  quokka-router/     (Phase 3) classifieur < 2 ms
  quokka-server/     (Phase 4) API OpenAI, MCP, Web UI
  quokka-telemetry/  (Phase 4) métriques
cli/                 binaire `quokka` (clap)
registry/            profiles.json — mapping matériel → GGUF
benches/             benchmarks criterion
```

## Licence

Apache-2.0 — voir [`LICENSE`](LICENSE).
