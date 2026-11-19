# 0001 — Bindings d'inférence : `llama-cpp-2` (FFI `llama.cpp`)

- Statut : Accepted
- Date : 2026-11-19

## Contexte

`quokka-core` doit charger des GGUF (Q4_K_M, IQ4_XS, Q8_0), gérer le KV-cache,
l'offload hybride CPU/GPU (`n_gpu_layers`) et streamer les tokens. Le moteur
devra supporter CUDA, Metal et Vulkan sur une matrice de compilation croisée
(§6 de ARCHITECTURE.md). Le budget « zéro allocation inutile / mmap » impose
un contrôle fin sur le cycle de vie des tenseurs et des contextes.

## Décision

Utiliser **`llama-cpp-2`** comme couche FFI vers `llama.cpp`, isolée derrière
une interface interne de `quokka-core`. Aucun autre moteur d'inférence
(Candle, burn, mistral.rs, ONNX generative) n'est retenu.

## Conséquences

### Positives
- Support natif et mature de CUDA / Metal / Vulkan / CPU SIMD (AVX2, NEON,
  AVX-512) sans écrire de kernels.
- `mmap` des poids et KV-cache persistant gérés en amont par llama.cpp.
- Binding Rust maintenu, généré depuis les headers C — la surface `unsafe`
  est confinée à un seul crate.

### Négatives / risques
- La compilation croisée CUDA/Metal exige une toolchain native par cible
  (nvcc, Xcode) — complexifie `release.yml`.
- L'API `llama.cpp` évolue vite : les montées de version du binding peuvent
  casser le build ; on épingle la version et on teste avant bump.
- `llama-cpp-2` tire `llama.cpp` en sous-module : la taille du tree de build
  est significative (atténuée par `strip = "symbols"` au release).

### Alternatives écartées
- `llama_cpp_sys` (bindings bruts) : plus de contrôle mais on réécrit la
  gestion des contextes/sequences à la main — surface d'erreurs trop grande.
- Candle / mistral.rs : pas de support Vulkan/Metal CPU-hybride équivalent,
  écosystème GGUF quantifié moins mature.
- ONNX Runtime pour la génération : viable pour de petits modèles, mais pas
  de KV-cache ni d'offload fin — réservé au classifieur du routeur (ADR-0002).
