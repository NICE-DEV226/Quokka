# 0003 — Stack de probe matériel : `sysinfo` + `nvml-wrapper` + `ash`

- Statut : Accepted
- Date : 2026-11-19

## Contexte

`quokka-hardware` doit rapporter RAM libre, cœurs, SIMD (AVX2/AVX-512/NEON)
et VRAM GPU (CUDA / Metal / Vulkan) au démarrage, en < quelques ms, sur
Windows, Linux et macOS, **sans paniquer ni exiger un pilote installé** —
une machine sans NVML ou sans ICD Vulkan doit dégrader proprement vers le
backend suivant, jamais échouer.

## Décision

Trois crates spécialisées, chacune dégradant en `Option`/`None` :

- **`sysinfo`** : RAM totale/disponible (`MemAvailable` sur Linux),
  topologie CPU, refresh ciblé `RefreshKind::nothing().with_cpu(...)`.
- **`nvml-wrapper`** : VRAM et nom des GPU NVIDIA via NVML ; binding safe,
  chargé dynamiquement — absent ⇒ `GpuBackend::Cuda` non candidat.
- **`ash`** : instance Vulkan minimale (create → enumerate physical
  devices → destroy) pour les GPU **discrets** uniquement ; les iGPU sont
  exclus pour ne pas gonfler la VRAM rapportée.
- **SIMD** : `std::arch::is_x86_feature_detected!` /
  `is_aarch64_feature_detected!` — std, zéro dépendance.

Ordre de probe : CUDA → Metal (macOS, mémoire unifiée) → Vulkan → CPU-only.

## Conséquences

### Positives
- Chaque détecteur est isolé dans `gpu/{cuda,vulkan,metal}.rs` avec des
  blocs `unsafe` documentés (`// SAFETY:`) uniquement côté FFI Vulkan.
- Aucun appel shell (`nvidia-smi`, `system_profiler`) : pas de spawn de
  processus, pas de parsing fragile, coût microseconde.
- La distinction VRAM dédiée / mémoire unifiée est explicite dans
  `GpuBackend::dedicated_vram_mb()` — la matrice de tiers ne promeut pas
  un Mac 16 Go en Workstation.

### Négatives / risques
- `ash` est verbeux : ~120 lignes pour une instance d'énumération.
- NVML ne voit que NVIDIA ; AMD passe par Vulkan (couvre l'usage).
- Sur Wayland/headless exotique, certains heaps Vulkan DEVICE_LOCAL sont
  des carve-outs iGPU — filtrés par `MIN_DEDICATED_HEAP_BYTES` (512 Mio)
  et le type `DISCRETE_GPU`.

### Alternatives écartées
- Parser `/proc`, `lspci`, `nvidia-smi` : fragile, lent, privilèges.
- `wgpu`/`vulkano` pour le probe : frameworks de rendu, surdimensionnés
  pour lire `VkPhysicalDeviceMemoryProperties`.
- `sysinfo` seul pour le GPU : ne remonte pas la VRAM par backend.
