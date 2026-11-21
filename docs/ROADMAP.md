# QUOKKA — Engineering Roadmap (v0.1 → v1.0)

Deep-study companion to [`ARCHITECTURE.md`](ARCHITECTURE.md). Where the
spec describes *what* Quokka is, this document describes *how we get there*:
sequenced work packages, hard dependencies, version risks discovered during
the ecosystem survey (Nov 2026), and the verification gate closing each phase.

---

## 0. Current state audit (as of commit `b09d97c`)

| Area | State | Evidence |
| --- | --- | --- |
| Workspace & CI | ✅ Done | `cargo test`/`clippy -D warnings`/`fmt`/`cargo-deny` green on `main` |
| `quokka-hardware` | ✅ Done | RAM (`MemAvailable`), CPU topology, SIMD runtime-detect, GPU probe CUDA→Metal→Vulkan→none, tier matrix, GGUF discovery (5 stores), HF downloader (ranged parallel + sha256 + resume) |
| Registry | ✅ Done | `profiles.json` embedded + `~/.quokka/profiles.json` override |
| CLI | ✅ Done | `hardware`, `pull`, `run`, `status`, `bench` live |
| Inference (`quokka-core`) | ✅ Done | `llama-cpp-2` pinned, mmap load, persistent `InferenceSession` per worker, KV prefix reuse (`cached_tokens` in usage), Q8 KV + flash attention, `GGML_CPU_REPACK=OFF` |
| Router (`quokka-router`) | ✅ Done | Stage 1 lexical + domain markers, `routing.toml` rules/`[policy]`, Stage 2 centroid-IDF classifier (ADR-0004, ONNX deferred), memory guard, `model=auto` + `X-Quokka-*` headers, corpus 137 prompts @ 100 %, criterion bench ~0,3–5,5 µs |
| Server / Web UI | ✅ Done | OpenAI JSON+SSE, `auto` model, embedded Svelte UI |
| KV-cache LZ4 offload | ✅ Done | `state_save_file`/`state_load_file` + LZ4, session_key = 1er msg user, LRU 8/model, `~/.quokka/cache/kv` — vérifié live : retour conversation → `cached_tokens` = prompt+généré restaurés |
| MCP / Telemetry | ✅ Done | `quokka mcp` stdio (rmcp 3.11) + `POST /v1/route` + `GET /telemetry` (quokka-telemetry: tokens, cache_hit_ratio, histogrammes routage) — vérifiés live |

### Known gaps to close inside Phase 2 prep

1. **No end-to-end download test.** `download::pull` has unit tests for
   segment planning and ETag parsing but has never pulled a real GGUF.
   Action: run `quokka pull flash` on a reference machine and record
   throughput in this doc.
2. **Parallel-mode resume shortcut.** Documented in `download.rs`: a stale
   `.part` is rewritten from zero. Acceptable for v0.1; revisit if users on
   flaky links report re-downloads of 9 GB experts.
3. **GPU probe coverage.** CUDA path is exercised on Linux/NVML; Vulkan and
   Metal paths are compile-checked but untested on real dGPU/iGPU hardware.
   Action: one manual run on an AMD/NVIDIA dGPU machine and one Apple
   Silicon Mac before Phase 2 sign-off.
4. **CI is Ubuntu-only.** `cargo-deny` graph targets include Windows/macOS
   triples but nothing compiles them. Add a smoke `cargo check` matrix in
   Phase 2 when `llama-cpp-2` feature flags make cross-compilation real.

---

## 1. Ecosystem survey — version risks (Nov 2026)

These facts change the plan versus what `ARCHITECTURE.md §3` assumed:

| Dependency | Latest | Risk | Mitigation |
| --- | --- | --- | --- |
| `llama-cpp-2` | `0.1.161` (2026-11-03) | **Does not follow semver.** Changelog-driven breakage is routine; bindgen also requires `clang`/`libclang` at build time. | Pin `=0.1.x` exactly; bump deliberately behind a CI build; isolate all FFI in `quokka-core`; keep `dynamic-link` feature as fallback for distros without CUDA toolkit. |
| `ort` (ONNX Runtime) | `2.0.0-rc.15` (2026-10-19) | Still **release candidate**, MSRV **1.92**, ships native `onnxruntime` binaries (CUDA 13 only since rc.13). | Phase 3 decision gate (§3.2): `ort` vs `rten` (pure Rust, no native dep) vs `candle`. If `ort` is chosen, workspace MSRV must move to 1.92. |
| `rmcp` (official MCP SDK) | `3.11.0` (2026-11-17) | MSRV **1.88** — above our current `rust-version = 1.85"`. Implements spec `2026-07-28`, back-compatible with `2025-11-25`. | Bump workspace `rust-version` to **1.92** in Phase 2 (single bump covering both `rmcp` and `ort`), or stage: 1.88 in Phase 4 only if `ort` is dropped. Prefer one bump. |
| `sysinfo` / `reqwest` / `tokio` | 0.33.1 / 0.12.28 / 1.54.1 | Stable, no action. | — |

**Decision: bump `rust-version` to 1.92 in Phase 2.** It unblocks `rmcp`
(1.88) and keeps `ort` rc.15 (1.92) available; the cost is dropping
distros shipping older toolchains, which is acceptable for a binary
distributed as prebuilt artifacts.

---

## 2. Phase 2 — `quokka-core` + minimal OpenAI server

**Goal:** `quokka run auto` loads the Flash model and answers
`POST /v1/chat/completions`. Expert model stays `mmap`'d on demand.

### Work packages

- **WP2.1 — MSRV bump + dep wiring.** `rust-version = "1.92"`, add
  `tokio`, `axum`, `tower-http`, `tokio-stream`, `llama-cpp-2` (pinned),
  `quokka-core` crate skeleton with `thiserror` error surface.
- **WP2.2 — Model pool.** `ModelPool` owning up to two `llama_model`
  handles (Flash pinned, Expert mmap). Placement hints come from
  `registry::Placement` already parsed. `n_gpu_layers` heuristic:
  `min(layers, floor(vram_mb / per_layer_mb))` with per-layer estimate
  from model metadata — never exceed VRAM minus a 512 MiB driver reserve.
- **WP2.3 — Inference session.** Chat-template application (use the
  model's embedded template via llama.cpp chat APIs, not hand-rolled
  strings), KV-cache per session id, `tokio::sync::Semaphore` around the
  compute slot (ARCHITECTURE §4.3).
- **WP2.4 — HTTP layer.** `axum` server: `GET /v1/models`,
  `POST /v1/chat/completions` with `stream: true` SSE, response headers
  `X-Quokka-Model-Used`, `X-Quokka-Route-Time-Ms`,
  `X-Quokka-Tokens-Per-Sec`.
- **WP2.5 — CLI `run`/`status`.** `quokka run auto` = probe → recommended
  pair → ensure downloaded → start server on `:8080`. `quokka status`
  prints live profile + loaded models.

### Hard rules

- Every `unsafe` block confined to `quokka-core` FFI wrappers, each with
  `// SAFETY:` (workspace lint already enforces).
- `llama-cpp-2` stays behind an internal trait (`InferenceBackend`) so a
  future `llama_cpp_sys` or `mistral.rs` swap doesn't leak.
- Model loading must be cancellable: a 9 GB mmap load shows progress,
  Ctrl-C leaves no half-mapped context.

### Verification gate

- `cargo test` + a `tests/integration/` smoke test that loads a **tiny**
  GGUF fixture (commit a ~50 MB test model or gate behind
  `QUOKKA_E2E=1`).
- `benches/tokens_per_sec.rs`: Flash model ≥ baseline tok/s on the
  reference machine; record numbers in this doc.
- Manual: `curl localhost:8080/v1/chat/completions` streamed output.

---

## 3. Phase 3 — Router + KV-cache offload

### 3.1 Stage 1 (lexical gate) — no dependencies

Token count, code fences, language keywords, `@flash`/`@expert`
overrides, `routing.toml` rules engine (`toml` crate). Target < 0.1 ms —
trivially met; add a criterion bench asserting it.

### 3.2 Stage 2 (semantic classifier) — decision gate

Choose the embedding runtime **before** writing code:

| Option | Pros | Cons |
| --- | --- | --- |
| `ort` rc.15 + all-MiniLM-L6-v2 INT8 | Matches ADR-0002; mature ops | RC status, native binary per platform, MSRV 1.92, ~100 MB runtime |
| `rten` (pure Rust) | Zero native deps, tiny binary | Younger; model format coverage must be verified for MiniLM |
| `candle` | Pure Rust, active | Heavier dep tree; INT8 quantized MiniLM support to verify |
| Heuristic-only fallback | Zero deps | ADR-0002 rejected: too many ambiguous prompts |

**Gate:** spike each viable option against a 20-prompt fixture corpus
measure p50/p99 latency on CPU. Pick the fastest that stays < 1.8 ms.
Record the outcome as **ADR-0004** (supersedes part of ADR-0002 if the
runtime changes).

### 3.3 KV-cache lifecycle

- LRU of sessions keyed by `session_id`; on eviction serialize with
  `lz4_flex` (pure Rust — avoids a C dep) to `~/.quokka/cache/kv/`.
- Restore target < 50 ms per ARCHITECTURE §4.3 — bench it.
- `MemoryPressureFallback`: re-read `available_ram_mb` before Expert
  load; below `expert_size + 1500 MiB` → Flash + `tracing::warn!` event.

### Verification gate — status

- ✅ `benches/router_bench.rs` : ~0,3–5,5 µs/appel, corpus 100 prompts
  = 203 µs — p99 très en dessous de 2 ms sur CPU de référence (i7-8565U).
- ✅ Corpus de 137 prompts labellisés : `crates/quokka-router/tests/
  routing_corpus.rs`, précision mesurée **100 %** — *mais itérée contre
  le classifieur* (sur-apprentissage possible, voir ADR-0004 ; un corpus
  externe reste à constituer).
- ✅ KV-cache LZ4 (§3.3) : `switch_session` sauvegarde/charge
  `state_save_file` compressé LZ4 dans `~/.quokka/cache/kv`, clé =
  FNV-1a du 1er message user, préfixe-token vérifié à la restauration,
  LRU 8 fichiers/modèle. Restauré état inclut les tokens générés.

*Last updated: 2026-11-20, after Phase 3 router completion (`b09d97c`).*

## 4. Phase 4 — MCP, Web UI, release engineering

- **MCP**: `rmcp` crate (`server` + `transport-io` features) exposing
  `quokka_route_info`, `quokka_summarize_local`. `quokka mcp` subcommand
  runs stdio transport; SSE transport optional.
- **Web UI**: `rust-embed` single-page dashboard — live tok/s, RAM/VRAM
  bars, mascot state, power-mode toggle, Flex Card PNG export.
- **Releases**: `release.yml` matrix per ARCHITECTURE §6
  (win-x64-cuda/vulkan, macos-arm64-metal, linux-x64-cuda/vulkan).
  `llama-cpp-2` feature flags map 1:1 (`cuda`, `vulkan`, `metal`) —
  expect the CUDA legs to need `cuda-toolkit` on runners; Vulkan leg is
  the safe universal default.
- **Telemetry**: `quokka-telemetry` crate — tok/s, RAM saved vs naive
  loading, routing histogram. Feeds both the Web UI and `X-Quokka-*`
  headers.

---

## 5. Cross-cutting concerns

- **Version pinning policy**: `llama-cpp-2` exact-pinned; `ort`/`rmcp`
  caret but reviewed monthly; everything else workspace-caret.
- **Unsafe budget**: today only `gpu/vulkan.rs`. Phase 2 adds FFI —
  keep total unsafe LOC < 5% of `quokka-core`, each block `// SAFETY:`.
- **MSRV**: single bump to 1.92 in WP2.1; update `Cargo.toml`,
  README badge, `deny.toml` graph if needed.
- **Testing strategy**: unit tests everywhere; E2E gated behind
  `QUOKKA_E2E=1` + tiny fixture model so CI stays fast and hermetic.
- **Docs**: every phase ends with this file updated (status column) and,
  for structural changes, a new ADR.

## 6. Risk register

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| `llama-cpp-2` breaking bump stalls Phase 2 | Medium | High | Pin + `dynamic-link` fallback; worst case drop to `llama_cpp_sys` (ADR-0001 superseded) |
| No pure-CPU path hits < 2 ms routing | Low | Medium | Stage-1 short-circuit already covers trivial prompts; measure before optimizing |
| CUDA cross-compile matrix explodes CI cost | Medium | Medium | Vulkan universal build first; CUDA legs nightly, not per-PR |
| `ort` rc never stabilizes | Medium | Low | Decision gate in §3.2 explicitly allows `rten`/`candle` |
| GGUF test fixture bloats repo | Low | Low | 50 MB cap or LFS/external URL |

---

*Last updated: 2026-11-19, after Phase 1 completion (`29e6cdf`).*
