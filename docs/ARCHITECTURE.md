# QUOKKA — Master Engineering & Architecture Specification (v0.1.0)
**Project:** Quokka — Smart Local AI Router & High-Performance Inference Engine  
**Target Audience:** Senior Systems Engineers (Rust/C++) & Autonomous Coding Agents  
**License:** Apache-2.0  

---

## 1. Executive Summary & Core Value Proposition
**Quokka** is a single-binary, zero-configuration local AI inference engine and semantic router built for consumer PCs. Unlike passive inference runners (Ollama, LM Studio) that require manual model selection and saturate system RAM/VRAM, Quokka acts as an **intelligent, hardware-aware traffic controller**:
1. **Hardware Auto-Tuning:** At startup, Quokka probes available RAM, CPU SIMD capabilities (AVX2/AVX-512/NEON), dedicated GPU VRAM (CUDA/ROCm/Metal), and integrated GPUs (Vulkan/DirectML). It automatically pulls and configures an optimal **Dual-Model Pair** (*Flash* + *Expert*) from a curated registry.
2. **Sub-2ms Semantic Routing:** Incoming prompts are classified in `< 2ms` using a lightweight heuristic + micro-embedding classifier. Simple queries route to the resident *Flash* model (1B–3B); complex reasoning/coding tasks route to the *Expert* model (8B–32B).
3. **Zero-Copy Memory Management (`mmap` + KV Cache Offloading):** Uses memory-mapped model weights (`mmap`) and dynamic KV-cache eviction to disk/RAM to swap between models instantaneously without triggering OS Out-Of-Memory (OOM) kills.
4. **Drop-in Universal Compatibility:** Exposes an **OpenAI-compatible REST/SSE API** (`http://localhost:8080/v1`) and a native **Model Context Protocol (MCP)** server for seamless integration with Cursor, Claude Code, VS Code, and Open WebUI.

---

## 2. Repository Structure (Cargo Workspace)
The repository follows strict modular separation of concerns:

```text
quokka/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                 # cargo test, clippy (-D warnings), fmt, cargo-deny
│   │   └── release.yml            # Cross-compile matrix (Linux/macOS/Windows + CUDA/Vulkan/Metal)
│   └── ISSUE_TEMPLATE/
├── brand/                         # Mascot SVGs, ASCII art, social banners
├── benches/                       # Criterion.rs benchmarks (router latency, token/s, RAM footprint)
├── crates/
│   ├── quokka-core/               # C++ FFI bindings (llama.cpp), mmap loader, KV-cache manager
│   ├── quokka-hardware/           # Hardware probe (sysinfo, nvml, vulkan), auto-tuner, HF downloader
│   ├── quokka-router/             # Sub-2ms intent classifier, routing rules engine (TOML/YAML)
│   ├── quokka-server/             # Axum HTTP server, OpenAI API, MCP server, embedded Web UI
│   └── quokka-telemetry/          # Metrics collector (tokens/sec, RAM saved, routing decisions)
├── cli/                           # Main binary entrypoint (`quokka` CLI via `clap`)
├── registry/
│   └── profiles.json              # Curated hardware-to-GGUF model mapping index
├── docs/
│   └── ARCHITECTURE.md            # This document
├── Cargo.toml                     # Workspace definition & LTO release profile
├── deny.toml                      # License & security audit rules
└── README.md
```

---

## 3. Technical Stack & Crate Dependencies

| Layer | Crate / Tech | Engineering Justification |
| :--- | :--- | :--- |
| **Async Runtime** | `tokio` (`full`) | High-concurrency task scheduling for HTTP streaming & background model prefetching. |
| **Inference Backend** | `llama-cpp-2` (C++ FFI) | Direct bindings to `llama.cpp` supporting CPU (AVX2/NEON), CUDA, Metal, and Vulkan out-of-the-box. |
| **Hardware Probing** | `sysinfo`, `nvml-wrapper`, `ash` | Cross-platform detection of free RAM, CPU cores/arch, NVIDIA VRAM, and Vulkan devices. |
| **Intent Classifier** | `ort` (ONNX Runtime) + Heuristics | Runs quantized `all-MiniLM-L6-v2` (INT8) or lexical/AST heuristics in `< 2ms` on CPU. |
| **HTTP / SSE / MCP** | `axum`, `tower-http`, `tokio-stream` | Zero-overhead OpenAI `/v1/chat/completions` streaming & JSON-RPC over stdio/SSE for MCP. |
| **Model Downloader** | `reqwest`, `futures-util`, `sha2` | Multi-segment parallel HTTP range downloads from Hugging Face with SHA-256 verification & resume. |
| **Embedded Web UI** | `rust-embed` | Compiles static HTML/JS/CSS dashboard directly into the single release binary. |
| **CLI** | `clap` (derive), `indicatif`, `console` | Ergonomic CLI (`run`, `pull`, `status`, `bench`) with ASCII mascot state rendering. |

---

## 4. Core Subsystems Specification

### 4.1 `quokka-hardware` — Hardware Probe & Auto-Tuner
Upon `quokka run auto`, the engine probes the host machine and computes a `HardwareProfile`:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub os: String,                  // "windows", "linux", "macos"
    pub arch: String,                // "x86_64", "aarch64"
    pub total_ram_mb: u64,
    pub available_ram_mb: u64,
    pub cpu_cores: usize,
    pub has_avx2: bool,
    pub has_neon: bool,
    pub gpu_backend: GpuBackend,     // Cuda { vram_mb }, Metal { unified_ram_mb }, Vulkan { vram_mb }, None
}
```

**Auto-Tuning Decision Matrix (`registry/profiles.json`):**
* **Tier 1 — Lightweight Laptop (`<= 8 GB RAM`, No dGPU):**
  * *Flash:* `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` (~1.1 GB, always resident in RAM).
  * *Expert:* `Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf` (~2.1 GB, `mmap` on-demand).
* **Tier 2 — Standard PC / Mac (`16 GB RAM` or `4–6 GB VRAM`):**
  * *Flash:* `Llama-3.2-3B-Instruct-Q4_K_M.gguf` (~2.0 GB, pinned in VRAM/RAM).
  * *Expert:* `Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` (~4.7 GB, partial GPU offload + `mmap`).
* **Tier 3 — Workstation (`>= 32 GB RAM` or `>= 12 GB VRAM`):**
  * *Flash:* `Qwen2.5-3B-Instruct-Q8_0.gguf` (~3.2 GB, 100% GPU).
  * *Expert:* `Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf` or `DeepSeek-R1-Distill-14B` (~9.0 GB, 100% GPU).

**Local Model Discovery (Zero-Duplicate Policy):**
Before downloading from Hugging Face, `quokka-hardware` scans default local directories (`~/.ollama/models`, `~/.cache/lm-studio`, `~/.cache/huggingface`) and symlinks compatible GGUF blobs if found.

---

### 4.2 `quokka-router` — Sub-2ms Semantic Router
Every incoming prompt passes through a two-stage pipeline before hitting the inference engine:

1. **Stage 1: Fast Lexical & Structural Gate (`< 0.1ms`)**
   * Checks token count, code fences (`` ``` ``), programming keywords (`fn `, `impl `, `SELECT `, `async `), or explicit user overrides (`@flash`, `@expert`).
   * Short conversational prompts (`< 25 tokens`, no code/reasoning markers) immediately route to **Flash**.
2. **Stage 2: Semantic Embedding Classifier (`< 1.8ms`)**
   * Computes a 384-dim embedding of the user prompt using an INT8 quantized ONNX model running on 2 CPU threads.
   * Calculates cosine similarity against pre-computed anchor centroids (`coding`, `math_logic`, `multi_step_analysis` vs `translation`, `summarization`, `chit_chat`).
   * **Memory Pressure Guard:** If `available_ram_mb < expert_model_size + 1500MB` (OS safety buffer), the router degrades gracefully to *Flash* and logs a `MemoryPressureFallback` event rather than risking system freeze.

**Declarative Routing Rules (`~/.quokka/routing.toml`):**
```toml
[policy]
mode = "balanced" # options: "eco", "balanced", "performance"
ram_safety_buffer_mb = 1536

[[rules]]
name = "code-generation"
match_keywords = ["rust", "python", "sql", "debug", "refactor", "architecture"]
target_tier = "expert"

[[rules]]
name = "quick-tasks"
max_input_tokens = 60
target_tier = "flash"
```

---

### 4.3 `quokka-core` — Zero-Copy Engine & Memory Orchestration
* **Memory-Mapped Loading (`mmap`):** Models are opened with `use_mmap = true` and `use_mlock = false` (except the *Flash* model if RAM permits). Switching from *Flash* to *Expert* requires zero disk I/O allocation overhead; the OS page cache handles paging seamlessly.
* **Context & KV-Cache Lifecycle:**
  * Maintains an LRU cache of active conversation sessions (`session_id`).
  * When switching models or handling multiple apps (e.g., Cursor + Open WebUI), inactive KV caches are serialized to `~/.quokka/cache/kv/` using LZ4 compression and restored in `< 50ms` when the session resumes.
* **Concurrency Control:** Uses a `tokio::sync::Semaphore` around GPU/CPU compute slots to prevent thread contention when multiple clients send simultaneous requests.

---

### 4.4 `quokka-server` — API & Protocol Specifications
1. **OpenAI-Compatible Endpoints (`http://127.0.0.1:8080/v1`):**
   * `POST /v1/chat/completions` (supports `stream: true` via SSE, `tools`/function calling, JSON mode).
   * `GET /v1/models` (exposes `quokka-auto`, `quokka-flash`, `quokka-expert`, plus any manually pulled models).
   * `POST /v1/embeddings` (powered by the internal ONNX/GGUF embedding worker).
   * *Custom Response Headers:* Injects `X-Quokka-Model-Used`, `X-Quokka-Route-Time-Ms`, and `X-Quokka-Tokens-Per-Sec` on every response for client debugging.
2. **Model Context Protocol (MCP) Server (`quokka mcp`):**
   * Exposes tools to AI agents (Claude Code, Cursor):
     * `quokka_route_info`: Returns current hardware load, active models, and routing stats.
     * `quokka_summarize_local`: Offloads large file summarization to the local *Flash* model to save cloud API tokens.
3. **Embedded Web UI (`http://127.0.0.1:8080/`):**
   * Single-page lightweight dashboard showing live tokens/sec, RAM/VRAM bar, mascot state (`Idle`, `Flash`, `Expert`, `Cool`), power mode toggle (`Eco` / `Balanced` / `Max`), and a 1-click **"Export Hardware Flex Card (PNG)"** button.

---

## 5. CLI UX & Mascot State Machine (`cli/`)
The CLI provides immediate feedback with ASCII mascot states:

* `quokka run auto` — Probes hardware, downloads optimal pair (if missing), starts server on `:8080`, and prints connection instructions.
* `quokka pull <model-id>` — Downloads a specific model from the registry or Hugging Face URL.
* `quokka status` — Displays hardware profile, loaded models, and live telemetry.
* `quokka bench` — Runs a standardized 3-prompt benchmark and outputs a shareable performance summary.

**Terminal Mascot States:**
* `(ᵔᴥᵔ) 📦 [Tuner]` — Scanning hardware & optimizing quantization...
* `(•ᴗ•) ⚡ [Flash]` — Routed to `Qwen2.5-3B` (1.2ms route | 118 tok/s)
* `(⌐■_■) 🧠 [Expert]` — Routed to `Qwen2.5-Coder-14B` (1.8ms route | 42 tok/s)
* `(•_•) 🍃 [Cool]` — Eco mode active (RAM saved: 5.4 GB)

---

## 6. Build, CI/CD & Cross-Compilation Matrix
To ensure maximum performance across consumer PCs, release binaries must be compiled with hardware-specific feature flags in `.github/workflows/release.yml`:

| Target Triple | Build Variant | Backend Flags |
| :--- | :--- | :--- |
| `x86_64-pc-windows-msvc` | `quokka-win-x64-cuda.exe` | `--features cuda` |
| `x86_64-pc-windows-msvc` | `quokka-win-x64-vulkan.exe` | `--features vulkan` (Universal AMD/Intel/NVIDIA) |
| `aarch64-apple-darwin` | `quokka-macos-arm64` | `--features metal` |
| `x86_64-unknown-linux-gnu` | `quokka-linux-x64-cuda` | `--features cuda` |
| `x86_64-unknown-linux-gnu` | `quokka-linux-x64-vulkan` | `--features vulkan` |

---

## 7. Phased Implementation Roadmap (For Agents & Contributors)

* **Phase 1 — Foundation & Hardware Probe (Week 1)**
  * Initialize Cargo workspace with `Cargo.toml` and `deny.toml`.
  * Implement `quokka-hardware` (`sysinfo` + GPU detection) and `registry/profiles.json`.
  * Build multi-threaded Hugging Face downloader with `indicatif` progress bars.
* **Phase 2 — Core Inference & OpenAI API (Week 2)**
  * Integrate `llama-cpp-2` in `quokka-core` with `mmap` loading and streaming token iterator.
  * Build `quokka-server` with `axum` implementing `/v1/chat/completions` (SSE streaming) and `/v1/models`.
* **Phase 3 — Sub-2ms Smart Router & KV Cache (Week 3)**
  * Implement Stage 1 lexical heuristics + Stage 2 ONNX/heuristic classifier in `quokka-router`.
  * Add `MemoryPressureFallback` and LZ4 KV-cache disk serialization in `quokka-core`.
* **Phase 4 — MCP Server, Embedded Web UI & Launch (Week 4)**
  * Add `quokka mcp` stdio/SSE endpoint for Cursor and Claude Code integration.
  * Embed lightweight HTML/CSS/JS dashboard via `rust-embed` with live telemetry and Flex Card generator.
  * Configure GitHub Actions release matrix and publish `v0.1.0`.

---

## 8. Engineering Standards & Definition of Done (DoD)
1. **Zero Panics in Production:** All FFI calls to `llama.cpp` and file/network operations must return `Result<T, QuokkaError>` using `thiserror` in libraries and `anyhow` in the CLI.
2. **Strict Clippy Compliance:** Code must pass `cargo clippy --workspace --all-targets -- -D warnings`.
3. **Latency Budget:** Router classification overhead must remain strictly `< 2.0ms` (verified via `benches/router_bench.rs`).
4. **Memory Safety:** No `unsafe` blocks outside `quokka-core` FFI boundaries, and every `unsafe` block must include a `// SAFETY:` invariant comment.
