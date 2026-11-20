# Quokka — Product Context

## What it is
Quokka is a local-first AI inference engine and intelligent router, written in
Rust with a llama.cpp (C++) backend. Single binary, zero configuration: it
probes the host hardware, picks a Flash/Expert model pair, serves an
OpenAI-compatible API (`/v1/chat/completions`, SSE streaming) and exposes an
embedded Web UI.

## Users
Developers and privacy-conscious users running LLMs on their own machine.
They want Ollama-class convenience without giving up control: no cloud, no
telemetry, predictable RAM usage, real hardware-aware behavior.

## Product truths (do not violate)
- Local-first: everything runs on `127.0.0.1`, no external calls at runtime.
- Zero-config: `quokka run` must work with no flags on supported hardware.
- Honest engineering: no marketing claims we cannot measure. Performance
  numbers come from real benchmarks on real hardware.
- Memory discipline: models are mmap'd; Quokka must stay a good citizen on
  16 GiB machines running a full desktop.

## Web UI role
The UI is the human face of `quokka run`: a chat surface over the existing
OpenAI-compatible endpoints. It is an Operate-mode surface — the visitor's
success is completing a conversation, fast. Brand lives in precise details,
not decoration.

## Constraints
- Embedded in the `quokka` binary (rust-embed); no separate server process.
- Built with Vite + Svelte; `ui/dist` is the committed-free build artifact
  embedded at compile time.
- Must work against `/v1/models` and `/v1/chat/completions` (SSE) as-is.
