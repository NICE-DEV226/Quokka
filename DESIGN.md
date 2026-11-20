# Quokka Web UI — Design System

Direction: **Open WebUI evolved**. Same structural grammar (dark rail,
light canvas, chat bubbles, model picker) but a Quokka identity: warm paper
canvas, deep forest-teal accent, mascot presence. Operate mode — scanability
and native chat expectations outrank expression.

## Color tokens (dark-first, light supported via tokens)

| Token | Dark | Light | Use |
|---|---|---|---|
| `--bg-canvas` | `#0f1413` | `#f7f5f0` | app background |
| `--bg-rail` | `#161c1a` | `#ece9e2` | sidebar |
| `--bg-surface` | `#1c2422` | `#ffffff` | cards, composer |
| `--bg-hover` | `#242e2b` | `#e4e0d6` | hover states |
| `--text-primary` | `#e8ece9` | `#1a1f1d` | body text |
| `--text-secondary` | `#9aa8a2` | `#5c6660` | labels, meta |
| `--accent` | `#4fd1a5` | `#0e7a55` | primary actions, user bubble |
| `--accent-ink` | `#06231a` | `#ffffff` | text on accent |
| `--border` | `#2a3532` | `#d8d3c7` | hairlines |

Rationale: Quokka = Australian marsupial, eucalyptus/warm-paper feel.
`#4fd1a5` is a eucalyptus mint — distinctive vs Open WebUI's teal and
Ollama's monochrome, still calm enough for long sessions.

## Typography
- UI: system stack (`system-ui, -apple-system, "Segoe UI", sans-serif`),
  14px base, 1.5 line-height.
- Assistant markdown: same stack, `pre` blocks get `--bg-rail` + 1px border,
  `code` gets accent-tinted background.
- No webfonts — zero-config, zero network at runtime.

## Layout (desktop-first, responsive)
- Left rail 260px: logo+mascot, "New chat", conversation list, footer
  (model count / status dot). Collapses to overlay drawer < 860px.
- Canvas: centered column max-width 820px. Empty state = mascot + greeting +
  3 suggestion chips. Composer pinned bottom, auto-growing textarea,
  send button accent-filled.
- Messages: user right-aligned accent bubble (rounded 18px, tail corner
  squared), assistant left-aligned plain text with subtle hover actions
  (copy). Streaming shows a pulsing caret block.

## Motion
- 150–200ms ease-out for rail/drawer, message fade-in 120ms.
- Streaming caret pulse 1s. `prefers-reduced-motion` disables all.

## Components
ChatView, Sidebar, ConversationList, Composer, MessageBubble,
Markdown-lite renderer (headings, lists, code fences, inline code, bold —
hand-rolled, no markdown dep unless size justifies marked.js).
ModelPicker: pill in composer footer, lists `/v1/models` ids.
