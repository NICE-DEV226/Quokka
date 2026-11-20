// Thin client for the Quokka OpenAI-compatible API.

export async function fetchModels() {
  const res = await fetch('./v1/models');
  if (!res.ok) throw new Error(`models: HTTP ${res.status}`);
  const data = await res.json();
  return data.data.map((m) => m.id);
}

export async function health() {
  try {
    const res = await fetch('./health');
    return res.ok;
  } catch {
    return false;
  }
}

// Streams chat completion tokens via SSE. Calls onToken(text) per delta,
// resolves with { content, finishReason, usage }.
export async function streamChat({ model, messages, signal, onToken }) {
  const res = await fetch('./v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, messages, stream: true }),
    signal,
  });
  if (!res.ok || !res.body) {
    const body = await res.text().catch(() => '');
    throw new Error(`chat: HTTP ${res.status} ${body.slice(0, 200)}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let content = '';
  let finishReason = null;
  let usage = null;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') continue;
      try {
        const json = JSON.parse(payload);
        const delta = json.choices?.[0]?.delta?.content;
        if (delta) {
          content += delta;
          onToken?.(delta);
        }
        if (json.choices?.[0]?.finish_reason) finishReason = json.choices[0].finish_reason;
        if (json.usage) usage = json.usage;
      } catch {
        // partial JSON across chunk boundary — ignore
      }
    }
  }
  return { content, finishReason, usage };
}
