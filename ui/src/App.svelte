<script>
  import { onMount } from 'svelte';
  import { fetchModels, health, streamChat } from './lib/api.js';
  import { renderMarkdown } from './lib/markdown.js';
  import Icon from './Icon.svelte';

  const LS_KEY = 'quokka.chats.v1';
  const LS_MODEL = 'quokka.model';

  let models = $state([]);
  let online = $state(false);
  let model = $state('');
  let chats = $state([]);
  let activeId = $state(null);
  let input = $state('');
  let streaming = $state(false);
  let streamText = $state('');
  let railOpen = $state(false);
  let abortCtrl = null;
  let scrollEl;

  const active = $derived(chats.find((c) => c.id === activeId) ?? null);

  onMount(async () => {
    chats = loadChats();
    model = localStorage.getItem(LS_MODEL) || '';
    online = await health();
    if (online) {
      try {
        models = await fetchModels();
        if (!model || !models.includes(model)) model = models[0] ?? '';
      } catch { /* server may be starting */ }
    }
  });

  $effect(() => {
    if (model) localStorage.setItem(LS_MODEL, model);
  });

  $effect(() => {
    if (active?.messages?.length && scrollEl) {
      scrollEl.scrollTop = scrollEl.scrollHeight;
    }
  });

  function loadChats() {
    try { return JSON.parse(localStorage.getItem(LS_KEY)) ?? []; }
    catch { return []; }
  }
  function saveChats() {
    localStorage.setItem(LS_KEY, JSON.stringify(chats.slice(0, 50)));
  }

  function newChat() {
    const chat = { id: crypto.randomUUID(), title: 'New chat', messages: [], updatedAt: Date.now() };
    chats = [chat, ...chats];
    activeId = chat.id;
    railOpen = false;
  }

  function selectChat(id) {
    activeId = id;
    railOpen = false;
  }

  function deleteChat(id, e) {
    e.stopPropagation();
    chats = chats.filter((c) => c.id !== id);
    if (activeId === id) activeId = chats[0]?.id ?? null;
    saveChats();
  }

  function chatTitle(messages) {
    const first = messages.find((m) => m.role === 'user')?.content ?? '';
    return first.slice(0, 48) + (first.length > 48 ? '…' : '') || 'New chat';
  }

  async function send() {
    const text = input.trim();
    if (!text || streaming || !model) return;
    input = '';

    let chat = active;
    if (!chat) {
      chat = { id: crypto.randomUUID(), title: 'New chat', messages: [], updatedAt: Date.now() };
      chats = [chat, ...chats];
      activeId = chat.id;
    }
    // Toujours muter via l'objet stocké dans $state (proxy), jamais la
    // référence brute — sinon le template lit une autre instance.
    chat = chats.find((c) => c.id === chat.id) ?? chat;
    const assistantMsg = { role: 'assistant', content: '' };
    chat.messages = [...chat.messages, { role: 'user', content: text }, assistantMsg];
    chat.title = chatTitle(chat.messages);
    chat.updatedAt = Date.now();
    chats = [...chats];
    saveChats();

    streaming = true;
    streamText = '';
    abortCtrl = new AbortController();
    const lastIdx = () => chat.messages.length - 1;
    const patchLast = (patch) => {
      chat.messages = chat.messages.map((m, i) =>
        i === lastIdx() ? { ...m, ...patch } : m,
      );
      chats = [...chats]; // notify même si `chat` est un objet brut
    };
    try {
      const history = chat.messages.slice(0, -1).map((m) => ({ role: m.role, content: m.content }));
      const result = await streamChat({
        model,
        messages: history,
        signal: abortCtrl.signal,
        onToken: (t) => { streamText += t; },
      });
      patchLast({
        content: streamText || result.content || '⚠ The model returned an empty response.',
        usage: result.usage,
      });
    } catch (err) {
      if (err.name !== 'AbortError') {
        patchLast({ content: `⚠ ${err.message}` });
      } else {
        chat.messages = chat.messages.slice(0, -1); // drop empty assistant msg
        chats = [...chats];
      }
    } finally {
      streaming = false;
      abortCtrl = null;
      saveChats();
    }
  }

  function shown(msg, i) {
    return (
      msg.content ||
      (streaming && i === active?.messages?.length - 1 ? streamText : '')
    );
  }

  function stop() {
    abortCtrl?.abort();
  }

  function onKeydown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }
</script>

<div class="shell">
  {#if railOpen}
    <div class="scrim" onclick={() => (railOpen = false)} role="presentation"></div>
  {/if}

  <aside class="rail" class:open={railOpen}>
    <div class="brand">
      <img src="quokka-mark.svg" alt="" class="mascot" />
      <span class="wordmark">Quokka</span>
      <button class="rail-close" onclick={() => (railOpen = false)} aria-label="Close"><Icon name="close" size={15} /></button>
    </div>

    <button class="new-chat" onclick={newChat}>
      <Icon name="plus" size={15} /> New chat
    </button>

    <nav class="chats">
      {#each chats as chat (chat.id)}
        <div
          class="chat-item"
          class:active={chat.id === activeId}
          role="button"
          tabindex="0"
          onclick={() => selectChat(chat.id)}
          onkeydown={(e) => e.key === 'Enter' && selectChat(chat.id)}
        >
          <span class="chat-title">{chat.title}</span>
          <button class="chat-del" onclick={(e) => deleteChat(chat.id, e)} aria-label="Delete chat"><Icon name="trash" size={14} /></button>
        </div>
      {:else}
        <p class="empty-hint">No conversations yet</p>
      {/each}
    </nav>

    <footer class="rail-foot">
      <span class="status-dot" class:off={!online}></span>
      {online ? 'quokka · online' : 'quokka · offline'}
    </footer>
  </aside>

  <main class="canvas">
    <header class="topbar">
      <button class="menu-btn" onclick={() => (railOpen = true)} aria-label="Menu"><Icon name="menu" size={19} /></button>
      <div class="model-picker">
        <select bind:value={model} disabled={!models.length} aria-label="Model">
          {#if !models.length}
            <option value="">{online ? 'loading…' : 'server offline'}</option>
          {/if}
          {#each models as m (m)}
            <option value={m}>{m}</option>
          {/each}
        </select>
      </div>
    </header>

    <div class="messages" bind:this={scrollEl}>
      {#if !active || active.messages.length === 0}
        <div class="empty">
          <div class="empty-stage">
            <div class="empty-halo"></div>
            <img src="quokka-mark.svg" alt="Quokka" class="empty-mascot" />
          </div>
          <h1>What are we curious about?</h1>
          <div class="chips">
            {#each [
              'Explain quantum computing simply',
              'Write a Python function to parse CSV',
              'Give me a haiku about the outback',
            ] as chip (chip)}
              <button class="chip" onclick={() => { input = chip; }}>{chip}</button>
            {/each}
          </div>
        </div>
      {:else}
        {#each active.messages as msg, i (i)}
          <div class="msg-row" class:user={msg.role === 'user'}>
            {#if msg.role === 'user'}
              <div class="bubble user">{msg.content}</div>
            {:else}
              <div class="assistant-side">
                <img
                  class="avatar"
                  class:thinking={streaming && i === active.messages.length - 1}
                  src="quokka-mark.svg"
                  alt=""
                />
                <div class="assistant-col">
                  <span class="author">quokka</span>
                  <div class="bubble assistant">
                    {#if shown(msg, i)}
                      <div class="md">{@html renderMarkdown(shown(msg, i))}</div>
                    {:else if streaming && i === active.messages.length - 1}
                      <span class="caret">▍</span>
                    {/if}
                    {#if msg.usage}
                      <div class="usage">{msg.usage.completion_tokens} tokens</div>
                    {/if}
                  </div>
                </div>
              </div>
            {/if}
          </div>
        {/each}
      {/if}
    </div>

    <div class="composer-wrap">
      <div class="composer">
        <textarea
          bind:value={input}
          onkeydown={onKeydown}
          placeholder={model ? 'Message Quokka…' : 'Start quokka run first'}
          rows="1"
          disabled={!model}
        ></textarea>
        {#if streaming}
          <button class="send stop" onclick={stop} aria-label="Stop"><Icon name="stop" size={12} /></button>
        {:else}
          <button class="send" onclick={send} disabled={!input.trim() || !model} aria-label="Send"><Icon name="send" size={16} /></button>
        {/if}
      </div>
      <p class="composer-hint">Enter to send · Shift+Enter for newline · runs 100% locally</p>
    </div>
  </main>
</div>

<style>
  .shell { display: flex; height: 100%; }
  .scrim { display: none; }

  .rail {
    width: 260px; flex-shrink: 0; background: var(--bg-rail);
    border-right: 1px solid var(--border);
    display: flex; flex-direction: column;
  }
  .brand { display: flex; align-items: center; gap: 10px; padding: 16px 14px 10px; }
  .mascot { width: 30px; height: 30px; border-radius: 8px; display: block; }
  .wordmark { font-weight: 650; font-size: 15px; letter-spacing: .2px; }
  .rail-close { display: none; margin-left: auto; color: var(--text-secondary); }
  .new-chat {
    margin: 6px 12px 12px; padding: 9px 12px; border-radius: var(--radius);
    background: var(--bg-surface); border: 1px solid var(--border);
    text-align: left; display: flex; align-items: center; gap: 8px;
    transition: background .15s;
  }
  .new-chat:hover { background: var(--bg-hover); }
  .plus { color: var(--accent); font-weight: 700; }

  .chats { flex: 1; overflow-y: auto; padding: 0 8px; }
  .chat-item {
    width: 100%; display: flex; align-items: center; gap: 6px;
    padding: 8px 10px; border-radius: 8px; text-align: left;
    color: var(--text-secondary); position: relative;
  }
  .chat-item:hover { background: var(--bg-hover); color: var(--text-primary); }
  .chat-item.active { background: var(--bg-surface); color: var(--text-primary); }
  .chat-title { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; }
  .chat-del { opacity: 0; color: var(--text-secondary); font-size: 15px; padding: 0 4px; }
  .chat-item:hover .chat-del { opacity: 1; }
  .chat-del:hover { color: var(--danger); }
  .empty-hint { padding: 12px; color: var(--text-secondary); font-size: 12px; }

  .rail-foot {
    padding: 12px 16px; border-top: 1px solid var(--border);
    font-size: 12px; color: var(--text-secondary);
    display: flex; align-items: center; gap: 7px;
  }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); animation: pulse 2.4s infinite; }
  .status-dot.off { background: var(--danger); animation: none; }

  .canvas { flex: 1; display: flex; flex-direction: column; min-width: 0; }
  .topbar {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 18px; border-bottom: 1px solid var(--border);
  }
  .menu-btn { display: none; font-size: 17px; color: var(--text-secondary); padding: 4px 8px; }
  .model-picker select {
    background: var(--bg-surface); color: var(--text-primary);
    border: 1px solid var(--border); border-radius: 999px;
    padding: 5px 12px; font-size: 12.5px; max-width: 320px;
  }

  .messages { flex: 1; overflow-y: auto; padding: 24px 20px 12px; }
  .empty {
    height: 100%; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 16px; text-align: center;
  }
  .empty-stage { position: relative; display: grid; place-items: center; }
  .empty-halo {
    position: absolute; width: 170px; height: 170px; border-radius: 50%;
    background: radial-gradient(circle, color-mix(in srgb, var(--accent) 22%, transparent), transparent 68%);
    animation: breathe 4.5s ease-in-out infinite;
  }
  .empty-mascot {
    width: 96px; height: 96px; border-radius: 24px; position: relative;
    box-shadow: 0 16px 48px rgba(0,0,0,.3), 0 0 0 1px var(--border);
    animation: float 5s ease-in-out infinite;
  }
  .empty h1 { font-size: 24px; font-weight: 600; letter-spacing: -.2px; }
  @keyframes float { 50% { transform: translateY(-7px); } }
  @keyframes breathe { 50% { transform: scale(1.12); opacity: .8; } }
  .chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; max-width: 560px; }
  .chip {
    padding: 8px 14px; border-radius: 999px; font-size: 13px;
    background: var(--bg-surface); border: 1px solid var(--border);
    color: var(--text-secondary); transition: all .15s;
  }
  .chip:hover { color: var(--text-primary); border-color: var(--accent-dim); }

  .msg-row { display: flex; margin-bottom: 20px; }
  .msg-row.user { justify-content: flex-end; }
  .assistant-side { display: flex; gap: 10px; width: 100%; }
  .avatar {
    width: 26px; height: 26px; border-radius: 8px; flex-shrink: 0;
    margin-top: 2px;
    box-shadow: 0 0 0 1px var(--border);
    transition: transform .2s;
  }
  .avatar.thinking { animation: think 1.1s ease-in-out infinite; }
  @keyframes think {
    0%, 100% { transform: rotate(-6deg) scale(1.04); }
    50% { transform: rotate(6deg) scale(1.04); }
  }
  .assistant-col { flex: 1; min-width: 0; }
  .author {
    display: block; font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: .8px;
    color: var(--text-secondary); margin-bottom: 3px;
  }
  .bubble { max-width: min(78%, 720px); }
  .bubble.user {
    background: var(--accent); color: var(--accent-ink);
    padding: 10px 16px; border-radius: 18px 18px 6px 18px;
    white-space: pre-wrap; word-break: break-word;
  }
  .bubble.assistant { padding: 2px 4px; width: 100%; max-width: none; }
  .composer:focus-within {
    border-color: var(--accent-dim);
    box-shadow: 0 4px 24px rgba(0,0,0,.18), 0 0 0 3px color-mix(in srgb, var(--accent) 18%, transparent);
  }
  .usage { margin-top: 8px; font-size: 11px; color: var(--text-secondary); }
  .caret { color: var(--accent); animation: pulse 1s infinite; }
  @keyframes pulse { 50% { opacity: .2; } }

  .composer-wrap {
    padding: 8px 20px 16px;
    background: linear-gradient(transparent, var(--bg-canvas) 30%);
  }
  .composer {
    max-width: 820px; margin: 0 auto;
    display: flex; align-items: flex-end; gap: 8px;
    background: var(--bg-surface); border: 1px solid var(--border);
    border-radius: 22px; padding: 10px 10px 10px 18px;
    box-shadow: 0 4px 24px rgba(0,0,0,.18);
  }
  .composer textarea {
    flex: 1; background: none; border: none; outline: none; resize: none;
    color: var(--text-primary); font: inherit; line-height: 1.5;
    max-height: 200px; padding: 6px 0;
  }
  .send {
    width: 36px; height: 36px; border-radius: 50%; flex-shrink: 0;
    background: var(--accent); color: var(--accent-ink);
    font-size: 16px; font-weight: 700;
    display: grid; place-items: center;
    transition: opacity .15s;
  }
  .send:disabled { opacity: .35; cursor: default; }
  .send.stop { background: var(--danger); color: #fff; }
  .composer-hint {
    max-width: 820px; margin: 8px auto 0; text-align: center;
    font-size: 11.5px; color: var(--text-secondary);
  }

  /* markdown */
  :global(.md p) { margin: 0 0 .7em; }
  :global(.md p:last-child) { margin-bottom: 0; }
  :global(.md h1), :global(.md h2), :global(.md h3), :global(.md h4) {
    margin: 1em 0 .5em; font-weight: 650; line-height: 1.3;
  }
  :global(.md h1) { font-size: 1.4em; } :global(.md h2) { font-size: 1.25em; }
  :global(.md h3) { font-size: 1.12em; } :global(.md h4) { font-size: 1em; }
  :global(.md ul), :global(.md ol) { margin: 0 0 .7em 1.4em; }
  :global(.md li) { margin: .2em 0; }
  :global(.md code) {
    font-family: var(--mono); font-size: .88em;
    background: var(--accent-dim); color: var(--text-primary);
    padding: .1em .35em; border-radius: 5px;
  }
  :global(.md pre) {
    background: var(--bg-code); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px 14px; margin: 0 0 .8em;
    overflow-x: auto;
  }
  :global(.md pre code) { background: none; padding: 0; font-size: .85em; }
  :global(.md a) { color: var(--accent); }
  :global(.md strong) { font-weight: 650; }

  @media (max-width: 860px) {
    .rail {
      position: fixed; inset: 0 auto 0 0; z-index: 20;
      transform: translateX(-100%); transition: transform .18s ease-out;
    }
    .rail.open { transform: translateX(0); }
    .rail-close { display: block; }
    .scrim {
      display: block; position: fixed; inset: 0; z-index: 10;
      background: rgba(0,0,0,.45);
    }
    .menu-btn { display: block; }
  }
  @media (prefers-reduced-motion: reduce) {
    * { animation: none !important; transition: none !important; }
  }
</style>
