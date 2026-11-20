// Minimal, safe markdown renderer: escapes HTML first, then applies a
// small set of transforms (fences, inline code, bold, italic, headings,
// lists, paragraphs, links). No innerHTML of raw input — XSS-safe.

const escapeHtml = (s) =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

function inline(text) {
  let s = escapeHtml(text);
  s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>');
  s = s.replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  return s;
}

export function renderMarkdown(src) {
  const blocks = [];
  const lines = src.replace(/\r\n/g, '\n').split('\n');
  let i = 0;
  let html = '';
  const push = (h) => { html += h; };

  while (i < lines.length) {
    const line = lines[i];

    // fenced code
    const fence = line.match(/^```(\w*)/);
    if (fence) {
      const lang = fence[1];
      const buf = [];
      i++;
      while (i < lines.length && !lines[i].startsWith('```')) buf.push(lines[i++]);
      i++; // closing fence
      push(`<pre><code class="lang-${lang}">${escapeHtml(buf.join('\n'))}</code></pre>`);
      continue;
    }

    // heading
    const h = line.match(/^(#{1,4})\s+(.*)/);
    if (h) {
      const level = h[1].length;
      push(`<h${level}>${inline(h[2])}</h${level}>`);
      i++;
      continue;
    }

    // unordered list
    if (/^\s*[-*]\s+/.test(line)) {
      push('<ul>');
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        push(`<li>${inline(lines[i].replace(/^\s*[-*]\s+/, ''))}</li>`);
        i++;
      }
      push('</ul>');
      continue;
    }

    // ordered list
    if (/^\s*\d+[.)]\s+/.test(line)) {
      push('<ol>');
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
        push(`<li>${inline(lines[i].replace(/^\s*\d+[.)]\s+/, ''))}</li>`);
        i++;
      }
      push('</ol>');
      continue;
    }

    // blank line separates paragraphs
    if (line.trim() === '') {
      i++;
      continue;
    }

    // paragraph: gather until blank or block start
    const buf = [];
    while (
      i < lines.length &&
      lines[i].trim() !== '' &&
      !lines[i].match(/^(#{1,4}\s|```|\s*[-*]\s|\s*\d+[.)]\s)/)
    ) {
      buf.push(lines[i++]);
    }
    push(`<p>${inline(buf.join(' '))}</p>`);
  }
  return html;
}
