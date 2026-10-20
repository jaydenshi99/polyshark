// Minimal Markdown → HTML for the lab's own docs (headings, lists, tables, code, links, emphasis).
import { esc } from './common.js';

function inline(s) {
  return esc(s)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, '$1<em>$2</em>')
    .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (m, text, href) => {
      const url = href.startsWith('#') || /^https?:/.test(href) ? href : '#';
      return `<a href="${url}"${url.startsWith('http') ? ' target="_blank" rel="noopener"' : ''}>${text}</a>`;
    });
}

const slug = s => s.toLowerCase().replace(/<[^>]+>/g, '').replace(/[^a-z0-9 -]/g, '').trim().replace(/\s+/g, '-');

export function markdown(src) {
  const lines = src.replace(/\r/g, '').split('\n');
  let html = '', i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line.startsWith('```')) {
      const body = [];
      for (i++; i < lines.length && !lines[i].startsWith('```'); i++) body.push(lines[i]);
      i++;
      html += `<pre class="block">${esc(body.join('\n'))}</pre>`;
      continue;
    }
    const h = line.match(/^(#{1,4})\s+(.*)/);
    if (h) {
      const n = h[1].length + 1, text = inline(h[2]);
      html += `<h${n} id="${slug(h[2])}">${text}</h${n}>`;
      i++;
      continue;
    }
    if (line.startsWith('|')) {
      const rows = [];
      for (; i < lines.length && lines[i].startsWith('|'); i++) rows.push(lines[i]);
      const cells = r => r.replace(/^\||\|$/g, '').split('|').map(c => c.trim());
      const head = cells(rows[0]), body = rows.slice(2).map(cells);
      html += `<div class="scroll"><table><thead><tr>${head.map(c => `<th>${inline(c)}</th>`).join('')}</tr></thead><tbody>${body.map(r => `<tr>${r.map(c => `<td class="wrap">${inline(c)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
      continue;
    }
    if (/^\s*([-*]|\d+\.)\s+/.test(line)) {
      const ordered = /^\s*\d+\./.test(line);
      const items = [];
      for (; i < lines.length && /^\s*([-*]|\d+\.)\s+|^\s{2,}\S/.test(lines[i]); i++) {
        if (/^\s*([-*]|\d+\.)\s+/.test(lines[i])) items.push(lines[i].replace(/^\s*([-*]|\d+\.)\s+/, ''));
        else items[items.length - 1] += ' ' + lines[i].trim();
      }
      html += `<${ordered ? 'ol' : 'ul'}>${items.map(x => `<li>${inline(x)}</li>`).join('')}</${ordered ? 'ol' : 'ul'}>`;
      continue;
    }
    if (!line.trim()) { i++; continue; }
    const para = [];
    for (; i < lines.length && lines[i].trim() && !/^(#|```|\||\s*([-*]|\d+\.)\s)/.test(lines[i]); i++) para.push(lines[i]);
    html += `<p>${inline(para.join(' '))}</p>`;
  }
  return html;
}
