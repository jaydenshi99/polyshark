// Shared helpers for the Match Lab pages.
export const TOKEN = document.querySelector('meta[name=lab-token]')?.content || '';
export const $ = id => document.getElementById(id);
export const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
export const collator = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });

export class ApiError extends Error {
  constructor(message, log) { super(message); this.log = log; }
}

export async function api(path, body) {
  const opts = body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json', 'X-Lab-Token': TOKEN }, body: JSON.stringify(body),
  };
  const res = await fetch('/api/' + path, opts);
  let data;
  try { data = await res.json(); } catch { throw new ApiError(`Request failed (${res.status})`); }
  if (!res.ok) throw new ApiError(data.error || `Request failed (${res.status})`, data.log);
  return data;
}

export async function uploadFile(path, file) {
  const res = await fetch('/api/' + path, {
    method: 'POST', headers: { 'X-Lab-Token': TOKEN, 'X-Filename': file.name, 'Content-Type': 'application/zip' }, body: file,
  });
  const data = await res.json();
  if (!res.ok) throw new ApiError(data.error || 'Upload failed', data.log);
  return data;
}

export function fmtDuration(s) {
  if (s == null || !isFinite(s)) return '—';
  s = Math.max(0, s);
  if (s < 1) return s < 0.01 ? '0s' : s.toFixed(1) + 's';
  if (s < 60) return Math.round(s) + 's';
  const m = Math.floor(s / 60), sec = Math.round(s % 60);
  if (m < 60) return `${m}m ${String(sec).padStart(2, '0')}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${String(m % 60).padStart(2, '0')}m`;
}

export function fmtTime(ts) {
  if (!ts) return '—';
  const d = new Date(ts * 1000), today = new Date();
  const time = d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
  return d.toDateString() === today.toDateString() ? time : d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) + ' ' + time;
}

export function fmtClock(ts) {
  return new Date(ts * 1000).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

export const pct = (x, d = 1) => x == null ? '—' : (100 * x).toFixed(d) + '%';
export const fmtNum = n => n == null ? '—' : Number(n).toLocaleString();
export const fmtMs = ms => ms == null ? '—' : ms < 0.001 ? (ms * 1e6).toFixed(0) + ' ns' : ms < 1 ? (ms * 1000).toFixed(ms < 0.01 ? 1 : 0) + ' µs' : ms < 1000 ? ms.toFixed(ms < 10 ? 2 : 0) + ' ms' : (ms / 1000).toFixed(2) + ' s';
export const fmtRate = r => r == null ? '—' : r >= 1e6 ? (r / 1e6).toFixed(2) + 'M' : r >= 1e4 ? (r / 1e3).toFixed(1) + 'k' : r >= 100 ? Math.round(r).toLocaleString() : r.toFixed(1);
export const plural = (n, w) => `${fmtNum(n)} ${w}${n === 1 ? '' : 's'}`;
export const fmtElo = (e, lo, hi) => {
  if (e == null || !isFinite(e)) return '—';
  const v = Math.round(e) || 0;
  const sign = v > 0 ? '+' : '';
  const range = lo != null && hi != null && isFinite(lo) && isFinite(hi) ? ` <small>${Math.round(lo)} … ${Math.round(hi)}</small>` : '';
  return `${sign}${v}${range}`;
};

export const scoreOf = rec => rec?.games ? (rec.wins + 0.5 * rec.draws) / rec.games : null;
export function scoreClass(s) {
  if (s == null) return '';
  if (s >= 0.45 && s <= 0.55) return s > 0.5 ? 'score-near-win' : s < 0.5 ? 'score-near-loss' : '';
  return s > 0.5 ? 'good' : 'bad';
}
export function recordCell(rec, extra = '') {
  const s = scoreOf(rec);
  if (s == null) return '<span class="small">—</span>';
  return `<strong class="${scoreClass(s)}">${pct(s)}</strong><small>${rec.wins} / ${rec.draws} / ${rec.losses} · ${plural(rec.games, 'game')}</small>${extra}`;
}
export function heatCell(rec) {
  const s = scoreOf(rec);
  if (s == null) return '<td><span class="small">—</span></td>';
  const k = 0.12 + 0.48 * Math.abs(s - 0.5) * 2;
  const bg = s > 0.5 ? `rgba(37,166,116,${k})` : s < 0.5 ? `rgba(203,76,76,${k})` : 'rgba(148,165,184,.10)';
  return `<td style="background:${bg}"><strong>${pct(s)}</strong><small>${rec.wins} / ${rec.draws} / ${rec.losses} · ${rec.games}</small></td>`;
}

const STATE_LABELS = {
  building: 'Building', queued: 'Queued', running: 'Running', paused: 'Paused', complete: 'Complete',
  cancelled: 'Cancelled', errors: 'Completed with failures', error: 'Build failed', done: 'Done', stopped: 'Stopped',
};
export function stateBadge(state, extra = '') {
  const live = ['running', 'building'].includes(state);
  return `<span class="badge ${esc(state)}${live ? ' live' : ''}">${live ? '<i class="dot"></i>' : ''}${esc(STATE_LABELS[state] || state)}${extra}</span>`;
}

// Bar chart (SVG) of [label, value] pairs.
export function svgBars(pairs, { width = 520, height = 120, color = '#5fae86', label = v => v } = {}) {
  if (!pairs.length) return '<p class="small">No data yet.</p>';
  const max = Math.max(...pairs.map(p => p[1]), 1), left = 44, bottom = 16, top = 8;
  const slot = (width - left) / pairs.length, bw = Math.max(1, Math.min(24, slot - 2));
  const bars = pairs.map(([k, v], i) => {
    const h = (height - bottom - top) * v / max, x = left + i * slot + (slot - bw) / 2, y = height - bottom - h;
    return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.max(h, v ? 1 : 0).toFixed(1)}" rx="2" fill="${color}"><title>${esc(label(k))}: ${fmtNum(v)}</title></rect>`;
  }).join('');
  const ticks = [pairs[0], pairs[Math.floor(pairs.length / 2)], pairs[pairs.length - 1]]
    .map((p, i) => { const idx = pairs.indexOf(p); return `<text x="${(left + idx * slot + slot / 2).toFixed(1)}" y="${height - 3}" text-anchor="${i === 0 ? 'start' : i === 2 ? 'end' : 'middle'}">${esc(label(p[0]))}</text>`; });
  return `<svg class="chart" viewBox="0 0 ${width} ${height}" style="max-width:${width}px" role="img">
<text x="${left - 4}" y="${top + 8}" text-anchor="end">${fmtRate(max)}</text><line x1="${left}" x2="${width}" y1="${top}" y2="${top}" stroke="#2c322e" stroke-dasharray="2 3"/>
${bars}<line x1="${left}" x2="${width}" y1="${height - bottom}" y2="${height - bottom}" stroke="#434b46"/>${[...new Set(ticks)].join('')}</svg>`;
}

// Multi-series line chart. series: [{name, color, points: [[x, y], ...]}]
export function svgLines(series, { width = 520, height = 140, yMin = null, yMax = null, xLabel = v => v, yLabel = v => v, zero = false } = {}) {
  const pts = series.flatMap(s => s.points);
  if (!pts.length) return '<p class="small">No data yet.</p>';
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1e-9);
  let y0 = yMin ?? Math.min(...ys), y1 = yMax ?? Math.max(...ys);
  if (y1 - y0 < 1e-9) { y0 -= 1; y1 += 1; }
  const left = 40, right = 8, top = 8, bottom = 18;
  const X = x => left + (x - x0) / (x1 - x0) * (width - left - right), Y = y => top + (1 - (y - y0) / (y1 - y0)) * (height - top - bottom);
  const paths = series.map(s => s.points.length ? `<path d="${s.points.map((p, i) => (i ? 'L' : 'M') + X(p[0]).toFixed(1) + ',' + Y(p[1]).toFixed(1)).join('')}" fill="none" stroke="${s.color}" stroke-width="1.6" stroke-linejoin="round"><title>${esc(s.name)}</title></path>` : '').join('');
  const zl = zero && y0 < 0 && y1 > 0 ? `<line x1="${left}" x2="${width - right}" y1="${Y(0)}" y2="${Y(0)}" stroke="#434b46" stroke-dasharray="3 3"/>` : '';
  return `<svg class="chart" viewBox="0 0 ${width} ${height}" style="max-width:${width}px">
<text x="${left - 5}" y="${top + 8}" text-anchor="end">${esc(yLabel(y1))}</text><text x="${left - 5}" y="${height - bottom}" text-anchor="end">${esc(yLabel(y0))}</text>
<line x1="${left}" x2="${left}" y1="${top}" y2="${height - bottom}" stroke="#434b46"/><line x1="${left}" x2="${width - right}" y1="${height - bottom}" y2="${height - bottom}" stroke="#434b46"/>${zl}
${paths}<text x="${left}" y="${height - 4}">${esc(xLabel(x0))}</text><text x="${width - right}" y="${height - 4}" text-anchor="end">${esc(xLabel(x1))}</text></svg>`;
}

export function store(key, value) { try { localStorage.setItem('polyshark.' + key, JSON.stringify(value)); } catch { } }
export function restore(key, fallback) {
  try { const v = localStorage.getItem('polyshark.' + key); return v == null ? fallback : JSON.parse(v); } catch { return fallback; }
}

// Tell the site shell (if any) to open the Position Analyser on a source.
export function openInAnalyser(source) {
  if (window.parent !== window) window.parent.postMessage({ type: 'analyse', source }, location.origin);
  else location.href = '/position/?' + new URLSearchParams({ src: JSON.stringify(source) });
}

export function tipFollow(el, html, ev) {
  if (!html) { el.hidden = true; return; }
  el.innerHTML = html;
  el.hidden = false;
  const pad = 14, r = el.getBoundingClientRect();
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - pad;
  if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - pad;
  el.style.left = x + 'px';
  el.style.top = y + 'px';
}
