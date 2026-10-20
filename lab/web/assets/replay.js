// Full-screen replay viewer used by the Match Lab and Self-play pages.
import { $, api, esc, fmtMs, openInAnalyser, store, restore, tipFollow } from './common.js';
import { Board, PLAYER_COLORS, actionArrow } from './board.js';

const CSS = `
.rv{display:flex;flex-direction:column;height:100%}
.rv-bar{display:flex;align-items:center;justify-content:space-between;gap:14px;padding:8px 16px;border-bottom:1px solid var(--line);flex-wrap:wrap}
.rv-title strong{font-size:14px}.rv-title .small{margin-left:10px}
.rv-bar label{display:flex;gap:6px;align-items:center}.rv-bar select{width:auto;padding:4px 8px}
.seg{display:inline-flex;border:1px solid var(--line2);border-radius:5px;overflow:hidden}.seg button{border:0;border-radius:0;padding:4px 10px;background:transparent;color:var(--muted)}
.seg button[aria-pressed=true]{background:#2a3a31;color:var(--text)}
.rv-body{flex:1;min-height:0;display:grid;grid-template-columns:minmax(0,1fr) 400px}
.rv-stage{position:relative;display:flex;flex-direction:column;min-width:0;min-height:0}
.rv-stage canvas{flex:1;min-height:0;width:100%;cursor:grab}
.rv-hud{position:absolute;left:14px;top:12px;display:flex;gap:8px;pointer-events:none}
.pill{background:rgba(22,24,25,.85);border:1px solid var(--line2);border-radius:20px;padding:3px 11px;font-size:12px}
.rv-controls{display:flex;align-items:center;gap:8px;padding:8px 14px;border-top:1px solid var(--line)}
.rv-controls button{padding:4px 9px}.rv-controls input[type=range]{flex:1;accent-color:var(--accent)}
.rv-controls select{width:auto;padding:3px 6px}
.rv-side{border-left:1px solid var(--line);overflow-y:auto;padding:12px 14px;display:flex;flex-direction:column;gap:14px;scrollbar-width:thin}
.pcards{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.pcard{border:1px solid var(--line);border-radius:6px;padding:8px 10px;font-size:12px}
.pcard.to-move{box-shadow:inset 0 2px 0 var(--pc)}
.pcard h4{margin:0 0 4px;font-size:13px;display:flex;justify-content:space-between;gap:6px;overflow-wrap:anywhere}
.pcard dl{display:grid;grid-template-columns:auto 1fr;gap:1px 8px;margin:0}.pcard dt{color:var(--muted)}.pcard dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}
.rv-now{border:1px solid var(--line);border-radius:6px;padding:8px 10px;font-size:12.5px}
.rv-now pre{margin:6px 0 0;max-height:140px;overflow:auto;font:11px ui-monospace,Menlo,monospace;color:var(--muted);white-space:pre-wrap}
.rv-graph svg{width:100%;height:auto;display:block;cursor:crosshair}
.rv-graph .seg{margin-bottom:6px}
.rv-log{font-size:12px;border:1px solid var(--line);border-radius:6px;max-height:42vh;overflow:auto;scrollbar-width:thin}
.rv-log .turn{position:sticky;top:0;background:#1b201d;padding:4px 8px;color:var(--muted);font-size:11px;border-bottom:1px solid var(--line)}
.rv-log button{display:flex;justify-content:space-between;gap:8px;width:100%;text-align:left;border:0;border-radius:0;background:transparent;padding:3px 8px;font-size:12px;color:var(--text)}
.rv-log button:hover{background:#232a26}.rv-log button.cur{background:#2a3a31}
.rv-log button small{color:var(--muted);font-variant-numeric:tabular-nums}
@media(max-width:900px){.rv-body{grid-template-columns:1fr;grid-template-rows:60vh auto}.rv-side{border-left:0;border-top:1px solid var(--line)}}
`;

export class ReplayViewer {
  constructor() {
    if (!document.getElementById('rv-style')) {
      const s = document.createElement('style');
      s.id = 'rv-style'; s.textContent = CSS;
      document.head.append(s);
    }
    this.dlg = document.createElement('dialog');
    this.dlg.className = 'viewer';
    this.dlg.innerHTML = `<div class="rv">
<header class="rv-bar"><div class="rv-title"><strong data-r="title">Replay</strong><span class="small" data-r="sub"></span></div>
<div class="row"><label class="small">View <select data-r="persp"><option value="-1">Everything</option><option value="0">Player 0's fog</option><option value="1">Player 1's fog</option><option value="mover">Side to move</option></select></label>
<div class="seg" data-r="modes"><button data-mode="iso">Iso</button><button data-mode="flat">Flat</button></div>
<button class="primary" data-r="analyse" title="Open this position in the Position Analyser">Analyse position</button>
<a data-r="download" class="small">JSON</a><a data-r="vis" class="small" title="Text replay for the raylib visualizer (--replay)">Visualizer file</a>
<button data-r="close">Close</button></div></header>
<div class="rv-body"><div class="rv-stage"><canvas data-r="canvas"></canvas><div class="rv-hud"><span class="pill" data-r="hud"></span><span class="pill" data-r="result"></span></div>
<div class="rv-controls"><button data-r="first" title="Start (Home)">⏮</button><button data-r="prevturn" title="Previous turn (Shift+←)">«</button><button data-r="prev" title="Previous action (←)">‹</button>
<button data-r="play" title="Play / pause (Space)">▶</button><button data-r="next" title="Next action (→)">›</button><button data-r="nextturn" title="Next turn (Shift+→)">»</button><button data-r="last" title="End (End)">⏭</button>
<input type="range" min="0" value="0" data-r="slider" aria-label="Step"><select data-r="speed" aria-label="Speed"><option value="2">2/s</option><option value="6" selected>6/s</option><option value="15">15/s</option><option value="40">40/s</option></select>
<span class="small num" data-r="step"></span></div></div>
<aside class="rv-side"><div class="pcards" data-r="players"></div><div class="rv-now" data-r="now"></div>
<div class="rv-graph"><div class="seg" data-r="metric"><button data-m="eval">Eval</button><button data-m="income">Income</button><button data-m="stars">Stars</button><button data-m="units">Units</button><button data-m="cities">Cities</button><button data-m="hp">HP</button></div><div data-r="graph"></div></div>
<div class="rv-log" data-r="log"></div></aside></div></div><div class="tip" data-r="tip" hidden></div>`;
    document.body.append(this.dlg);
    this.r = {};
    this.dlg.querySelectorAll('[data-r]').forEach(el => { this.r[el.dataset.r] = el; });
    this.board = new Board(this.r.canvas, { mode: restore('replay.mode', 'iso') });
    this.board.onHover = (t, e) => tipFollow(this.r.tip, t >= 0 ? this.board.describeTile(t) : '', e);
    this.metric = restore('replay.metric', 'eval');
    this.frames = new Map();
    this.pending = new Map();
    this.k = 0;
    this.timer = null;
    this._wire();
  }

  _wire() {
    const r = this.r;
    r.close.onclick = () => this.close();
    this.dlg.addEventListener('close', () => this._stop());
    r.first.onclick = () => this.go(0);
    r.last.onclick = () => this.go(this.steps - 1);
    r.prev.onclick = () => this.go(this.k - 1);
    r.next.onclick = () => this.go(this.k + 1);
    r.prevturn.onclick = () => this.go(this._turnStart(this.k - 1));
    r.nextturn.onclick = () => this.go(this._nextTurn(this.k));
    r.play.onclick = () => (this.timer ? this._stop() : this._play());
    r.speed.onchange = () => { if (this.timer) { this._stop(); this._play(); } };
    r.slider.oninput = () => this.go(+r.slider.value);
    r.persp.value = restore('replay.persp', '-1');
    r.persp.onchange = () => { store('replay.persp', r.persp.value); this.render(); };
    for (const b of r.modes.querySelectorAll('button')) b.onclick = () => { store('replay.mode', b.dataset.mode); this.board.setMode(b.dataset.mode); this._modes(); };
    for (const b of r.metric.querySelectorAll('button')) b.onclick = () => { this.metric = b.dataset.m; store('replay.metric', this.metric); this._graph(); };
    r.analyse.onclick = () => {
      this.close();
      openInAnalyser({ kind: this.src.kind, id: this.src.id, game: this.src.game, step: this.k });
    };
    r.log.addEventListener('click', e => { const b = e.target.closest('button[data-k]'); if (b) this.go(+b.dataset.k + 1); });
    this.dlg.addEventListener('keydown', e => {
      if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
      const k = { ArrowLeft: () => e.shiftKey ? r.prevturn.click() : r.prev.click(), ArrowRight: () => e.shiftKey ? r.nextturn.click() : r.next.click(),
        Home: () => this.go(0), End: () => this.go(this.steps - 1), ' ': () => r.play.click() }[e.key];
      if (k) { e.preventDefault(); k(); }
    });
  }
  _modes() { for (const b of this.r.modes.querySelectorAll('button')) b.setAttribute('aria-pressed', String(b.dataset.mode === this.board.mode)); }

  async open(src, { title = 'Replay', step = null } = {}) {
    // src: {kind: 'matches'|'selfplay', id, game}
    this.src = src;
    this.frames.clear();
    this.pending.clear();
    this.r.title.textContent = title;
    this.r.sub.textContent = 'Loading…';
    this.r.log.innerHTML = '';
    this.r.graph.innerHTML = '';
    this.r.players.innerHTML = '';
    this._modes();
    if (!this.dlg.open) this.dlg.showModal();
    const base = `${src.kind}/${encodeURIComponent(src.id)}/games/${src.game}`;
    this.base = base;
    this.r.download.href = `/api/${base}/replay?download=1`;
    this.r.vis.href = `/api/${base}/visualizer`;
    try {
      if (!this.rules) this.rules = await api('rules');
      this.board.setRules(this.rules);
      const data = await api(`${base}/timeline`);
      this.replay = data.replay;
      this.tl = data.timeline;
      this.steps = this.tl.steps;
      this.r.vis.hidden = this.replay.start?.kind !== 'generated';
      this.names = (this.replay.players || []).map(p => p.name);
      const res = this.replay.result || {};
      const w = res.winner;
      this.r.result.textContent = w == null ? `Draw · ${res.reason || ''}` : `${this.names[w] || 'P' + w} wins · ${res.reason}${res.forfeit ? ' (' + res.forfeit.kind + ')' : ''}`;
      this.r.sub.textContent = `${this.names.map((n, i) => `P${i} ${n}`).join(' vs ')} · ${this.tl.steps - 1} actions · ${res.turns ?? '?'} turns` +
        (this.tl.verified === false ? ' · ⚠ hash mismatch' : this.tl.verified ? ' · replay verified' : '');
      this.r.slider.max = this.steps - 1;
      this._buildLog();
      this.go(step ?? 0, true);
    } catch (e) {
      this.r.sub.textContent = e.message;
    }
  }
  close() { this._stop(); if (this.dlg.open) this.dlg.close(); }

  async _frame(k) {
    if (this.frames.has(k)) return this.frames.get(k);
    const start = Math.floor(k / 64) * 64;
    if (!this.pending.has(start)) {
      this.pending.set(start, api(`${this.base}/frames?from=${start}&count=64`).then(d => {
        d.states.forEach((s, i) => this.frames.set(d.from + i, s));
      }));
    }
    await this.pending.get(start);
    return this.frames.get(k);
  }

  async go(k, force = false) {
    if (!this.tl) return;
    k = Math.max(0, Math.min(this.steps - 1, k));
    if (k === this.k && !force && this.state) return;
    this.k = k;
    this.r.slider.value = k;
    const want = k;
    const s = await this._frame(k);
    if (this.k !== want) return;
    this.state = s;
    // Warm the next chunk while playing forward.
    if (k % 64 > 40 && k + 64 < this.steps) this._frame(Math.floor(k / 64) * 64 + 64);
    this.render();
  }

  render() {
    const s = this.state, tl = this.tl, k = this.k;
    if (!s) return;
    const pv = this.r.persp.value;
    const persp = pv === 'mover' ? s.current_player : +pv;
    this.board.setState(s, persp);
    const last = k > 0 ? tl.actions[k - 1] : null;
    const next = k < this.steps - 1 ? tl.actions[k] : null;
    const arrows = [];
    if (last) { const a = actionArrow(last, '#ffd84f'); if (a) arrows.push(a); }
    if (next) { const a = actionArrow(next, 'rgba(160,230,255,.85)'); if (a) arrows.push({ ...a, dash: true, alpha: 0.6 }); }
    this.board.setOverlay({ arrows });
    const mover = tl.movers[k];
    this.r.hud.innerHTML = `Turn ${tl.turns[k]} · <span style="color:${PLAYER_COLORS[mover]}">P${mover} ${esc(this.names[mover] || '')}</span> to move`;
    this.r.step.textContent = `${k} / ${this.steps - 1}`;
    this._players(s, mover);
    this._now(last, k - 1);
    this._graph();
    const cur = this.r.log.querySelector('.cur');
    if (cur) cur.classList.remove('cur');
    const btn = this.r.log.querySelector(`button[data-k="${k - 1}"]`);
    if (btn) { btn.classList.add('cur'); btn.scrollIntoView({ block: 'nearest' }); }
  }

  _players(s, mover) {
    const st = this.tl.stats, k = this.k;
    const techName = id => this.rules?.techs?.[id]?.name;
    this.r.players.innerHTML = [0, 1].map(p => {
      const techs = [];
      for (let t = 1; t < 32; t++) if ((s.players[p].techs_mask >> t) & 1) techs.push(techName(t) || t);
      return `<div class="pcard${mover === p ? ' to-move' : ''}" style="--pc:${PLAYER_COLORS[p]}"><h4><span style="color:${PLAYER_COLORS[p]}">P${p}</span><span>${esc(this.names[p] || '')}</span></h4>
<dl><dt>Stars</dt><dd>${st.stars[k][p]}★</dd><dt>Income</dt><dd>+${st.income[k][p]}</dd><dt>Cities</dt><dd>${st.cities[k][p]}</dd><dt>Units</dt><dd>${st.units[k][p]} <span class="muted">(${st.hp[k][p]} hp)</span></dd></dl>
<div class="small" style="margin-top:4px">${techs.length ? esc(techs.join(', ')) : 'No techs'}</div></div>`;
    }).join('');
  }

  _now(action, i) {
    if (!action) { this.r.now.innerHTML = '<span class="muted">Start position</span>'; return; }
    const d = this.replay.decisions?.[i] || {};
    const p = d.p ?? this.tl.movers[i];
    const evalTxt = d.eval != null ? ` · eval <strong>${(+d.eval).toFixed(3)}</strong>` : '';
    this.r.now.innerHTML = `<div><span style="color:${PLAYER_COLORS[p]}">P${p}</span> ${esc(action.label)}</div>
<div class="small">Action ${i + 1} · ${fmtMs(d.ms)}${evalTxt}</div>${d.info ? `<pre>${esc(JSON.stringify(d.info, null, 1))}</pre>` : ''}`;
  }

  _buildLog() {
    const tl = this.tl, dec = this.replay.decisions || [];
    let html = '', lastKey = '';
    for (let i = 0; i < tl.actions.length; i++) {
      const key = `${tl.turns[i]}-${tl.movers[i]}`;
      if (key !== lastKey) {
        const p = tl.movers[i];
        html += `<div class="turn">Turn ${tl.turns[i]} · <span style="color:${PLAYER_COLORS[p]}">P${p} ${esc(this.names[p] || '')}</span></div>`;
        lastKey = key;
      }
      const d = dec[i] || {};
      html += `<button data-k="${i}"><span>${esc(tl.actions[i].label)}</span><small>${d.eval != null ? (+d.eval).toFixed(2) + ' · ' : ''}${fmtMs(d.ms)}</small></button>`;
    }
    this.r.log.innerHTML = html || '<div class="turn">No actions</div>';
  }

  _graph() {
    for (const b of this.r.metric.querySelectorAll('button')) b.setAttribute('aria-pressed', String(b.dataset.m === this.metric));
    const tl = this.tl;
    if (!tl) return;
    const W = 370, H = 130, left = 30, right = 6, top = 6, bottom = 14, n = this.steps;
    let series = [], y0, y1, label = v => v;
    if (this.metric === 'eval') {
      const dec = this.replay.decisions || [];
      const pts = [[], []];
      dec.forEach((d, i) => { if (d.eval != null) pts[d.p].push([i, d.p === 0 ? +d.eval : -d.eval]); });
      series = [{ c: PLAYER_COLORS[0], pts: pts[0] }, { c: PLAYER_COLORS[1], pts: pts[1] }];
      y0 = -1; y1 = 1; label = v => v.toFixed(1);
      if (!pts[0].length && !pts[1].length) {
        this.r.graph.innerHTML = '<p class="small">No evals recorded. Bots can send <code>"eval"</code> with each action (P0\'s perspective is shown, P1\'s evals are negated).</p>';
        return;
      }
    } else {
      const arr = tl.stats[this.metric];
      series = [0, 1].map(p => ({ c: PLAYER_COLORS[p], pts: arr.map((v, i) => [i, v[p]]) }));
      y0 = 0; y1 = Math.max(1, ...arr.flat());
    }
    const X = i => left + i / Math.max(1, n - 1) * (W - left - right), Y = v => top + (1 - (v - y0) / (y1 - y0)) * (H - top - bottom);
    const paths = series.map(s => s.pts.length ? `<path d="${s.pts.map((p, i) => (i ? 'L' : 'M') + X(p[0]).toFixed(1) + ',' + Y(p[1]).toFixed(1)).join('')}" fill="none" stroke="${s.c}" stroke-width="1.6"/>` : '').join('');
    const turnLines = [];
    for (let i = 1; i < n; i++) if (tl.turns[i] !== tl.turns[i - 1] && tl.turns[i] % 5 === 0) turnLines.push(`<line x1="${X(i)}" x2="${X(i)}" y1="${top}" y2="${H - bottom}" stroke="#262c28"/><text x="${X(i) + 2}" y="${H - 3}">t${tl.turns[i]}</text>`);
    const zero = y0 < 0 ? `<line x1="${left}" x2="${W - right}" y1="${Y(0)}" y2="${Y(0)}" stroke="#434b46" stroke-dasharray="3 3"/>` : '';
    this.r.graph.innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="chart"><g font-size="9" fill="#a0a6a3">${turnLines.join('')}<text x="${left - 4}" y="${top + 7}" text-anchor="end">${label(y1)}</text><text x="${left - 4}" y="${H - bottom}" text-anchor="end">${label(y0)}</text></g>
${zero}${paths}<line x1="${X(this.k)}" x2="${X(this.k)}" y1="${top}" y2="${H - bottom}" stroke="#ffd84f" stroke-width="1.2"/></svg>`;
    const svg = this.r.graph.querySelector('svg');
    svg.onclick = e => {
      const r = svg.getBoundingClientRect(), x = (e.clientX - r.left) / r.width * W;
      this.go(Math.round((x - left) / (W - left - right) * (n - 1)));
    };
  }

  _turnStart(k) {
    const tl = this.tl;
    if (k <= 0) return 0;
    const key = t => `${tl.turns[t]}-${tl.movers[t]}`;
    let i = k;
    while (i > 0 && key(i - 1) === key(k)) i--;
    return i;
  }
  _nextTurn(k) {
    const tl = this.tl, key = t => `${tl.turns[t]}-${tl.movers[t]}`;
    let i = k;
    while (i < this.steps - 1 && key(i) === key(k)) i++;
    return i;
  }
  _play() {
    if (this.k >= this.steps - 1) this.go(0);
    this.r.play.textContent = '❚❚';
    const tick = () => {
      if (this.k >= this.steps - 1) return this._stop();
      this.go(this.k + 1);
      this.timer = setTimeout(tick, 1000 / +this.r.speed.value);
    };
    this.timer = setTimeout(tick, 1000 / +this.r.speed.value);
  }
  _stop() { clearTimeout(this.timer); this.timer = null; this.r.play.textContent = '▶'; }
}
