// Board renderer for serialised Polyshark states.
//  * "iso": the raylib visualizer's isometric projection and sprites ((0,0) at the bottom corner).
//  * "flat": a top-down grid for analysis ((0,0) top-left).
// Perspective -1 shows everything; 0/1 hide what that player has not explored.
import { esc } from './common.js';

const DEFAULT_RULES = {
  terrain: ['None', 'Field', 'Forest', 'Mountain', 'Water', 'Village'],
  resources: ['None', 'Fruit', 'Crop', 'Animal', 'Metal'].map((name, id) => ({ id, name })),
  buildings: ['None', 'Mine', 'Farm', 'Road'],
  units: ['None', 'Warrior', 'Archer', 'Rider', 'Defender', 'Giant'].map((name, id) => ({ id, name })),
};
const UNIT_SPRITES = { Warrior: 'warrior', Archer: 'archer', Rider: 'rider', Defender: 'defender', Giant: 'giant' };
const SPRITE_FILES = {
  grass: 'terrain/grass.png', mountain: 'terrain/mountain.png', forest: 'terrain/imperius/Imperius_forest.png',
  village: 'terrain/village.png', lighthouse: 'terrain/lighthouse.png', Fruit: 'terrain/imperius/Imperius_fruit.png',
  Crop: 'terrain/crop.png', Animal: 'terrain/imperius/Imperius_game.png', Metal: 'terrain/metal.png', star: 'star.png',
};
for (let l = 1; l <= 9; l++) SPRITE_FILES['city' + l] = `cities/lvl${l}.png`;
for (const [name, file] of Object.entries(UNIT_SPRITES)) {
  SPRITE_FILES[`u0${name}`] = `units/blue/${file}_blue.png`;
  SPRITE_FILES[`u1${name}`] = `units/red/${file}_red.png`;
}
// Offsets in tile fractions and scale multipliers, copied from visualizer.cpp.
const XF_UNIT = { Warrior: [-0.03, -0.04, 0.70], Archer: [0.05, -0.05, 0.73], Rider: [0, -0.08, 0.62], Defender: [0, -0.08, 0.73], Giant: [0, -0.15, 0.76] };
const XF_RES = { Fruit: [0, -0.4, 0.7], Crop: [0, -0.2, 1.4], Animal: [0, -0.35, 0.4], Metal: [0, -0.38, 0.53], forest: [0.02, -0.1, 1.5], village: [0, -0.3, 0.9], lighthouse: [0, -0.35, 0.4] };
const XF_CITY = [[0, -0.24, 0.42], [0, -0.2, 0.5], [0, -0.16, 0.6], [0, -0.1, 0.65], [0, -0.08, 0.82], [0, -0.08, 0.86], [0, -0.08, 0.9], [0, -0.08, 0.95], [0, -0.08, 1.0]];

export const sprites = {};
let spritesLoading = null;
function loadSprites(onReady) {
  if (spritesLoading) return spritesLoading.then(onReady);
  spritesLoading = Promise.all(Object.entries(SPRITE_FILES).map(([key, file]) => new Promise(resolve => {
    const img = new Image();
    img.onload = () => {
      // Alpha bounding box, so sprites anchor on their visible pixels like the visualizer.
      const c = document.createElement('canvas');
      c.width = img.width; c.height = img.height;
      const g = c.getContext('2d', { willReadFrequently: true });
      g.drawImage(img, 0, 0);
      const d = g.getImageData(0, 0, img.width, img.height).data;
      let x0 = img.width, x1 = -1, y0 = img.height, y1 = -1;
      for (let y = 0; y < img.height; y += 2) for (let x = 0; x < img.width; x += 2) {
        if (d[(y * img.width + x) * 4 + 3] > 16) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
      }
      sprites[key] = { img, x0, x1: x1 + 1, y0, y1: y1 + 1 };
      resolve();
    };
    img.onerror = () => resolve();
    img.src = '/sprites/' + file;
  })));
  return spritesLoading.then(onReady);
}

export const PLAYER_COLORS = ['#5f9bf0', '#ec6a5e'];
const PLAYER_RGB = [[95, 155, 240], [236, 106, 94]];
const TERRAIN_FLAT = { 1: '#4d7a3a', 2: '#2f5a33', 3: '#6f736d', 4: '#25527d', 5: '#93754a' };

export class Board {
  constructor(canvas, { rules = null, mode = 'iso' } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.rules = rules || DEFAULT_RULES;
    this.mode = mode;
    this.state = null;
    this.perspective = -1;
    this.overlay = {};
    this.view = { s: 1, tx: 0, ty: 0 };
    this.onClick = null;
    this.onHover = null;
    this.hover = -1;
    this._drag = null;
    loadSprites(() => this.draw());
    new ResizeObserver(() => this.draw()).observe(canvas);
    canvas.addEventListener('mousemove', e => this._move(e));
    canvas.addEventListener('mouseleave', e => { this.hover = -1; this.onHover?.(-1, e); this.draw(); });
    canvas.addEventListener('mousedown', e => { this._drag = { x: e.clientX, y: e.clientY, tx: this.view.tx, ty: this.view.ty, moved: false }; });
    window.addEventListener('mouseup', e => {
      const d = this._drag;
      this._drag = null;
      if (d && !d.moved && e.target === canvas) this.onClick?.(this.tileAt(e), e);
    });
    canvas.addEventListener('wheel', e => {
      e.preventDefault();
      const r = canvas.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
      const k = Math.exp(-e.deltaY * 0.0015), s = Math.min(6, Math.max(0.4, this.view.s * k));
      const f = s / this.view.s;
      this.view.tx = px - (px - this.view.tx) * f;
      this.view.ty = py - (py - this.view.ty) * f;
      this.view.s = s;
      this.draw();
    }, { passive: false });
    canvas.addEventListener('dblclick', () => this.fit());
  }

  setRules(rules) { if (rules) { this.rules = rules; this.draw(); } }
  setMode(mode) { this.mode = mode; this.fit(); }
  fit() { this.view = { s: 1, tx: 0, ty: 0 }; this.draw(); }
  setState(state, perspective = this.perspective) {
    const resize = !this.state || state?.map_size !== this.state.map_size;
    this.state = state;
    this.perspective = perspective;
    if (resize) this.view = { s: 1, tx: 0, ty: 0 };
    this.draw();
  }
  setPerspective(p) { this.perspective = p; this.draw(); }
  setOverlay(o) { this.overlay = o || {}; this.draw(); }

  // ------------------------------------------------------------ queries --
  unitName(type) { const u = this.rules.units[type]; return (u && (u.name || u)) || '?'; }
  resourceName(r) { const x = this.rules.resources[r]; return (x && (x.name || x)) || '?'; }
  explored(p, i) {
    const w = this.state?.players?.[p]?.visible;
    return !!w && ((w[i >> 4] >> (i & 15)) & 1) === 1;
  }
  hidden(i) {
    const t = this.state.tiles[i];
    if (t.fog || t.terrain === 0) return true;
    return this.perspective >= 0 && !this.explored(this.perspective, i);
  }
  unitAt(i) { const t = this.state.tiles[i]; return t.unit_id != null ? this._units.get(t.unit_id) : null; }
  cityAt(i) { const t = this.state.tiles[i]; return t.city_id != null ? this.state.cities[t.city_id] : null; }
  territory(i) {
    const t = this.state.tiles[i];
    if (t.territory != null) return t.territory;
    if (t.border_city_id != null) return this.state.cities[t.border_city_id]?.owner ?? null;
    return null;
  }
  describeTile(i) {
    if (!this.state || i < 0) return '';
    const n = this.state.map_size, x = i % n, y = Math.floor(i / n);
    const head = `<strong>(${x},${y})</strong> <span class="muted">tile ${i}</span>`;
    if (this.hidden(i)) return head + '<br><span class="muted">Unexplored</span>';
    const t = this.state.tiles[i], parts = [this.rules.terrain[t.terrain] || '?'];
    if (t.resource) parts.push(this.resourceName(t.resource));
    if (t.building) parts.push(this.rules.buildings[t.building]);
    const terr = this.territory(i);
    let html = `${head}<br>${esc(parts.join(' · '))}${terr != null ? ` · <span class="p${terr}">P${terr} territory</span>` : ''}`;
    const c = this.cityAt(i);
    if (c) html += `<br><span class="p${c.owner}">City L${c.level}</span> · pop ${c.population}/${c.level + 1} · units ${c.units_owned}/${c.level + 1}${c.is_capital ? ' · capital' : ''}${c.has_walls ? ' · walls' : ''}${c.has_workshop ? ' · workshop' : ''}${c.is_sieged ? ' · <span class="bad">sieged</span>' : ''}${c.pending_upgrade ? ' · upgrade pending' : ''}`;
    const u = this.unitAt(i);
    if (u) html += `<br><span class="p${u.owner}">${esc(this.unitName(u.type))}</span> ${u.hp}/${u.max_hp} hp · mp ${u.mp}${u.has_attacked ? ' · attacked' : ''} · kills ${u.kills}${u.promotion_ready ? ' · veteran ready' : ''}`;
    return html;
  }

  // ------------------------------------------------------------ geometry --
  _size() {
    const r = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
    return { w: r.width, h: r.height, dpr };
  }
  _layout(w, h) {
    const n = this.state.map_size;
    if (this.mode === 'flat') {
      const S = Math.floor(Math.min((w - 40) / n, (h - 36) / n));
      return { n, S, ox: (w - S * n) / 2 + 8, oy: (h - S * n) / 2 + 10 };
    }
    // Room above the grid for tall sprites (mountains, cities).
    const W = Math.min(w * 0.96 / n, (h * 0.9 / n) / 0.6 / 1.05);
    const H = W * 0.6;
    const gridH = n * H;
    const oy = (h - gridH) / 2 + gridH - H / 2 + H * 0.35;
    return { n, W, H, ox: w / 2, oy };
  }
  center(L, i) {
    const n = L.n, c = i % n, r = Math.floor(i / n);
    if (this.mode === 'flat') return [L.ox + (c + 0.5) * L.S, L.oy + (r + 0.5) * L.S];
    return [L.ox + (c - r) * L.W / 2, L.oy - (c + r) * L.H / 2];
  }
  tileAt(e) {
    if (!this.state || !this._L) return -1;
    const rect = this.canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left - this.view.tx) / this.view.s, my = (e.clientY - rect.top - this.view.ty) / this.view.s;
    const L = this._L, n = L.n;
    let r, c;
    if (this.mode === 'flat') { c = Math.floor((mx - L.ox) / L.S); r = Math.floor((my - L.oy) / L.S); }
    else {
      const u = 2 * (mx - L.ox) / L.W, s = 2 * (L.oy - my) / L.H;
      c = Math.floor((s + u) / 2 + 0.5); r = Math.floor((s - u) / 2 + 0.5);
    }
    return r >= 0 && c >= 0 && r < n && c < n ? r * n + c : -1;
  }
  _move(e) {
    if (this._drag) {
      const dx = e.clientX - this._drag.x, dy = e.clientY - this._drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 4) this._drag.moved = true;
      if (this._drag.moved) { this.view.tx = this._drag.tx + dx; this.view.ty = this._drag.ty + dy; this.draw(); }
      return;
    }
    const t = this.tileAt(e);
    if (t !== this.hover) { this.hover = t; this.draw(); }
    this.onHover?.(t, e);
  }

  // ------------------------------------------------------------- drawing --
  draw() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = 0; this._draw(); });
  }
  _draw() {
    const { w, h, dpr } = this._size();
    const g = this.ctx;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (!this.state) return;
    this._units = new Map((this.state.units || []).map(u => [u.id, u]));
    const L = this._L = this._layout(w, h);
    g.setTransform(dpr * this.view.s, 0, 0, dpr * this.view.s, dpr * this.view.tx, dpr * this.view.ty);
    if (this.mode === 'flat') this._drawFlat(g, L); else this._drawIso(g, L);
    this._drawOverlay(g, L);
  }

  _diamond(g, cx, cy, W, H) {
    g.beginPath(); g.moveTo(cx, cy - H / 2); g.lineTo(cx + W / 2, cy); g.lineTo(cx, cy + H / 2); g.lineTo(cx - W / 2, cy); g.closePath();
  }
  _sprite(g, key, cx, cy, L, xf, base) {
    const sp = sprites[key];
    if (!sp) return false;
    const bw = sp.x1 - sp.x0;
    const scale = L.W * base * xf[2] / bw;
    const ax = cx + xf[0] * L.W, ay = cy + xf[1] * L.H + L.H / 2;
    g.drawImage(sp.img, ax - (sp.x0 + sp.x1) / 2 * scale, ay - sp.y1 * scale, sp.img.width * scale, sp.img.height * scale);
    return true;
  }
  _tileEdges(L, i, cx, cy) {
    // Grid neighbour → diamond edge (iso) or square edge (flat), as two points.
    const n = L.n, c = i % n, r = Math.floor(i / n);
    if (this.mode === 'flat') {
      const x0 = L.ox + c * L.S, y0 = L.oy + r * L.S, S = L.S;
      return [[c + 1, r, [x0 + S, y0], [x0 + S, y0 + S]], [c - 1, r, [x0, y0], [x0, y0 + S]], [c, r + 1, [x0, y0 + S], [x0 + S, y0 + S]], [c, r - 1, [x0, y0], [x0 + S, y0]]];
    }
    const T = [cx, cy - L.H / 2], R = [cx + L.W / 2, cy], B = [cx, cy + L.H / 2], Lf = [cx - L.W / 2, cy];
    return [[c + 1, r, T, R], [c, r + 1, Lf, T], [c - 1, r, B, Lf], [c, r - 1, R, B]];
  }
  _borders(g, L, i, cx, cy) {
    const own = this.territory(i);
    if (own == null) return;
    const n = L.n;
    g.lineWidth = Math.max(1.5, (L.W || L.S) * 0.045);
    g.strokeStyle = `rgba(${PLAYER_RGB[own].join(',')},0.95)`;
    for (const [nc, nr, a, b] of this._tileEdges(L, i, cx, cy)) {
      const j = nc >= 0 && nr >= 0 && nc < n && nr < n ? nr * n + nc : -1;
      if (j >= 0 && !this.hidden(j) && this.territory(j) === own) continue;
      // Inset toward the tile centre so neighbouring borders sit side by side.
      const k = 0.1, ax = a[0] + (cx - a[0]) * k, ay = a[1] + (cy - a[1]) * k, bx = b[0] + (cx - b[0]) * k, by = b[1] + (cy - b[1]) * k;
      g.beginPath(); g.moveTo(ax, ay); g.lineTo(bx, by); g.stroke();
    }
    g.fillStyle = `rgba(${PLAYER_RGB[own].join(',')},0.10)`;
    if (this.mode === 'flat') g.fillRect(L.ox + (i % n) * L.S, L.oy + Math.floor(i / n) * L.S, L.S, L.S);
    else { this._diamond(g, cx, cy, L.W, L.H); g.fill(); }
  }

  _drawIso(g, L) {
    const n = L.n, st = this.state;
    for (let sum = 2 * (n - 1); sum >= 0; sum--) {
      for (let r = 0; r < n; r++) {
        const c = sum - r;
        if (c < 0 || c >= n) continue;
        const i = r * n + c, t = st.tiles[i], [cx, cy] = this.center(L, i);
        if (this.hidden(i)) {
          this._diamond(g, cx, cy, L.W, L.H);
          g.fillStyle = '#141615'; g.fill();
          g.strokeStyle = '#1d201e'; g.lineWidth = 1; g.stroke();
          continue;
        }
        if (t.terrain === 4) { this._diamond(g, cx, cy, L.W, L.H); g.fillStyle = '#1e5082'; g.fill(); }
        else if (t.terrain !== 3) { if (!this._sprite(g, 'grass', cx, cy, L, [0, 0, 1], 1)) { this._diamond(g, cx, cy, L.W, L.H); g.fillStyle = '#5a8f35'; g.fill(); } }
        if (t.terrain === 3) {
          this._borders(g, L, i, cx, cy);
          if (!this._sprite(g, 'mountain', cx, cy, L, [0, 0, 1], 1)) { this._diamond(g, cx, cy, L.W, L.H); g.fillStyle = '#777'; g.fill(); }
        } else this._borders(g, L, i, cx, cy);
        if (t.terrain === 2) this._sprite(g, 'forest', cx, cy, L, XF_RES.forest, 0.5);
        if (t.terrain === 5 && t.city_id == null) this._sprite(g, 'village', cx, cy, L, XF_RES.village, 0.5);
        if ((c === 0 || c === n - 1) && (r === 0 || r === n - 1)) this._sprite(g, 'lighthouse', cx, cy, L, XF_RES.lighthouse, 0.5);
        if (t.resource) { const rn = this.resourceName(t.resource); if (XF_RES[rn]) this._sprite(g, rn, cx, cy, L, XF_RES[rn], 0.5); }
        if (t.building) this._buildingTag(g, cx, cy, L, t.building);
        if (t.city_id != null) {
          const city = st.cities[t.city_id], lvl = Math.min(9, Math.max(1, city.level));
          this._sprite(g, 'city' + lvl, cx, cy, L, XF_CITY[lvl - 1], 1);
        }
      }
    }
    // Labels and units last, so tall sprites never cover them.
    for (let sum = 2 * (n - 1); sum >= 0; sum--) {
      for (let r = 0; r < n; r++) {
        const c = sum - r;
        if (c < 0 || c >= n) continue;
        const i = r * n + c;
        if (this.hidden(i)) continue;
        const t = st.tiles[i], [cx, cy] = this.center(L, i);
        if (t.city_id != null) this._cityPlate(g, cx, cy + L.H * 0.4, L.W * 0.9, st.cities[t.city_id]);
        if (t.unit_id != null) this._unitIso(g, L, cx, cy, this._units.get(t.unit_id));
      }
    }
  }
  _buildingTag(g, cx, cy, L, b) {
    const label = { 1: 'Mine', 2: 'Farm', 3: 'Road' }[b] || '';
    g.font = `600 ${Math.max(8, L.W * 0.1)}px system-ui`;
    g.textAlign = 'center';
    g.fillStyle = 'rgba(0,0,0,.55)';
    g.fillText(label, cx + 1, cy + L.H * 0.3 + 1);
    g.fillStyle = '#f3e6c4';
    g.fillText(label, cx, cy + L.H * 0.3);
  }
  _cityPlate(g, cx, cy, size, city) {
    const fs = Math.max(8, size * 0.11);
    g.font = `700 ${fs}px system-ui`;
    const label = `${city.is_capital ? '★ ' : ''}L${city.level}`;
    const tw = g.measureText(label).width, pips = city.level + 1, pw = Math.max(3, fs * 0.42), gap = 2;
    const width = Math.max(tw + 10, pips * (pw + gap) + 8), h = fs * 1.9;
    g.fillStyle = 'rgba(16,18,17,.82)';
    g.strokeStyle = PLAYER_COLORS[city.owner] || '#999';
    g.lineWidth = 1.2;
    roundRect(g, cx - width / 2, cy - h / 2, width, h, 3); g.fill(); g.stroke();
    g.fillStyle = '#eef0ed'; g.textAlign = 'center'; g.textBaseline = 'alphabetic';
    g.fillText(label, cx, cy - h / 2 + fs * 0.95);
    for (let k = 0; k < pips; k++) {
      g.fillStyle = k < city.population ? (PLAYER_COLORS[city.owner] || '#ccc') : '#3a403c';
      g.fillRect(cx - (pips * (pw + gap) - gap) / 2 + k * (pw + gap), cy + h / 2 - pw - 3, pw, pw * 0.8);
    }
    if (city.is_sieged) { g.fillStyle = '#ffaaa3'; g.fillText('!', cx + width / 2 + fs * 0.4, cy); }
  }
  _unitIso(g, L, cx, cy, u) {
    if (!u) return;
    const name = this.unitName(u.type), key = `u${u.owner}${name}`, sp = sprites[key];
    const ready = this.state.current_player === u.owner && (u.mp > 0 || !u.has_attacked);
    g.save();
    g.translate(cx, cy + L.H * 0.14); g.scale(1, 0.6);
    g.beginPath(); g.arc(0, 0, L.W * 0.21, 0, Math.PI * 2);
    g.fillStyle = ready ? 'rgba(80,220,255,.22)' : 'rgba(0,0,0,.25)'; g.fill();
    if (ready) { g.strokeStyle = 'rgba(80,220,255,.75)'; g.lineWidth = 1.5; g.stroke(); }
    g.restore();
    let figTop = cy - L.H * 0.55, figLeft = cx - L.W * 0.2;
    if (sp) {
      // The visualizer's scale, ×1.3 so units read clearly at web sizes; the figure stands on the tile centre.
      const xf = XF_UNIT[name] || [0, 0, 0.75];
      const scale = (L.W * 0.95) / sp.img.height * xf[2] * 1.3;
      const w = sp.img.width * scale, h = sp.img.height * scale;
      const xoff = ((sp.x0 + sp.x1) / 2 - sp.img.width / 2) * scale, yoff = ((sp.y0 + sp.y1) / 2 - sp.img.height / 2) * scale;
      const lift = (sp.y1 - sp.y0) * scale * 0.32;
      const x = cx - w / 2 - xoff + xf[0] * L.W, y = cy - h / 2 - yoff + xf[1] * L.H - lift;
      g.drawImage(sp.img, x, y, w, h);
      figTop = y + sp.y0 * scale;
      figLeft = x + sp.x0 * scale;
    } else {
      g.beginPath(); g.arc(cx, cy - L.H * 0.2, L.W * 0.16, 0, Math.PI * 2); g.fillStyle = PLAYER_COLORS[u.owner]; g.fill();
    }
    this._hp(g, figLeft, figTop + L.W * 0.06, L.W, u);
  }
  _hp(g, x, y, size, u) {
    const fs = Math.max(9, size * 0.13);
    g.font = `700 ${fs}px system-ui`;
    g.textAlign = 'center';
    const txt = String(u.hp);
    g.fillStyle = 'rgba(0,0,0,.8)'; g.fillText(txt, x + 1, y + 1);
    g.fillStyle = u.hp < u.max_hp ? '#ffd0b0' : '#ffffff'; g.fillText(txt, x, y);
    if (u.promotion_ready || u.kills >= 3) { g.fillStyle = '#e1bc69'; g.fillText('★', x + fs * 0.9, y); }
  }

  _drawFlat(g, L) {
    const n = L.n, S = L.S, st = this.state;
    g.font = `11px system-ui`; g.fillStyle = '#6f7673'; g.textAlign = 'center';
    for (let k = 0; k < n; k++) {
      g.fillText(k, L.ox + (k + 0.5) * S, L.oy - 5);
      g.textAlign = 'right'; g.fillText(k, L.ox - 6, L.oy + (k + 0.5) * S + 4); g.textAlign = 'center';
    }
    for (let i = 0; i < n * n; i++) {
      const x = L.ox + (i % n) * S, y = L.oy + Math.floor(i / n) * S, t = st.tiles[i];
      if (this.hidden(i)) { g.fillStyle = '#141615'; g.fillRect(x, y, S, S); g.strokeStyle = '#1d201e'; g.strokeRect(x + .5, y + .5, S - 1, S - 1); continue; }
      g.fillStyle = TERRAIN_FLAT[t.terrain] || '#444'; g.fillRect(x, y, S, S);
      g.strokeStyle = 'rgba(0,0,0,.25)'; g.lineWidth = 1; g.strokeRect(x + .5, y + .5, S - 1, S - 1);
      const glyph = { 2: '♣', 3: '▲', 5: '⌂' }[t.terrain];
      if (glyph && t.city_id == null) { g.fillStyle = 'rgba(255,255,255,.28)'; g.font = `${S * 0.42}px system-ui`; g.fillText(glyph, x + S * 0.5, y + S * 0.62); }
    }
    for (let i = 0; i < n * n; i++) if (!this.hidden(i)) { const [cx, cy] = this.center(L, i); this._borders(g, L, i, cx, cy); }
    for (let i = 0; i < n * n; i++) {
      if (this.hidden(i)) continue;
      const x = L.ox + (i % n) * S, y = L.oy + Math.floor(i / n) * S, t = st.tiles[i];
      if (t.resource) {
        const rn = this.resourceName(t.resource), col = { Fruit: '#e05d5d', Crop: '#e8cf5a', Animal: '#b98a5a', Metal: '#c8ccd0' }[rn] || '#fff';
        g.beginPath(); g.arc(x + S * 0.2, y + S * 0.2, S * 0.11, 0, Math.PI * 2); g.fillStyle = col; g.fill();
        g.fillStyle = '#111'; g.font = `700 ${S * 0.13}px system-ui`; g.fillText(rn[0], x + S * 0.2, y + S * 0.245);
      }
      if (t.building) { g.fillStyle = '#f3e6c4'; g.font = `600 ${S * 0.15}px system-ui`; g.fillText({ 1: 'Mine', 2: 'Farm', 3: 'Road' }[t.building] || '', x + S * 0.5, y + S * 0.93); }
      if (t.city_id != null) {
        const c = st.cities[t.city_id];
        g.fillStyle = 'rgba(16,18,17,.75)'; g.strokeStyle = PLAYER_COLORS[c.owner]; g.lineWidth = 2;
        roundRect(g, x + S * 0.12, y + S * 0.12, S * 0.76, S * 0.76, 4); g.fill(); g.stroke();
        g.fillStyle = '#eef0ed'; g.font = `700 ${S * 0.16}px system-ui`; g.fillText(`${c.is_capital ? '★' : ''}L${c.level} · ${c.population}/${c.level + 1}`, x + S * 0.5, y + S * 0.3);
      }
      if (t.unit_id != null) {
        const u = this._units.get(t.unit_id);
        if (!u) continue;
        const ready = st.current_player === u.owner && (u.mp > 0 || !u.has_attacked);
        g.beginPath(); g.arc(x + S * 0.5, y + S * 0.64, S * 0.22, 0, Math.PI * 2);
        g.fillStyle = PLAYER_COLORS[u.owner]; g.fill();
        g.lineWidth = ready ? 2.5 : 1; g.strokeStyle = ready ? '#bff3ff' : 'rgba(0,0,0,.5)'; g.stroke();
        g.fillStyle = '#0d1110'; g.font = `800 ${S * 0.2}px system-ui`; g.fillText(this.unitName(u.type)[0], x + S * 0.5, y + S * 0.71);
        g.fillStyle = u.hp < u.max_hp ? '#ffd0b0' : '#fff'; g.font = `700 ${S * 0.17}px system-ui`;
        g.fillText(u.hp, x + S * 0.82, y + S * 0.95);
      }
    }
  }

  _drawOverlay(g, L) {
    const o = this.overlay, flat = this.mode === 'flat';
    const ring = (i, color, fill) => {
      const [cx, cy] = this.center(L, i);
      if (flat) { g.strokeStyle = color; g.lineWidth = 3; g.strokeRect(L.ox + (i % L.n) * L.S + 3, L.oy + Math.floor(i / L.n) * L.S + 3, L.S - 6, L.S - 6); if (fill) { g.fillStyle = fill; g.fillRect(L.ox + (i % L.n) * L.S, L.oy + Math.floor(i / L.n) * L.S, L.S, L.S); } return; }
      g.save(); g.translate(cx, cy); g.scale(1, L.H / L.W);
      g.beginPath(); g.arc(0, 0, L.W * 0.22, 0, Math.PI * 2);
      g.lineWidth = L.W * 0.07; g.strokeStyle = color; g.stroke();
      g.beginPath(); g.arc(0, 0, L.W * 0.08, 0, Math.PI * 2); g.fillStyle = color; g.fill();
      g.restore();
    };
    const outline = (i, color, width = 3) => {
      const [cx, cy] = this.center(L, i);
      g.lineWidth = width; g.strokeStyle = color;
      if (flat) g.strokeRect(L.ox + (i % L.n) * L.S + 1.5, L.oy + Math.floor(i / L.n) * L.S + 1.5, L.S - 3, L.S - 3);
      else { this._diamond(g, cx, cy, L.W, L.H); g.stroke(); }
    };
    for (const h of o.tint || []) {
      const [cx, cy] = this.center(L, h.tile);
      g.fillStyle = h.color;
      if (flat) g.fillRect(L.ox + (h.tile % L.n) * L.S, L.oy + Math.floor(h.tile / L.n) * L.S, L.S, L.S);
      else { this._diamond(g, cx, cy, L.W, L.H); g.fill(); }
    }
    for (const i of o.moves || []) ring(i, 'rgba(80,220,255,.75)');
    for (const i of o.attacks || []) ring(i, 'rgba(240,90,90,.85)');
    for (const i of o.targets || []) ring(i, 'rgba(225,188,105,.9)');
    if (o.selected != null && o.selected >= 0) outline(o.selected, '#ffd84f', 3);
    if (this.hover >= 0) outline(this.hover, 'rgba(255,255,255,.55)', 1.5);
    for (const a of o.arrows || []) this._arrow(g, L, a);
  }
  _arrow(g, L, a) {
    const size = L.W || L.S;
    const pts = (a.path && a.path.length > 1 ? a.path : [a.from, a.to]).map(i => this.center(L, i));
    if (pts.length < 2) {
      // Single-tile actions (train, recover, upgrade…): a marker on the tile.
      const [cx, cy] = this.center(L, a.from ?? a.to);
      g.beginPath(); g.arc(cx, cy, size * 0.12, 0, Math.PI * 2); g.lineWidth = 3; g.strokeStyle = a.color || '#ffd84f'; g.stroke();
      return;
    }
    g.save();
    g.strokeStyle = g.fillStyle = a.color || '#ffd84f';
    g.lineWidth = a.width || Math.max(2.5, size * 0.06);
    g.lineCap = 'round'; g.lineJoin = 'round';
    if (a.dash) g.setLineDash([size * 0.12, size * 0.1]);
    g.globalAlpha = a.alpha ?? 0.9;
    g.beginPath();
    pts.forEach(([x, y], k) => (k ? g.lineTo(x, y) : g.moveTo(x, y)));
    const [x1, y1] = pts[pts.length - 1], [x0, y0] = pts[pts.length - 2];
    const ang = Math.atan2(y1 - y0, x1 - x0), head = size * 0.2;
    g.stroke();
    g.setLineDash([]);
    g.beginPath(); g.moveTo(x1, y1);
    g.lineTo(x1 - head * Math.cos(ang - 0.45), y1 - head * Math.sin(ang - 0.45));
    g.lineTo(x1 - head * Math.cos(ang + 0.45), y1 - head * Math.sin(ang + 0.45));
    g.closePath(); g.fill();
    if (a.label) {
      g.font = `700 ${Math.max(10, size * 0.14)}px system-ui`; g.textAlign = 'center';
      const mx = (pts[0][0] + x1) / 2, my = (pts[0][1] + y1) / 2 - 4;
      g.globalAlpha = 1; g.fillStyle = 'rgba(0,0,0,.75)'; g.fillText(a.label, mx + 1, my + 1); g.fillStyle = a.color || '#ffd84f'; g.fillText(a.label, mx, my);
    }
    g.restore();
  }
}

function roundRect(g, x, y, w, h, r) {
  g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
}

// Arrow/overlay helpers for an action object {type, from, to, param, path}.
export function actionArrow(a, color, label) {
  if (!a) return null;
  if (a.type === 'Move') return { from: a.from, to: a.to, path: a.path, color, label };
  if (a.type === 'Attack') return { from: a.from, to: a.to, color: color || '#f05a5a', label, dash: true };
  if (a.type === 'TrainUnit' || a.type === 'Recover') return { from: a.from, color, label };
  if (a.type === 'HarvestResource' || a.type === 'CaptureCity') return { from: a.to, color, label };
  return null;
}
