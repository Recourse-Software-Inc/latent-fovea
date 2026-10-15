// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// Masked-diffusion view (MDLM). The answer starts as [MASK] slots; each denoising step the
// model reads the whole sequence at once (attention runs both ways) and some slots lock in.
// Steps are computed a little ahead in the worker and replayed layer by layer, like the
// autoregressive view: the field shows one slot at a time (the one that just changed,
// or the one you click), the text shows every slot's current best guess, and in 3-D each
// masked slot sits at the centroid of its guesses until it snaps onto a token.
import { ForwardView } from './field.js';

export class DiffusionView extends ForwardView {
  constructor(ctx) {
    super(ctx);
    this.canFly = false;  // no left-to-right path to follow: every slot fills in at once
    this.canAttribute = false;  // attribution and the neuron inspector explain one next word
    this.steps = []; this.snaps = new Map(); this.lensCache = new Map();
    this.cur = 0; this.sel = -1; this.nSteps = 0; this.lastStep = -1; this.inflight = false;
  }

  onWorker(m) {
    if (m.type === 'ready') {
      this.maskRow = m.maskRow;
      super.onWorker(m);
      this.$('fwdStatus').textContent = `${this.modelName} ready · up to ${m.maxT} positions`;
    } else if (m.type === 'dstep') this.onStep(m);
    else if (m.type === 'dlens') { this.lensCache.set(`${m.snapId}:${m.p}`, m.lens); this.hcache = null; }
    else super.onWorker(m);
  }

  run(text) {
    if (!this.ready) return;
    const ids = this.tok.encode(text);
    const prompt = ids.map((i) => this.rowOf[i]).filter((r) => r >= 0).slice(0, this.maxT - 8);
    if (!prompt.length) return;
    const N = Math.max(4, Math.min(+this.$('fwdCount').value || 32, this.maxT - prompt.length));
    this.promptLen = prompt.length;
    this.initial = [...prompt, ...Array(N).fill(this.maskRow)];
    this.rows = this.initial.slice();
    this.nSteps = N;  // one step per slot on average, as in the MDLM paper's small-step regime
    this.steps = []; this.snaps.clear(); this.lensCache.clear(); this.hcache = null;
    this.cur = 0; this.sel = -1; this.lastStep = -1; this.beat = 0; this.playing = true; this.inflight = false;
    this.worker.postMessage({ type: 'dreset', rows: this.initial, steps: this.nSteps, greedy: this.$('fwdSample').value === 'greedy', seed: (Math.random() * 2 ** 31) | 0 });
    this.requestStep(0);
    document.body.classList.add('running');
    this.showPanel(true);
    this.renderChips(); this.syncButtons();
    const c = [0, 0, 0];
    for (const r of prompt) for (let k = 0; k < 3; k++) c[k] += this.points[3 * r + k] / prompt.length;
    this.flyToPoint(c, 2.4);
  }

  requestStep(i) {
    if (this.inflight || i > this.nSteps || this.lastStep >= 0) return;
    this.inflight = true;
    this.worker.postMessage({ type: 'dstep', i });
  }

  onStep(m) {
    this.inflight = false;
    const prev = m.i ? this.steps[m.i - 1] : null;
    m.before = prev ? prev.rows : this.initial;
    if (m.snap) this.snaps.set(m.snapId, m.snap);
    for (const p in m.lens) this.lensCache.set(`${m.snapId}:${p}`, m.lens[p]);
    if (m.forwarded) this.msEma = this.msEma ? this.msEma * 0.7 + m.ms * 0.3 : m.ms;
    this.steps[m.i] = m;
    if (m.left === 0 || m.cleanup) this.lastStep = m.i;
    else this.requestStep(m.i + 1);  // compute ahead while the replay runs
    if (m.i === this.cur) { this.beat = m.forwarded ? 0 : this.L + 1; this.lastBeat = performance.now(); }
    this.renderChips();
  }

  // Replay: layers 0..L for steps that ran the model, then one "unmask" beat per step.
  tick(now) {
    if (!this.active || !this.ready) return;
    const st = this.steps[this.cur];
    if (!st || !(this.playing || this.stepOnce)) return;
    const ms = this.speed(), R = this.L + 1;
    const hold = this.beat === R ? Math.max(ms * 3, 120) : ms;  // linger on the unmask beat
    if (now - this.lastBeat < hold && !this.stepOnce) return;
    this.stepOnce = false; this.lastBeat = now;
    if (this.beat < R) { this.beat = ms === 0 ? R : this.beat + 1; this.hcache = null; return; }
    if (this.cur >= this.lastStep && this.lastStep >= 0) { this.playing = false; this.syncButtons(); return; }
    const next = this.steps[this.cur + 1];
    if (!next) return;  // still computing
    this.cur++; this.beat = next.forwarded ? 0 : R; this.hcache = null;
    this.renderChips();
  }

  // Rows as shown: before this step's unmasking until its last beat.
  shownRows() {
    const st = this.steps[this.cur];
    if (!st) return this.initial || [];
    return this.beat >= this.L + 1 ? st.rows : st.before;
  }
  focusSlot() {
    const st = this.steps[this.cur];
    if (this.sel >= 0) return this.sel;
    if (st && st.changes.length) return st.changes[0][0];
    const rows = this.shownRows();
    for (let p = this.promptLen; p < rows.length; p++) if (rows[p] === this.maskRow) return p;
    return rows.length - 1;
  }

  view() {
    this.rows = this.shownRows();
    const st = this.steps[this.cur];
    if (!st) return { pos: 0, h: null, beat: 0 };
    const q = this.focusSlot(), snap = this.snaps.get(st.snapId);
    if (!snap) return { pos: 0, h: null, beat: 0 };
    const key = `${this.cur}:${q}:${this.lensCache.has(`${st.snapId}:${q}`)}`;
    if (!this.hcache || this.hcache.key !== key) {
      const { L, H, F } = this, n = snap.n;
      const att = new Float32Array(L * H * n);
      for (let l = 0; l < L; l++) for (let h = 0; h < H; h++) {
        const o = ((l * H + h) * n + q) * n;
        for (let k = 0; k < n; k++) att[(l * H + h) * n + k] = snap.att[o + k] / 255;
      }
      const mlp = new Float32Array(L * F);
      for (let l = 0; l < L; l++) for (let i = 0; i < F; i++) mlp[l * F + i] = snap.mlp[(l * n + q) * F + i] / 25;
      let lens = this.lensCache.get(`${st.snapId}:${q}`);
      if (!lens) {
        lens = Array(L + 1).fill(null);
        const b = st.beliefs[q];
        if (b) lens[L] = b;
        if (b && st.snapId === this.latestSnap() && !this.lensAsked?.has(`${st.snapId}:${q}`)) {
          (this.lensAsked ||= new Set()).add(`${st.snapId}:${q}`);
          this.worker.postMessage({ type: 'dlens', p: q, snapId: st.snapId });
        }
      }
      this.hcache = { key, h: { att, headN: snap.headN.subarray(q * L * H, (q + 1) * L * H), mlp, lens } };
    }
    return { pos: snap.n - 1, h: this.hcache.h, beat: Math.min(this.beat, this.L + 1) };
  }
  latestSnap() { let s = 0; for (const st of this.steps) if (st && st.snapId > s) s = st.snapId; return s; }

  hudText(v) {
    const st = this.steps[this.cur];
    if (!st) return '';
    const left = this.shownRows().filter((r) => r === this.maskRow).length;
    const phase = v.beat <= this.L ? (v.beat === 0 ? 'embedding' : `layer ${v.beat - 1}`) : (st.changes.length ? `unmasked ${st.changes.length}` : 'no change');
    return `step ${this.cur + 1} of ${this.nSteps + 1} · ${left} still masked · slot ${this.focusSlot()} · ${phase}`;
  }

  guess(p) {  // best current guess for a masked slot, from the step being shown
    const st = this.steps[this.cur];
    const b = st && st.beliefs[p];
    return b ? { row: b.rows[0], p: b.probs[0] } : null;
  }
  fresh(p) {
    const st = this.steps[this.cur];
    return st && this.beat >= this.L + 1 && st.changes.some(([q]) => q === p);
  }

  renderChips() {
    const rows = this.shownRows(), sel = this.focusSlot();
    this.$('fwdChips').innerHTML = rows.map((r, p) => {
      const masked = r === this.maskRow, g = masked ? this.guess(p) : null;
      const cls = ['chip', p < this.promptLen ? 'prompt' : 'gen', masked ? 'masked' : '', p === sel ? 'on' : '', this.fresh(p) ? 'fresh' : ''].join(' ');
      const label = masked ? (g ? this.disp(this.tokens[g.row]) : '▒') : this.disp(this.tokens[r]);
      return `<button type="button" class="${cls}" data-pos="${p}" title="${masked ? 'masked slot: best guess' : ''}">${esc(label)}</button>`;
    }).join('');
  }
  bindUi() {
    super.bindUi();
    // Clicking a chip picks the slot the field shows (instead of scrubbing positions).
    this.$('fwdChips').addEventListener('click', (e) => {
      if (!(this instanceof DiffusionView)) return;
      const b = e.target.closest('[data-pos]'); if (!b) return;
      e.stopImmediatePropagation();
      const p = +b.dataset.pos; this.sel = this.sel === p ? -1 : p; this.hcache = null; this.renderChips();
    }, true);
  }

  renderOutput() {
    if (!this.steps.length) return;
    const rows = this.shownRows(), t = (r) => esc(this.tokens[r]);
    let html = `<span class="p">${rows.slice(0, this.promptLen).map(t).join('')}</span>`;
    for (let p = this.promptLen; p < rows.length; p++) {
      const r = rows[p];
      if (r !== this.maskRow) html += `<span class="${this.fresh(p) ? 'now' : 'g'}">${t(r)}</span>`;
      else {
        const g = this.guess(p);
        html += g ? `<span class="m" style="opacity:${(0.25 + 0.6 * g.p).toFixed(2)}">${t(g.row)}</span>` : '<span class="m">▒</span>';
      }
    }
    if (this.playing) html += '<span class="caret"></span>';
    if (html !== this.lastOut) { this.lastOut = html; const el = this.$('qOut'); el.innerHTML = html; }
  }

  // 3-D: prompt tokens where they are; masked slots at the centroid of their guesses;
  // locked slots on their token. Arcs show the focus slot's attention at the current layer.
  drawOverlay(ov, W, H) {
    if (!this.active) return;
    const v = this.view(); if (!v.h) return;
    const rows = this.shownRows(), d = this.dpr, P = this.points, L = this.L, NH = this.H, n = rows.length;
    const at = (x, y, z) => this.project(x, y, z);
    const pos = rows.map((r, p) => {
      if (r !== this.maskRow) return at(P[3 * r], P[3 * r + 1], P[3 * r + 2]);
      const st = this.steps[this.cur], b = st && st.beliefs[p];
      if (!b) return null;
      const c = [0, 0, 0]; let s = 0;
      b.rows.forEach((rr, j) => { const w = b.probs[j]; s += w; for (let k = 0; k < 3; k++) c[k] += P[3 * rr + k] * w; });
      return at(c[0] / s, c[1] / s, c[2] / s);
    });
    const q = this.focusSlot(), layer = v.beat >= 1 && v.beat <= L ? v.beat - 1 : -1;
    ov.font = `${10.5 * d}px "JetBrains Mono", monospace`; ov.textBaseline = 'middle';
    if (layer >= 0 && pos[q]) {
      const sel = this.pinnedHead || this.focusHead;
      for (let j = 0; j < n; j++) {
        if (j === q || !pos[j]) continue;
        let w = 0;
        if (sel && sel.l === layer) w = v.h.att[(layer * NH + sel.h) * n + j];
        else { for (let k = 0; k < NH; k++) w += v.h.att[(layer * NH + k) * n + j]; w /= NH; }
        if (w < 0.02) continue;
        const a = pos[q], b = pos[j];
        const mx = (a[0] + b[0]) / 2, my = Math.min(a[1], b[1]) - 50 * d - Math.hypot(a[0] - b[0], a[1] - b[1]) * 0.15;
        ov.strokeStyle = `rgba(53,224,230,${Math.min(0.95, 0.15 + w * 1.2)})`; ov.lineWidth = (0.6 + w * 5) * d;
        ov.beginPath(); ov.moveTo(a[0], a[1]); ov.quadraticCurveTo(mx, my, b[0], b[1]); ov.stroke();
      }
    }
    ov.lineWidth = 1 * d; ov.strokeStyle = 'rgba(232,230,245,0.16)'; ov.beginPath();
    let first = true;
    for (const p of pos) { if (!p) { first = true; continue; } if (first) ov.moveTo(p[0], p[1]); else ov.lineTo(p[0], p[1]); first = false; }
    ov.stroke();
    rows.forEach((r, p) => {
      const s = pos[p]; if (!s) return;
      const masked = r === this.maskRow, isQ = p === q, prompt = p < this.promptLen;
      ov.globalAlpha = masked ? 0.7 : 1;
      if (masked) { ov.strokeStyle = isQ ? '#35e0e6' : 'rgba(157,143,245,0.9)'; ov.lineWidth = 1.2 * d; ov.beginPath(); ov.arc(s[0], s[1], (isQ ? 6 : 4) * d, 0, Math.PI * 2); ov.stroke(); }
      else { ov.fillStyle = isQ ? '#35e0e6' : prompt ? '#e8e6f5' : '#9d8ff5'; ov.beginPath(); ov.arc(s[0], s[1], (isQ ? 5.5 : 3.5) * d, 0, Math.PI * 2); ov.fill(); }
      if (this.fresh(p)) { ov.strokeStyle = '#fff'; ov.lineWidth = 1.5 * d; ov.beginPath(); ov.arc(s[0], s[1], 10 * d, 0, Math.PI * 2); ov.stroke(); }
      const g = masked ? this.guess(p) : null;
      const label = masked ? (g ? `${this.disp(this.tokens[g.row])}?` : '') : this.disp(this.tokens[r]);
      ov.fillStyle = masked ? 'rgba(154,151,179,0.9)' : '#e8e6f5';
      ov.fillText(label, s[0] + 8 * d, s[1]);
      ov.globalAlpha = 1;
    });
  }
}

function esc(s) { return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
