// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// Forward-pass view: the model runs in a worker; its computation is replayed layer by layer
// at a speed you choose, across the whole "parametric field" (every layer x 12 heads + MLP),
// and as a path of logit-lens guesses through the 3-D embedding space.
import { GPT2Tokenizer, HFTokenizer } from './bpe.js';
import { dataFetch } from './store.js';
import { SAES } from './sae.js';

const BEATS = { live: 0, slow: 160, slower: 520, glacial: 1500 };

export class ForwardView {
  constructor(ctx) {
    Object.assign(this, ctx);  // $, DATA, tokens, points, cats, dpr, project, disp, flyToPoint, catColors
    this.active = false; this.ready = false; this.loading = false;
    this.hist = []; this.rows = []; this.promptLen = 0;
    this.playing = false; this.pending = false; this.beat = 99; this.lastBeat = 0; this.viewPos = -1;
    this.toGenerate = 0; this.gaze = null; this.focusHead = null; this.stepOnce = false;
    this.canFly = true; this.flying = false;  // fly-through: the camera follows the thought path
    try { this.flying = localStorage.getItem('lf-fly') === '1'; } catch {}
    // Attribution (A): colour the field by each part's direct push toward a word. `pick` is a
    // logit-lens guess chosen to explain instead of the word written next. `nrn` is the open
    // neuron inspector: { l, sel }. gains/nvocab cache the worker's per-neuron answers.
    this.canAttribute = true; this.attrOn = false; this.pick = null; this.nrn = null;
    this.gains = new Map(); this.nvocab = new Map(); this.nrnKey = '';
    this.asked = new Set();  // foveated readouts already requested from the worker: 'lens:pos:depth', 'attr:pos:row'
    // Interventions: parts knocked out (pending until Rerun) and the spec the current run used.
    // The reference run ("A"), to compare the current run with: pinned by hand, or the unchanged
    // run a Rerun with interventions keeps. { kind: 'pin' | 'iv', label, rows, promptLen, gen,
    // hist, p0: per position, the probability it gave the word it wrote next }.
    this.ko = { heads: new Set(), mlps: new Set(), neurons: new Set() }; this.ivRun = null; this.ref = null; this.ivVer = 0;
    // Features (SAE): the set for this model, encodings per 'depth:pos', features per 'depth:j',
    // Neuronpedia's descriptions per 'source:j', and the latest download note.
    this.sae = null; this.saeRes = new Map(); this.saeFeat = new Map(); this.np = new Map(); this.npVer = 0; this.saeMsg = '';
    try { this.attrOn = localStorage.getItem('lf-attr') === '1'; } catch {}
    // Views of the same pass: the field, the logit lens as a river, attention as flow; and the
    // hidden states as a space of their own in the 3-D scene (hs: its projection, refitted as the
    // run grows; shown eases toward goal).
    this.fview = 'field'; this.rollout = true;
    this.hs = { on: false, goal: null, shown: null, n: 0, share: 0, busy: false, t: 0, id: 0, hover: null };
    try { this.fview = ['river', 'flow'].includes(localStorage.getItem('lf-fview')) ? localStorage.getItem('lf-fview') : 'field'; this.rollout = localStorage.getItem('lf-roll') !== '0'; } catch {}
    this.canvas = this.$('field'); this.fc = this.canvas.getContext('2d');
    this.panel = this.$('fwdPanel');
    try { Object.assign(this, JSON.parse(localStorage.getItem('lf-panel')) || {}); } catch {}  // {collapsed, panelH}
    this.bindUi();
    this.bindPanel();
  }

  // ---------- the panel: show, hide (P), resize by its grip ----------
  showPanel(on) {
    this.panel.hidden = !on;
    this.panel.classList.toggle('collapsed', !!this.collapsed);
    this.$('fwdTab').hidden = !(on && this.collapsed);
    if (this.panelH) this.panel.style.height = `${Math.min(this.panelH, innerHeight * 0.82)}px`;
    if (on && !this.collapsed) requestAnimationFrame(() => this.resize());
  }
  setCollapsed(c) {
    this.collapsed = c;
    this.savePanel();
    this.showPanel(this.running);
    if (!c) this.lastOut = null;
  }
  toggleCollapsed() { if (this.running) this.setCollapsed(!this.collapsed); }
  savePanel() { try { localStorage.setItem('lf-panel', JSON.stringify({ collapsed: !!this.collapsed, panelH: this.panelH || 0 })); } catch {} }
  // How far (in NDC) to lift the 3-D scene so it sits in the space the panel leaves free.
  sceneShift() {
    if (!this.running || this.collapsed || this.panel.hidden) return 0;
    const top = this.panel.getBoundingClientRect().top, reserve = 110;  // query bar + answer text
    const cy = (reserve + top) / 2;
    return 1 - (2 * cy) / innerHeight;
  }
  bindPanel() {
    const $ = this.$, grip = $('fwdGrip');
    $('fwdHide').addEventListener('click', () => this.setCollapsed(true));
    $('fwdTab').addEventListener('click', () => this.setCollapsed(false));
    let drag = null;
    grip.addEventListener('pointerdown', (e) => {
      grip.setPointerCapture(e.pointerId);
      drag = { y: e.clientY, h: this.panel.getBoundingClientRect().height };
      this.panel.classList.add('dragging');
    });
    grip.addEventListener('pointermove', (e) => {
      if (!drag) return;
      this.panelH = Math.round(Math.max(180, Math.min(innerHeight * 0.82, drag.h + (drag.y - e.clientY))));
      this.panel.style.height = `${this.panelH}px`;
    });
    const end = () => { if (!drag) return; drag = null; this.panel.classList.remove('dragging'); this.savePanel(); this.resize(); };
    grip.addEventListener('pointerup', end); grip.addEventListener('pointercancel', end);
    grip.addEventListener('dblclick', () => this.setCollapsed(true));
    grip.addEventListener('keydown', (e) => {
      const h = this.panel.getBoundingClientRect().height;
      if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
        e.preventDefault();
        this.panelH = Math.round(Math.max(180, Math.min(innerHeight * 0.82, h + (e.key === 'ArrowUp' ? 32 : -32))));
        this.panel.style.height = `${this.panelH}px`; this.savePanel();
      } else if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); this.setCollapsed(true); }
    });
  }

  // ---------- lifecycle ----------
  enter() {
    this.active = true;
    this.$('fwdFlyRow').hidden = !this.canFly;
    this.$('fwdAttrRow').hidden = !this.canAttribute;
    this.$('fwdWhyRow').hidden = !this.canAttribute;
    this.$('fwdIvRow').hidden = !this.canAttribute;
    this.$('fwdCmpRow').hidden = !this.canAttribute;
    this.$('fwdStatesRow').hidden = !this.canAttribute; this.$('fwdViews').hidden = !this.canAttribute;
    if (!this.canAttribute) this.fview = 'field';
    this.syncButtons();  // toggles show their remembered state before the model is ready
    this.showPanel(this.rows.length > 0);
    document.body.classList.toggle('running', this.rows.length > 0);
    this.resize();
  }
  exit() { this.active = false; this.panel.hidden = true; this.$('fwdTab').hidden = true; document.body.classList.remove('running'); this.playing = false; this.syncButtons(); }
  get running() { return this.rows.length > 0; }

  // Start fetching the weights as soon as someone starts typing.
  prefetch() { if (!this.ready && !this.loading) this.load(); }
  submit(text) {
    text = (text || '').trim();
    if (!text) return;
    if (this.ivRun) { this.worker.postMessage({ type: 'intervene', spec: {} }); this.ivRun = null; }  // a new query starts unchanged
    if (this.ref && this.ref.kind === 'iv') this.setRef(null);  // a pinned run stays, to compare the new text with
    this.queued = text;
    if (this.ready) { this.queued = null; this.run(text); }
    else this.prefetch();
  }
  backendPref() { try { return localStorage.getItem('lf-backend') || 'auto'; } catch { return 'auto'; } }
  setStatus(t) { this.$('fwdStatus').textContent = t; this.$('qStatus').textContent = t; }

  async load() {
    this.loading = true;
    this.setStatus(`Loading ${this.modelName} (${this.modelMB} MB, once)…`);
    try {
      // Models built in the browser keep Hugging Face's tokenizer.json; built-in ones vocab + merges.
      const tj = this.DATA.startsWith('opfs:') ? await dataFetch(this.DATA + 'model/tokenizer.json') : null;
      if (tj && tj.ok) this.tok = new HFTokenizer(await tj.json());
      else {
        const [vocab, merges] = await Promise.all([
          dataFetch(this.DATA + 'model/vocab.json').then((r) => { if (!r.ok) throw new Error('This data set has no model weights.'); return r.json(); }),
          dataFetch(this.DATA + 'model/merges.txt').then((r) => r.text()),
        ]);
        this.tok = new GPT2Tokenizer(vocab, merges);
      }
    } catch (e) { this.setStatus(e.message); this.loading = false; return; }
    this.worker = new Worker('forward-worker.js', { type: 'module' });
    this.worker.onmessage = (e) => this.onWorker(e.data);
    this.worker.postMessage({ type: 'load', data: this.DATA.startsWith('opfs:') ? this.DATA : new URL(this.DATA, location.href).href,
      stream: new URLSearchParams(location.search).has('stream'),  // ?stream forces weight streaming (testing)
      backend: new URLSearchParams(location.search).get('backend') || this.backendPref() });  // 'cpu' keeps it off the GPU
  }

  onWorker(m) {
    if (m.type === 'gain') { this.gains.set(`${m.l}:${m.row}`, m.gain); this.nrnKey = ''; return; }
    if (m.type === 'nvocab') { this.nvocab.set(`${m.l}:${m.i}`, m); this.nrnKey = ''; return; }
    if (m.type === 'lens') { const h = this.hist[m.pos]; if (h) h.lens[m.depth] = m.lens; this.nrnKey = ''; return; }
    if (m.type === 'diff') { const h = this.hist[m.pos]; if (h) h.diff = m.d; return; }
    if (m.type === 'hidden') { this.onHidden(m); return; }
    if (m.type === 'saeProgress') { this.saeMsg = m.text; this.nrnKey = ''; return; }
    if (m.type === 'sae') { this.saeRes.set(`${m.depth}:${m.pos}`, m); this.saeMsg = ''; this.nrnKey = ''; return; }
    if (m.type === 'saeFeat') { this.saeFeat.set(`${m.depth}:${m.j}`, m); this.nrnKey = ''; return; }
    if (m.type === 'attr') {  // one more word's attribution at an earlier position
      const h = this.hist[m.pos]; if (!h || !h.targets || h.targets.includes(m.row)) return;
      const t = new Uint32Array(h.targets.length + 1); t.set(h.targets); t[h.targets.length] = m.row;
      const a = new Float32Array(h.attr.length + m.rec.length); a.set(h.attr); a.set(m.rec, h.attr.length);
      h.targets = t; h.attr = a; this.nrnKey = ''; return;
    }
    if (m.type === 'progress') this.setStatus(`Loading ${this.modelName}… ${(m.p * 100).toFixed(0)}%`);
    else if (m.type === 'ready') {
      Object.assign(this, { L: m.L, H: m.H, F: m.F, maxT: m.maxT, rowOf: m.rowOf, simd: m.simd });
      this.ready = true; this.loading = false;
      this.setStatus('');
      this.streaming = !!m.streaming;
      this.sae = m.sae || null; this.$('fwdFeatRow').hidden = !this.sae;
      this.fidelityNote().then((note) => { if (note && this.ready) this.$('fwdStatus').textContent += m.q8 ? ` · the 16-bit model is ${note}; at 8 bits its predictions move slightly` : ` · ${note}${m.backend === 'gpu' ? '; the GPU matches the CPU kernel to within 0.001' : ''}`; });
      this.backend = m.backend;
      const q8 = m.q8 && m.qerr ? `, 8-bit weights (${(m.qerr.all * 100).toFixed(2)}% from the published ones)` : '';
      const where = m.backend === 'gpu' ? ` · on your GPU${m.gpuName ? ` (${m.gpuName})` : ''}${q8}` : m.gpuNote ? ` · ${m.gpuNote}` : '';
      this.q8 = !!m.q8;
      this.$('fwdStatus').textContent = `${this.modelName} ready${where} · up to ${m.maxT} tokens${m.streaming ? ' · weights streamed from this browser\'s storage, one layer at a time' : ''}`;
      const sel = this.$('fwdBackend');
      this.$('fwdBackendRow').hidden = !m.canGpu;
      sel.value = ['cpu', 'gpu8'].includes(this.backendPref()) ? this.backendPref() : 'auto';
      this.syncButtons();
      if (this.queued) { const q = this.queued; this.queued = null; this.run(q); }
    } else if (m.type === 'captured') {
      this.pending = false;
      this.hist[m.pos] = m;
      this.msEma = this.msEma ? this.msEma * 0.7 + m.ms * 0.3 : m.ms;
      if (m.pos < this.promptLen - 1) this.request(m.pos + 1);          // prompt: no replay
      else { this.beat = m.pos === this.promptLen - 1 && this.speed() === 0 ? this.L + 1 : 0; this.lastBeat = performance.now(); }
      this.renderChips();
    } else if (m.type === 'error') { this.setStatus(m.message); this.pending = false; this.playing = false; this.syncButtons(); }
  }

  request(pos) {
    this.pending = true;
    const b = this.ref, watch = b && pos + 1 < b.rows.length ? b.rows[pos + 1] : undefined;  // A's next word, matched by index
    this.worker.postMessage({ type: 'step', id: pos, row: this.rows[pos], pos, lensMode: 2, next: this.rows[pos + 1], watch });
  }

  // ---------- control ----------
  speed() { return BEATS[this.$('fwdSpeed').value] ?? 160; }
  run(text, count) {
    if (!this.ready) return;
    this.promptText = text;
    const ids = this.tok.encode(text).slice(0, this.maxT - 2);
    this.rows = ids.map((i) => this.rowOf[i]).filter((r) => r >= 0);
    if (!this.rows.length) return;
    this.promptLen = this.rows.length; this.hist = []; this.viewPos = -1; this.asked.clear(); this.pick = null;
    this.saeRes.clear(); this.saeFeat.clear();
    Object.assign(this.hs, { goal: null, shown: null, n: 0, fresh: true, hover: null }); this.flowCache = null; this.runId = (this.runId || 0) + 1;
    this.toGenerate = count ?? +this.$('fwdCount').value;
    this.playing = true; this.beat = 0;
    document.body.classList.add('running');
    this.showPanel(true);
    this.renderChips(); this.syncButtons();
    this.request(0);
    const c = [0, 0, 0];
    for (const r of this.rows) for (let k = 0; k < 3; k++) c[k] += this.points[3 * r + k] / this.rows.length;
    this.flyToPoint(c, 2.2);
  }
  sampleNext(h) {
    const lens = h.lens[this.L];
    if (this.$('fwdSample').value === 'greedy') return lens.rows[0];
    const T = 0.8, w = lens.probs.map((p) => Math.pow(Math.max(p, 1e-9), 1 / T));
    let r = Math.random() * w.reduce((a, b) => a + b, 0);
    for (let j = 0; j < w.length; j++) { r -= w[j]; if (r <= 0) return lens.rows[j]; }
    return lens.rows[0];
  }
  advanceToNextToken() {
    const last = this.hist[this.rows.length - 1];
    const generated = this.rows.length - this.promptLen;
    if (!last || generated >= this.toGenerate || this.rows.length >= this.maxT - 1) { this.playing = false; this.syncButtons(); return; }
    this.rows.push(this.sampleNext(last));
    this.request(this.rows.length - 1);
    this.renderChips();
  }
  tick(now) {
    if (!this.active || !this.ready || this.pending) return;
    const lastPos = this.rows.length - 1;
    if (!this.hist[lastPos]) return;
    const go = this.playing || this.stepOnce;
    if (!go) return;
    const ms = this.speed();
    if (now - this.lastBeat < ms && !this.stepOnce) return;
    this.stepOnce = false; this.lastBeat = now;
    const R = this.L + 1;  // readout beat
    if (this.beat < R) { this.beat = ms === 0 ? R : this.beat + 1; return; }
    this.advanceToNextToken();
  }

  bindUi() {
    const $ = this.$;
    $('fwdPause').addEventListener('click', () => { this.playing = !this.playing; if (this.playing) this.continueIfDone(); if (this.playing && this.beat >= this.L + 1 && !this.pending) this.lastBeat = 0; this.syncButtons(); });
    $('fwdStep').addEventListener('click', () => { this.playing = false; this.continueIfDone(); this.stepOnce = true; this.syncButtons(); });
    $('fwdFly').addEventListener('click', () => this.toggleFly());
    $('fwdChips').addEventListener('click', (e) => {
      const b = e.target.closest('[data-pos]'); if (!b) return;
      const p = +b.dataset.pos; this.viewPos = this.viewPos === p ? -1 : p; this.renderChips();
    });
    const c = this.canvas;
    c.addEventListener('pointermove', (e) => { const r = c.getBoundingClientRect(); this.gaze = { x: (e.clientX - r.left) * this.dpr, y: (e.clientY - r.top) * this.dpr }; });
    c.addEventListener('pointerleave', () => { this.gaze = null; });
    c.addEventListener('click', (e) => {
      const g = this.geom, r = c.getBoundingClientRect();
      if (this.fview !== 'field') {  // river and flow: a click picks the token under the pointer
        const p = this.fview === 'flow' && this.flowGeom ? this.flowCol((e.clientX - r.left) * this.dpr) : -1;
        if (p >= 0 && this.hist[p]) { this.viewPos = p === this.rows.length - 1 ? -1 : p; this.renderChips(); }
        return;
      }
      if (g) {  // which row and column was clicked
        const x = (e.clientX - r.left) * this.dpr, y = (e.clientY - r.top) * this.dpr;
        const depth = g.L - Math.floor((y - g.pad) / g.rowH);
        if (depth >= 1 && depth <= g.L && x >= g.mlpX && x <= g.mlpX + g.mlpW && this.canAttribute) { this.openNeurons(depth - 1); return; }
        if (depth >= 0 && depth <= g.L && x < g.pad + g.labW && this.sae) { this.openFeatures(depth); return; }
        if (depth >= 0 && depth <= g.L && x >= g.lensX && this.canAttribute) { this.pickLens(depth); return; }
      }
      if (e.shiftKey && this.focusHead && this.canAttribute) { this.toggleKO('heads', `${this.focusHead.l}:${this.focusHead.h}`); return; }
      this.pinnedHead = this.focusHead && (!this.pinnedHead || this.pinnedHead.l !== this.focusHead.l || this.pinnedHead.h !== this.focusHead.h) ? { ...this.focusHead } : null;
    });
    $('fwdAttr').addEventListener('click', () => this.toggleAttr());
    $('fwdBackend').addEventListener('change', (e) => {  // a different engine means loading the model again
      try { localStorage.setItem('lf-backend', e.target.value); } catch {}
      location.reload();
    });
    $('fwdStates').addEventListener('click', () => this.toggleStates());
    $('fwdRoll').addEventListener('click', () => { this.rollout = !this.rollout; try { localStorage.setItem('lf-roll', this.rollout ? '1' : '0'); } catch {} this.syncButtons(); });
    $('fwdViews').addEventListener('click', (e) => { const b = e.target.closest('[data-v]'); if (b) this.setView(b.dataset.v); });
    $('fwdIvRun').addEventListener('click', () => this.rerun());
    $('fwdIvClear').addEventListener('click', () => this.clearKO());
    $('fwdPin').addEventListener('click', () => this.togglePin());
    $('fwdWhy').addEventListener('change', () => this.explainWord($('fwdWhy').value));
    $('fwdWhy').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); this.explainWord($('fwdWhy').value); } });
    const nrn = $('nrn');
    nrn.addEventListener('click', (e) => {
      if (e.target.closest('.x')) { this.nrn = null; nrn.hidden = true; return; }
      const fb = e.target.closest('[data-f]');
      if (fb && this.nrn && this.nrn.kind === 'feat') { const j = +fb.dataset.f; this.nrn.sel = this.nrn.sel === j ? -1 : j; this.nrnKey = ''; return; }
      const ko = e.target.closest('[data-ko]');
      if (ko) { this.toggleKO(ko.dataset.ko, ko.dataset.key); return; }
      const b = e.target.closest('[data-n]'); if (!b || !this.nrn) return;
      const i = +b.dataset.n; this.nrn.sel = this.nrn.sel === i ? -1 : i; this.nrnKey = '';
      if (this.nrn.sel >= 0 && !this.nvocab.has(`${this.nrn.l}:${i}`)) this.worker.postMessage({ type: 'nvocab', l: this.nrn.l, i });
    });
    new ResizeObserver(() => this.resize()).observe(c);
  }
  // A finished pass: Play and Step layer carry on writing, `Tokens` more at a time.
  continueIfDone() {
    if (!this.rows.length || this.pending || this.beat < this.L + 1) return;
    const generated = this.rows.length - this.promptLen;
    if (generated >= this.toGenerate) this.toGenerate = generated + Math.max(1, +this.$('fwdCount').value || 1);
  }
  toggleAttr() {
    if (!this.canAttribute) return;
    this.attrOn = !this.attrOn;
    try { localStorage.setItem('lf-attr', this.attrOn ? '1' : '0'); } catch {}
    this.nrnKey = ''; this.syncButtons();
  }
  setView(v) {
    if (!this.canAttribute || !['field', 'river', 'flow'].includes(v)) return;
    this.fview = v;
    try { localStorage.setItem('lf-fview', v); } catch {}
    this.pinnedHead = null; this.syncButtons();
  }
  // ---------- the hidden-state space ----------
  toggleStates() {
    if (!this.canAttribute) return;
    this.hs.on = !this.hs.on;
    if (this.hs.on) { this.hs.fresh = true; this.askHidden(true); this.flyToPoint([0, 0, 0], 2.1); }
    this.syncButtons();
  }
  // How many positions have been captured, from the first on.
  captured() { let n = 0; while (n < this.rows.length && this.hist[n]) n++; return n; }
  // Ask the worker for the projection when there are new states: at once when a run is done, at
  // most every 1.5 s while one plays (each refit runs in the worker, a few hundred ms at most).
  askHidden(now = false) {
    const hs = this.hs, n = this.captured();
    if (!hs.on || !this.worker || hs.busy || !n || n === hs.n) return;
    const t = performance.now();
    if (!now && (this.playing || this.pending) && t - hs.t < 1500) return;
    hs.busy = true; hs.t = t; hs.id++;
    this.worker.postMessage({ type: 'hidden', id: hs.id, n, fresh: !!hs.fresh });
    hs.fresh = false;
  }
  // Place the states in the scene: centred, scaled so nearly all of them fit where the word cloud was.
  onHidden(m) {
    const hs = this.hs; hs.busy = false;
    if (m.id !== hs.id) return;
    const xyz = m.xyz, N = xyz.length / 3, c = [0, 0, 0];
    for (let j = 0; j < N; j++) for (let k = 0; k < 3; k++) c[k] += xyz[3 * j + k] / N;
    const rad = new Float32Array(N);
    for (let j = 0; j < N; j++) rad[j] = Math.hypot(xyz[3 * j] - c[0], xyz[3 * j + 1] - c[1], xyz[3 * j + 2] - c[2]);
    const rTop = Float32Array.from(rad).sort()[Math.min(N - 1, Math.floor(N * 0.995))] || 1, k = 0.9 / rTop;  // nearly every state in view
    const goal = new Float32Array(N * 3);
    for (let j = 0; j < N; j++) for (let a = 0; a < 3; a++) goal[3 * j + a] = (xyz[3 * j + a] - c[a]) * k;
    const old = hs.shown, shown = new Float32Array(N * 3);
    if (old) shown.set(old.subarray(0, Math.min(old.length, shown.length)));
    for (let j = old ? old.length / 3 : 0; j < N; j++) shown.set(goal.subarray(3 * j, 3 * j + 3), 3 * j);  // new states appear in place
    Object.assign(hs, { goal, shown, n: m.n, share: m.share, ms: m.ms });
  }
  cloudDim() { return this.hs.on && this.active ? 0.18 : 1; }
  // A click in the scene: put the token whose state is under the pointer on screen.
  sceneClick() {
    const t = this.hs.on && this.hs.hover; if (!t) return false;
    this.viewPos = t.p === this.rows.length - 1 ? -1 : t.p; this.renderChips();
    return true;
  }
  drawStates(ov) {
    const hs = this.hs, v = this.view(); if (!hs.shown || !v.h) return false;
    const d = this.dpr, L = this.L, D = L + 1, n = Math.min(hs.n, this.captured()), { pos, beat } = v;
    const cur = Math.min(beat, L), ease = 0.12;
    for (let j = 0; j < n * D * 3; j++) hs.shown[j] += (hs.goal[j] - hs.shown[j]) * ease;
    const S = hs.shown, scr = new Array(n * D);
    for (let j = 0; j < n * D; j++) scr[j] = this.project(S[3 * j], S[3 * j + 1], S[3 * j + 2]);
    const live = (p) => p === this.rows.length - 1 && this.viewPos < 0;  // the token being computed: drawn up to its replayed depth
    // Every other token's path, faint, in two batched strokes: attention's part, then the MLP's.
    const split = (p, l) => { const w = this.hist[p].wr; if (!w || Number.isNaN(w[2 * l + 1])) return 1; const a = w[2 * l], b = w[2 * l + 1]; return a + b > 0 ? a / (a + b) : 0.5; };
    const faint = n > 60 ? 0.16 : 0.3;
    for (const [col, part] of [['53,224,230', 0], ['169,150,255', 1]]) {
      ov.strokeStyle = `rgba(${col},${faint})`; ov.lineWidth = 1 * d; ov.beginPath();
      for (let p = 0; p < n; p++) {
        if (p === pos) continue;
        for (let l = 0; l < L; l++) {
          const a = scr[p * D + l], b = scr[p * D + l + 1]; if (!a || !b) continue;
          const f = split(p, l), m = [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
          if (part === 0) { ov.moveTo(a[0], a[1]); ov.lineTo(m[0], m[1]); } else { ov.moveTo(m[0], m[1]); ov.lineTo(b[0], b[1]); }
        }
      }
      ov.stroke();
    }
    // The slice at the depth on screen: every token's state there.
    ov.fillStyle = 'rgba(232,230,245,0.7)';
    for (let p = 0; p < n; p++) { const s = scr[p * D + cur]; if (s && p !== pos) ov.fillRect(s[0] - 1.5 * d, s[1] - 1.5 * d, 3 * d, 3 * d); }
    // The token on screen: its whole path, each layer's step as thick as the change it made.
    const h = this.hist[pos], top = live(pos) ? cur : L;
    // A layer's change: how far it moved the state, compared with the layer that moved it most.
    const chg = (l) => (h.wr ? h.wr[2 * l] + (Number.isNaN(h.wr[2 * l + 1]) ? 0 : h.wr[2 * l + 1]) : 0);
    let cmax = 1e-6; for (let l = 0; l < L; l++) cmax = Math.max(cmax, chg(l));
    for (let l = 0; l < top; l++) {
      const a = scr[pos * D + l], b = scr[pos * D + l + 1]; if (!a || !b) continue;
      const ch = h.wr ? Math.sqrt(chg(l) / cmax) : 0.3;
      const f = split(pos, l), m = [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
      ov.lineWidth = (1.2 + 5 * Math.min(1, ch)) * d; ov.lineCap = 'round';
      ov.strokeStyle = '#35e0e6'; ov.beginPath(); ov.moveTo(a[0], a[1]); ov.lineTo(m[0], m[1]); ov.stroke();
      ov.strokeStyle = '#a996ff'; ov.beginPath(); ov.moveTo(m[0], m[1]); ov.lineTo(b[0], b[1]); ov.stroke();
    }
    ov.lineCap = 'butt';
    for (let dp = 0; dp <= top; dp++) {
      const s = scr[pos * D + dp]; if (!s) continue;
      const t = dp / L;
      ov.fillStyle = dp === cur ? '#fff' : `rgb(${Math.round(169 - 116 * t)},${Math.round(150 + 74 * t)},${Math.round(255 - 25 * t)})`;
      ov.beginPath(); ov.arc(s[0], s[1], (dp === cur ? 5 : 2.6) * d, 0, Math.PI * 2); ov.fill();
    }
    // Labels: the token at its start, and what its state would say at the depth on screen.
    ov.font = `${10.5 * d}px "JetBrains Mono", monospace`; ov.textBaseline = 'middle';
    const word = (r) => this.disp(this.tokens[r]);
    const s0 = scr[pos * D], sc = scr[pos * D + cur];
    if (s0) { ov.fillStyle = '#e8e6f5'; ov.fillText(`${word(this.rows[pos])} · embedding`, s0[0] + 8 * d, s0[1]); }
    const lz = h.lens[cur];
    if (sc && cur > 0) { ov.fillStyle = '#fff'; ov.fillText(`layer ${cur - 1}${lz ? ` → ${word(lz.rows[0])} ${(lz.probs[0] * 100).toFixed(0)}%` : ''}`, sc[0] + 9 * d, sc[1] - 10 * d); }
    if (n <= 32) {  // short texts: name every token at the slice
      ov.fillStyle = 'rgba(232,230,245,0.55)'; ov.font = `${9.5 * d}px "JetBrains Mono", monospace`;
      for (let p = 0; p < n; p++) { const s = scr[p * D + cur]; if (s && p !== pos) ov.fillText(word(this.rows[p]), s[0] + 5 * d, s[1] + 7 * d); }
    }
    // Hover: the state nearest the pointer.
    const g = this.pointer && this.pointer(); let best = null, bd = 12 * d;
    if (g) for (let j = 0; j < n * D; j++) { const s = scr[j]; if (!s) continue; const e = Math.hypot(s[0] - g.x, s[1] - g.y); if (e < bd) { bd = e; best = j; } }
    hs.hover = best == null ? null : { p: Math.floor(best / D), dp: best % D };
    if (hs.hover) {
      const { p, dp } = hs.hover, s = scr[best], hp = this.hist[p], lp = hp.lens[dp];
      if (!lp && !this.asked.has(`lens:${p}:${dp}`)) { this.asked.add(`lens:${p}:${dp}`); this.worker.postMessage({ type: 'lens', pos: p, depth: dp }); }
      const txt = `${word(this.rows[p])} · token ${p + 1} · ${dp === 0 ? 'embedding' : `after layer ${dp - 1}`}${lp ? ` · would say ${word(lp.rows[0])} ${(lp.probs[0] * 100).toFixed(0)}%` : ''}`;
      ov.font = `${10.5 * d}px "JetBrains Mono", monospace`;
      const w = ov.measureText(txt).width + 14 * d, x = Math.min(s[0] + 10 * d, this.FWscene - w - 4 * d), y = s[1] + 16 * d;
      ov.fillStyle = 'rgba(14,14,24,0.92)'; ov.fillRect(x, y - 9 * d, w, 18 * d);
      ov.strokeStyle = 'rgba(53,224,230,0.5)'; ov.lineWidth = 1 * d; ov.strokeRect(x, y - 9 * d, w, 18 * d);
      ov.fillStyle = '#e8e6f5'; ov.fillText(txt, x + 7 * d, y);
      ov.strokeStyle = '#fff'; ov.beginPath(); ov.arc(s[0], s[1], 6 * d, 0, Math.PI * 2); ov.stroke();
    }
    return true;
  }
  toggleFly() {
    if (!this.canFly) return;
    this.flying = !this.flying;
    try { localStorage.setItem('lf-fly', this.flying ? '1' : '0'); } catch {}
    this.syncButtons();
  }
  syncButtons() {
    const $ = this.$;
    $('fwdFly').setAttribute('aria-pressed', String(this.flying && this.canFly));
    $('fwdAttr').setAttribute('aria-pressed', String(this.attrOn && this.canAttribute));
    $('fwdStates').setAttribute('aria-pressed', String(this.hs.on && this.canAttribute));
    $('fwdRoll').hidden = this.fview !== 'flow'; $('fwdRoll').setAttribute('aria-pressed', String(this.rollout));
    for (const b of $('fwdViews').querySelectorAll('[data-v]')) b.setAttribute('aria-selected', String(b.dataset.v === this.fview));
    $('fwdPause').textContent = this.playing ? 'Pause' : 'Play';
    for (const id of ['fwdPause', 'fwdStep']) $(id).disabled = !this.ready || !this.rows.length;
    if (this.canAttribute) { this.syncIv(); this.syncRef(); }
  }
  resize() {
    const r = this.canvas.getBoundingClientRect();
    this.FW = Math.max(100, Math.round(r.width * this.dpr)); this.FH = Math.max(100, Math.round(r.height * this.dpr));
    this.canvas.width = this.FW; this.canvas.height = this.FH;
  }

  // The thought path's point for one logit-lens readout: the probability-weighted centroid of its guesses.
  lensCentroid(lz) {
    const P = this.points, c = [0, 0, 0]; let s = 0;
    lz.rows.forEach((r, j) => { const w = lz.probs[j]; s += w; for (let k = 0; k < 3; k++) c[k] += P[3 * r + k] * w; });
    return s > 0 ? c.map((v) => v / s) : null;
  }
  // The deepest readout on screen so far (big vocabularies skip some depths).
  lensAt(h, beat) {
    for (let dp = Math.min(beat, this.L); dp >= 0; dp--) if (h.lens[dp]) return h.lens[dp];
    return null;
  }
  // Fly-through: where the camera is heading. The thought path's point at the depth on screen,
  // so for each word the camera travels from the token toward the word the model settles on.
  flyGoal() {
    if (!this.flying || !this.canFly || !this.active || !this.rows.length) return null;
    const { h, pos, beat } = this.view(); if (!h) return null;
    if (this.hs.on) {  // in the hidden-state space, follow the token's own state
      const S = this.hs.shown, j = pos * (this.L + 1) + Math.min(beat, this.L);
      return S && pos < this.hs.n ? [S[3 * j], S[3 * j + 1], S[3 * j + 2]] : null;
    }
    const lz = this.lensAt(h, beat), c = lz && this.lensCentroid(lz);
    if (c) return c;
    const P = this.points, r = this.rows[pos];
    return [P[3 * r], P[3 * r + 1], P[3 * r + 2]];
  }

  // Which position is on screen, and how far its replay has got (0 = embedding .. L = last layer, L + 1 = readout).
  view() {
    const latest = this.rows.length - 1;
    let pos = this.viewPos >= 0 ? this.viewPos : latest;
    while (pos >= 0 && !this.hist[pos]) pos--;
    const beat = this.viewPos >= 0 || pos < latest || pos < this.promptLen - 1 ? this.L + 1 : Math.min(this.beat, this.L + 1);
    return { pos, h: this.hist[pos], beat };
  }

  renderChips() {
    const { pos } = this.view();
    const html = this.rows.map((r, p) => {
      const differs = this.ref && (p >= this.ref.rows.length || this.ref.rows[p] !== r);  // not A's token at this position
      const cls = ['chip', p < this.promptLen ? 'prompt' : 'gen', p === pos ? 'on' : '', !this.hist[p] ? 'wait' : '', differs ? 'diff' : ''].join(' ');
      return `<button type="button" class="${cls}" data-pos="${p}">${esc(this.disp(this.tokens[r]))}</button>`;
    }).join('');
    this.$('fwdChips').innerHTML = html;
    const el = this.$('fwdChips'); el.scrollLeft = el.scrollWidth;
  }

  // The answer as it forms: prompt dim, generated text bright, the next guess while it is being read out.
  renderOutput() {
    if (!this.rows.length) return;
    const t = (r) => esc(this.tokens[r]);
    const p = this.rows.slice(0, this.promptLen).map(t).join('');
    const b = this.ref;
    let dv = -1;  // the first generated token that differs from the reference run
    if (b) for (let p = this.promptLen; p < this.rows.length; p++) if (this.rows[p] !== b.rows[p]) { dv = p; break; }
    const sameText = b && b.promptLen === this.promptLen && b.rows.slice(0, b.promptLen).every((r, j) => r === this.rows[j]);
    const g = this.rows.slice(this.promptLen).map((r, j) => (this.promptLen + j === dv ? `<span class="dv">${t(r)}</span>` : t(r))).join('');
    const last = this.hist[this.rows.length - 1];
    const forming = this.playing && last && this.beat >= 1 && last.lens[this.L]
      ? `<span class="now">${t((last.lens[Math.min(this.beat, this.L)] || last.lens[this.L]).rows[0])}</span>` : '';
    const baseLine = !b ? '' : `<span class="base"><b>${b.label}${sameText && dv < 0 && !this.playing ? ', same answer' : ''}:</b> ${sameText ? '' : `<span class="p">${b.rows.slice(0, b.promptLen).map(t).join('')}</span>`}${b.rows.slice(b.promptLen).map(t).join('')}${b.rows.length && b.promptLen !== this.promptLen ? ' <i>(a different number of tokens: positions are matched by index)</i>' : ''}</span>`;
    const html = `<span class="p">${p}</span><span class="g">${g}</span>${forming}${this.playing ? '<span class="caret"></span>' : ''}${baseLine}`;
    if (html !== this.lastOut) { this.lastOut = html; const el = this.$('qOut'); el.innerHTML = html; el.scrollTop = el.scrollHeight; }
  }

  // How closely this model's numbers were checked against the reference implementation.
  async fidelityNote() {
    if (!this.repo) return this.family === 'masked-diffusion' ? 'matches its reference implementation to about 0.00001' : 'matches its reference implementation to 4 decimal places';
    try {
      const v = (await (await fetch('verified.json')).json()).models || {}, r = v[this.repo];
      return r ? `checked against transformers: probabilities within ${r.prob}` : 'a supported design, not checked against transformers one by one';
    } catch { return ''; }
  }

  // ---------- comparing runs ----------
  // Keep the run on screen as the reference ('pin': "Run A"; 'iv': the unchanged model), or drop it.
  setRef(kind) {
    if (!kind) { this.ref = null; this.worker.postMessage({ type: 'pinA', clear: true }); this.syncRef(); return; }
    const L = this.L;
    const p0 = this.rows.map((r, p) => { const lz = this.hist[p] && this.hist[p].lens[L], nx = this.rows[p + 1]; if (!lz || nx == null) return null; const j = lz.rows.indexOf(nx); return j >= 0 ? lz.probs[j] : null; });
    this.ref = { kind, label: kind === 'pin' ? 'Run A' : 'Unchanged model', rows: this.rows.slice(), promptLen: this.promptLen, gen: this.rows.length - this.promptLen, hist: this.hist.slice(), p0 };
    this.worker.postMessage({ type: 'pinA' });
    this.syncRef();
  }
  togglePin() { if (this.ref && this.ref.kind === 'pin') this.setRef(null); else if (this.rows.length && !this.playing) this.setRef('pin'); }
  syncRef() {
    const pinned = this.ref && this.ref.kind === 'pin';
    this.$('fwdPin').textContent = pinned ? 'Unpin A' : 'Pin as A';
    this.$('fwdPin').disabled = !pinned && (!this.rows.length || this.playing || !this.ready);
    const n = this.$('fwdPinNote');
    n.textContent = this.ref ? `A: ${this.ref.rows.map((r) => this.tokens[r]).join('').slice(0, 60)}` : '';
    n.title = this.ref ? n.textContent : '';
    n.classList.toggle('on', !!this.ref);
    this.lastOut = null; this.renderChips();
  }

  // ---------- interventions ----------
  toggleKO(kind, key) {
    const s = this.ko[kind]; if (s.has(key)) s.delete(key); else s.add(key);
    this.ivVer++; this.nrnKey = ''; this.syncIv();
  }
  koAtGaze() { const f = this.pinnedHead || this.focusHead; if (f && this.canAttribute) this.toggleKO('heads', `${f.l}:${f.h}`); }
  ivSpec() {
    const sp = (k) => [...this.ko[k]].map((x) => x.split(':').map(Number));
    return { heads: sp('heads').map(([l, h]) => [l, h, 0]), mlps: sp('mlps').map(([l]) => [l, 0]), neurons: sp('neurons').map(([l, i]) => [l, i, 0]) };
  }
  koCount() { return this.ko.heads.size + this.ko.mlps.size + this.ko.neurons.size; }
  syncIv() {
    const n = this.koCount(), applied = this.ivRun ? JSON.stringify(this.ivRun) === JSON.stringify(this.ivSpec()) : n === 0;
    const el = this.$('fwdIvCount');
    el.textContent = n ? `${n} knocked out${applied ? '' : ' · rerun'}` : 'No changes';
    el.classList.toggle('on', n > 0);
    this.$('fwdIvRun').disabled = !this.ready || !this.promptText || (applied && n > 0) || (n === 0 && !this.ivRun);
    this.$('fwdIvClear').disabled = n === 0 && !this.ivRun;
  }
  // Rerun the text through the changed model, keeping the unchanged run to compare with.
  rerun() {
    if (!this.promptText || !this.ready) return;
    if (!this.ivRun) this.setRef('iv');  // the run on screen is unchanged: it becomes the reference
    const spec = this.ivSpec(), any = this.koCount() > 0, gen = this.ref ? this.ref.gen : undefined;
    this.worker.postMessage({ type: 'intervene', spec });
    this.ivRun = any ? spec : null;
    if (!any && this.ref && this.ref.kind === 'iv') this.setRef(null);
    this.run(this.promptText, gen);
    this.syncIv();
  }
  clearKO() {
    for (const s of Object.values(this.ko)) s.clear();
    this.ivVer++; this.nrnKey = '';
    if (this.ivRun) this.rerun(); else this.syncIv();
  }

  // ---------- attribution ----------
  // The word to explain at position pos: a picked logit-lens guess, else the word written next,
  // else the model's final top guess.
  attrTarget(h, pos) {
    if (this.pick && this.pick.pos === pos) return this.pick.row;  // attribution may still be on its way
    const next = this.rows[pos + 1];
    if (next != null && h.targets && h.targets.includes(next)) return next;
    return h.lens[this.L] ? h.lens[this.L].rows[0] : null;
  }
  // { row, heads, mlps, emb, rest, logit, inv } for word `row` at a captured position, or null.
  attrFor(h, row) {
    if (!h.targets || row == null) return null;
    const j = h.targets.indexOf(row); if (j < 0) return null;
    const L = this.L, NH = this.H, per = L * NH + L + 4, a = h.attr.subarray(j * per, (j + 1) * per);
    return { row, heads: a.subarray(0, L * NH), mlps: a.subarray(L * NH, L * NH + L), emb: a[L * NH + L], rest: a[L * NH + L + 1], logit: a[L * NH + L + 2], inv: a[L * NH + L + 3] };
  }
  // Foveated readouts: detail is computed only where someone looks. The fovea is the depth row
  // under the pointer, else the depth being replayed; a missing logit lens there, or a missing
  // attribution for the word being explained, is asked of the worker once.
  foveate(v, geom) {
    if (!this.canAttribute || !v.h || !this.worker) return;
    const { h, pos, beat } = v, L = this.L;
    let depth = Math.min(beat, L);
    if (this.gaze && geom) {
      const d = L - Math.floor((this.gaze.y - geom.pad) / geom.rowH);
      if (d >= 0 && d <= L) depth = d;
    }
    const k = `lens:${pos}:${depth}`;
    if (!h.lens[depth] && !this.asked.has(k)) { this.asked.add(k); this.worker.postMessage({ type: 'lens', pos, depth }); }
    if (this.attrOn) this.ensureAttr(h, pos, this.attrTarget(h, pos));
  }
  ensureAttr(h, pos, row) {
    if (row == null || !h.targets || h.targets.includes(row)) return;
    const k = `attr:${pos}:${row}`;
    if (!this.asked.has(k)) { this.asked.add(k); this.worker.postMessage({ type: 'attr', pos, row }); }
  }
  // "Why not Paris?": explain any word at the token on screen. A word typed without a leading
  // space gets one, as it would appear mid-sentence.
  explainWord(text) {
    const { h, pos } = this.view();
    if (!text.trim() || !h || !this.tok) { this.pick = null; this.nrnKey = ''; return; }
    const ids = this.tok.encode(/^\s/.test(text) || pos < 0 ? text : ' ' + text);
    const row = ids.length ? this.rowOf[ids[0]] : -1;
    if (row == null || row < 0) return;
    this.pick = { pos, row };
    if (!this.attrOn) this.toggleAttr();
    this.ensureAttr(h, pos, row); this.nrnKey = '';
  }
  pickLens(depth) {
    const { h, pos } = this.view(); const lz = h && h.lens[depth]; if (!lz) return;
    const row = lz.rows[0];
    this.pick = this.pick && this.pick.pos === pos && this.pick.row === row ? null : { pos, row };
    if (!this.attrOn) this.toggleAttr();
    this.nrnKey = '';
  }

  // ---------- features (SAE) ----------
  // depth: 0 = the embedding .. L = after the last layer; the SAE reads the state at that depth.
  openFeatures(depth) {
    this.nrn = this.nrn && this.nrn.kind === 'feat' && this.nrn.depth === depth ? null : { kind: 'feat', depth, sel: -1 };
    this.$('nrn').hidden = !this.nrn; this.nrnKey = '';
  }
  // Neuronpedia's description of a feature, fetched once (only the layer and number are sent).
  npDescribe(depth, j) {
    const src = SAES[this.sae].neuronpedia(depth, j), key = `${src.source}:${j}`;
    if (this.np.has(key)) return this.np.get(key);
    this.np.set(key, null);
    fetch(`https://www.neuronpedia.org/api/feature/${src.model}/${src.source}/${j}`).then((r) => (r.ok ? r.json() : null))
      .then((f) => { this.np.set(key, (f && f.explanations && f.explanations[0] && f.explanations[0].description) || ''); this.npVer++; this.nrnKey = ''; })
      .catch(() => { this.np.set(key, ''); this.npVer++; this.nrnKey = ''; });
    return null;
  }
  renderFeatures(v) {
    const el = this.$('nrn'), { depth, sel } = this.nrn, { pos } = v, L = this.L, spec = SAES[this.sae];
    if (!v.h || !spec) { el.hidden = true; return; }
    const r = this.saeRes.get(`${depth}:${pos}`), f = sel >= 0 ? this.saeFeat.get(`${depth}:${sel}`) : null;
    if (!r && !this.asked.has(`sae:${depth}:${pos}`)) { this.asked.add(`sae:${depth}:${pos}`); this.worker.postMessage({ type: 'sae', depth, pos }); }
    if (sel >= 0 && !f && !this.asked.has(`saeF:${depth}:${sel}`)) { this.asked.add(`saeF:${depth}:${sel}`); this.worker.postMessage({ type: 'saeFeat', depth, j: sel, n: this.rows.length }); }
    const key = `feat|${depth}|${pos}|${sel}|${!!r}|${!!f}|${this.saeMsg}|${this.npVer}|${this.hist.length}`;
    if (key === this.nrnKey) return;
    this.nrnKey = key;
    const word = (row) => esc(this.disp(this.tokens[row]));
    const info = (html) => `<span class="info" tabindex="0" role="note">i<span class="tip left">${html}</span></span>`;
    const where = depth === 0 ? 'The embedding' : `After layer ${depth - 1}`;  // as the field's row labels say
    let html = `<header><b>${where} · features</b>${info(`<b>Features, not neurons.</b> A sparse autoencoder (SAE) is a second, small network, trained to rewrite the model's state at one layer as a sum of a few <b>features</b> from a learned dictionary of ${spec.n.toLocaleString()} directions. A single neuron responds to many unrelated things; a feature tends to mean one thing, so it is easier to name. This dictionary was trained by ${spec.by} (${spec.license}). The descriptions come from Neuronpedia, written by an AI model from each feature's strongest examples: good guesses, not ground truth. Derived: the SAE is a model of the model, and the line below says how much of this token's state it explains.`)}<button type="button" class="x" aria-label="Close">✕</button></header>`;
    html += `<p class="note">At “${word(this.rows[pos])}” (token ${pos + 1}) · ${spec.n.toLocaleString()} features, streamed once from Hugging Face</p>`;
    if (!r) html += `<p class="note">${esc(this.saeMsg || 'Working out the features…')}</p>`;
    else {
      html += `<div class="eff">${r.l0} of ${spec.n.toLocaleString()} features are active. Together they rebuild this token's state to a cosine of <b>${r.cos.toFixed(3)}</b>, missing <b>${(r.relErr * 100).toFixed(0)}%</b> of its size. ${info('How faithful the features are here. A cosine of 1 means the rebuilt state points exactly the same way as the real one; the miss is the size of what the features leave out, relative to the state. Whatever they miss is invisible in this view.')}</div>`;
      if (pos === 0) html += `<p class="note warn">GPT-2's first token is unusual: this SAE was trained with a start-of-text token in front, which this tool doesn't add, so read the first token's features with care.</p>`;
      const amax = r.top.length ? r.top[0][1] : 1;
      html += `<div class="sec">Most active here ${info('The features firing hardest at this token, with what Neuronpedia says each one responds to. Pick one to see where it fires across the text and what it writes.')}</div><ol>`;
      for (const [j, a] of r.top) {
        const dsc = this.npDescribe(depth, j);
        html += `<li><button type="button" data-f="${j}" aria-current="${j === sel}"><span>#${j}</span><span class="bar"><i style="left:0;width:${Math.min(100, (a / amax) * 100)}%;background:#35e0e6"></i></span><span class="num">${a.toFixed(2)}</span></button><div class="fx">${dsc == null ? '…' : dsc ? esc(dsc) : '<i>no description</i>'}</div></li>`;
      }
      html += '</ol>';
    }
    if (sel >= 0) {
      const np = spec.neuronpedia(depth, sel);
      html += `<div class="sec">Feature #${sel} · where it fires ${info('This feature\'s activation at every token of the text so far: the brighter, the stronger. The outlined token is the one on screen.')}</div>`;
      if (!f) html += '<p class="note">Reading the text…</p>';
      else {
        let fmax = 1e-6; for (const x of f.acts) if (x > fmax) fmax = x;
        html += `<div class="text">${this.rows.map((row, p) => { const x = f.acts[p], t = Number.isNaN(x) ? 0 : Math.min(1, x / fmax); return `<span class="${p === pos ? 'here' : ''}" style="background:rgba(53,224,230,${(t * 0.75).toFixed(2)})" title="${Number.isNaN(x) ? 'not run yet' : x.toFixed(3)}">${esc(this.tokens[row])}</span>`; }).join('')}</div>`;
        html += `<div class="sec">What it writes ${info('The words this feature\'s direction pushes <span class="c">up</span> and <span class="a">down</span> most when it fires, through the final normalisation and output layer; what later layers do with it is not counted. Derived.')}</div>`;
        if (f.up) html += `<div class="words up">${f.up.map((w) => `<span>${word(w.row)}</span>`).join('')}</div><div class="words down" style="margin-top:6px">${f.down.map((w) => `<span>${word(w.row)}</span>`).join('')}</div>`;
      }
      html += `<p style="margin-top:10px"><a class="np" href="https://www.neuronpedia.org/${np.model}/${np.source}/${sel}" target="_blank" rel="noopener">Open feature #${sel} on Neuronpedia ↗</a></p>`;
    } else if (r) html += '<p class="note" style="margin-top:12px">Pick a feature to see where it fires across the text and what it writes.</p>';
    const scroll = el.scrollTop;
    el.innerHTML = html; el.hidden = false; el.scrollTop = scroll;
  }

  // ---------- neuron inspector ----------
  openNeurons(l) {
    this.nrn = this.nrn && this.nrn.l === l ? null : { l, sel: -1 };
    this.$('nrn').hidden = !this.nrn; this.nrnKey = '';
  }
  // Rebuild the drawer when what it shows changes (position, word, layer, answers from the worker).
  renderNeurons(v) {
    if (this.nrn && this.nrn.kind === 'feat') { this.renderFeatures(v); return; }
    const el = this.$('nrn');
    if (!this.nrn || !v.h || this.hist.length === 0) { if (!el.hidden) el.hidden = true; return; }
    const { l, sel } = this.nrn, { h, pos } = v, F = this.F;
    const target = this.attrTarget(h, pos), a = this.attrFor(h, target);
    const gain = target != null ? this.gains.get(`${l}:${target}`) : null;
    if (target != null && !gain && !this.gainAsked?.has(`${l}:${target}`)) {
      (this.gainAsked ||= new Set()).add(`${l}:${target}`);
      this.worker.postMessage({ type: 'gain', l, row: target });
    }
    const key = `${l}|${sel}|${pos}|${target}|${!!gain}|${this.nvocab.has(`${l}:${sel}`)}|${this.hist.length}|${this.ivVer}`;
    if (key === this.nrnKey) return;
    this.nrnKey = key;
    const act = h.mlp.subarray(l * F, (l + 1) * F), word = (r) => esc(this.disp(this.tokens[r]));
    const tw = target != null ? `“${word(target)}”` : 'the next word';
    const gelu = /^(gpt2|gpt_neox)$/.test(this.arch);
    const info = (html, left = true) => `<span class="info" tabindex="0" role="note">i<span class="tip${left ? ' left' : ''}">${html}</span></span>`;
    const bar = (x, m, pos) => { const w = Math.min(50, Math.abs(x) / (m || 1) * 50); return `<span class="bar"><i style="${x >= 0 ? `left:50%;width:${w}%;background:${pos}` : `right:50%;width:${w}%;background:#a996ff`}"></i></span>`; };
    const effOf = (i) => (gain && a ? act[i] * gain[i] * a.inv : null);
    const fmt = (x) => (x == null ? '' : `${x >= 0 ? '+' : ''}${x.toFixed(2)}`);
    const koN = (i) => this.ko.neurons.has(`${l}:${i}`);
    const row = (i, m) => { const e = effOf(i); return `<li><button type="button" data-n="${i}" aria-current="${i === sel}"><span>${koN(i) ? '✕' : '#'}${i}</span>${bar(act[i], m, '#35e0e6')}<span class="num ${e == null ? '' : e >= 0 ? 'pos' : 'neg'}">${e == null ? act[i].toFixed(2) : fmt(e)}</span></button></li>`; };
    let active = 0, amax = 1e-6; for (let i = 0; i < F; i++) { const x = Math.abs(act[i]); if (x > 1) active++; if (x > amax) amax = x; }
    const byAct = Array.from({ length: F }, (_, i) => i).sort((p, q) => Math.abs(act[q]) - Math.abs(act[p])).slice(0, 8);
    let html = `<header><b>Layer ${l} · MLP</b>${info(`<b>The MLP is the part of each layer that works on one token at a time.</b> This layer has ${F.toLocaleString()} neurons. Each one reads the model's running state and fires by some amount: its <b>activation</b>, <span class="c">cyan</span> when positive, <span class="v">violet</span> when negative. ${gelu ? 'Here a neuron\'s activation is one projection passed through GELU.' : 'Here a neuron\'s activation is a gated product (SwiGLU): one projection passed through SiLU, times another.'} Whatever a neuron fires, it writes back along its own fixed direction, its <b>output weights</b>. Real: every number here comes straight from the model.`)}<button type="button" class="x" aria-label="Close the neuron inspector">✕</button></header>`;
    html += `<div class="fwd-row" style="margin:4px 0 6px"><button type="button" data-ko="mlps" data-key="${l}" aria-pressed="${this.ko.mlps.has(`${l}`)}">${this.ko.mlps.has(`${l}`) ? 'Restore' : 'Knock out'} this MLP</button>${this.ivRun || this.koCount() ? '<span class="note">then Rerun, under Playback</span>' : ''}</div>`;
    html += `<p class="note">At “${word(this.rows[pos])}” (token ${pos + 1}) · ${active.toLocaleString()} of ${F.toLocaleString()} neurons above 1.0</p>`;
    if (a) html += `<div class="eff">This layer's MLP pushed ${tw} by <b class="${a.mlps[l] >= 0 ? 'pos' : 'neg'}">${fmt(a.mlps[l])}</b>, out of a score of ${a.logit.toFixed(2)} above the average word.</div>`;
    html += `<div class="sec">Most active here ${info('The neurons firing hardest at this token, either way. Firing hard is not the same as mattering: a neuron can fire strongly and write almost nothing toward the word. The number on the right is its effect on the word when that is known, else its activation.')}</div><ol>${byAct.map((i) => row(i, amax)).join('')}</ol>`;
    if (gain && a) {
      const eff = Array.from({ length: F }, (_, i) => [effOf(i), i]).sort((p, q) => q[0] - p[0]);
      const emax = Math.max(1e-6, Math.abs(eff[0][0]), Math.abs(eff[F - 1][0]));
      const effRow = ([e, i]) => `<li><button type="button" data-n="${i}" aria-current="${i === sel}"><span>${koN(i) ? '✕' : '#'}${i}</span>${bar(e, emax, '#35e0e6')}<span class="num ${e >= 0 ? 'pos' : 'neg'}">${fmt(e)}</span></button></li>`;
      html += `<div class="sec">Toward ${tw} ${info(`Each neuron's <b>direct effect</b> on the word's score: its activation times how much its output weights point at that word, through the final normalisation. Added up over all ${F.toLocaleString()} neurons, plus the layer's bias, this is the layer's share above. Direct path only: what later layers make of it is counted in those layers.`)}</div><ol>${eff.slice(0, 6).map(effRow).join('')}</ol>`;
      html += `<div class="sec">Against ${tw}</div><ol>${eff.slice(-4).reverse().map(effRow).join('')}</ol>`;
    } else if (target != null) html += `<p class="note">Working out each neuron's effect on ${tw}…</p>`;
    if (sel >= 0) {
      const vals = []; let vmax = 1e-6;
      for (let p = 0; p < this.rows.length; p++) { const hp = this.hist[p]; const x = hp ? hp.mlp[l * F + sel] : null; vals.push(x); if (x != null) vmax = Math.max(vmax, Math.abs(x)); }
      const tok = vals.map((x, p) => {
        const t = x == null ? 0 : Math.min(1, Math.abs(x) / vmax), bg = x == null ? 'transparent' : x >= 0 ? `rgba(53,224,230,${(t * 0.75).toFixed(2)})` : `rgba(169,150,255,${(t * 0.75).toFixed(2)})`;
        return `<span class="${p === pos ? 'here' : ''}" style="background:${bg}" title="${x == null ? 'not run yet' : x.toFixed(3)}">${esc(this.tokens[this.rows[p]])}</span>`;
      }).join('');
      const e = effOf(sel), nv = this.nvocab.get(`${l}:${sel}`);
      html += `<div class="sec">Neuron #${sel} · where it fires ${info('This neuron\'s activation at every token of the text so far: the brighter the background, the stronger it fired (<span class="c">cyan</span> positive, <span class="v">violet</span> negative). Neurons usually respond to a pattern, such as a kind of word, a position or a topic, so reading across the text is the quickest way to guess what it detects. Many respond to several unrelated things at once. The outlined token is the one on screen.')}</div>`;
      html += `<div class="text">${tok}</div>`;
      html += `<div class="fwd-row" style="margin:8px 0 0"><button type="button" data-ko="neurons" data-key="${l}:${sel}">${koN(sel) ? 'Restore' : 'Knock out'} neuron #${sel}</button></div>`;
      html += `<div class="eff" style="margin-top:8px">Here: activation <b>${act[sel].toFixed(3)}</b>${e == null ? '' : ` · effect on ${tw} <b class="${e >= 0 ? 'pos' : 'neg'}">${fmt(e)}</b>`}</div>`;
      html += `<div class="sec">What it writes ${info('The words this neuron\'s output weights push <span class="c">up</span> and <span class="a">down</span> most whenever it fires positively (the other way round when negative). Worked out from its weights through the final normalisation and output layer, so it ignores what later layers do with its output. Derived.')}</div>`;
      html += nv ? `<div class="words up">${nv.up.map((w) => `<span>${word(w.row)}</span>`).join('')}</div><div class="words down" style="margin-top:6px">${nv.down.map((w) => `<span>${word(w.row)}</span>`).join('')}</div>` : '<p class="note">Reading its output weights…</p>';
    } else html += '<p class="note" style="margin-top:12px">Pick a neuron to see where it fires across the text and which words it writes toward.</p>';
    const scroll = el.scrollTop;
    el.innerHTML = html; el.hidden = false; el.scrollTop = scroll;
  }

  // ---------- the field ----------
  draw(now) {
    if (!this.active) return;
    this.tick(now);
    this.renderOutput();
    if (this.collapsed) return;  // hidden: keep playing, skip drawing the field
    const c = this.fc, H = this.FH, d = this.dpr;
    c.clearRect(0, 0, this.FW, H);
    const nrnEl = this.$('nrn'), W = this.FW - (this.nrn && !nrnEl.hidden ? nrnEl.offsetWidth * d : 0);  // leave room for the neuron drawer
    const v = this.view();
    const hd = this.$('fwdHud');
    if (!v.h) { hd.textContent = this.ready ? '' : ''; return; }
    const hsNote = this.hs.on && this.hs.goal ? ` · states: 3 axes show ${(this.hs.share * 100).toFixed(0)}% of their variation` : '';
    if (this.fview === 'river' || this.fview === 'flow') {
      if (this.fview === 'river') this.drawRiver(v, W); else this.drawFlow(v, W);
      hd.textContent = this.viewHud + hsNote;
      this.renderNeurons(v);
      return;
    }
    const { h, pos, beat } = v, L = this.L, NH = this.H;
    const tgt = this.attrOn && this.canAttribute ? this.attrTarget(h, pos) : null, A = tgt != null ? this.attrFor(h, tgt) : null;
    // The reference run at the same position: its capture, and how far this run's state has moved from it.
    const rh = this.ref && this.ref.hist[pos], D = rh && h.diff;
    // Where the two runs part ways: the first depth reaching a quarter of the largest difference.
    const dMax = D ? Math.max(...D) : 0, partsAt = D && dMax > 0.005 ? D.findIndex((x) => x >= Math.max(0.005, dMax * 0.25)) : -1;
    if (rh && !h.diff && this.worker && !this.asked.has(`diff:${pos}`)) { this.asked.add(`diff:${pos}`); this.worker.postMessage({ type: 'diff', pos }); }
    hd.textContent = this.hudText ? this.hudText(v) : `token ${pos + 1} · ${beat <= this.L ? (beat === 0 ? 'embedding' : `layer ${beat - 1}`) : 'readout'}`
      + (A ? ` · why “${this.disp(this.tokens[tgt]).trim() || this.disp(this.tokens[tgt])}”${this.pick && this.pick.pos === pos ? ' (picked)' : ''}` : '')
      + (h.watch && this.ref ? ` · ${this.ref.kind === 'pin' ? 'A\'s' : 'unchanged'} next “${this.disp(this.tokens[h.watch.row])}” ${this.ref.p0[pos] != null ? `${(this.ref.p0[pos] * 100).toFixed(1)}% → ` : 'now '}${(h.watch.p * 100).toFixed(1)}%` : '')
      + (partsAt >= 0 ? ` · parts ways from A at ${partsAt === 0 ? 'the embedding' : `layer ${partsAt - 1}`}` : '') + hsNote;
    // Attribution colours share one scale: the largest single push, either way.
    let amax = 1e-6;
    if (A) { for (const x of A.heads) amax = Math.max(amax, Math.abs(x)); for (const x of A.mlps) amax = Math.max(amax, Math.abs(x)); }
    const push = (x, a0 = 0.1, a1 = 0.85) => { const t = Math.min(1, Math.abs(x) / amax); return x >= 0 ? `rgba(53,224,230,${a0 + a1 * t})` : `rgba(242,181,107,${a0 + a1 * t})`; };

    const pad = 8 * d, labW = 34 * d, rowH = (H - 2 * pad) / (L + 1);
    const headW = Math.min(W * 0.4, (rowH - 3 * d) * NH * 1.6), cellW = headW / NH;
    const mlpX = pad + labW + headW + 14 * d, mlpW = Math.min(W * 0.26, this.F / 8 * d * 0.9);
    const lensX = mlpX + mlpW + 14 * d, lensW = W - lensX - pad;
    const effW = A && lensW > 230 * d ? Math.min(150 * d, lensW * 0.36) : 0, effX = W - pad - effW;  // per-layer total push
    const dW = D && lensW > (effW ? 360 : 200) * d ? Math.min(120 * d, lensW * 0.28) : 0, dX = (effW ? effX - 10 * d : W - pad) - dW;  // Δ from A
    this.geom = { L, pad, rowH, mlpX, mlpW, lensX, labW };
    this.foveate(v, this.geom);
    const n = pos + 1;
    const hmax = []; for (let l = 0; l < L; l++) { let m = 1e-6; for (let k = 0; k < NH; k++) m = Math.max(m, h.headN[l * NH + k]); hmax.push(m); }
    const rowY = (depth) => pad + (L - depth) * rowH;  // depth 0 (embedding) at the bottom

    // Foveated field: the head nearest the gaze gets full detail; the rest show summaries.
    let focus = null, best = Infinity;
    const R = 70 * d;
    c.font = `${9.5 * d}px "JetBrains Mono", monospace`; c.textBaseline = 'middle';
    for (let depth = 0; depth <= L; depth++) {
      const y = rowY(depth), reached = depth <= beat, current = depth === beat && beat <= L;
      c.globalAlpha = reached ? 1 : 0.18;
      c.fillStyle = current ? '#35e0e6' : '#66637e';
      c.fillText(depth === 0 ? 'EMB' : `L${depth - 1}`, pad, y + rowH / 2);
      if (this.nrn && this.nrn.kind === 'feat' && this.nrn.depth === depth) { c.strokeStyle = '#fff'; c.lineWidth = 1.2 * d; c.strokeRect(pad - 3 * d, y + 2 * d, labW - 6 * d, rowH - 4 * d); }
      if (current) { c.fillStyle = 'rgba(53,224,230,0.08)'; c.fillRect(pad + labW - 4 * d, y, W - labW - 2 * pad + 4 * d, rowH); }
      if (depth > 0) {
        const l = depth - 1;
        for (let k = 0; k < NH; k++) {
          const x = pad + labW + k * cellW, s = Math.min(cellW, rowH) - 3 * d;
          const cx = x + s / 2, cy = y + rowH / 2;
          const g = this.gaze ? Math.hypot(this.gaze.x - cx, this.gaze.y - cy) : Infinity;
          if (g < best && g < R) { best = g; focus = { l, h: k, x: cx, y: cy }; }
          const strength = h.headN[l * NH + k] / hmax[l];
          c.fillStyle = A ? push(A.heads[l * NH + k], 0.06, 0.9) : `rgba(108,92,231,${0.12 + 0.75 * strength})`;
          c.fillRect(x, y + 1.5 * d, s, s);
          // Summary: attention over the last keys as a comb (newest on the right).
          const att = h.att.subarray((l * NH + k) * n, (l * NH + k + 1) * n), m = Math.min(n, 8);
          c.fillStyle = A ? 'rgba(255,255,255,0.55)' : '#35e0e6';
          for (let j = 0; j < m; j++) {
            const w = att[n - m + j], bh = w * (s - 4 * d);
            c.fillRect(x + 2 * d + j * (s - 4 * d) / m, y + 1.5 * d + s - 2 * d - bh, Math.max(1, (s - 4 * d) / m - 1 * d), bh);
          }
          if (this.pinnedHead && this.pinnedHead.l === l && this.pinnedHead.h === k) { c.strokeStyle = '#fff'; c.lineWidth = 1.5 * d; c.strokeRect(x, y + 1.5 * d, s, s); }
          if (rh && rh.att && rh.att.length === h.att.length) {  // attention unlike A's: an amber outline
            let tv = 0; const o = (l * NH + k) * n;
            for (let j = 0; j < n; j++) tv += Math.abs(h.att[o + j] - rh.att[o + j]);
            tv *= 0.5;
            if (tv > 0.05) { c.strokeStyle = `rgba(242,181,107,${Math.min(0.95, 0.2 + tv)})`; c.lineWidth = 1.2 * d; c.strokeRect(x - 1 * d, y + 0.5 * d, s + 2 * d, s + 2 * d); }
          }
          if (this.ko.heads.has(`${l}:${k}`)) {  // knocked out: an amber cross
            c.strokeStyle = '#f2b56b'; c.lineWidth = 1.5 * d; c.beginPath();
            c.moveTo(x + 2 * d, y + 3.5 * d); c.lineTo(x + s - 2 * d, y + s - 0.5 * d); c.moveTo(x + s - 2 * d, y + 3.5 * d); c.lineTo(x + 2 * d, y + s - 0.5 * d); c.stroke();
          }
        }
        // MLP: every neuron of the layer, 8 rows.
        const F = this.F, act = h.mlp.subarray(l * F, (l + 1) * F), rows = 8, cols = Math.ceil(F / rows);
        const img = c.createImageData(cols, rows);
        for (let i = 0; i < F; i++) {
          const a = act[i], t = Math.min(1, Math.abs(a) / 2.5) ** 0.7;
          const col = a >= 0 ? [53, 224, 230] : [143, 110, 255];
          img.data[4 * i] = 8 + col[0] * t; img.data[4 * i + 1] = 8 + col[1] * t; img.data[4 * i + 2] = 14 + col[2] * t; img.data[4 * i + 3] = 255;
        }
        if (!this.mlpCanvas || this.mlpCanvas.width !== cols) this.mlpCanvas = new OffscreenCanvas(cols, rows);
        this.mlpCanvas.getContext('2d').putImageData(img, 0, 0);
        c.imageSmoothingEnabled = false;
        c.drawImage(this.mlpCanvas, mlpX, y + 1.5 * d, mlpW, rowH - 4 * d);
        if (A) { c.strokeStyle = push(A.mlps[l], 0.15, 0.85); c.lineWidth = (1 + 2.5 * Math.min(1, Math.abs(A.mlps[l]) / amax)) * d; c.strokeRect(mlpX, y + 1.5 * d, mlpW, rowH - 4 * d); }
        if (this.nrn && this.nrn.l === l) { c.strokeStyle = '#fff'; c.lineWidth = 1.5 * d; c.strokeRect(mlpX - 2 * d, y, mlpW + 4 * d, rowH - 1 * d); }
        if (this.ko.mlps.has(`${l}`)) {  // knocked out: hatched
          c.strokeStyle = 'rgba(242,181,107,0.85)'; c.lineWidth = 1 * d; c.beginPath();
          for (let hx = 0; hx < mlpW + rowH; hx += 10 * d) { c.moveTo(mlpX + hx, y + 1.5 * d); c.lineTo(mlpX + hx - rowH, y + rowH - 2.5 * d); }
          c.save(); c.beginPath(); c.rect(mlpX, y, mlpW, rowH); c.clip(); c.stroke(); c.restore();
        }
      }
      if (dW) {  // how far this run's state has moved from A's here: 0 = the same
        const v = D[depth], bw = Math.min(1, v / Math.max(dMax, 1e-6)) * dW * 0.7;
        c.fillStyle = 'rgba(255,255,255,0.06)'; c.fillRect(dX, y + rowH * 0.3, dW * 0.7, rowH * 0.4);
        c.fillStyle = partsAt >= 0 && depth >= partsAt ? 'rgba(242,181,107,0.85)' : 'rgba(143,130,238,0.35)'; c.fillRect(dX, y + rowH * 0.3, Math.max(1 * d, bw), rowH * 0.4);
        c.fillStyle = 'rgba(232,230,245,0.75)'; c.fillText(v < 0.001 ? '0' : v.toFixed(v < 0.1 ? 3 : 2), dX + dW * 0.72 + 4 * d, y + rowH / 2);
        if (depth === partsAt) { c.fillStyle = '#f2b56b'; c.fillText('◀', dX - 10 * d, y + rowH / 2); }
      }
      if (A && effW) {  // this depth's total direct push: the embedding, or the layer's heads + MLP
        let t = depth === 0 ? A.emb : A.mlps[depth - 1];
        if (depth > 0) for (let k = 0; k < NH; k++) t += A.heads[(depth - 1) * NH + k];
        const mid = effX + effW * 0.5, bw = Math.min(1, Math.abs(t) / (amax * 2)) * effW * 0.5;
        c.fillStyle = 'rgba(255,255,255,0.08)'; c.fillRect(mid, y + rowH * 0.2, 1 * d, rowH * 0.6);
        c.fillStyle = push(t, 0.3, 0.6); c.fillRect(t >= 0 ? mid : mid - bw, y + rowH * 0.3, bw, rowH * 0.4);
        c.fillStyle = 'rgba(232,230,245,0.8)'; c.textAlign = t >= 0 ? 'right' : 'left';
        c.fillText(`${t >= 0 ? '+' : ''}${t.toFixed(1)}`, t >= 0 ? mid - 4 * d : mid + 4 * d, y + rowH / 2); c.textAlign = 'left';
      }
      // Logit lens: what the model would say if it stopped at this depth.
      const lz = h.lens[depth];
      if (lz && lensW > 60 * d) {
        const p = lz.probs[0], w = Math.max(2 * d, p * (lensW * 0.45));
        c.fillStyle = 'rgba(143,130,238,0.55)'; c.fillRect(lensX, y + rowH * 0.25, w, rowH * 0.5);
        if (A && lz.rows[0] === tgt) { c.strokeStyle = 'rgba(53,224,230,0.8)'; c.lineWidth = 1 * d; c.strokeRect(lensX, y + rowH * 0.25, w, rowH * 0.5); }
        c.fillStyle = depth === L ? '#fff' : 'rgba(232,230,245,0.85)';
        c.fillText(`${this.disp(this.tokens[lz.rows[0]])}  ${(p * 100).toFixed(0)}%`, lensX + 6 * d, y + rowH / 2);
      } else if (lensW > 60 * d && this.canAttribute) {  // periphery: not read out yet; it is, once looked at
        c.fillStyle = 'rgba(143,130,238,0.22)';
        for (let j = 0; j < 3; j++) c.fillRect(lensX + (6 + j * 7) * d, y + rowH / 2 - 1 * d, 2 * d, 2 * d);
      }
    }
    c.globalAlpha = 1;
    c.fillStyle = '#66637e';
    const fit = (long, short, w) => (c.measureText(long).width <= w ? long : short);
    c.fillText(fit('HEADS · write strength + attention to recent tokens', `HEADS · ${NH} per layer`, headW + 10 * d), pad + labW, pad / 2 + 1);
    c.fillText(fit(`MLP · ${this.F.toLocaleString()} neurons${this.canAttribute ? ' · click a row' : ''}`, fit(`MLP · ${this.F.toLocaleString()} neurons`, 'MLP', mlpW + 10 * d), mlpW + 10 * d), mlpX, pad / 2 + 1);
    const sparse = h.lens.filter(Boolean).length < L + 1;  // big vocabularies: the lens runs at a spread of depths
    if (lensW > 60 * d) c.fillText(fit(`LOGIT LENS · top guess${sparse ? ' · point at a row to read it' : ''}`, 'LOGIT LENS', lensW), lensX, pad / 2 + 1);
    if (effW) { c.textAlign = 'center'; c.fillText(`PUSH · score ${A.logit.toFixed(1)}`, effX + effW / 2, pad / 2 + 1); c.textAlign = 'left'; }
    if (dW) c.fillText(fit('Δ A · how far from A', 'Δ A', dW + 10 * d), dX, pad / 2 + 1);

    this.focusHead = focus;
    const show = this.pinnedHead || focus;
    if (show) this.drawLoupe(show, h, n, A);
    this.renderNeurons(v);
  }

  // ---------- the river: the logit lens over depth ----------
  // Bottom to top, the words the model would say if it stopped at each depth, each band as wide
  // as its probability; the empty space above is the rest of the vocabulary. The white line is
  // how unsure it is: e^entropy, "as unsure as choosing among N words".
  drawRiver(v, W) {
    const c = this.fc, d = this.dpr, H = this.FH, L = this.L, { h, pos, beat } = v;
    const word = (r) => this.disp(this.tokens[r]);
    if (this.worker && !this.playing) for (let dp = 0; dp <= L; dp++) {  // fill in the depths a big vocabulary skipped
      const k = `lens:${pos}:${dp}`;
      if (!h.lens[dp] && !this.asked.has(k)) { this.asked.add(k); this.worker.postMessage({ type: 'lens', pos, depth: dp }); }
    }
    const live = pos === this.rows.length - 1 && this.viewPos < 0, cur = Math.min(beat, L), last = live ? cur : L;
    const avail = []; for (let dp = 0; dp <= last; dp++) if (h.lens[dp]) avail.push(dp);
    const pad = 8 * d, x0 = pad + 40 * d, x1 = W - pad - 120 * d, yTop = pad + 14 * d, eh = Math.max(24 * d, (H - yTop) * 0.16);
    const yR0 = yTop + eh + 10 * d, yR1 = H - pad - 16 * d, xOf = (dp) => x0 + (dp / L) * (x1 - x0), yOf = (p) => yR1 - p * (yR1 - yR0);
    c.font = `${9.5 * d}px "JetBrains Mono", monospace`; c.textBaseline = 'middle';
    // Axes: probability on the left, depth along the bottom.
    c.fillStyle = '#66637e'; c.textAlign = 'right';
    for (const p of [0, 0.5, 1]) { c.fillText(`${p * 100}%`, x0 - 6 * d, yOf(p)); c.fillRect(x0, yOf(p), x1 - x0, 1); }
    c.textAlign = 'center';
    const every = Math.max(1, Math.ceil((L + 1) / Math.max(1, (x1 - x0) / (30 * d))));
    for (let dp = 0; dp <= L; dp += every) { c.fillStyle = dp === cur && live ? '#35e0e6' : '#66637e'; c.fillText(dp === 0 ? 'EMB' : `L${dp - 1}`, xOf(dp), yR1 + 10 * d); }
    c.textAlign = 'left';
    c.fillText(`LOGIT LENS OVER DEPTH · “${word(this.rows[pos])}” (token ${pos + 1})`, pad, pad / 2 + 4 * d);
    if (!avail.length) { this.viewHud = `token ${pos + 1} · reading the logit lens…`; return; }
    // The words: the eight that ever do best get a colour; the rest of each top ten is grey.
    const best = new Map();
    for (const dp of avail) h.lens[dp].rows.forEach((r, j) => best.set(r, Math.max(best.get(r) || 0, h.lens[dp].probs[j])));
    const lastLz = h.lens[avail[avail.length - 1]];
    const pAt = (lz, r) => { const j = lz.rows.indexOf(r); return j >= 0 ? lz.probs[j] : 0; };
    const named = [...best.keys()].sort((a, b) => best.get(b) - best.get(a)).slice(0, 8)
      .sort((a, b) => pAt(lastLz, b) - pAt(lastLz, a) || best.get(b) - best.get(a));
    const PAL = ['#35e0e6', '#a996ff', '#f2b56b', '#ff7ab6', '#7ee08a', '#5aa9ff', '#e8d36b', '#ff8a6b'];
    const bands = named.map((r, i) => ({ r, col: PAL[i] })).concat([{ r: -1, col: 'rgba(154,151,179,0.28)' }]);
    // Stack each depth's probabilities, bottom up, in the same order everywhere.
    const stack = avail.map((dp) => {
      const lz = h.lens[dp]; let y = 0; const out = [];
      for (const b of bands) {
        const p = b.r >= 0 ? pAt(lz, b.r) : lz.probs.reduce((s, q, j) => s + (named.includes(lz.rows[j]) ? 0 : q), 0);
        out.push([y, y + p]); y += p;
      }
      return out;
    });
    const smooth = (pts, back) => {  // a ribbon edge through the columns, eased between them
      const seq = back ? pts.slice().reverse() : pts;
      seq.forEach(([x, y], i) => { if (i === 0) { if (back) c.lineTo(x, y); else c.moveTo(x, y); return; } const [px, py] = seq[i - 1], mx = (px + x) / 2; c.bezierCurveTo(mx, py, mx, y, x, y); });
    };
    bands.forEach((b, i) => {
      const lo = avail.map((dp, k) => [xOf(dp), yOf(stack[k][i][0])]), hi = avail.map((dp, k) => [xOf(dp), yOf(stack[k][i][1])]);
      if (avail.length === 1) { lo.push([lo[0][0] + 6 * d, lo[0][1]]); hi.push([hi[0][0] + 6 * d, hi[0][1]]); }
      c.beginPath(); smooth(lo, false); smooth(hi, true); c.closePath();
      c.globalAlpha = b.r >= 0 ? 0.78 : 1; c.fillStyle = b.col; c.fill(); c.globalAlpha = 1;
    });
    // Columns not read out yet (big vocabularies read a spread of depths first).
    c.fillStyle = 'rgba(143,130,238,0.35)';
    for (let dp = 0; dp <= last; dp++) if (!h.lens[dp]) for (let j = 0; j < 3; j++) c.fillRect(xOf(dp) - 1 * d, yR1 - (8 + j * 7) * d, 2 * d, 2 * d);
    // Names: inside each band where it is widest, and at the right edge with the final share.
    c.font = `${10 * d}px "JetBrains Mono", monospace`;
    bands.forEach((b, i) => {
      if (b.r < 0) return;
      let k = 0; stack.forEach((st, j) => { if (st[i][1] - st[i][0] > stack[k][i][1] - stack[k][i][0]) k = j; });
      const [a0, a1] = stack[k][i], hpx = (a1 - a0) * (yR1 - yR0);
      if (hpx > 12 * d && k < stack.length - 1) { c.fillStyle = '#08080e'; const x = Math.min(Math.max(xOf(avail[k]) + 4 * d, x0 + 4 * d), x1 - 70 * d); c.fillText(word(b.r), x, yOf((a0 + a1) / 2)); }
      const [l0, l1] = stack[stack.length - 1][i];
      if ((l1 - l0) * (yR1 - yR0) > 9 * d) { c.fillStyle = b.col; c.fillText(`${word(b.r)} ${((l1 - l0) * 100).toFixed(0)}%`, xOf(avail[avail.length - 1]) + 8 * d, yOf((l0 + l1) / 2)); }
    });
    // Uncertainty: e^entropy on a log scale, from 1 word to the whole vocabulary.
    const Vn = Math.log(this.tokens.length || 50257), eY = (H0) => yTop + eh - Math.min(1, H0 / Vn) * eh;
    c.fillStyle = 'rgba(255,255,255,0.04)'; c.fillRect(x0, yTop, x1 - x0, eh);
    c.strokeStyle = '#fff'; c.lineWidth = 1.4 * d; c.beginPath();
    avail.forEach((dp, k) => { const y = eY(h.lens[dp].entropy); if (k === 0) c.moveTo(xOf(dp), y); else c.lineTo(xOf(dp), y); });
    c.stroke();
    c.fillStyle = '#9a97b3'; c.textAlign = 'right'; c.font = `${9 * d}px "JetBrains Mono", monospace`;
    c.fillText('unsure', x0 - 6 * d, yTop + eh * 0.35); c.textAlign = 'left';
    const eL = Math.exp(lastLz.entropy);
    c.fillStyle = '#e8e6f5'; c.textAlign = 'right';
    c.fillText(`${live && cur < L ? 'now' : 'at the end'}: as unsure as choosing among ${eL < 10 ? eL.toFixed(1) : Math.round(eL).toLocaleString()} words`, x1 - 6 * d, yTop + 8 * d); c.textAlign = 'left';
    if (live && cur <= L) { c.fillStyle = 'rgba(53,224,230,0.8)'; c.fillRect(xOf(cur) - 0.75 * d, yTop, 1.5 * d, yR1 - yTop); }
    // Hover: the depth under the pointer, its top five.
    const g = this.gaze;
    if (g && g.x >= x0 - 10 * d && g.x <= x1 + 10 * d) {
      let dp = avail[0]; for (const q of avail) if (Math.abs(xOf(q) - g.x) < Math.abs(xOf(dp) - g.x)) dp = q;
      const lz = h.lens[dp], lines = [`${dp === 0 ? 'embedding' : `after layer ${dp - 1}`} · as unsure as ${Math.exp(lz.entropy) < 10 ? Math.exp(lz.entropy).toFixed(1) : Math.round(Math.exp(lz.entropy)).toLocaleString()} words`]
        .concat(lz.rows.slice(0, 5).map((r, j) => `${word(r)}  ${(lz.probs[j] * 100).toFixed(1)}%`));
      c.fillStyle = 'rgba(255,255,255,0.5)'; c.fillRect(xOf(dp) - 0.5 * d, yR0, 1 * d, yR1 - yR0);
      c.font = `${10 * d}px "JetBrains Mono", monospace`;
      const bw = Math.max(...lines.map((t) => c.measureText(t).width)) + 16 * d, lh = 15 * d, bx = Math.min(xOf(dp) + 10 * d, W - bw - 4 * d), by = yR0;
      c.fillStyle = 'rgba(14,14,24,0.94)'; c.fillRect(bx, by, bw, lh * lines.length + 8 * d);
      c.strokeStyle = 'rgba(53,224,230,0.5)'; c.lineWidth = 1 * d; c.strokeRect(bx, by, bw, lh * lines.length + 8 * d);
      c.font = `${10 * d}px "JetBrains Mono", monospace`;
      lines.forEach((t, j) => { c.fillStyle = j ? '#e8e6f5' : '#35e0e6'; c.fillText(t, bx + 8 * d, by + 4 * d + lh * (j + 0.5)); });
    }
    // In words: when did it make up its mind?
    const top = lastLz.rows[0];
    let settle = avail[avail.length - 1];
    for (let k = avail.length - 1; k >= 0 && h.lens[avail[k]].rows[0] === top; k--) settle = avail[k];
    const firstSeen = avail.find((dp) => h.lens[dp].rows.includes(top));
    this.viewHud = `token ${pos + 1} · ${live && cur < L ? `at layer ${cur - 1} it leads with` : 'settles on'} “${word(top)}”${!(live && cur < L) ? ` from ${settle === 0 ? 'the embedding' : `layer ${settle - 1}`} on` : ''}${firstSeen != null && firstSeen < settle ? `, in its top ten from ${firstSeen === 0 ? 'the embedding' : `layer ${firstSeen - 1}`}` : ''}`;
  }

  // ---------- the flow: where each token's information came from ----------
  // A token's attention at layer l, averaged over the layer's heads, cached on its capture.
  abar(q) {
    const h = this.hist[q]; if (!h) return null;
    if (h.abar) return h.abar;
    const L = this.L, NH = this.H, n = q + 1, a = new Float32Array(L * n);
    for (let l = 0; l < L; l++) for (let k = 0; k < NH; k++) { const o = (l * NH + k) * n; for (let j = 0; j < n; j++) a[l * n + j] += h.att[o + j] / NH; }
    return (h.abar = a);
  }
  // What attention actually moves: each weight times the size of the value vector it reads
  // (Kobayashi et al., 2020), averaged over heads, each row scaled to sum to 1. Many models park
  // most of their attention on the first token, whose value is tiny: weights alone overstate it.
  aflow(q) {
    const h = this.hist[q]; if (!h) return null;
    if (h.aflow) return h.aflow;
    const L = this.L, NH = this.H, n = q + 1;
    if (!h.vn || !h.vn.length || this.hist.slice(0, n).some((x) => !x || !x.vn || !x.vn.length)) return this.abar(q);
    const a = new Float32Array(L * n);
    for (let l = 0; l < L; l++) {
      let s = 0;
      for (let k = 0; k < NH; k++) { const o = (l * NH + k) * n; for (let j = 0; j < n; j++) { const w = h.att[o + j] * this.hist[j].vn[l * NH + k]; a[l * n + j] += w; s += w; } }
      if (s > 0) for (let j = 0; j < n; j++) a[l * n + j] /= s;
    }
    return (h.aflow = a);
  }
  // Attention rollout for token pos: r[dp][k] is how much of pos's state after the last layer
  // traces back to token k at depth dp, with each layer's residual connection counted as half
  // (Abnar and Zuidema, 2020). Each r[dp] sums to 1.
  rolloutFor(pos) {
    const L = this.L, r = new Array(L + 1);
    r[L] = new Float32Array(pos + 1); r[L][pos] = 1;
    for (let l = L - 1; l >= 0; l--) {
      const nx = r[l + 1], out = new Float32Array(pos + 1);
      for (let q = 0; q <= pos; q++) {
        const w = nx[q]; if (w < 1e-6) continue;
        out[q] += 0.5 * w;
        const a = this.aflow(q); if (!a) { out[q] += 0.5 * w; continue; }
        const n = q + 1, row = a.subarray(l * n, (l + 1) * n);
        for (let k = 0; k < n; k++) out[k] += 0.5 * w * row[k];
      }
      r[l] = out;
    }
    return r;
  }
  flowCol(x) { const g = this.flowGeom; if (!g) return -1; const p = Math.floor((x - g.x0) / g.colW); return p >= 0 && p < g.n ? p : -1; }
  drawFlow(v, W) {
    const c = this.fc, d = this.dpr, H = this.FH, L = this.L, D = L + 1, { pos, beat } = v, n = this.captured();
    const word = (r) => this.disp(this.tokens[r]);
    const pad = 8 * d, labW = 34 * d, x0 = pad + labW, x1 = W - pad, yTop = pad + 14 * d, yBot = H - pad - 40 * d;
    const rowH = (yBot - yTop) / D, colW = (x1 - x0) / Math.max(1, n);
    const xOf = (p) => x0 + (p + 0.5) * colW, yOf = (dp) => yTop + (L - dp + 0.5) * rowH;
    this.flowGeom = { x0, colW, n };
    // The edges, drawn once per run length and size, then reused every frame.
    const key = `${this.runId}|${n}|${W}|${H}`;
    if (!this.flowCache || this.flowCache.key !== key) {
      const t0 = performance.now(), oc = new OffscreenCanvas(Math.max(1, W), Math.max(1, H)), o = oc.getContext('2d');
      const buckets = Array.from({ length: 6 }, () => []);  // segments grouped by strength, one stroke each
      let segs = 0;
      for (let q = 1; q < n; q++) {
        const a = this.aflow(q), m = q + 1;
        for (let l = 0; l < L; l++) {
          const row = a.subarray(l * m, (l + 1) * m), top = [];
          for (let k = 0; k < q; k++) if (row[k] >= 0.03) top.push(k);
          top.sort((p1, p2) => row[p2] - row[p1]);
          for (const k of top.slice(0, 4)) { buckets[Math.min(5, Math.floor(row[k] * 6))].push(k, l, q); segs++; }
        }
      }
      buckets.forEach((b, i) => {
        if (!b.length) return;
        o.strokeStyle = `rgba(169,150,255,${(0.06 + i * 0.1).toFixed(2)})`; o.lineWidth = (0.6 + i * 0.35) * d; o.beginPath();
        for (let j = 0; j < b.length; j += 3) { const k = b[j], l = b[j + 1], q = b[j + 2]; o.moveTo(xOf(k), yOf(l)); o.lineTo(xOf(q), yOf(l + 1)); }
        o.stroke();
      });
      o.fillStyle = 'rgba(232,230,245,0.35)';
      const ds = Math.max(1, Math.min(3 * d, colW * 0.25));
      for (let p = 0; p < n; p++) for (let dp = 0; dp <= L; dp++) o.fillRect(xOf(p) - ds / 2, yOf(dp) - ds / 2, ds, ds);
      this.flowCache = { key, oc, segs, ms: performance.now() - t0 };
    }
    // The depth being replayed, then the cached edges.
    const live = pos === this.rows.length - 1 && this.viewPos < 0, cur = Math.min(beat, L);
    if (live && cur <= L) { c.fillStyle = 'rgba(53,224,230,0.07)'; c.fillRect(x0, yOf(cur) - rowH / 2, x1 - x0, rowH); }
    c.drawImage(this.flowCache.oc, 0, 0);
    c.font = `${9.5 * d}px "JetBrains Mono", monospace`; c.textBaseline = 'middle';
    const every = Math.max(1, Math.ceil(D / Math.max(1, (yBot - yTop) / (12 * d))));
    for (let dp = 0; dp <= L; dp += every) { c.fillStyle = dp === cur && live ? '#35e0e6' : '#66637e'; c.fillText(dp === 0 ? 'EMB' : `L${dp - 1}`, pad, yOf(dp)); }
    c.fillStyle = '#66637e';
    c.fillText(`ATTENTION FLOW · ${n} tokens × ${L} layers · strongest links (≥ 3%, top 4 per token and layer)`, pad, pad / 2 + 4 * d);
    // The token on screen: its column, and with rollout, the cone of tokens feeding it.
    c.fillStyle = 'rgba(53,224,230,0.08)'; c.fillRect(xOf(pos) - colW / 2, yTop, colW, yBot - yTop);
    let R = null;
    if (this.rollout && pos < n) {
      const rk = `${this.runId}|${n}|${pos}`;
      if (!this.rollCache || this.rollCache.key !== rk) this.rollCache = { key: rk, r: this.rolloutFor(pos) };
      R = this.rollCache.r;
      // The first token only attends to itself, so whatever reaches it stays: rollout piles up
      // there. Shares are shown among the other tokens; the first token's is given apart, in grey.
      const rest = (dp) => (pos > 0 ? Math.max(1e-6, 1 - R[dp][0]) : 1);
      for (let l = 0; l < L; l++) {  // the links inside the cone
        const nx = R[l + 1], z = rest(l + 1);
        for (let q = 1; q <= pos; q++) {
          const w = nx[q] / z; if (w < 0.01) continue;
          const a = this.aflow(q), m = q + 1;
          for (let k = 0; k < q; k++) {
            const f = 0.5 * w * a[l * m + k]; if (f < 0.004) continue;
            c.strokeStyle = k === 0 ? 'rgba(154,151,179,0.9)' : 'rgba(53,224,230,0.9)';
            c.globalAlpha = Math.min(0.9, 0.15 + f * 5) * (k === 0 ? 0.5 : 1); c.lineWidth = (0.6 + Math.min(4, f * 20)) * d;
            c.beginPath(); c.moveTo(xOf(k), yOf(l)); c.lineTo(xOf(q), yOf(l + 1)); c.stroke();
          }
        }
      }
      c.globalAlpha = 1;
      const shrink = Math.min(1, colW / (6 * d) + 0.3);
      for (let dp = 0; dp <= L; dp++) for (let k = 0; k <= pos; k++) {
        const w = k === 0 && pos > 0 ? 0.04 : R[dp][k] / rest(dp); if (R[dp][k] < 0.004) continue;
        c.fillStyle = k === 0 && pos > 0 ? '#66637e' : '#35e0e6';
        c.beginPath(); c.arc(xOf(k), yOf(dp), (0.8 + 5 * Math.sqrt(w)) * d * shrink, 0, Math.PI * 2); c.fill();
      }
      let bmax = 1e-6; for (let k = pos > 0 ? 1 : 0; k <= pos; k++) bmax = Math.max(bmax, R[0][k]);  // where it draws from, at the embedding
      c.fillStyle = '#35e0e6';
      for (let k = pos > 0 ? 1 : 0; k <= pos; k++) { const bh = (R[0][k] / bmax) * 14 * d; c.fillRect(xOf(k) - Math.max(1, colW * 0.35), yBot + 16 * d - bh, Math.max(1.5 * d, colW * 0.7), bh); }
    }
    // Token names along the bottom, where there is room; always the token on screen.
    c.textAlign = 'center';
    const room = colW >= 28 * d;
    for (let p = 0; p < n; p++) {
      if (!room && p !== pos && !(R && R[0][p] >= 0.08)) continue;
      c.fillStyle = p === pos ? '#35e0e6' : p < this.promptLen ? 'rgba(232,230,245,0.8)' : '#c9c1ff';
      const t = word(this.rows[p]); c.fillText(room ? t.slice(0, Math.max(2, Math.floor(colW / (6.2 * d)))) : t, xOf(p), yBot + 26 * d);
    }
    c.textAlign = 'left';
    // Hover: what the state under the pointer read from, at that layer.
    const g = this.gaze; let hov = null;
    if (g) { const p = this.flowCol(g.x), dp = L - Math.floor((g.y - yTop) / rowH); if (p >= 0 && dp >= 1 && dp <= L) hov = { p, dp }; }
    if (hov) {
      const m = hov.p + 1, l = hov.dp - 1, row = this.aflow(hov.p).subarray(l * m, (l + 1) * m), raw = this.abar(hov.p).subarray(l * m, (l + 1) * m);
      const top = Array.from(row).map((w, k) => [w, k]).sort((p1, p2) => p2[0] - p1[0]).slice(0, 4);
      const lines = [`“${word(this.rows[hov.p])}” at layer ${l} took in`].concat(top.map(([w, k]) => `${k === hov.p ? 'itself' : `“${word(this.rows[k])}”`} @${k}  ${(w * 100).toFixed(0)}% · attention ${(raw[k] * 100).toFixed(0)}%`));
      c.strokeStyle = '#fff'; c.lineWidth = 1 * d; c.beginPath(); c.arc(xOf(hov.p), yOf(hov.dp), 5 * d, 0, Math.PI * 2); c.stroke();
      c.font = `${10 * d}px "JetBrains Mono", monospace`;
      const bw = Math.max(...lines.map((t) => c.measureText(t).width)) + 16 * d, lh = 15 * d, bx = Math.min(xOf(hov.p) + 12 * d, W - bw - 4 * d), by = Math.max(yTop, Math.min(yOf(hov.dp) - 20 * d, yBot - lh * lines.length));
      c.fillStyle = 'rgba(14,14,24,0.94)'; c.fillRect(bx, by, bw, lh * lines.length + 8 * d);
      c.strokeStyle = 'rgba(53,224,230,0.5)'; c.strokeRect(bx, by, bw, lh * lines.length + 8 * d);
      c.font = `${10 * d}px "JetBrains Mono", monospace`;
      lines.forEach((t, j) => { c.fillStyle = j ? '#e8e6f5' : '#35e0e6'; c.fillText(t, bx + 8 * d, by + 4 * d + lh * (j + 0.5)); });
    }
    if (R) {
      const z = pos > 0 ? Math.max(1e-6, 1 - R[0][0]) : 1, pct = (w) => `${((w / z) * 100).toFixed(0)}%`;
      const src = Array.from(R[0]).map((w, k) => [w, k]).filter(([, k]) => k !== pos && k > 0).sort((p1, p2) => p2[0] - p1[0]).slice(0, 3);
      this.viewHud = `token ${pos + 1} “${word(this.rows[pos])}” · draws on itself ${pct(R[0][pos])}${src.length ? `, ${src.map(([w, k]) => `“${word(this.rows[k])}” ${pct(w)}`).join(', ')}` : ''}${pos > 0 ? ` · ${(R[0][0] * 100).toFixed(0)}% rests on the first token` : ''}`;
    } else this.viewHud = `token ${pos + 1} “${word(this.rows[pos])}” · ${this.flowCache.segs.toLocaleString()} links`;
  }

  drawLoupe(f, h, n, A) {
    const c = this.fc, d = this.dpr, NH = this.H;
    const att = h.att.subarray((f.l * NH + f.h) * n, (f.l * NH + f.h + 1) * n);
    const top = Array.from(att).map((w, j) => [w, j]).sort((a, b) => b[0] - a[0]).slice(0, 6);
    const w = 250 * d, lineH = 16 * d, hh = (3 + top.length + (A ? 1 : 0)) * lineH;
    let x = (f.x ?? this.FW / 2) + 16 * d, y = (f.y ?? this.FH / 2) - hh / 2;
    x = Math.min(x, this.FW - w - 6 * d); y = Math.max(6 * d, Math.min(y, this.FH - hh - 6 * d));
    c.fillStyle = 'rgba(14,14,24,0.94)'; c.strokeStyle = 'rgba(53,224,230,0.5)'; c.lineWidth = 1 * d;
    c.fillRect(x, y, w, hh); c.strokeRect(x, y, w, hh);
    c.fillStyle = '#35e0e6'; c.font = `${10 * d}px "JetBrains Mono", monospace`;
    c.fillText(`LAYER ${f.l} · HEAD ${f.h}${this.pinnedHead ? ' · PINNED' : ''}`, x + 8 * d, y + lineH * 0.8);
    c.fillStyle = '#9a97b3';
    c.fillText(`writes |Δ| = ${h.headN[f.l * NH + f.h].toFixed(2)} · attends to`, x + 8 * d, y + lineH * 1.8);
    if (A) {
      const p = A.heads[f.l * NH + f.h];
      c.fillStyle = p >= 0 ? '#35e0e6' : '#f2b56b';
      c.fillText(`pushes “${this.disp(this.tokens[A.row])}” ${p >= 0 ? '+' : ''}${p.toFixed(2)} of score ${A.logit.toFixed(1)}`, x + 8 * d, y + lineH * (2.9 + top.length));
    }
    top.forEach(([wt, j], i) => {
      const yy = y + lineH * (2.9 + i);
      c.fillStyle = 'rgba(53,224,230,0.35)'; c.fillRect(x + 8 * d, yy - 5 * d, wt * (w - 16 * d), 10 * d);
      c.fillStyle = '#e8e6f5';
      c.fillText(`${this.disp(this.tokens[this.rows[j]])}  ${(wt * 100).toFixed(0)}%  @${j}`, x + 12 * d, yy);
    });
  }

  // ---------- 3-D overlay: the sequence, the thought path, the attention arcs ----------
  drawOverlay(ov, W, H) {
    if (!this.active) return;
    this.FWscene = W;
    if (this.hs.on) { this.askHidden(); this.drawStates(ov); return; }
    const v = this.view(); if (!v.h) return;
    const { h, pos, beat } = v, d = this.dpr, L = this.L, NH = this.H, n = pos + 1;
    const P = this.points;
    const scr = (r) => this.project(P[3 * r], P[3 * r + 1], P[3 * r + 2]);
    ov.lineWidth = 1 * d; ov.font = `${10.5 * d}px "JetBrains Mono", monospace`; ov.textBaseline = 'middle';
    // Sequence path
    ov.strokeStyle = 'rgba(232,230,245,0.18)'; ov.beginPath();
    let first = true;
    for (let p = 0; p <= pos; p++) { const s = scr(this.rows[p]); if (!s) { first = true; continue; } if (first) ov.moveTo(s[0], s[1]); else ov.lineTo(s[0], s[1]); first = false; }
    ov.stroke();
    // Thought path: at each depth, the probability-weighted centroid of the logit-lens guesses.
    const path = [];
    for (let dp = 0; dp <= Math.min(beat, L); dp++) {
      const lz = h.lens[dp]; if (!lz) continue;
      const c = this.lensCentroid(lz); if (c) path.push(this.project(c[0], c[1], c[2]));
    }
    // Attention arcs from the current depth to every earlier token.
    const layer = beat >= 1 && beat <= L ? beat - 1 : (beat === L + 1 ? L - 1 : -1);
    const from = path[path.length - 1] || scr(this.rows[pos]);
    if (layer >= 0 && from) {
      const sel = this.pinnedHead || this.focusHead;
      for (let j = 0; j < n; j++) {
        let w = 0;
        if (sel && sel.l === layer) w = h.att[(layer * NH + sel.h) * n + j];
        else { for (let k = 0; k < NH; k++) w += h.att[(layer * NH + k) * n + j]; w /= NH; }
        if (w < 0.02) continue;
        const to = scr(this.rows[j]); if (!to) continue;
        const mx = (from[0] + to[0]) / 2, my = Math.min(from[1], to[1]) - 60 * d - Math.hypot(from[0] - to[0], from[1] - to[1]) * 0.15;
        ov.strokeStyle = `rgba(53,224,230,${Math.min(0.95, 0.15 + w * 1.2)})`; ov.lineWidth = (0.6 + w * 5) * d;
        ov.beginPath(); ov.moveTo(from[0], from[1]); ov.quadraticCurveTo(mx, my, to[0], to[1]); ov.stroke();
      }
    }
    ov.lineWidth = 1.5 * d; ov.strokeStyle = 'rgba(143,130,238,0.9)'; ov.beginPath();
    path.forEach((p, i) => { if (!p) return; if (i === 0) ov.moveTo(p[0], p[1]); else ov.lineTo(p[0], p[1]); });
    ov.stroke();
    path.forEach((p, i) => { if (!p) return; ov.fillStyle = i === path.length - 1 ? '#fff' : 'rgba(143,130,238,0.9)'; ov.beginPath(); ov.arc(p[0], p[1], (i === path.length - 1 ? 4.5 : 2.5) * d, 0, Math.PI * 2); ov.fill(); });
    if (this.flying && this.canFly) {  // fly-through: name the guesses the path is passing through
      const lz = this.lensAt(h, beat);
      if (lz) lz.rows.slice(0, 4).forEach((r, j) => {
        const s = scr(r); if (!s) return;
        ov.globalAlpha = Math.min(1, 0.35 + lz.probs[j] * 2); ov.fillStyle = '#c9c1ff';
        ov.beginPath(); ov.arc(s[0], s[1], 2.5 * d, 0, Math.PI * 2); ov.fill();
        ov.fillText(`${this.disp(this.tokens[r])}  ${(lz.probs[j] * 100).toFixed(0)}%`, s[0] + 7 * d, s[1] + 11 * d);
      });
      ov.globalAlpha = 1;
    }
    // Token markers + labels
    for (let p = 0; p <= Math.min(pos + 1, this.rows.length - 1); p++) {
      const s = scr(this.rows[p]); if (!s) continue;
      const isPred = p === pos + 1;
      ov.fillStyle = p === pos ? '#35e0e6' : (isPred ? '#ffffff' : (p < this.promptLen ? '#e8e6f5' : '#9d8ff5'));
      ov.beginPath(); ov.arc(s[0], s[1], (p === pos ? 5 : 3.5) * d, 0, Math.PI * 2); ov.fill();
      ov.fillText(`${this.disp(this.tokens[this.rows[p]])}${isPred ? '  ← predicted' : ''}`, s[0] + 8 * d, s[1]);
    }
    if (beat === L + 1 && h.lens[L] && pos === this.rows.length - 1) {  // the guess, before it is committed
      const r = h.lens[L].rows[0], s = scr(r);
      if (s) { ov.strokeStyle = '#fff'; ov.setLineDash([3 * d, 4 * d]); ov.beginPath(); ov.arc(s[0], s[1], 9 * d, 0, Math.PI * 2); ov.stroke(); ov.setLineDash([]); ov.fillStyle = '#fff'; ov.fillText(`${this.disp(this.tokens[r])}  ${(h.lens[L].probs[0] * 100).toFixed(0)}%`, s[0] + 12 * d, s[1] - 12 * d); }
    }
  }
}

function esc(s) { return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }
