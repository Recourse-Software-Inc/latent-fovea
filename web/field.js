// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// Forward-pass view: the model runs in a worker; its computation is replayed layer by layer
// at a speed you choose, across the whole "parametric field" (every layer x 12 heads + MLP),
// and as a path of logit-lens guesses through the 3-D embedding space.
import { GPT2Tokenizer, HFTokenizer } from './bpe.js';
import { dataFetch } from './store.js';
import { SAES } from './sae.js';
import { NeuronBook } from './neuronbook.js';
import { NeuronGraph } from './ngraph-view.js';

const BEATS = { live: 0, slow: 160, slower: 520, glacial: 1500 };

// the page-address keys a shared link uses (app.js drops them when the model changes)
export const SHARED_KEYS = ['ask', 'gen', 'dec', 'p', 'n', 'g', 'gk', 'w', 'iv', 'h', 'm'];

export class ForwardView {
  constructor(ctx) {
    Object.assign(this, ctx);  // $, DATA, tokens, points, cats, dpr, project, disp, flyToPoint, catColors
    this.active = false; this.ready = false; this.loading = false;
    this.hist = []; this.rows = []; this.promptLen = 0; this.lead = 0;  // lead: 1 when the run starts with <|endoftext|>
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
    // neurons: key 'l:i' -> { s, add, label }: knocked out (s 0), or steered (activation * s + add).
    this.ko = { heads: new Set(), mlps: new Set(), neurons: new Map() }; this.ivRun = null; this.ref = null; this.ivVer = 0;
    // Reading one neuron: its book (strongest examples, both poles, its shape; null when the model
    // has none), what feeds it at a token ('pos:l:j'), what reads it ('l:i'), a reader's own test
    // sentence, and whether "where it fires" shows the activation or the input before it.
    this.book = null; this.bookData = new Map(); this.lk = new Map(); this.rd = new Map(); this.probe = null; this.showInput = false; this.caps = {};
    this.graph = null;  // the neuron graph for one word: { pos, row, at, data | null, progress }
    this.traces = new Map();  // finished graphs, 'pos:row:ivVer' -> data, so reopening one is instant
    // The reader's own map: a trail of neurons visited (back and forward), and their names and
    // notes, kept in this browser per model and exportable.
    this.trail = []; this.trailIx = -1; this.notes = new Map(); this.urlDone = false;
    this.shared = new URLSearchParams(location.search);  // a shared address, read once before anything rewrites it
    this.expand = null;  // an example opened to its whole passage
    try { this.blind = localStorage.getItem('lf-blind') === '1'; } catch { this.blind = false; }  // reading blind, kept across visits
    this.band = null; this.bandWin = new Map(); this.showMine = false; this.pendingUrl = null;  // a band of its range opened; the reader's own index
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
  submit(text, { shared = false } = {}) {
    text = (text || '').trim();
    if (!text) return;
    // the reader's own text: whatever a shared link still had waiting is dropped, never applied to it
    if (!shared) { this.urlDone = true; this.pendingIv = null; this.pendingUrl = null; this.shared = new URLSearchParams(); }
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
    // replies about the text on screen: one from an earlier run (or the unchanged model, before a
    // Rerun) still on its way would show the wrong model's numbers
    if (m.run != null && m.run !== this.runId && ['links', 'readers', 'sae', 'saeFeat', 'hidden', 'lens', 'attr', 'diff', 'error'].includes(m.type)) { if (m.type === 'hidden') this.hs.busy = false; return; }
    if (m.type === 'gain') { this.gains.set(`${m.l}:${m.row}`, m.gain); this.nrnKey = ''; return; }
    if (m.type === 'links') { this.lk.set(`${m.pos}:${m.l}:${m.j}`, m); this.nrnKey = ''; return; }
    if (m.type === 'readers') { this.rd.set(`${m.pos}:${m.l}:${m.i}`, m); this.nrnKey = ''; return; }
    if (m.type === 'probe') {
      if (m.kind === 'expand') { if (this.expand && this.expand.id === m.id) { this.expand.result = m; this.nrnKey = ''; } return; }
      if (this.probe && this.probe.id === m.id) { this.probe.result = m; this.nrnKey = ''; }
      return;
    }
    if (m.type === 'graphProgress') { if (m.runId === this.textId && this.graph && !this.graph.data) { this.graph.progress = `${m.k} of ${m.n} neurons traced`; this.graphView.render(this); } return; }
    if (m.type === 'graph') {
      if (m.runId !== this.textId) return;  // a trace of an earlier text
      if (!m.error) this.traces.set(`${m.pos}:${m.row}:${m.ivVer}`, m);
      if (this.graph && this.graph.pos === m.pos && this.graph.row === m.row) { this.graph.data = m; this.graphView.render(this); }
      this.tryPendingIv();  // a shared link: now change the model
      return;
    }
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
      this.caps = m.caps || {};
      this.loadNotes();
      this.book = null; this.bookData.clear();
      if (!this.repo && !this.DATA.startsWith('opfs:')) NeuronBook.open(new URL(this.DATA, location.href).href, dataFetch).then((b) => { this.book = b; this.nrnKey = ''; });
      this.fidelityNote().then((note) => { if (note && this.ready) this.$('fwdStatus').textContent += m.q8 ? ` · the 16-bit model is ${note}; at 8 bits its predictions move slightly` : ` · ${note}${m.backend === 'gpu' ? '; the GPU matches the CPU kernel to within 0.001' : ''}`; });
      this.backend = m.backend;
      const q8 = m.q8 && m.qerr ? `, 8-bit weights (${(m.qerr.all * 100).toFixed(2)}% from the published ones)` : '';
      const where = m.backend === 'gpu' ? ` · on your GPU${m.gpuName ? ` (${m.gpuName})` : ''}${q8}` : m.gpuNote ? ` · ${m.gpuNote}` : '';
      this.q8 = !!m.q8;
      this.$('fwdStatus').textContent = `${this.modelName} ready${where} · up to ${m.maxT} tokens${m.streaming ? ' · weights streamed from this browser\'s storage, one layer at a time' : ''}`;
      const sel = this.$('fwdBackend');
      this.$('fwdBackendRow').hidden = !m.canGpu;
      sel.value = ['cpu', 'gpu8'].includes(this.backendPref()) ? this.backendPref() : 'auto';
      if (this.notice) this.$('fwdStatus').textContent += ` · ${this.notice}`;
      this.syncButtons();
      if (this.queued) { const q = this.queued; this.queued = null; this.run(q); }
    } else if (m.type === 'captured') {
      if (m.run !== this.runId) return;  // a step of an earlier run, still in the worker's queue
      this.pending = false;
      // inputs before the activation function: every token on small models, else the latest only
      if (m.pre && this.L * this.F > 65536) for (const h of this.hist) if (h) { h.pre = null; h.preU = null; }
      this.hist[m.pos] = m;
      if (!this.urlDone && m.pos >= this.promptLen - 1) this.applyUrl();
      else if (this.pendingUrl) this.flushUrl();
      this.msEma = this.msEma ? this.msEma * 0.7 + m.ms * 0.3 : m.ms;
      if (m.pos < this.promptLen - 1) this.request(m.pos + 1);          // prompt: no replay
      else { this.beat = m.pos === this.promptLen - 1 && this.speed() === 0 ? this.L + 1 : 0; this.lastBeat = performance.now(); }
      this.renderChips();
    } else if (m.type === 'error') { this.setStatus(m.message); this.pending = false; this.playing = false; this.syncButtons(); }
  }

  request(pos) {
    this.pending = true;
    const b = this.ref, watch = b && pos + 1 < b.rows.length ? b.rows[pos + 1] : undefined;  // A's next word, matched by index
    this.worker.postMessage({ type: 'step', id: pos, run: this.runId, row: this.rows[pos], pos, lensMode: 2, next: this.rows[pos + 1], watch });
  }

  // ---------- control ----------
  speed() { return BEATS[this.$('fwdSpeed').value] ?? 160; }
  run(text, count) {
    if (!this.ready) return;
    if (!this.tok.encode(text).some((i) => this.rowOf[i] >= 0)) return;  // nothing this model can read: keep the run on screen
    this.promptText = text;
    // GPT-2-family runs start after <|endoftext|>, as the model saw each document in training and
    // as the neuron book and Test do, so a value here is the one the book would give
    const eot = this.eotRow(); this.lead = eot != null ? 1 : 0;
    const ids = this.tok.encode(text).slice(0, this.maxT - 2 - this.lead);
    this.rows = ids.map((i) => this.rowOf[i]).filter((r) => r >= 0);
    if (!this.rows.length) return;
    if (this.lead) this.rows.unshift(eot);
    this.promptLen = this.rows.length; this.hist = []; this.viewPos = -1; this.asked.clear(); this.pick = null;
    this.saeRes.clear(); this.saeFeat.clear(); this.lk.clear(); this.rd.clear();
    Object.assign(this.hs, { goal: null, shown: null, n: 0, fresh: true, hover: null, busy: false, id: (this.hs.id || 0) + 1 }); this.flowCache = null; this.runId = (this.runId || 0) + 1;
    // a new text makes old traces meaningless; a rerun of the same text keeps the graph, which is
    // how its knockouts are tested (each trace remembers which model it was made on)
    if (text !== this.graphText) { this.textId = (this.textId || 0) + 1; this.traces.clear(); if (this.graph) { this.graph = null; this.graphView.el.hidden = true; } this.graphView.putBack(); this.graphView.note = ''; this.graphText = text; }
    setTimeout(() => this.syncUrl(), 0);
    this.toGenerate = count ?? +this.$('fwdCount').value;
    this.playing = true; this.beat = 0;
    document.body.classList.add('running');
    this.showPanel(true);
    this.renderChips(); this.syncButtons();
    this.request(0);
    const c = [0, 0, 0];
    const shown = this.rows.slice(this.lead);
    for (const r of shown) for (let k = 0; k < 3; k++) c[k] += this.points[3 * r + k] / shown.length;
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
    if (!last || generated >= this.toGenerate || this.rows.length >= this.maxT - 1) { this.playing = false; this.syncButtons(); if (this.pendingUrl) this.flushUrl(true); this.tryPendingIv(); return; }
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
      const p = +b.dataset.pos; this.viewPos = this.viewPos === p ? -1 : p; this.renderChips(); this.syncUrl();
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
      if (e.target.closest('.x')) { this.nrn = null; nrn.hidden = true; this.syncUrl(); return; }
      const fb = e.target.closest('[data-f]');
      if (fb && this.nrn && this.nrn.kind === 'feat') { const j = +fb.dataset.f; this.nrn.sel = this.nrn.sel === j ? -1 : j; this.nrnKey = ''; return; }
      const ko = e.target.closest('[data-ko]');
      if (ko) { this.toggleKO(ko.dataset.ko, ko.dataset.key); return; }
      const st = e.target.closest('[data-steer]');
      if (st) {
        const [s, a, pre] = st.dataset.steer.split(','), t = this.nrnTop;
        const add = a === 'max' ? (t && t.key === st.dataset.key ? t.top : NaN) : +a;
        if (Number.isFinite(add)) this.steer(st.dataset.key, +s, add, st.dataset.label, +pre || 0);
        return;
      }
      const xb = e.target.closest('[data-expand]');
      if (xb) { const [kind, k] = xb.dataset.expand.split(':'); this.openPassage(kind, +k); return; }
      if (e.target.closest('[data-retry]') && this.nrn) { this.bookData.delete(`${this.nrn.l}:${this.nrn.sel}`); this.nrnKey = ''; return; }
      if (e.target.closest('[data-blind]')) { this.blind = e.target.closest('[data-blind]').checked; try { localStorage.setItem('lf-blind', this.blind ? '1' : '0'); } catch {} this.nrnKey = ''; return; }
      const go = e.target.closest('[data-goto]');
      if (go) { const [l, i] = go.dataset.goto.split(':').map(Number); this.gotoNeuron(l, i); return; }
      if (e.target.closest('[data-inputview]')) { this.showInput = !this.showInput; this.nrnKey = ''; return; }
      if (e.target.closest('[data-graph]')) { this.openGraph(); return; }
      const tr = e.target.closest('[data-trail]');
      if (tr) { this.trailGo(tr.dataset.trail === 'back' ? this.trailIx - 1 : tr.dataset.trail === 'fwd' ? this.trailIx + 1 : +tr.dataset.trail); return; }
      if (e.target.closest('[data-notes="export"]')) { this.exportNotes(); return; }
      if (e.target.closest('[data-mine]')) { this.showMine = !this.showMine; this.nrnKey = ''; return; }
      const bb = e.target.closest('[data-band]');
      if (bb) { const k = +bb.dataset.band; this.band = this.band && this.band.key === `${this.nrn.l}:${this.nrn.sel}` && this.band.k === k ? null : { key: `${this.nrn.l}:${this.nrn.sel}`, k }; this.nrnKey = ''; return; }
      if (e.target.closest('[data-notes="import"]')) { this.importNotes(); return; }
      const br = e.target.closest('[data-bandretry]');
      if (br) { this.bandWin.delete(br.dataset.bandretry); this.book.cache.delete(`b${br.dataset.bandretry}`); this.nrnKey = ''; return; }
      if (e.target.closest('[data-probe]')) { this.runProbe(nrn.querySelector('#nrnProbe').value); return; }
      const b = e.target.closest('[data-n]'); if (!b || !this.nrn) return;
      const i = +b.dataset.n;
      if (this.nrn.sel === i) { this.nrn.sel = -1; this.nrnKey = ''; this.syncUrl(); } else this.gotoNeuron(this.nrn.l, i);
    });
    nrn.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && e.target.id === 'nrnProbe') { e.preventDefault(); this.runProbe(e.target.value); }
      if (e.key === 'Enter' && e.target.id === 'nrnLabel') { e.preventDefault(); e.target.blur(); }
      if (e.key === 'Enter' && e.target.id === 'nrnGoto') {  // "L5 880", "5:880", "5.880", "#880" (this layer)
        e.preventDefault(); const m = /^\s*(?:L?(\d+)\s*[ .:#/]\s*#?)?(\d+)\s*$/i.exec(e.target.value);
        const l = m && m[1] != null ? +m[1] : this.nrn ? this.nrn.l : 0, i = m ? +m[2] : -1;
        this.gotoMsg = !m ? 'Write a layer and a neuron, like L5 880.' : l >= this.L ? `This model has layers 0 to ${this.L - 1}.` : i >= this.F ? `Each layer has neurons 0 to ${this.F - 1}.` : '';
        if (!this.gotoMsg) { e.target.value = ''; this.gotoNeuron(l, i); } else this.nrnKey = '';
      }
    });
    nrn.addEventListener('input', (e) => {  // a neuron's name or note is saved as it is typed, so a redraw can't lose it
      if (!this.nrn || this.nrn.sel == null || this.nrn.sel < 0 || !/^nrn(Label|Note)$/.test(e.target.id)) return;
      clearTimeout(this.noteT); const l = this.nrn.l, i = this.nrn.sel, f = e.target.id === 'nrnLabel' ? 'label' : 'note', v = e.target.value;
      this.noteT = setTimeout(() => this.setNote(l, i, f, v, true), 250);
    });
    nrn.addEventListener('change', (e) => {  // a neuron's name or note, saved as it is written
      if (!this.nrn || this.nrn.sel == null || this.nrn.sel < 0) return;
      if (/^nrn(Label|Note)$/.test(e.target.id)) clearTimeout(this.noteT);  // the trimmed text wins over a pending keystroke save
      if (e.target.id === 'nrnLabel' || e.target.id === 'nrnNote') this.setNote(this.nrn.l, this.nrn.sel, e.target.id === 'nrnLabel' ? 'label' : 'note', e.target.value);
    });
    addEventListener('keydown', (e) => {  // [ and ] walk the trail of neurons read
      if (nrn.hidden || /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName) || e.metaKey || e.ctrlKey) return;
      if (e.key === '[') { e.preventDefault(); this.trailGo(this.trailIx - 1); }
      if (e.key === ']') { e.preventDefault(); this.trailGo(this.trailIx + 1); }
    });
    addEventListener('storage', (e) => {  // another tab named a neuron (or revealed one) on this model
      const k = this.notesKey();
      if (!this.ready || (e.key !== k && e.key !== `${k}:revealed`)) return;  // not another model's notes
      const keep = this.revealed; this.loadNotes(); for (const k of keep) this.revealed.add(k);
      this.nrnKey = ''; if (this.graph && this.graph.data) this.graphView.render(this);
    });
    this.graphView = new NeuronGraph(this);
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
  tokName(p) { return p < this.lead ? 'the document start' : `token ${p + 1 - this.lead}`; }  // your text counts from 1
  captured() { let n = 0; while (n < this.rows.length && this.hist[n]) n++; return n; }
  // Ask the worker for the projection when there are new states: at once when a run is done, at
  // most every 1.5 s while one plays (each refit runs in the worker, a few hundred ms at most).
  askHidden(now = false) {
    const hs = this.hs, n = this.captured();
    if (!hs.on || !this.worker || hs.busy || !n || n === hs.n) return;
    const t = performance.now();
    if (!now && (this.playing || this.pending) && t - hs.t < 1500) return;
    hs.busy = true; hs.t = t; hs.id++;
    this.worker.postMessage({ type: 'hidden', run: this.runId, id: hs.id, n, fresh: !!hs.fresh });
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
      if (!lp && !this.asked.has(`lens:${p}:${dp}`)) { this.asked.add(`lens:${p}:${dp}`); this.worker.postMessage({ type: 'lens', run: this.runId, pos: p, depth: dp }); }
      const txt = `${word(this.rows[p])} · ${this.tokName(p)} · ${dp === 0 ? 'embedding' : `after layer ${dp - 1}`}${lp ? ` · would say ${word(lp.rows[0])} ${(lp.probs[0] * 100).toFixed(0)}%` : ''}`;
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
      if (p < this.lead) return `<button type="button" class="${cls} doc" data-pos="${p}" title="&lt;|endoftext|&gt;: the start of a document, added in front as GPT-2 saw documents in training and as the neuron book reads them">⟨doc⟩</button>`;
      return `<button type="button" class="${cls}" data-pos="${p}">${esc(this.disp(this.tokens[r]))}</button>`;
    }).join('');
    this.$('fwdChips').innerHTML = html;
    const el = this.$('fwdChips'); el.scrollLeft = el.scrollWidth;
  }

  // The answer as it forms: prompt dim, generated text bright, the next guess while it is being read out.
  renderOutput() {
    if (!this.rows.length) return;
    const t = (r) => esc(this.tokens[r]);
    const p = this.rows.slice(this.lead, this.promptLen).map(t).join('');
    const b = this.ref;
    let dv = -1;  // the first generated token that differs from the reference run
    if (b) for (let p = this.promptLen; p < this.rows.length; p++) if (this.rows[p] !== b.rows[p]) { dv = p; break; }
    const sameText = b && b.promptLen === this.promptLen && b.rows.slice(0, b.promptLen).every((r, j) => r === this.rows[j]);
    const g = this.rows.slice(this.promptLen).map((r, j) => (this.promptLen + j === dv ? `<span class="dv">${t(r)}</span>` : t(r))).join('');
    const last = this.hist[this.rows.length - 1];
    const forming = this.playing && last && this.beat >= 1 && last.lens[this.L]
      ? `<span class="now">${t((last.lens[Math.min(this.beat, this.L)] || last.lens[this.L]).rows[0])}</span>` : '';
    const baseLine = !b ? '' : `<span class="base"><b>${b.label}${sameText && dv < 0 && !this.playing ? ', same answer' : ''}:</b> ${sameText ? '' : `<span class="p">${b.rows.slice(this.lead, b.promptLen).map(t).join('')}</span>`}${b.rows.slice(b.promptLen).map(t).join('')}${b.rows.length && b.promptLen !== this.promptLen ? ' <i>(a different number of tokens: positions are matched by index)</i>' : ''}</span>`;
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
    const s = this.ko[kind];
    if (kind === 'neurons') { const c = s.get(key); if (c && c.s === 0 && !c.add && !c.pre) s.delete(key); else s.set(key, { s: 0, add: 0, pre: 0, label: 'knocked out' }); }
    else if (s.has(key)) s.delete(key); else s.add(key);
    this.ivVer++; this.nrnKey = ''; this.syncIv();
  }
  // Steer neuron 'l:i': activation * s + add at every position (the same choice again restores it).
  steer(key, s, add, label, pre = 0) {
    const cur = this.ko.neurons.get(key);
    if (cur && cur.s === s && cur.add === add && (cur.pre || 0) === pre) this.ko.neurons.delete(key); else this.ko.neurons.set(key, { s, add, pre, label });
    this.ivVer++; this.nrnKey = ''; this.syncIv();
  }
  koAtGaze() { const f = this.pinnedHead || this.focusHead; if (f && this.canAttribute) this.toggleKO('heads', `${f.l}:${f.h}`); }
  ivSpec() {
    const sp = (k) => [...this.ko[k]].map((x) => x.split(':').map(Number));
    const neurons = [...this.ko.neurons].map(([k, v]) => { const [l, i] = k.split(':').map(Number); return [l, i, v.s, v.add, v.pre || 0]; });
    return { heads: sp('heads').map(([l, h]) => [l, h, 0]), mlps: sp('mlps').map(([l]) => [l, 0]), neurons };
  }
  koCount() { return this.ko.heads.size + this.ko.mlps.size + this.ko.neurons.size; }
  syncIv() {
    const n = this.koCount(), applied = this.ivRun ? JSON.stringify(this.ivRun) === JSON.stringify(this.ivSpec()) : n === 0;
    const el = this.$('fwdIvCount');
    const steered = [...this.ko.neurons.values()].filter((v) => v.s !== 0 || v.add || v.pre).length;
    el.textContent = n ? `${n - steered ? `${n - steered} knocked out` : ''}${n - steered && steered ? ', ' : ''}${steered ? `${steered} steered` : ''}${applied ? '' : ' · rerun'}` : 'No changes';
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
    if (any && this.$('fwdSample').value !== 'greedy') this.setStatus('Sample decoding: the rerun\'s words can differ from the unchanged run\'s by chance, not only by your change. Choose Greedy to compare the two.');
  }
  clearKO() {
    for (const s of Object.values(this.ko)) s.clear();
    if (this.graphView) { this.graphView.saved = null; this.graphView.note = ''; }  // nothing set aside to bring back
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
    if (!h.lens[depth] && !this.asked.has(k)) { this.asked.add(k); this.worker.postMessage({ type: 'lens', run: this.runId, pos, depth }); }
    if (this.attrOn) this.ensureAttr(h, pos, this.attrTarget(h, pos));
  }
  ensureAttr(h, pos, row) {
    if (row == null || !h.targets || h.targets.includes(row)) return;
    const k = `attr:${pos}:${row}`;
    if (!this.asked.has(k)) { this.asked.add(k); this.worker.postMessage({ type: 'attr', run: this.runId, pos, row }); }
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
    if (!r && !this.asked.has(`sae:${depth}:${pos}`)) { this.asked.add(`sae:${depth}:${pos}`); this.worker.postMessage({ type: 'sae', run: this.runId, depth, pos }); }
    // over the tokens run so far (later ones aren't this run's yet), asked again as the run grows
    const cap = this.captured();
    if (sel >= 0 && cap && (!f || f.acts.length < cap) && !this.asked.has(`saeF:${depth}:${sel}:${cap}`)) { this.asked.add(`saeF:${depth}:${sel}:${cap}`); this.worker.postMessage({ type: 'saeFeat', run: this.runId, depth, j: sel, n: cap }); }
    const key = `feat|${depth}|${pos}|${sel}|${!!r}|${!!f}|${this.saeMsg}|${this.npVer}|${this.hist.length}`;
    if (key === this.nrnKey) return;
    this.nrnKey = key;
    const word = (row) => esc(this.disp(this.tokens[row]));
    const info = (html) => `<span class="info" tabindex="0" role="note">i<span class="tip left">${html}</span></span>`;
    const where = depth === 0 ? 'The embedding' : `After layer ${depth - 1}`;  // as the field's row labels say
    let html = `<header><b>${where} · features</b>${info(`<b>Features: a model of the model.</b> A sparse autoencoder (SAE) is a second, small network, trained to rewrite the model's state at one layer (the residual stream, which has no neurons of its own) as a sum of a few <b>features</b> from a learned dictionary of ${spec.n.toLocaleString()} directions. Useful for asking what is in this token's state; to see how the network itself computes, read its neurons first (handbook, Chapter 11). This dictionary was trained by ${spec.by} (${spec.license}). The descriptions come from Neuronpedia, written by an AI model from each feature's strongest examples: good guesses, not ground truth. Derived: the SAE is a model of the model, and the line below says how much of this token's state it explains.`)}<button type="button" class="x" aria-label="Close">✕</button></header>`;
    html += `<p class="note">At “${word(this.rows[pos])}” (${this.tokName(pos)}) · ${spec.n.toLocaleString()} features, streamed once from Hugging Face</p>`;
    if (!r) html += `<p class="note">${esc(this.saeMsg || 'Working out the features…')}</p>`;
    else {
      html += `<div class="eff">${r.l0} of ${spec.n.toLocaleString()} features are active. Together they rebuild this token's state to a cosine of <b>${r.cos.toFixed(3)}</b>, missing <b>${(r.relErr * 100).toFixed(0)}%</b> of its size. ${info('How faithful the features are here. A cosine of 1 means the rebuilt state points exactly the same way as the real one; the miss is the size of what the features leave out, relative to the state. Whatever they miss is invisible in this view.')}</div>`;
      if (pos < this.lead) html += `<p class="note warn">This is the document-start token, where the model parks attention: its features say little about the text.</p>`;
      else if (pos === 0) html += `<p class="note warn">The first token is unusual: this SAE was trained with a start-of-text token in front, which this model's run doesn't have, so read the first token's features with care.</p>`;
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
        let fmax = 1e-6; for (let p = this.lead; p < f.acts.length; p++) if (f.acts[p] > fmax) fmax = f.acts[p];  // not the document start's outsized state
        html += `<div class="text">${this.rows.map((row, p) => { if (p < this.lead) return '<span class="doc" title="the start of the document">⟨doc⟩</span>'; const x = p < f.acts.length ? f.acts[p] : NaN, t = Number.isNaN(x) ? 0 : Math.min(1, x / fmax); return `<span class="${p === pos ? 'here' : ''}" style="background:rgba(53,224,230,${(t * 0.75).toFixed(2)})" title="${Number.isNaN(x) ? 'not run yet' : x.toFixed(3)}">${esc(this.tokens[row])}</span>`; }).join('')}</div>`;
        html += `<div class="sec">What it writes ${info('The words this feature\'s direction pushes <span class="c">up</span> and <span class="a">down</span> most when it fires, through the final normalisation and output layer; what later layers do with it is not counted. Derived.')}</div>`;
        if (f.up) html += `<div class="words up">${f.up.map((w) => `<span>${word(w.row)}</span>`).join('')}</div><div class="words down" style="margin-top:6px">${f.down.map((w) => `<span>${word(w.row)}</span>`).join('')}</div>`;
      }
      html += `<p style="margin-top:10px"><a class="np" href="https://www.neuronpedia.org/${np.model}/${np.source}/${sel}" target="_blank" rel="noopener">Open feature #${sel} on Neuronpedia ↗</a></p>`;
    } else if (r) html += '<p class="note" style="margin-top:12px">Pick a feature to see where it fires across the text and what it writes.</p>';
    this.redraw(el, html);
  }
  // Replace a drawer's contents, keeping its scroll and the keyboard's place: a field being typed
  // in keeps its text and caret, a focused button its focus (found again by its data attributes).
  redraw(el, html) {
    const scroll = el.scrollTop, fa = document.activeElement, inside = fa && fa !== el && el.contains(fa);
    const keep = inside && fa.id ? { id: fa.id, v: fa.value, s: fa.selectionStart, e: fa.selectionEnd } : null;
    const attrs = inside && !fa.id ? [...fa.attributes].filter((a) => a.name.startsWith('data-')).map((a) => `[${a.name}="${CSS.escape(a.value)}"]`).join('') : '';
    el.innerHTML = html; el.hidden = false; el.scrollTop = scroll;
    if (keep) { const b = el.querySelector(`#${keep.id}`); if (b) { b.value = keep.v; b.focus(); try { b.setSelectionRange(keep.s, keep.e); } catch {} } }
    else if (attrs) { const b = el.querySelector(`${fa.tagName.toLowerCase()}${attrs}`); if (b) b.focus({ preventScroll: true }); }
  }

  // ---------- neuron inspector ----------
  openNeurons(l) {
    this.nrn = this.nrn && this.nrn.l === l ? null : { l, sel: -1 };
    this.$('nrn').hidden = !this.nrn; this.nrnKey = ''; this.syncUrl();
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
    const bk = sel >= 0 && this.book ? this.bookData.get(`${l}:${sel}`) : null;
    const key = `${l}|${sel}|${pos}|${target}|${!!gain}|${this.nvocab.has(`${l}:${sel}`)}|${this.hist.length}|${this.ivVer}|${!!this.book}|${bk && bk.ready}|${this.lk.has(`${pos}:${l}:${sel}`)}|${this.rd.has(`${pos}:${l}:${sel}`)}|${this.probe && this.probe.result ? this.probe.id : ''}|${this.showInput}|${this.trailIx}|${this.trail.length}|${this.blind}|${this.expand ? `${this.expand.id}${!!this.expand.result}` : ''}|${this.notes.size}|${this.band ? this.band.k : ''}|${this.showMine}|${(this.bandWin.get(`${l}:${sel}`) || {}).ready}`;
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
    const row = (i, m) => { const e = effOf(i), lb = this.label(l, i); return `<li><button type="button" data-n="${i}" aria-current="${i === sel}"${lb ? ` title="${esc(lb)}" class="named"` : ''}><span>${koN(i) ? '✕' : '#'}${i}</span>${bar(act[i], m, '#35e0e6')}<span class="num ${e == null ? '' : e >= 0 ? 'pos' : 'neg'}">${e == null ? act[i].toFixed(2) : fmt(e)}</span></button></li>`; };
    let active = 0, amax = 1e-6; for (let i = 0; i < F; i++) { const x = Math.abs(act[i]); if (x > 1) active++; if (x > amax) amax = x; }
    const byAct = Array.from({ length: F }, (_, i) => i).sort((p, q) => Math.abs(act[q]) - Math.abs(act[p])).slice(0, 8);
    let html = `<header><b>Layer ${l} · MLP</b>${info(`<b>The MLP is the part of each layer that works on one token at a time.</b> This layer has ${F.toLocaleString()} neurons. Each one reads the model's running state and fires by some amount: its <b>activation</b>, <span class="c">cyan</span> when positive, <span class="v">violet</span> when negative. ${gelu ? 'Here a neuron\'s activation is one projection passed through GELU.' : 'Here a neuron\'s activation is a gated product (SwiGLU): one projection passed through SiLU, times another.'} Whatever a neuron fires, it writes back along its own fixed direction, its <b>output weights</b>. Real: every number here comes straight from the model.`)}<button type="button" class="x" aria-label="Close the neuron inspector">✕</button></header>`;
    html += this.mapBar();
    html += `<div class="fwd-row" style="margin:4px 0 6px"><button type="button" data-ko="mlps" data-key="${l}" aria-pressed="${this.ko.mlps.has(`${l}`)}">${this.ko.mlps.has(`${l}`) ? 'Restore' : 'Knock out'} this MLP</button>${this.ivRun || this.koCount() ? '<span class="note">then Rerun, under Playback</span>' : ''}</div>`;
    html += `<p class="note">At “${word(this.rows[pos])}” (${this.tokName(pos)}) · ${active.toLocaleString()} of ${F.toLocaleString()} neurons above 1.0</p>`;
    if (a) html += `<div class="eff">This layer's MLP pushed ${tw} by <b class="${a.mlps[l] >= 0 ? 'pos' : 'neg'}">${fmt(a.mlps[l])}</b>, out of a score of ${a.logit.toFixed(2)} above the average word.</div>`;
    if (a && this.caps.links) html += `<div class="fwd-row" style="margin:4px 0 2px"><button type="button" data-graph>Trace the neurons behind ${tw}</button>${info('A graph of the model\'s own neurons that carry this word at this token: which ones push it directly, which feed those, and through which attention heads other tokens come in. Built from exact splits of each neuron\'s input; knock a group out and Rerun to test it.')}</div>`;
    html += `<div class="sec">Most active here ${info('The neurons firing hardest at this token, either way. Firing hard is not the same as mattering: a neuron can fire strongly and write almost nothing toward the word. The number on the right is its effect on the word when that is known, else its activation.')}</div><ol>${byAct.map((i) => row(i, amax)).join('')}</ol>`;
    if (gain && a) {
      const eff = Array.from({ length: F }, (_, i) => [effOf(i), i]).sort((p, q) => q[0] - p[0]);
      const emax = Math.max(1e-6, Math.abs(eff[0][0]), Math.abs(eff[F - 1][0]));
      const effRow = ([e, i]) => `<li><button type="button" data-n="${i}" aria-current="${i === sel}"${this.label(l, i) ? ` title="${esc(this.label(l, i))}" class="named"` : ''}><span>${koN(i) ? '✕' : '#'}${i}</span>${bar(e, emax, '#35e0e6')}<span class="num ${e >= 0 ? 'pos' : 'neg'}">${fmt(e)}</span></button></li>`;
      html += `<div class="sec">Toward ${tw} ${info(`Each neuron's <b>direct effect</b> on the word's score: its activation times how much its output weights point at that word, through the final normalisation. Added up over all ${F.toLocaleString()} neurons, plus the layer's bias, this is the layer's share above. Direct path only: what later layers make of it is counted in those layers.`)}</div><ol>${eff.slice(0, 6).map(effRow).join('')}</ol>`;
      html += `<div class="sec">Against ${tw}</div><ol>${eff.slice(-4).reverse().map(effRow).join('')}</ol>`;
    } else if (target != null) html += `<p class="note">Working out each neuron's effect on ${tw}…</p>`;
    if (sel >= 0) html += this.neuronSection(v, l, sel, act, effOf, tw, info, fmt, word);
    else html += '<p class="note" style="margin-top:12px">Pick a neuron to read it: where it fires, its strongest examples on both sides, what feeds it, what reads it, and what it writes.</p>';
    this.redraw(el, html);
  }



  // The top of the drawer: the trail of neurons read, a go-to box, and the reader's own index.
  mapBar() {
    const crumbs = this.trail.map((t, k) => ({ t, k })).slice(Math.max(0, this.trailIx - 4), this.trailIx + 3);
    let html = `<div class="trail"><button type="button" class="mini" data-trail="back" ${this.trailIx > 0 ? '' : 'disabled'} title="Back ([)">‹</button><button type="button" class="mini" data-trail="fwd" ${this.trailIx < this.trail.length - 1 ? '' : 'disabled'} title="Forward (])">›</button>${crumbs.map(({ t, k }) => `<button type="button" class="crumb" data-trail="${k}" aria-current="${k === this.trailIx}">${this.nname(t.l, t.i)}</button>`).join('<span class="sep">›</span>')}`;
    html += `<input id="nrnGoto" class="goto" type="text" placeholder="go to L5 880" aria-label="Go to a neuron: layer and number"><button type="button" class="mini" data-mine aria-pressed="${this.showMine}" title="The neurons you have named in this model">yours (${this.notes.size})</button><label class="check blind mini" title="Hide each neuron's examples, shape, neighbours, words and activations until you have written a reading and tested it (kept on until you turn it off)"><input type="checkbox" data-blind ${this.blind ? 'checked' : ''}> blind</label><span class="notes-io"><button type="button" class="mini" data-notes="export" title="Download every neuron you have named, as one JSON file (with a Markdown summary inside)" ${this.notes.size ? '' : 'disabled'}>export</button><button type="button" class="mini" data-notes="import" title="Load notes exported before">import</button></span></div>${this.gotoMsg ? `<p class="note warn">${esc(this.gotoMsg)}</p>` : ''}`;
    if (this.showMine) {
      const list = [...this.notes].map(([k, v]) => { const [l, i] = k.split(':').map(Number); return { l, i, v }; }).sort((a, b) => a.l - b.l || a.i - b.i);
      html += list.length ? `<ol class="mine">${list.map(({ l, i, v }) => `<li><button type="button" class="link" data-goto="${l}:${i}">L${l} #${i}</button> <span>${esc(v.label || '')}</span>${v.note ? `<span class="nn">${esc(v.note.length > 80 ? v.note.slice(0, 79) + '…' : v.note)}</span>` : ''}</li>`).join('')}</ol>` : '<p class="note">No neurons named yet: give one a name below its number, and it is listed here, in the graph and in every list.</p>';
    }
    return html;
  }

  // A few random tokens from one band of a neuron's range (the book's band samples), each with the
  // text before it: where the meaning radiates out from its strongest examples.
  bandSection(l, n, k, d, tokOf, shade) {
    const key = `${l}:${n}`;
    let st = this.bandWin.get(key);
    if (!st) {
      st = { ready: false, list: [] }; this.bandWin.set(key, st);
      this.book.bands(l, n).then((all) => Promise.all(all.map(async (b) => ({ ...b, ...(await this.book.window(b.at)) }))))
        .then((list) => { st.list = list; st.ready = true; this.nrnKey = ''; })
        .catch((e) => { st.err = e.message; st.ready = true; this.nrnKey = ''; });  // kept until Try again
    }
    const lo = 1 / 2 ** (k + 1), hi = 1 / 2 ** k;
    let html = `<div class="sec">${k === 0 ? 'Above half its max' : `Between ${lo === 1 / 4096 ? '0' : `1/${2 ** (k + 1)}`} and 1/${2 ** k} of its max`}</div>`;
    if (!st.ready) return html + '<p class="note">Reading them…</p>';
    if (st.err) return html + `<p class="note">Couldn't read them (${esc(st.err)}). <button type="button" class="mini" data-bandretry="${key}">Try again</button></p>`;
    const list = st.list.map((b, ix) => ({ ...b, ix })).filter((b) => b.band === k);
    if (!list.length) return html + '<p class="note">No samples kept for this band.</p>';
    const mx = Math.max(d.max, 1e-6), ex = this.expand && this.expand.key === key && this.expand.kind === 'band' ? this.expand : null;
    const li = (b) => `<li><span class="num pos" title="activation">${b.v.toFixed(2)}<small>${(100 * b.v / mx).toFixed(1)}%</small></span><span class="text ex">${b.ids.map((id, j) => (id === this.book.meta.eot ? '<span class="doc">⟨doc⟩</span>' : `<span class="${j === b.k ? 'here' : ''}" style="${j === b.k ? `background:${shade(b.v, mx)}` : ''}">${esc(tokOf(id))}</span>`)).join('')}</span><button type="button" class="mini more" data-expand="band:${b.ix}" title="The whole passage up to here, run through the model, every token shaded">${ex && ex.k === b.ix ? '−' : '↑'}</button></li>${ex && ex.k === b.ix ? this.passageHtml(ex, false, mx, shade) : ''}`;
    return html + `<ol class="exs">${list.map(li).join('')}</ol><p class="note">Random tokens from this band, not its strongest${this.book.meta.band_spacing ? `, no two within ${this.book.meta.spacing || this.book.meta.window} tokens in one document` : '; several may come from one document'}. ↑ shades the whole passage, to see why it fires here.</p>`;
  }
  // An opened passage, shaded token by token (by its input for the other side)
  passageHtml(ex, input, scale, shade) {
    if (!ex.result) return '<li class="passage"><p class="note">Running the passage…</p></li>';
    const r = ex.result;
    return `<li class="passage">${ex.cut ? `<p class="note">The first ${ex.cut} tokens of this document don't fit the browser kernel's window, so this runs on the ${r.rows.length} tokens around the example; its value may differ from the book's.</p>` : ''}<div class="text">${r.rows.map((rw, p) => `<span class="${p === ex.at - 1 ? 'here' : ''}" style="background:${shade(input ? r.pres[p] : r.acts[p], scale)}" title="${(input ? r.pres[p] : r.acts[p]).toFixed(3)}">${esc(this.tokens[rw])}</span>`).join('')}</div></li>`;
  }
  // ---------- reading one neuron ----------
  gotoNeuron(l, i, push = true) {
    this.nrn = { l, sel: i }; this.$('nrn').hidden = false; this.nrnKey = ''; this.gotoMsg = '';
    if (!this.nvocab.has(`${l}:${i}`)) this.worker.postMessage({ type: 'nvocab', l, i });
    if (push) {
      const cur = this.trail[this.trailIx];
      if (!cur || cur.l !== l || cur.i !== i) { this.trail = this.trail.slice(0, this.trailIx + 1); this.trail.push({ l, i }); this.trailIx = this.trail.length - 1; }
    }
    this.$('nrn').scrollTop = 0;
    this.syncUrl();
  }
  trailGo(ix) {
    if (ix < 0 || ix >= this.trail.length) return;
    this.trailIx = ix; const t = this.trail[ix];
    this.gotoNeuron(t.l, t.i, false);
  }
  // ---------- the reader's notes ----------
  notesKey() { return `lf-notes:${this.repo || this.modelName || this.DATA}`; }
  loadNotes() {
    this.notes = new Map(); this.revealed = new Set();
    try { for (const [k, v] of Object.entries(JSON.parse(localStorage.getItem(this.notesKey()) || '{}'))) this.notes.set(k, v); } catch {}
    try { for (const k of JSON.parse(localStorage.getItem(`${this.notesKey()}:revealed`) || '[]')) this.revealed.add(k); } catch {}
  }
  // a neuron read blind stays revealed once a reading came before a test, whatever is edited after
  // (merged with what another tab may have saved since, never overwriting it)
  reveal(k) {
    if (this.revealed.has(k)) return; this.revealed.add(k);
    try { const key = `${this.notesKey()}:revealed`, cur = new Set(JSON.parse(localStorage.getItem(key) || '[]')); for (const x of this.revealed) cur.add(x); localStorage.setItem(key, JSON.stringify([...cur])); this.revealed = cur; } catch {}
  }
  // Saves only the neurons changed here, onto what is stored now: another tab on the same model
  // keeps its own names (the storage event below brings them into this one).
  saveNotes(keys) {
    try {
      const cur = JSON.parse(localStorage.getItem(this.notesKey()) || '{}');
      for (const k of keys) { const v = this.notes.get(k); if (v) cur[k] = v; else delete cur[k]; }
      localStorage.setItem(this.notesKey(), JSON.stringify(cur));
      this.notes = new Map(Object.entries(cur));
    } catch (e) { this.setStatus(`Couldn't save your notes in this browser (${e.message}): export them to keep them.`); }
  }
  setNote(l, i, field, text, typing = false) {
    const k = `${l}:${i}`, n = { ...(this.notes.get(k) || {}) };
    n[field] = typing ? text : text.trim(); n.at = new Date().toISOString();
    if (!n.label && !n.note) this.notes.delete(k); else this.notes.set(k, n);
    this.saveNotes([k]); if (!typing) this.nrnKey = '';
    if (this.graph && this.graph.data) this.graphView.render(this);
  }
  label(l, i) { const n = this.notes.get(`${l}:${i}`); return n && n.label ? n.label : ''; }
  // "L5 #880 “label”" for lists; the label is what the reader has called it
  nname(l, i, withLayer = true) { const lb = this.label(l, i); return `${withLayer ? `L${l} ` : ''}#${i}${lb ? ` “${esc(lb.length > 28 ? lb.slice(0, 27) + '…' : lb)}”` : ''}`; }
  exportNotes() {  // one file: the notes as JSON, with a readable Markdown summary inside
    if (!this.notes.size) { this.setStatus('No named neurons to export yet.'); return; }
    const model = this.repo || this.modelName, list = [...this.notes].map(([k, v]) => { const [l, i] = k.split(':').map(Number); return { layer: l, neuron: i, ...v }; }).sort((a, b) => a.layer - b.layer || a.neuron - b.neuron);
    const markdown = `# Neuron notes: ${model}\n\n${list.map((n) => `- **L${n.layer} #${n.neuron}**${n.label ? ` ${n.label}` : ''}${n.note ? `\n  ${n.note.replace(/\n/g, '\n  ')}` : ''}`).join('\n')}\n`;
    const json = JSON.stringify({ schema: 'latent-fovea-notes/1', model, layers: this.L, neurons_per_layer: this.F, exported: new Date().toISOString(), neurons: list, markdown }, null, 2);
    const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([json], { type: 'application/json' }));
    a.download = `neuron-notes-${String(model).replace(/[^\w.-]+/g, '_')}.json`; document.body.append(a); a.click(); setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
    this.setStatus(`Exported ${list.length} named neuron${list.length > 1 ? 's' : ''}.`);
  }
  importNotes() {  // merges by the newer note; refuses notes made on a model of another shape
    const inp = document.createElement('input'); inp.type = 'file'; inp.accept = '.json,application/json';
    inp.onchange = async () => {
      try {
        const d = JSON.parse(await inp.files[0].text());
        if (!Array.isArray(d.neurons)) throw new Error('no neurons in this file');
        if ((d.layers && d.layers !== this.L) || (d.neurons_per_layer && d.neurons_per_layer !== this.F)) throw new Error(`these notes are for a model with ${d.layers} layers of ${d.neurons_per_layer} neurons (${d.model}); this one has ${this.L} of ${this.F}`);
        const model = this.repo || this.modelName;
        let n = 0; const changed = [];
        for (const x of d.neurons) {
          if (!Number.isInteger(x.layer) || !Number.isInteger(x.neuron) || x.layer < 0 || x.layer >= this.L || x.neuron < 0 || x.neuron >= this.F) continue;
          if (!String(x.label || '').trim() && !String(x.note || '').trim()) continue;  // nothing written
          const k = `${x.layer}:${x.neuron}`, cur = this.notes.get(k);
          if (cur && cur.at && x.at && cur.at > x.at) continue;  // keep the newer
          this.notes.set(k, { label: String(x.label || '').slice(0, 80), note: String(x.note || '').slice(0, 4000), at: x.at || new Date().toISOString() }); n++; changed.push(k);
        }
        this.saveNotes(changed); this.nrnKey = ''; if (this.graph && this.graph.data) this.graphView.render(this);
        this.setStatus(`Imported ${n} neuron note${n === 1 ? '' : 's'}${d.model && d.model !== model ? ` (made on ${d.model})` : ''}.`);
      } catch (e) { this.setStatus(`Couldn't import those notes: ${e.message}`); }
    };
    inp.click();
  }
  // ---------- a shareable address ----------
  // #<model>?ask=<text>&gen=<tokens>[&dec=sample]&p=<token>&n=<layer>.<neuron>&g=1&gk=<0|1>&w=<word row>
  // and the interventions in force at the last Rerun: iv=<l>:<i>:<scale>:<add>:<pre>_…, h=<l>.<h>_…,
  // m=<l>_…, so a link reproduces a finding. gk says whether the graph was traced on the changed
  // model (1) or the unchanged one (0).
  static encodeIv(neurons) { return neurons.map((x) => x.map((y) => +(+y).toFixed(4)).join(':')).join('_'); }
  static decodeIv(s) {
    return (s || '').split('_').filter(Boolean).map((t) => t.split(':').map(Number))
      .filter((x) => x.length >= 2 && x.length <= 5 && x.every(Number.isFinite) && Number.isInteger(x[0]) && Number.isInteger(x[1]));
  }
  syncUrl() {
    if (!this.urlDone) return;  // the shared address is still being applied
    try {
      const u = new URL(location.href), q = u.searchParams;
      for (const k of SHARED_KEYS) q.delete(k);
      if (this.modelHash) u.hash = this.modelHash;  // the model it was made on
      if (this.promptText && this.promptText.length <= 400) {
        q.set('ask', this.promptText);
        if (this.toGenerate != null) q.set('gen', this.toGenerate);
        if (this.$('fwdSample').value !== 'greedy') q.set('dec', this.$('fwdSample').value);
        const v = this.view(), sel = this.nrn && this.nrn.sel >= 0 ? this.nrn : null;
        if (v.h && this.viewPos >= 0) q.set('p', v.pos);
        if (sel) q.set('n', `${sel.l}.${sel.sel}`);
        if (this.graph && !this.graphView.el.hidden) { q.set('g', '1'); q.set('gk', this.graph.ko ? '1' : '0'); q.set('w', this.graph.row); q.set('p', this.graph.pos); }
        const sp = this.ivRun || null;
        if (sp) {
          if (sp.neurons.length) q.set('iv', ForwardView.encodeIv(sp.neurons));
          if (sp.heads.length) q.set('h', sp.heads.map(([l, h]) => `${l}.${h}`).join('_'));
          if (sp.mlps.length) q.set('m', sp.mlps.map(([l]) => l).join('_'));
        }
      } else if (this.promptText && !this.longNoted) { this.longNoted = true; this.setStatus('This prompt is over 400 characters, so the page\'s address can\'t carry it: a link won\'t reopen it.'); }
      history.replaceState(history.state, '', u);
    } catch {}
  }
  // Once the shared text has run: restore its interventions (and rerun), then the token, the
  // neuron and the graph, each when the run has reached it. A graph traced on the unchanged
  // model (gk=0) is opened first, and the interventions applied after it is traced.
  applyUrl() {
    this.urlDone = true;
    const q = this.shared;
    const num = (x) => x.split('.').map(Number);
    const iv = ForwardView.decodeIv(q.get('iv')), hs = (q.get('h') || '').split('_').filter(Boolean).map(num), ms = (q.get('m') || '').split('_').filter(Boolean).map(Number);
    const p = q.get('p'), n = q.get('n'), g = q.get('g') === '1', w = q.get('w') != null ? +q.get('w') : null;
    const changes = (iv.length || hs.length || ms.length) && !this.ivRun;
    // the changes wait for the unchanged run to finish writing, so the rerun writes as much and
    // the two can be compared word by word
    if (changes && g && q.get('gk') === '0') {  // trace on this unchanged run, then change the model
      this.pendingIv = { iv, hs, ms, graph: true };
      this.shared = new URLSearchParams([...q].filter(([k]) => ['p', 'n'].includes(k)));
      this.pendingUrl = { p: p != null ? +p : null, n: null, g, w };
      this.flushUrl();
      return;
    }
    if (changes) {
      this.pendingIv = { iv, hs, ms };
      this.shared = new URLSearchParams([...q].filter(([k]) => !['iv', 'h', 'm'].includes(k)));  // applied: the rerun brings the rest
      this.tryPendingIv();
      return;
    }
    this.pendingUrl = { p: p != null ? +p : null, n, g, w };
    this.flushUrl();
  }
  tryPendingIv() {
    const u = this.pendingIv;
    if (!u || this.playing || (u.graph && !(this.graph && this.graph.data))) return;
    this.pendingIv = null; this.applyIv(u);
  }
  applyIv({ iv, hs, ms }) {
    for (const [l, i, sc = 0, add = 0, pre = 0] of iv) if (l >= 0 && l < this.L && i >= 0 && i < this.F) this.ko.neurons.set(`${l}:${i}`, { s: sc, add, pre, label: sc === 0 && !add && !pre ? 'knocked out' : sc === 0 ? `held at ${add} (shared link)` : `steered (shared link)` });
    for (const [l, h] of hs) if (l < this.L && h < this.H) this.ko.heads.add(`${l}:${h}`);
    for (const l of ms) if (l < this.L) this.ko.mlps.add(`${l}`);
    this.ivVer++; this.urlDone = false; this.rerun();  // the shared state applies to the rerun
  }
  flushUrl(ended = false) {  // apply what the run has reached; a token the model hasn't written yet waits for it
    const u = this.pendingUrl; if (!u) return;
    const reached = u.p == null || (u.p < this.hist.length && this.hist[u.p]);
    if (!reached && !ended) return;
    if (!reached && this.pendingIv) this.pendingIv.graph = false;  // no graph to wait for
    if (!reached) this.setStatus(`The shared link points at token ${u.p + 1 - this.lead}, which this run didn't reach${this.$('fwdSample').value !== 'greedy' ? ' (sampled text differs from run to run)' : ''}.`);
    else if (u.p != null) this.viewPos = u.p;  // held there, even while the run writes on
    if (u.n) { const [l, i] = u.n.split('.').map(Number); if (l >= 0 && l < this.L && i >= 0 && i < this.F) this.gotoNeuron(l, i); }
    if (u.g && reached) setTimeout(() => this.openGraph(u.w), 300);
    this.pendingUrl = null;
  }
  bookFor(l, i) {
    if (!this.book) return null;
    const k = `${l}:${i}`;
    if (!this.bookData.has(k)) {
      const rec = { ready: false };
      this.bookData.set(k, rec);
      const done = () => { rec.ready = true; this.nrnKey = ''; if (this.graph && this.graph.data) { clearTimeout(this.graphT); this.graphT = setTimeout(() => this.graphView.render(this), 120); } };
      this.book.get(l, i).then((d) => { rec.d = d; done(); }).catch((e) => { rec.err = e.message; done(); });
    }
    return this.bookData.get(k);
  }
  runProbe(text) {
    const { l, sel } = this.nrn || {};
    if (sel == null || sel < 0 || !text || !text.trim()) return;
    const rows = this.tok.encode(text).map((i) => this.rowOf[i]).filter((r) => r >= 0).slice(0, this.maxT - 1);
    if (!rows.length) return;
    this.probe = { id: (this.probe ? this.probe.id : 0) + 1, l, i: sel, text, result: null, at: new Date().toISOString() };
    const eot = this.eotRow();  // start after <|endoftext|>, as the corpus examples do, not on the attention sink
    this.worker.postMessage({ type: 'probe', id: this.probe.id, l, i: sel, rows: eot != null ? [eot, ...rows] : rows, lead: eot != null ? 1 : 0 });
    this.nrnKey = '';
  }
  eotRow() { const r = this.tok instanceof GPT2Tokenizer && this.rowOf ? this.rowOf[50256] : null; return r != null && r >= 0 ? r : null; }
  // An example's whole passage, from its document's start, run through the unchanged model.
  async openPassage(kind, k) {
    const { l, sel } = this.nrn || {}, key = `${l}:${sel}`;
    if (this.expand && this.expand.key === key && this.expand.kind === kind && this.expand.k === k) { this.expand = null; this.nrnKey = ''; return; }
    const rec = this.bookData.get(key), bw = this.bandWin.get(key);
    const ex = kind === 'band' ? bw && bw.list[k] : rec && rec.d && rec.d[kind] && rec.d[kind][k]; if (!ex) return;
    const id = (this.expand ? this.expand.id : 0) + 1;
    this.expand = { id, key, kind, k, at: 0, result: null }; this.nrnKey = '';
    try {
      const pg = await this.book.passage(ex.at);
      if (!this.expand || this.expand.id !== id) return;
      // the passage from its document's start (ids[0] is <|endoftext|>); if it doesn't fit the
      // kernel's window, keep the tokens nearest the example, after a fresh <|endoftext|>
      let ids = pg.ids, at = pg.at;
      if (ids.length > this.maxT) { const keep = this.maxT - 1, cut = ids.length - keep; ids = [ids[0], ...ids.slice(cut)]; at -= cut - 1; this.expand.cut = cut - 1; }
      this.expand.at = at;
      const rows = ids.map((t) => this.rowOf[t]).map((r) => (r == null || r < 0 ? 0 : r));
      this.worker.postMessage({ type: 'probe', kind: 'expand', id, l, i: sel, rows, lead: 1 });
    } catch (e) { this.expand = null; this.setStatus(`Couldn't read that passage: ${e.message}`); }
  }
  openGraph(want) {
    const v = this.view(); if (!v.h) return;
    const row = want != null && want >= 0 ? want : this.attrTarget(v.h, v.pos); if (row == null) return;
    const key = `${v.pos}:${row}:${this.ivVer}`, done = this.traces.get(key);  // traces are cleared with each new run
    this.graph = { pos: v.pos, row, key, at: this.tokens[this.rows[v.pos]] || '', data: done || null, progress: 'Tracing…' };
    this.graphView.show();
    if (!done) this.worker.postMessage({ type: 'graph', pos: v.pos, row, runId: this.textId, ivVer: this.ivVer });
    const r = this.ivRun;  // the graph says which model it was traced on: the one last rerun
    this.graph.ko = r ? r.neurons.length + r.heads.length + r.mlps.length : 0;
    this.syncUrl();
  }
  // The selected neuron, read several ways (see the handbook, §11.3).
  neuronSection(v, l, sel, act, effOf, tw, info, fmt, word) {
    const { pos } = v, F = this.F, key = `${l}:${sel}`, swiglu = this.caps.act === 'swiglu';
    const tokOf = (id) => { const r = this.rowOf[id]; return r == null || r < 0 ? '' : this.tokens[r]; };
    const shade = (x, m) => { const t = Math.min(1, Math.abs(x) / (m || 1)); return x >= 0 ? `rgba(53,224,230,${(t * 0.75).toFixed(2)})` : `rgba(169,150,255,${(t * 0.75).toFixed(2)})`; };
    const bar = (x, m) => `<span class="bar"><i style="${x >= 0 ? `left:50%;width:${Math.min(50, Math.abs(x) / m * 50)}%;background:#35e0e6` : `right:50%;width:${Math.min(50, Math.abs(x) / m * 50)}%;background:#a996ff`}"></i></span>`;
    // the reader's name and note for this one
    const note = this.notes.get(key) || {};
    let html = `<div class="sec">Neuron L${l} #${sel}</div>`;
    html += `<input id="nrnLabel" class="nlabel" type="text" maxlength="80" placeholder="Name it: the distinction it draws" value="${esc(note.label || '')}" aria-label="Your name for this neuron"><textarea id="nrnNote" class="nnote" rows="2" placeholder="Your reading, written before you test it, and what would break it" aria-label="Your notes on this neuron">${esc(note.note || '')}</textarea>`;
    // blind reading: the neuron's examples, words and neighbours stay hidden until a test has run
    const pr = this.probe && this.probe.l === l && this.probe.i === sel && this.probe.kind !== 'expand' ? this.probe : null;
    // revealed once a reading has been written and a sentence tested after it; it stays revealed
    if (pr && pr.result && note.note && note.at && note.at <= pr.at) this.reveal(key);
    const hidden = this.blind && !this.revealed.has(key);
    if (this.blind) html += `<p class="note">${hidden ? 'Reading blind' : 'Reading blind: revealed by your test'} ${info('Read blind (at the top of the drawer, kept on until you turn it off) hides each neuron\'s examples, shape, neighbours, the words it writes, its activations in this text and its maximum, until you have written your reading in the note and then tested a sentence of your own below; after that it stays revealed. A test only counts if the reading came first. The layer\'s lists above still show its value at the token on screen, among the others.')}</p>`;
    // what it does in this text, and how strongly: values are shown against the neuron's own range
    const bk = this.bookFor(l, sel), stats = bk && bk.ready && bk.d ? bk.d : null;
    // its range in this text, read from the unchanged run when a changed one is on screen
    const base = this.ivRun && this.ref && this.ref.kind === 'iv' ? this.ref.hist : this.hist;
    let seen = 0, seenMin = 0, preMin = Infinity;
    for (let p = 1; p < base.length; p++) { const hp = base[p]; if (!hp) continue; const x = hp.mlp[l * F + sel]; seen = Math.max(seen, x); seenMin = Math.min(seenMin, x); if (hp.pre) preMin = Math.min(preMin, hp.pre[l * F + sel]); }
    const loading = !!this.book && !(bk && bk.ready);  // its book entry is on its way: its max and trough aren't known yet
    const top = stats ? stats.max : seen, dead = top <= 1e-3, cur = this.ko.neurons.get(key);
    this.nrnTop = { key, top: +top.toFixed(3) };  // what "Hold at its max" sets, kept out of the page while reading blind
    // steering, kept inside what the network itself produces: knock out, set to its maximum, or
    // push it the other way only as far as the network itself takes it: GELU to the lowest value
    // its own input reaches in the corpus (never below GELU's trough, -0.17), SwiGLU to its most
    // negative value seen in this text
    const gelu = (x) => 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
    // the lowest input it reaches (the book's, else this text's when the input was recorded); GELU bottoms out at -0.17 near -0.75
    const lowIn = stats ? stats.minIn : Number.isFinite(preMin) ? preMin : -Infinity;
    const trough = +gelu(Math.max(lowIn, -0.7518)).toFixed(3);
    const neg = swiglu ? { s: 0, add: +seenMin.toFixed(3), pre: 0 } : { s: 0, add: trough, pre: 0 };
    const negOff = loading || (swiglu ? seenMin > -1e-3 : trough > -1e-3);
    const isOn = (o) => !!cur && cur.s === o.s && cur.add === o.add && (cur.pre || 0) === o.pre;
    const sb = (o, label, text, off, attr) => `<button type="button" data-steer="${attr || `${o.s},${o.add},${o.pre}`}" data-key="${key}" data-label="${label}" aria-pressed="${isOn(o)}" ${off ? 'disabled' : ''}>${text}</button>`;
    const hb = hidden ? 'Hold at its max' : `Hold at its max (${top.toFixed(2)})`;
    const where = stats ? 'in the corpus' : 'in this text';
    const negText = swiglu ? 'Hold at its minimum' : `Hold at its trough${hidden ? '' : ` (${trough === -0.17 ? '−0.17' : trough.toFixed(2).replace('-', '−')})`}`;
    const negHelp = swiglu ? `<b>Hold at its minimum</b> sets it to its most negative value in this text${hidden ? '' : ` (${neg.add})`}.` : Number.isFinite(lowIn) ? `<b>Hold at its trough</b> sets it to the most negative value it reaches ${where}: −0.17, GELU's own trough, for a neuron whose input reaches −0.75 (where GELU bottoms out), else what its lowest input gives (this one${hidden ? '' : `: ${trough}`}); its readers never get less from it, and knocking it out is close to the same thing.` : '<b>Hold at its trough</b> sets it to −0.17, the lowest value GELU can give (its input wasn\'t recorded on this model, so its own lowest isn\'t known); knocking it out is close to the same thing.';
    if (loading) html += '<p class="note">Reading its book: holding it at its max or trough waits for its values there.</p>';
    html += `<div class="fwd-row" style="margin:6px 0 0;flex-wrap:wrap;gap:4px">${sb({ s: 0, add: 0, pre: 0 }, 'knocked out', 'Knock out')}${sb({ s: 0, add: this.nrnTop.top, pre: 0 }, 'held at its maximum', hb, dead || loading, '0,max,0')}${sb(neg, swiglu ? 'held at its minimum' : 'held at its trough', negText, negOff)}${info(`Steering, at every token; then <b>Rerun</b> (under Playback) runs the same text through the changed model, beside the unchanged run. <b>Hold at its max</b> sets the activation to its strongest ${where} everywhere, so a silent neuron speaks. ${negHelp} Every token at once is far from anything the neuron does naturally: compare with the same change to a random neuron of the same layer before reading much into the result.${dead ? ` This neuron never rises above 0.001 ${where}.` : ''}`)}</div>`;
    if (cur) html += `<p class="note">${cur.label}${this.ivRun && JSON.stringify(this.ivRun) === JSON.stringify(this.ivSpec()) ? ' · applied' : ' · press Rerun'}</p>`;
    else if (this.ivRun && this.ivRun.neurons.some(([a, b]) => a === l && b === sel)) html += '<p class="note">Changed in the run on screen (by a graph test) but not among your changes: Rerun puts back your own.</p>';
    // here: activation, input, gate and value
    const h = v.h, a0 = act[sel], e = effOf(sel);
    const pre = h.pre ? h.pre[l * F + sel] : null, preU = h.preU ? h.preU[l * F + sel] : null;
    let here = `Here: activation <b>${a0.toFixed(3)}</b>${top > 1e-3 ? ` (${(100 * a0 / top).toFixed(0)}% of its max)` : ''}`;
    if (swiglu && pre != null && preU != null) {
      const sg = (x) => (x >= 0 ? '+' : '−'), big = Math.abs(a0) > 0.1 * (top || 1);
      const regime = !big ? '' : pre >= 0 && preU >= 0 ? ' · its main side (gate open, value positive)' : pre >= 0 ? ' · its negative side (gate open, value negative)' : preU < 0 ? ' · <b>double negative</b>: gate and value both negative, so it fires positive for a reason unlike its main one' : ' · a leak through a closed gate';
      here += ` = SiLU(gate ${pre.toFixed(2)}) × value ${preU.toFixed(2)} (${sg(pre)}×${sg(preU)})${regime}`;
    } else if (pre != null) here += ` · input ${pre.toFixed(3)}${pre < -2 ? ' (far below zero)' : ''}`;
    if (e != null) here += ` · effect on ${tw} <b class="${e >= 0 ? 'pos' : 'neg'}">${fmt(e)}</b> (logits)`;
    html += `<div class="eff" style="margin-top:8px">${hidden ? 'Here: hidden while reading blind' : here}</div>`;
    if (pos === 0) html += `<p class="note warn">${this.lead ? 'This is the document-start token (⟨doc⟩), where the model parks attention' : 'This is the first token, where the model parks attention'}: values here are often outsized and say little about the neuron${this.lead ? '; the book leaves this position out' : ''}.</p>`;
    // where it fires in this text: activation, or the input before the activation function
    const canInput = this.hist.some((hp) => hp && hp.pre);
    const valOf = (hp) => (!hp ? null : this.showInput && hp.pre ? hp.pre[l * F + sel] : hp.mlp[l * F + sel]);
    let vmax = this.showInput ? (stats ? Math.abs(stats.minIn) : 1e-6) : Math.max(top, 1e-6);
    for (let p = 1; p < this.rows.length; p++) { const x = valOf(this.hist[p]); if (x != null) vmax = Math.max(vmax, Math.abs(x)); }
    html += `<div class="sec">Where it fires ${canInput ? `<button type="button" class="mini" data-inputview>${this.showInput ? 'show activation' : 'show input'}</button>` : ''} ${info(`This neuron's ${this.showInput ? 'input before its activation function, which shows its negative side even where GELU flattens it' : 'activation'} at every token of the text so far: <span class="c">cyan</span> positive, <span class="v">violet</span> negative, as bright as it is against ${stats ? 'its strongest in the corpus' : 'its strongest here'}, so a weak firing looks weak. Neurons often fire not on the word that carries a meaning but next to it: on the word before the thing it predicts, or on punctuation where the model sums up what came before. The outlined token is the one on screen.`)}</div>`;
    html += hidden ? '<p class="note">Hidden while reading blind.</p>' : `<div class="text">${this.rows.map((r, p) => { if (p < this.lead) return '<span class="doc" title="the start of the document">⟨doc⟩</span>'; const x = valOf(this.hist[p]); return `<span class="${p === pos ? 'here' : ''}" style="background:${x == null ? 'transparent' : shade(x, vmax)}" title="${x == null ? 'not run yet' : x.toFixed(3)}">${esc(this.tokens[r])}</span>`; }).join('')}</div>`;
    // the book: strongest examples, its other side, its shape
    if (this.book && !hidden) {
      const m = this.book.meta, B = m.before, pad = m.pad ?? 65535;
      const ex = this.expand && this.expand.key === key ? this.expand : null;
      const exLine = (e, kind, k, scale, unit) => {
        const toks = Array.from(e.ids, (id, j) => (id === pad ? '' : id === m.eot ? '<span class="doc" title="start of the document">⟨doc⟩</span>' : `<span class="${j === B ? 'here' : ''}" style="background:${shade(e.acts[j], scale)}" title="${unit} ${e.acts[j].toFixed(3)}">${esc(tokOf(id))}</span>`)).join('');
        const v0 = e.acts[B], open = ex && ex.kind === kind && ex.k === k;
        let li = `<li><span class="num ${v0 >= 0 ? 'pos' : 'neg'}" title="${unit}">${v0.toFixed(2)}<small>${scale > 1e-3 ? `${(100 * Math.abs(v0) / scale).toFixed(0)}%` : ''}</small></span><span class="text ex">${toks}</span>${e.at !== 0xffffffff ? `<button type="button" class="mini more" data-expand="${kind}:${k}" title="The whole passage up to here, run through the model">${open ? '−' : '↑'}</button>` : ''}</li>`;
        if (open) li += this.passageHtml(ex, kind === 'negIn', kind === 'negIn' ? Math.abs(stats.minIn) : scale, shade);
        return li;
      };
      const spacing = m.spacing || m.window;
      html += `<div class="sec">Strongest examples ${info(`The ${m.pos} tokens that drive this neuron hardest out of ${(m.tokens / 1e6).toFixed(1)} million tokens of ${esc(m.corpus.name)} (${esc(m.corpus.license)}), no two within ${spacing} tokens of each other, each with the ${B} tokens before it (⟨doc⟩ marks a document's start). Numbers are its activation, and that as a share of its maximum. Read the whole line, not just the outlined token: the meaning is often in what came before; ↑ runs the whole passage up to it.`)}</div>`;
      if (!bk.ready) html += '<p class="note">Reading its examples…</p>';
      else if (bk.err) html += `<p class="note">Couldn't read its examples (${esc(bk.err)}). <button type="button" class="mini" data-retry>Try again</button></p>`;
      else {
        const d = bk.d, mx = Math.max(d.max, 1e-6);
        html += d.max > 1e-3 ? `<ol class="exs">${d.pos.map((x, k) => exLine(x, 'pos', k, mx, 'activation')).join('')}</ol>` : '<p class="note">Never above 0.001 in the corpus: this neuron barely or never fires.</p>';
        // the strongest few can be a sliver of what it does: early layers run hot in a document's first tokens
        const early = d.pos.filter((e) => e.at !== 0xffffffff && e.at % m.ctx < 8).length;
        if (d.max > 1e-3 && early >= Math.ceil(d.pos.length / 2)) html += `<p class="note warn">${early} of these ${d.pos.length} sit in a document's first few tokens, where early layers run hot, while ${d.half.toLocaleString()} tokens reach half its max: they may be a sliver of what it does. ${(this.book.meta.band_first ?? 1) === 0 ? 'Open the top band of its shape below, or test' : 'Test'} it on a case mid-sentence.</p>`;
        if (swiglu && d.negOut && d.negOut.length) {  // for GELU these would all sit at its -0.17 trough: ties, not a list
          html += `<div class="sec">Where its output is most negative ${info('The tokens where its activation is most negative: what the neurons that read it receive from its negative side.')}</div>`;
          html += `<ol class="exs">${d.negOut.map((x, k) => exLine(x, 'negOut', k, Math.max(1e-6, Math.abs(d.negOut[0].acts[B])), 'activation')).join('')}</ol>`;
        }
        const im = Math.max(1e-6, Math.abs(d.minIn));
        html += `<div class="sec">Its other side ${info(swiglu ? 'The tokens with its most negative input: what pushes it hardest the other way.' : 'The tokens with its most negative input before GELU (numbers are that input): the votes against it at their strongest. GELU turns these into almost exactly zero, so its readers get between −0.17 and 0 from this side; the distinction it draws is in what lands here compared with its strongest examples.')}</div>`;
        html += d.minIn > -0.5 ? '<p class="note">Never pushed far below zero: its input stays above −0.5 everywhere in the corpus.</p>' : d.negIn.length ? `<ol class="exs">${d.negIn.map((x, k) => exLine(x, 'negIn', k, im, 'input')).join('')}</ol>` : '<p class="note">Nothing pushes it strongly negative.</p>';
        // shape: tokens per halving band of its own maximum, and how often it fires strongly
        const bins = d.bins, bmax = Math.max(1, ...bins), pages = d.tokens / 400, positive = bins.reduce((s, c) => s + c, 0);
        const per = d.half ? pages / d.half : Infinity;
        const freq = !d.half ? 'never reaches half its maximum in the corpus' : per >= 1 ? `reaches half its maximum or more about once every ${per >= 10 ? Math.round(per) : per.toFixed(1)} pages` : `reaches half its maximum or more about ${(1 / per).toFixed(1)} times a page`;
        const share = d.nonpos != null ? positive / (positive + d.nonpos) : null;
        html += `<div class="sec">Its shape ${info('How many tokens in the corpus land in each band of this neuron\'s own range, from its maximum (left) down by halves to 1/4096 of it (right), on a log scale; only tokens where it is positive. Where its tokens sit in its own range; open a band to see what the middle of its range responds to. A page is about 400 tokens (the essay counts one token in 100 as a quarter of a page).')}</div>`;
        const nb = this.book.meta.bands || 0, b0 = this.book.meta.band_first ?? 1, sel_ = this.band && this.band.key === key ? this.band.k : -1;
        html += `<div class="shape">${bins.map((c, k) => { const t = `${k === bins.length - 1 ? 'at most 1/4096' : k === 0 ? 'above 1/2' : `1/${2 ** (k + 1)} to 1/${2 ** k}`} of its max: ${c.toLocaleString()} tokens`, h = `height:${c ? Math.max(4, (Math.log10(c + 1) / Math.log10(bmax + 1)) * 100) : 0}%`; return k >= b0 && k < b0 + nb && c ? `<button type="button" class="bandbar${k === sel_ ? ' on' : ''}" data-band="${k}" style="${h}" title="${t}: click for a few of them" aria-label="${t}: show a few of them" aria-pressed="${k === sel_}"></button>` : `<i style="${h}" title="${t}"></i>`; }).join('')}</div>`;
        html += `<p class="note">Max ${d.max.toFixed(2)}${share != null ? ` · positive on ${(100 * share).toFixed(1)}% of tokens` : ''} · ${freq}.${nb ? ` Click a band to see a few random tokens from it${b0 === 0 ? ': the top one shows what it mostly does when it fires hard, the lower ones' : ':'} the middle of its range, where a reading should still hold.` : ''}</p>`;
        if (sel_ >= b0) html += this.bandSection(l, sel, sel_, d, tokOf, shade);
      }
    } else if (this.book && hidden) html += '<p class="note">Reading blind: its examples, shape, neighbours and words appear after your first test below.</p>';
    // built from, here: each earlier part's exact share of its input at this token
    if (this.caps.links && !hidden) {
      const lk = this.lk.get(`${pos}:${l}:${sel}`);
      if (!lk && !this.asked.has(`lk:${pos}:${l}:${sel}`)) { this.asked.add(`lk:${pos}:${l}:${sel}`); this.worker.postMessage({ type: 'links', run: this.runId, pos, l, j: sel }); }
      html += `<div class="sec">What feeds it here ${info(`Its input at this token split exactly into one share per earlier part: each earlier neuron (its activation times the weight between them), each attention head (what it moved here from this and earlier tokens), and the token's own embedding; what is left is biases. ${swiglu ? 'A SwiGLU neuron has two inputs, its gate and its value; both are split. ' : ''}Click a neuron to read it.`)}</div>`;
      if (!lk) html += '<p class="note">Splitting its input…</p>';
      else if (lk.none) html += `<p class="note">${esc(lk.none)}</p>`;
      else {
        const split = (blk, title) => {
          const parts = [];
          for (let k = 0; k < blk.neurons.length; k++) if (blk.neurons[k]) parts.push([blk.neurons[k], `n:${Math.floor(k / F)}:${k % F}`]);
          for (let k = 0; k < blk.heads.length; k++) if (blk.heads[k]) parts.push([blk.heads[k], `h:${Math.floor(k / this.H)}:${k % this.H}`]);
          parts.push([blk.emb, 'e']);
          let sum = 0; for (const [x] of parts) sum += x;
          parts.sort((p, q) => Math.abs(q[0]) - Math.abs(p[0]));
          const pm = Math.max(1e-6, Math.abs(parts[0][0]));
          const pRow = ([x, k]) => {
            const [t, a, b] = k.split(':');
            const label = t === 'n' ? `<button type="button" class="link" data-goto="${a}:${b}" title="${esc(this.label(+a, +b))}">${this.nname(+a, +b)}</button>` : t === 'h' ? `head L${a}.${b}` : 'embedding';
            return `<li><span class="lbl">${label}</span>${bar(x, pm)}<span class="num ${x >= 0 ? 'pos' : 'neg'}">${fmt(x)}</span></li>`;
          };
          return `${title ? `<p class="note sub">${title}</p>` : ''}<ol class="parts">${parts.slice(0, 8).map(pRow).join('')}</ol><p class="note">Input ${blk.total.toFixed(2)} = these, ${parts.length - 8 > 0 ? `${(parts.length - 8).toLocaleString()} smaller shares` : 'nothing else'}, and ${(blk.total - sum).toFixed(2)} of biases.</p>`;
        };
        html += lk.up ? split(lk, 'Its gate') + split(lk.up, 'Its value') : split(lk, '');
        const live = this.ivRun && this.ivRun.neurons.find(([a, b]) => a === l && b === sel);
        if (live && live[4]) html += `<p class="note">You steer its input by ${live[4] >= 0 ? '+' : ''}${(+live[4]).toFixed(2)}: the input it acts on is ${(lk.total + live[4]).toFixed(2)}, the split above plus your steer.</p>`;
        else if (live) html += `<p class="note">${this.ko.neurons.get(key) ? `You have set its activation (${this.ko.neurons.get(key).label})` : 'Its activation is set in the run on screen'}: the split above is the input it would have had; what it writes is the value you set.</p>`;
      }
      // read by: later neurons whose input weights line up with its output (weights only)
      const rd = this.rd.get(`${pos}:${key}`);
      if (!rd && !this.asked.has(`rd:${pos}:${key}`)) { this.asked.add(`rd:${pos}:${key}`); this.worker.postMessage({ type: 'readers', run: this.runId, pos, l, i: sel }); }
      if (rd && rd.into) {
        const list = []; for (let k = 0; k < rd.into.length; k++) if (rd.into[k]) list.push([rd.into[k], Math.floor(k / F), k % F]);
        list.sort((p, q) => Math.abs(q[0]) - Math.abs(p[0]));
        if (list.length) {
          const rm = Math.abs(list[0][0]);
          html += `<div class="sec">What reads it ${info('Later neurons whose input weights line up most with this one\'s output, either way, per unit of its activation, scaled by each reader\'s normalisation at the token on screen (late layers divide their input by more): where what it writes goes next. A strongly negative reader may be gated by it, this one firing keeping that one quiet; check what feeds that reader at a token to see.')}</div><ol class="parts">${list.slice(0, 6).map(([x, ll, ii]) => `<li><span class="lbl"><button type="button" class="link" data-goto="${ll}:${ii}" title="${esc(this.label(ll, ii))}">${this.nname(ll, ii)}</button></span>${bar(x, rm)}<span class="num ${x >= 0 ? 'pos' : 'neg'}">${fmt(x)}</span></li>`).join('')}</ol>`;
        }
      }
    }
    // what it writes (the output weights through the final norm and unembedding)
    if (!hidden) {
      const nv = this.nvocab.get(key);
      html += `<div class="sec">What it writes ${info('The words this neuron\'s output weights push <span class="c">up</span> and <span class="a">down</span> most whenever it fires positively (the other way round when negative). Worked out from its weights through the final normalisation and output layer, so it ignores what later layers do with its output. Many neurons write nothing word-like: their job is to feed later neurons.')}</div>`;
      html += nv ? `<div class="words up">${nv.up.map((w) => `<span>${word(w.row)}</span>`).join('')}</div><div class="words down" style="margin-top:6px">${nv.down.map((w) => `<span>${word(w.row)}</span>`).join('')}</div>` : '<p class="note">Reading its output weights…</p>';
    }
    // test it: a sentence of the reader's own, on the unchanged model
    if (this.caps.pre) {
      html += `<div class="sec">Test it ${info(`A test only counts if your reading came first: write it in the note above, then a sentence meant to fire this neuron as hard as possible, ideally with <b>Read blind</b> on. Copying an example proves nothing. If your reading is right, the sentence should come near its strongest activation${stats ? ` in the corpus${hidden ? '' : ` (${stats.max.toFixed(2)})`}` : ''}, or beat it; also try one that should <i>not</i> fire it, a near miss. The test runs on the unchanged model, whatever is steered.`)}</div>`;
      html += `<div class="fwd-row probe"><input id="nrnProbe" type="text" placeholder="A sentence you think will fire it" value="${pr ? esc(pr.text) : ''}" aria-label="A test sentence"><button type="button" data-probe>Test</button></div>`;
      if (pr && !pr.result) html += '<p class="note">Running…</p>';
      else if (pr && pr.result) {
        const r = pr.result; let best = -Infinity; for (const x of r.acts) best = Math.max(best, x);
        const sc = hidden ? Math.max(best, 1e-6) : Math.max(top, best, 1e-6);  // shading against the max would give it away
        html += `<div class="text">${r.rows.map((rw, p) => `<span style="background:${shade(r.acts[p], sc)}" title="${r.acts[p].toFixed(3)}${r.preUs[p] != null ? ` · gate ${r.pres[p].toFixed(2)} × value ${r.preUs[p].toFixed(2)}` : ` · input ${r.pres[p].toFixed(2)}`}">${esc(this.tokens[rw])}</span>`).join('')}</div>`;
        html += hidden ? `<p class="note">Strongest here ${best.toFixed(2)}. Write your reading in the note first, then test again: a test only reveals the neuron, and how close you came to its strongest, if the reading came first.</p>` : `<p class="note">Strongest here ${best.toFixed(2)}${stats && stats.max > 1e-3 ? ` · ${(100 * best / stats.max).toFixed(0)}% of its strongest in ${(stats.tokens / 1e6).toFixed(1)}M corpus tokens${best > stats.max ? ' · beats every corpus example' : ''}` : ''}.</p>`;
      }
    }
    html += `<p class="note" style="margin-top:10px">Reading it: what one distinction does it draw between its two sides? Try to say it without listing nouns. <a href="real.html#ch-neurons" data-explain>How to read a neuron</a></p>`;
    return html;
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
    if (rh && !h.diff && this.worker && !this.asked.has(`diff:${pos}`)) { this.asked.add(`diff:${pos}`); this.worker.postMessage({ type: 'diff', run: this.runId, pos }); }
    hd.textContent = this.hudText ? this.hudText(v) : `${this.tokName(pos)} · ${beat <= this.L ? (beat === 0 ? 'embedding' : `layer ${beat - 1}`) : 'readout'}`
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
      if (!h.lens[dp] && !this.asked.has(k)) { this.asked.add(k); this.worker.postMessage({ type: 'lens', run: this.runId, pos, depth: dp }); }
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
    c.fillText(`LOGIT LENS OVER DEPTH · “${word(this.rows[pos])}” (${this.tokName(pos)})`, pad, pad / 2 + 4 * d);
    if (!avail.length) { this.viewHud = `${this.tokName(pos)} · reading the logit lens…`; return; }
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
    this.viewHud = `${this.tokName(pos)} · ${live && cur < L ? `at layer ${cur - 1} it leads with` : 'settles on'} “${word(top)}”${!(live && cur < L) ? ` from ${settle === 0 ? 'the embedding' : `layer ${settle - 1}`} on` : ''}${firstSeen != null && firstSeen < settle ? `, in its top ten from ${firstSeen === 0 ? 'the embedding' : `layer ${firstSeen - 1}`}` : ''}`;
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
      this.viewHud = `${this.tokName(pos)} “${word(this.rows[pos])}” · draws on itself ${pct(R[0][pos])}${src.length ? `, ${src.map(([w, k]) => `“${word(this.rows[k])}” ${pct(w)}`).join(', ')}` : ''}${pos > 0 ? ` · ${(R[0][0] * 100).toFixed(0)}% rests on ${this.lead ? '⟨doc⟩' : 'the first token'}` : ''}`;
    } else this.viewHud = `${this.tokName(pos)} “${word(this.rows[pos])}” · ${this.flowCache.segs.toLocaleString()} links`;
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
