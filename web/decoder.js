// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// Loads a model built in the browser into the general decoder kernel (kernel/dec.c) and
// exposes its recordings with the same interface as forward.js, plus lens on demand.

export const CFG_KEYS = ['V', 'C', 'L', 'NH', 'NKV', 'HD', 'F', 'NCTX', 'NORM', 'NORM_BIAS', 'POS', 'ROT', 'MLP',
  'QKV_BIAS', 'O_BIAS', 'MLP_BIAS', 'QK_NORM', 'PARALLEL', 'TIED', 'STREAM'];

export class Decoder {
  static async create(simdBytes, baseBytes) {
    const simd = simdBytes && WebAssembly.validate(simdBytes);
    const { instance } = await WebAssembly.instantiate(simd ? simdBytes : baseBytes, {});
    const d = new Decoder();
    d.ex = instance.exports; d.simd = simd;
    return d;
  }

  // manifest.cfg holds the kernel config; files are fp16 in the kernel's slot order.
  // opts.openLayer(name) -> a sync reader for a layer file streams the weights: they stay in the
  // browser's storage and each layer is read into one shared buffer as the pass reaches it, so a
  // model only needs memory for its word tables and one layer (see kernel/dec.c, STREAM_).
  async load(getBuf, manifest, ids, onProgress = () => {}, opts = {}) {
    const ex = this.ex, cfg = { ...manifest.cfg, STREAM: opts.openLayer ? 1 : 0 };
    const mem = () => ex.memory.buffer;
    const c = new Uint32Array(mem(), (ex.ptr_cfg() >>> 0), CFG_KEYS.length);
    CFG_KEYS.forEach((k, i) => { c[i] = cfg[k] | 0; });
    new Float32Array(mem(), (ex.ptr_eps() >>> 0), 1)[0] = cfg.EPS;
    new Float32Array(mem(), (ex.ptr_inv_freq() >>> 0), 256).set((cfg.INV_FREQ || []).slice(0, 256));
    const rc = ex.init();
    if (rc === 2) throw new Error('This model is built in a way this tool can\'t run yet.');
    if (rc !== 0) throw new Error('Not enough memory to run this model in the browser.');
    this.m = manifest; this.L = cfg.L; this.H = cfg.NH; this.C = cfg.C; this.F = cfg.F; this.V = cfg.V;
    this.maxT = ex.max_t();
    const stageN = ex.stage_halves();
    const feed = (which, buf) => {
      const src = new Uint16Array(buf);
      for (let off = 0; off < src.length; off += stageN) {
        const n = Math.min(stageN, src.length - off);
        new Uint16Array(mem(), (ex.ptr_stage() >>> 0), n).set(src.subarray(off, off + n));
        if (ex.load(which, off, n) !== 0) throw new Error(`could not load part ${which} of the model`);
      }
    };
    this.streaming = !!opts.openLayer;
    if (this.streaming) {  // keep a reader per layer file; nothing is loaded until a pass needs it
      this.layerBytes = ex.layer_halves() * 2; this.inBuf = -1;
      this.readers = await Promise.all(manifest.layer_files.map((f) => opts.openLayer('model/' + f)));
      const bad = this.readers.findIndex((r) => r.getSize() !== this.layerBytes);
      if (bad >= 0) throw new Error(`layer ${bad} of this model is incomplete; remove it and download it again`);
    }
    const jobs = [...(this.streaming ? [] : manifest.layer_files.map((f, l) => ['model/' + f, l])), ['model/misc.f16', 100], ['vectors.f16', 200]];
    if (!cfg.TIED) jobs.push(['model/lmhead.f16', 201]);
    let done = 0;
    for (const [name, which] of jobs) {
      onProgress(done / jobs.length, name);
      feed(which, await getBuf(name));
      done++;
    }
    onProgress(1, 'ready');
    let maxId = 0;  // a loop, not Math.max(...ids): large arrays overflow the call stack
    for (let i = 0; i < ids.length; i++) if (ids[i] > maxId) maxId = ids[i];
    this.rowOf = new Int32Array(maxId + 1).fill(-1);
    ids.forEach((id, row) => { this.rowOf[id] = row; });
    this.ids = ids;
  }

  step(row, pos) {
    const ex = this.ex;
    if (!this.streaming) {
      if (ex.step(row, pos) !== 0) throw new Error(`position ${pos} is past the ${this.maxT}-token window`);
      return;
    }
    if (ex.step_begin(row, pos) !== 0) throw new Error(`position ${pos} is past the ${this.maxT}-token window`);
    for (let l = 0; l < this.L; l++) { this.loadLayer(l); ex.step_layer(l); }
    ex.step_end();
  }
  // Streaming: read layer l's weights from storage into the kernel's shared layer buffer.
  loadLayer(l) {
    if (!this.streaming || this.inBuf === l) return;
    const view = new Uint8Array(this.ex.memory.buffer, (this.ex.ptr_layer_buf() >>> 0), this.layerBytes);
    const n = this.readers[l].read(view, { at: 0 });
    if (n !== this.layerBytes) throw new Error(`could not read layer ${l} from the browser's storage`);
    this.inBuf = l;
  }
  close() { (this.readers || []).forEach((r) => { try { r.close(); } catch {} }); this.readers = null; }
  // Which depths get a logit-lens readout: all of them when that is cheap, else an even
  // spread (always including the last), keeping the cost near `budget` multiply-adds.
  lensDepths(budget = 1.6e9) {
    const per = this.V * this.C, all = this.L + 1;
    const n = Math.max(2, Math.min(all, Math.floor(budget / per)));
    if (n >= all) return Array.from({ length: all }, (_, d) => d);
    const s = new Set([this.L]);
    for (let j = 0; j < n - 1; j++) s.add(Math.round((j * this.L) / (n - 1)));
    return [...s].sort((a, b) => a - b);
  }
  lens(depth) {
    const ex = this.ex, K = 10;
    ex.lens(depth);
    const p = new Float32Array(ex.memory.buffer, (ex.ptr_lens_p() >>> 0), K);
    const i = new Uint32Array(ex.memory.buffer, (ex.ptr_lens_i() >>> 0), K);
    return { rows: Array.from(i), probs: Array.from(p), entropy: ex.lens_entropy() };
  }
  attRow(l, h, q) { return new Float32Array(this.ex.memory.buffer, (this.ex.ptr_att() >>> 0) + ((l * this.H + h) * this.maxT) * 4, q + 1); }
  headNorms() { return new Float32Array(this.ex.memory.buffer, (this.ex.ptr_headn() >>> 0), this.L * this.H); }
  mlp() { return new Float32Array(this.ex.memory.buffer, (this.ex.ptr_mlp() >>> 0), this.L * this.F); }
  residual(depth) { return new Float32Array(this.ex.memory.buffer, (this.ex.ptr_res() >>> 0) + depth * this.C * 4, this.C); }
  // What layer l's MLP wrote into the stream at the last step (null on an older cached kernel).
  mlpOut(l) { return this.ex.ptr_mout ? new Float32Array(this.ex.memory.buffer, (this.ex.ptr_mout() >>> 0) + l * this.C * 4, this.C) : null; }
  // The size of each head's value vector at position pos, every layer (null on an older cached kernel).
  valueNorms(pos) { const ex = this.ex; if (!ex.vnorm) return null; const o = new Float32Array(this.L * this.H); for (let l = 0; l < this.L; l++) for (let h = 0; h < this.H; h++) o[l * this.H + h] = ex.vnorm(l, pos, h); return o; }

  // Interventions, applied from the next step on: { heads: [[l, h, scale]], mlps: [[l, scale]],
  // neurons: [[l, i, scale]] }; scale 0 knocks a part out. An empty spec restores the model.
  setInterventions(spec = {}) {
    const ex = this.ex;
    if (!ex.iv_clear) return false;
    ex.iv_clear();
    for (const [l, h, s] of spec.heads || []) ex.iv_head(l, h, s);
    for (const [l, s] of spec.mlps || []) ex.iv_mlp(l, s);
    for (const [l, i, s] of spec.neurons || []) ex.iv_neuron(l, i, s);
    return true;
  }
  // After the final logit lens of a step: the probability of word `row` (a tile row).
  probOf(row) { return this.ex.prob_of ? this.ex.prob_of(row) : null; }
  // Any direction (an SAE feature's): the words it pushes up most and down most.
  vocabOf(vec) {
    const ex = this.ex, K = 10;
    if (!ex.vocab_q) return null;
    new Float32Array(ex.memory.buffer, (ex.ptr_qvec() >>> 0), this.C).set(vec);
    ex.vocab_q();
    const p = new Float32Array(ex.memory.buffer, (ex.ptr_nv_p() >>> 0), 2 * K), r = new Uint32Array(ex.memory.buffer, (ex.ptr_nv_i() >>> 0), 2 * K);
    const pick = (a, b) => Array.from({ length: b - a }, (_, j) => ({ row: r[a + j], score: p[a + j] }));
    return { up: pick(0, K), down: pick(K, 2 * K) };
  }
  // Logit lens on any residual-stream vector (a stored depth of an earlier token, say).
  lensOf(vec) {
    const ex = this.ex, K = 10;
    if (!ex.lens_q) return null;
    new Float32Array(ex.memory.buffer, (ex.ptr_qvec() >>> 0), this.C).set(vec);
    ex.lens_q();
    const p = new Float32Array(ex.memory.buffer, (ex.ptr_lens_p() >>> 0), K);
    const i = new Uint32Array(ex.memory.buffer, (ex.ptr_lens_i() >>> 0), K);
    return { rows: Array.from(i), probs: Array.from(p), entropy: ex.lens_entropy() };
  }
  // Direct logit attribution of word `row` (a tile row) at the most recent position: each
  // head's and each MLP's share of that word's final logit (see the kernel).
  attribute(row) {
    const ex = this.ex, L = this.L, H = this.H;
    if (!ex.attribute || ex.attribute(row) !== 0) return null;  // an older kernel: no attribution
    const a = new Float32Array(ex.memory.buffer, (ex.ptr_attr() >>> 0), L * H + L + 4).slice();
    return { heads: a.subarray(0, L * H), mlps: a.subarray(L * H, L * H + L), emb: a[L * H + L], rest: a[L * H + L + 1], logit: a[L * H + L + 2], inv: a[L * H + L + 3] };
  }
  // Per-neuron direct effect on word `row` per unit activation, for layer l (before 1/scale).
  neuronGain(l, row) {
    const ex = this.ex;
    if (!ex.neuron_gain) return null;
    this.loadLayer(l);
    if (ex.neuron_gain(l, row) !== 0) return null;
    return new Float32Array(ex.memory.buffer, (ex.ptr_gain() >>> 0), this.F).slice();
  }
  // The words neuron i of layer l pushes up most and down most.
  neuronVocab(l, i) {
    const ex = this.ex, K = 10;
    if (!ex.neuron_vocab) return null;
    this.loadLayer(l);
    if (ex.neuron_vocab(l, i) !== 0) return null;
    const p = new Float32Array(ex.memory.buffer, (ex.ptr_nv_p() >>> 0), 2 * K), r = new Uint32Array(ex.memory.buffer, (ex.ptr_nv_i() >>> 0), 2 * K);
    const pick = (a, b) => Array.from({ length: b - a }, (_, j) => ({ row: r[a + j], score: p[a + j] }));
    return { up: pick(0, K), down: pick(K, 2 * K) };
  }
}
