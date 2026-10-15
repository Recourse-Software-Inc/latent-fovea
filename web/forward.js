// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// Loads GPT-2 into the forward-pass kernel and exposes its recordings.
// Shared by the browser app and tools/check_forward.mjs.

export class Forward {
  static async create(simdBytes, baseBytes) {
    const simd = simdBytes && WebAssembly.validate(simdBytes);
    const { instance } = await WebAssembly.instantiate(simd ? simdBytes : baseBytes, {});
    const f = new Forward();
    f.ex = instance.exports; f.simd = simd;
    return f;
  }

  // getBuf(name) -> ArrayBuffer of a data file; ids: tile row -> token id.
  async load(getBuf, manifest, ids, onProgress = () => {}) {
    const ex = this.ex;
    this.m = manifest;
    this.L = manifest.n_layer; this.H = manifest.n_head; this.C = manifest.d_model; this.F = manifest.d_mlp;
    this.V = ids.length;
    const rc = ex.init(this.V, manifest.n_ctx, manifest.n_layer);
    if (rc === 2) throw new Error(`This model has ${manifest.n_layer} layers; this viewer runs up to 12 for it.`);
    if (rc !== 0) throw new Error('Not enough memory to load the model.');
    this.maxT = ex.max_t();
    const stageN = ex.stage_floats();
    const feed = (which, buf) => {
      const src = new Uint16Array(buf);
      for (let off = 0; off < src.length; off += stageN) {
        const n = Math.min(stageN, src.length - off);
        new Uint16Array(ex.memory.buffer, ex.ptr_stage(), n).set(src.subarray(off, off + n));
        ex.load(which, off, n);
      }
    };
    const jobs = [...manifest.layer_files.map((f, l) => ['model/' + f, l]), ['model/misc.f16', 100], ['vectors.f16', 200]];
    let done = 0;
    for (const [name, which] of jobs) {
      onProgress(done / jobs.length, name);
      feed(which, await getBuf(name));
      done++;
    }
    onProgress(1, 'ready');
    let maxId = 0;  // a loop, not Math.max(...ids): 50k arguments overflow WebKit's stack
    for (let i = 0; i < ids.length; i++) if (ids[i] > maxId) maxId = ids[i];
    this.rowOf = new Int32Array(maxId + 1).fill(-1);
    ids.forEach((id, row) => { this.rowOf[id] = row; });
    this.ids = ids;
  }

  // Run one position. lensMode: 0 none, 1 final only, 2 every depth.
  step(row, pos, lensMode = 2) {
    if (this.ex.step(row, pos, lensMode) !== 0) throw new Error(`position ${pos} is past the ${this.maxT}-token window`);
  }

  // Views into the kernel's recordings for the most recent position.
  lens(depth) {
    const ex = this.ex, K = 10;
    const p = new Float32Array(ex.memory.buffer, ex.ptr_lens_p() + depth * K * 4, K);
    const i = new Uint32Array(ex.memory.buffer, ex.ptr_lens_i() + depth * K * 4, K);
    return { rows: Array.from(i), probs: Array.from(p), entropy: new Float32Array(ex.memory.buffer, ex.ptr_lens_h(), this.L + 1)[depth] };
  }
  attRow(l, h, q) {  // probabilities over keys 0..q
    const T = this.maxT;
    return new Float32Array(this.ex.memory.buffer, this.ex.ptr_att() + (((l * this.H + h) * T + q) * T) * 4, q + 1);
  }
  headNorms() { return new Float32Array(this.ex.memory.buffer, this.ex.ptr_headn(), this.L * this.H); }
  mlp() { return new Float32Array(this.ex.memory.buffer, this.ex.ptr_mlp(), this.L * this.F); }
  residual(depth) { return new Float32Array(this.ex.memory.buffer, this.ex.ptr_res() + depth * this.C * 4, this.C); }
  // What layer l's MLP wrote into the stream at the last step (null on an older cached kernel).
  mlpOut(l) { return this.ex.ptr_mout ? new Float32Array(this.ex.memory.buffer, this.ex.ptr_mout() + l * this.C * 4, this.C) : null; }
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
    const slot = ex.lens_q();
    const p = new Float32Array(ex.memory.buffer, (ex.ptr_lens_p() >>> 0) + slot * K * 4, K);
    const i = new Uint32Array(ex.memory.buffer, (ex.ptr_lens_i() >>> 0) + slot * K * 4, K);
    return { rows: Array.from(i), probs: Array.from(p), entropy: new Float32Array(ex.memory.buffer, (ex.ptr_lens_h() >>> 0), slot + 1)[slot] };
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
    if (!ex.neuron_gain || ex.neuron_gain(l, row) !== 0) return null;
    return new Float32Array(ex.memory.buffer, (ex.ptr_gain() >>> 0), this.F).slice();
  }
  // The words neuron i of layer l pushes up most and down most.
  neuronVocab(l, i) {
    const ex = this.ex, K = 10;
    if (!ex.neuron_vocab || ex.neuron_vocab(l, i) !== 0) return null;
    const p = new Float32Array(ex.memory.buffer, (ex.ptr_nv_p() >>> 0), 2 * K), r = new Uint32Array(ex.memory.buffer, (ex.ptr_nv_i() >>> 0), 2 * K);
    const pick = (a, b) => Array.from({ length: b - a }, (_, j) => ({ row: r[a + j], score: p[a + j] }));
    return { up: pick(0, K), down: pick(K, 2 * K) };
  }
}
