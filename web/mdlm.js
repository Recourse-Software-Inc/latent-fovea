// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// Loads MDLM into the masked-diffusion kernel and exposes its recordings.
// Shared by the worker and tools/check_mdlm.mjs.

export class MDLM {
  static async create(simdBytes, baseBytes) {
    const simd = simdBytes && WebAssembly.validate(simdBytes);
    const { instance } = await WebAssembly.instantiate(simd ? simdBytes : baseBytes, {});
    const m = new MDLM();
    m.ex = instance.exports; m.simd = simd;
    return m;
  }

  // getBuf(name) -> ArrayBuffer; ids: tile row -> token id.
  async load(getBuf, manifest, ids, onProgress = () => {}) {
    const ex = this.ex;
    this.m = manifest; this.L = manifest.n_layer; this.H = manifest.n_head; this.C = manifest.d_model; this.F = manifest.d_mlp;
    this.V = ids.length; this.maskRow = manifest.mask_row;
    const rc = ex.init(this.V, this.L, this.maskRow);
    if (rc === 2) throw new Error(`This model has ${this.L} blocks; this viewer runs up to 12 for it.`);
    if (rc !== 0) throw new Error('Not enough memory to load the model.');
    this.maxP = ex.max_p(); this.K = ex.top_k();
    const stageN = ex.stage_floats();
    const feed16 = (which, buf) => {
      const src = new Uint16Array(buf);
      for (let off = 0; off < src.length; off += stageN) {
        const n = Math.min(stageN, src.length - off);
        new Uint16Array(ex.memory.buffer, ex.ptr_stage(), n).set(src.subarray(off, off + n));
        ex.load(which, off, n);
      }
    };
    const feed32 = (which, buf) => {
      const src = new Float32Array(buf), cap = stageN >> 1;
      for (let off = 0; off < src.length; off += cap) {
        const n = Math.min(cap, src.length - off);
        new Float32Array(ex.memory.buffer, ex.ptr_stage_f32(), n).set(src.subarray(off, off + n));
        ex.load(which, off, n);
      }
    };
    const jobs = [...manifest.layer_files.map((f, l) => ['model/' + f, l, feed16]), ['model/mods.f32', 100, feed32],
      ['model/final.f32', 101, feed32], ['vectors.f16', 200, feed16], ['model/unembed.f16', 201, feed16]];
    let done = 0;
    for (const [name, which, feed] of jobs) { onProgress(done / jobs.length, name); feed(which, await getBuf(name)); done++; }
    onProgress(1, 'ready');
    let maxId = 0;  // a loop, not Math.max(...ids): 50k arguments overflow WebKit's stack
    for (let i = 0; i < ids.length; i++) if (ids[i] > maxId) maxId = ids[i];
    this.rowOf = new Int32Array(maxId + 1).fill(-1);
    ids.forEach((id, row) => { this.rowOf[id] = row; });
    this.ids = ids;
  }

  // rows: tile rows for every position (maskRow = masked). Runs one full forward pass.
  forward(rows) {
    const ex = this.ex, n = rows.length;
    if (n > this.maxP) throw new Error(`sequences are limited to ${this.maxP} tokens`);
    new Uint32Array(ex.memory.buffer, ex.ptr_rows(), n).set(rows);
    if (ex.forward(n) !== 0) throw new Error('forward failed');
    this.n = n;
  }
  belief(p) {  // top-k for a masked position after forward()
    const ex = this.ex, K = this.K;
    return {
      rows: Array.from(new Uint32Array(ex.memory.buffer, ex.ptr_topi() + p * K * 4, K)),
      probs: Array.from(new Float32Array(ex.memory.buffer, ex.ptr_topp() + p * K * 4, K)),
      entropy: new Float32Array(ex.memory.buffer, ex.ptr_ent(), this.maxP)[p],
    };
  }
  sample(p, u, greedy) { return this.ex.sample(p, u, greedy ? 1 : 0); }
  attRow(l, h, q) {
    const T = this.maxP;
    return new Float32Array(this.ex.memory.buffer, this.ex.ptr_att() + (((l * this.H + h) * T + q) * T) * 4, this.n);
  }
  headNorms(p) {  // [L][NH] for position p
    const out = new Float32Array(this.L * this.H), all = new Float32Array(this.ex.memory.buffer, this.ex.ptr_headn(), this.L * this.maxP * this.H);
    for (let l = 0; l < this.L; l++) out.set(all.subarray((l * this.maxP + p) * this.H, (l * this.maxP + p + 1) * this.H), l * this.H);
    return out;
  }
  mlp(p) {  // [L][F] for position p
    const out = new Float32Array(this.L * this.F), all = new Float32Array(this.ex.memory.buffer, this.ex.ptr_mlp(), this.L * this.maxP * this.F);
    for (let l = 0; l < this.L; l++) out.set(all.subarray((l * this.maxP + p) * this.F, (l * this.maxP + p + 1) * this.F), l * this.F);
    return out;
  }
  lens(p) {  // logit lens at every depth for position p
    this.ex.lens(p);
    const K = this.K, mem = this.ex.memory.buffer, out = [];
    const P = new Float32Array(mem, this.ex.ptr_lens_p(), (this.L + 1) * K), I = new Uint32Array(mem, this.ex.ptr_lens_i(), (this.L + 1) * K);
    const Hn = new Float32Array(mem, this.ex.ptr_lens_h(), this.L + 1);
    for (let d = 0; d <= this.L; d++) out.push({ rows: Array.from(I.subarray(d * K, d * K + K)), probs: Array.from(P.subarray(d * K, d * K + K)), entropy: Hn[d] });
    return out;
  }
}
