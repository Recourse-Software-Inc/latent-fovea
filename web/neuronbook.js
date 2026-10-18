// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// A model's neuron book (tools/build_neuron_examples.py): every MLP neuron's strongest examples
// on both sides over a public corpus, and the shape of its activations. One neuron's records are
// contiguous in each file, so it costs a few small Range requests, made only when it is opened;
// an example's whole passage is one more, made only when asked for.

export class NeuronBook {
  // base: the model's data folder URL (ending in '/'). Resolves to null when it has no book.
  static async open(base, dataFetch) {
    try {
      const m = await dataFetch(base + 'meta.json').then((x) => (x.ok ? x.json() : {})).catch(() => ({}));
      if (!m.neuron_book) return null;  // set by tools/build_neuron_examples.py
      const r = await dataFetch(base + m.neuron_book);
      if (!r.ok) return null;
      const meta = await r.json();
      const b = new NeuronBook();
      const sub = m.neuron_book.replace(/[^/]*$/, '');  // the book's folder, e.g. 'neurons/'
      Object.assign(b, { base, sub, dir: base + sub, dataFetch, meta, suffix: m.binary_suffix || '', parts: m.parts || {}, cache: new Map() });
      b.K = meta.pos + meta.neg + (meta.negout || 0);
      return b;
    } catch { return null; }
  }
  async range(name, start, bytes) {
    const parts = this.parts[this.sub + name];  // a file the stager split into parts (tools/stage_static.py)
    if (parts) return this.rangeParts(parts, start, bytes);
    const r = await this.dataFetch(`${this.dir}${name}${this.suffix}`, { headers: { Range: `bytes=${start}-${start + bytes - 1}` } });
    if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
    const buf = await r.arrayBuffer();
    // a host that ignores Range sends the whole file: take this neuron's slice of it
    return buf.byteLength === bytes ? buf : buf.slice(start, start + bytes);
  }
  async rangeParts(parts, start, bytes) {  // a file the stager split: read across its parts
    const out = new Uint8Array(bytes);
    let at = 0;
    for (const p of parts) {
      const a = Math.max(start, at), b = Math.min(start + bytes, at + p.bytes);
      if (a < b) {
        const r = await this.dataFetch(`${this.base}${p.file}${this.suffix}`, { headers: { Range: `bytes=${a - at}-${b - at - 1}` } });
        if (!r.ok) throw new Error(`${p.file}: HTTP ${r.status}`);
        const buf = new Uint8Array(await r.arrayBuffer());
        out.set(buf.byteLength === b - a ? buf : buf.subarray(a - at, b - at), a - start);
      }
      at += p.bytes;
    }
    return out.buffer;
  }
  // { pos, negIn, negOut: [{ ids, acts, at }], max, minIn, half, nonpos, bins, tokens }
  get(l, n) {
    const k = `${l}:${n}`;
    if (!this.cache.has(k)) this.cache.set(k, this.load(l, n).catch((e) => { this.cache.delete(k); throw e; }));  // a failed read is tried again next time
    return this.cache.get(k);
  }
  async load(l, n) {
    const { pos, neg, window: W } = this.meta, negout = this.meta.negout || 0, K = this.K, NS = this.meta.stats.length;
    const v2 = (this.meta.schema || '').endsWith('/2');
    const [ib, ab, sb, tb] = await Promise.all([
      this.range(`L${l}.ids.u16`, n * K * W * 2, K * W * 2),
      this.range(`L${l}.acts.f16`, n * K * W * 2, K * W * 2),
      this.range(`L${l}.stats.f32`, n * NS * 4, NS * 4),
      v2 ? this.range(`L${l}.at.u32`, n * K * 4, K * 4) : Promise.resolve(null),
    ]);
    const ids = new Uint16Array(ib), acts = f16(new Uint16Array(ab)), st = new Float32Array(sb), at = tb ? new Uint32Array(tb) : null;
    const ex = (k) => ({ ids: ids.subarray(k * W, (k + 1) * W), acts: acts.subarray(k * W, (k + 1) * W), at: at ? at[k] : 0xffffffff });
    const real = (e) => e.at !== 0xffffffff || (e.ids[this.meta.before] !== (this.meta.pad ?? 65535) && e.acts[this.meta.before] !== 0);
    const range = (a, b) => Array.from({ length: b - a }, (_, k) => ex(a + k)).filter(real);
    const off = v2 ? 4 : 3;
    return {
      pos: range(0, pos), negIn: range(pos, pos + neg), negOut: range(pos + neg, pos + neg + negout),
      max: st[0], minIn: st[1], half: st[2], nonpos: v2 ? st[3] : null, bins: Array.from(st.subarray(off)), tokens: this.meta.tokens,
    };
  }
  // A few random examples from each band (band 0: above half its max; each next one half the one above):
  // [{ band, at, v }], the middle of the neuron's range, where a reading can be checked.
  async bands(l, n) {
    const { bands = 0, band_samples: R = 0 } = this.meta;
    if (!bands || !R) return [];
    const k = `b${l}:${n}`;
    if (!this.cache.has(k)) this.cache.set(k, (async () => {
      const [ab, vb] = await Promise.all([this.range(`L${l}.bands.u32`, n * bands * R * 4, bands * R * 4), this.range(`L${l}.bandv.f16`, n * bands * R * 2, bands * R * 2)]);
      const at = new Uint32Array(ab), v = f16(new Uint16Array(vb)), out = [];
      const first = this.meta.band_first ?? 1;  // books before the top band was sampled start at band 1
      for (let b = 0; b < bands; b++) for (let j = 0; j < R; j++) { const q = b * R + j; if (at[q] !== 0xffffffff) out.push({ band: b + first, at: at[q], v: v[q] }); }
      return out;
    })().catch((e) => { this.cache.delete(k); throw e; }));
    return this.cache.get(k);
  }
  // The token ids around one corpus position, inside its document: { ids, k } with ids[k] the token.
  async window(at) {
    const { ctx, before, after, pad } = this.meta, doc = Math.floor(at / ctx), t = at - doc * ctx;
    const a = Math.max(0, t - before), b = Math.min(ctx, t + after + 1);
    const ids = Array.from(new Uint16Array(await this.range('corpus.u16', (doc * ctx + a) * 2, (b - a) * 2))).filter((x) => x !== pad);
    return { ids, k: t - a };
  }
  // An example's passage: its document from the start up to `after` tokens past it (token ids).
  async passage(at) {
    const { ctx, after, pad } = this.meta, doc = Math.floor(at / ctx), t = at - doc * ctx;
    const end = Math.min(ctx, t + after + 1);
    const ids = new Uint16Array(await this.range('corpus.u16', doc * ctx * 2, end * 2));
    let n = ids.length; while (n > 0 && ids[n - 1] === pad) n--;
    return { ids: Array.from(ids.subarray(0, n)), at: t };
  }
}
function f16(u) {  // fp16 bits -> f32
  const out = new Float32Array(u.length);
  for (let i = 0; i < u.length; i++) {
    const h = u[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    out[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return out;
}
