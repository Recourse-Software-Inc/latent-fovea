// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// Minimal Hugging Face Hub client for the browser: model info, file fetches, and
// tensor reads out of safetensors files by HTTP Range (one tensor at a time, so a
// model never has to be downloaded as one blob). Works in workers and on the page.

export const HF = 'https://huggingface.co';

export class Hub {
  constructor(repo, { token = '', revision = 'main' } = {}) {
    this.repo = repo; this.revision = revision;
    this.headers = token ? { Authorization: `Bearer ${token}` } : {};
  }
  url(file) { return `${HF}/${this.repo}/resolve/${this.revision}/${file}`; }

  async get(file, range) {
    const headers = { ...this.headers };
    if (range) headers.Range = `bytes=${range[0]}-${range[1]}`;
    let r;
    for (let attempt = 0; ; attempt++) {  // Hugging Face rate-limits by IP: back off and retry
      r = await fetch(this.url(file), { headers });
      if ((r.status !== 429 && r.status < 500) || attempt >= 5) break;
      const wait = Math.min(60, +(r.headers.get('Retry-After') || 0) || 2 ** attempt);
      await new Promise((ok) => setTimeout(ok, wait * 1000));
    }
    if (r.status === 429) throw new Error('Hugging Face is rate-limiting this network. Try again in a few minutes.');
    if (r.status === 401 || r.status === 403) throw new Error(`${this.repo} is gated or private: accept its licence on Hugging Face and give a token with read access.`);
    if (!r.ok) throw new Error(`${file}: HTTP ${r.status}`);
    if (range && r.status !== 206) throw new Error(`${file}: the server ignored the byte range`);
    return r;
  }
  async json(file) { return (await this.get(file)).json(); }
  async text(file) { return (await this.get(file)).text(); }
  async has(file) { return (this.files || []).includes(file); }

  // Repo metadata from the Hub API: files, parameter counts, gating, config, licence.
  async info() {
    const q = ['safetensors', 'config', 'siblings', 'gated', 'cardData', 'downloads', 'sha'].map((e) => `expand[]=${e}`).join('&');
    const r = await fetch(`${HF}/api/models/${this.repo}?${q}`, { headers: this.headers });
    if (r.status === 404 || r.status === 401) throw new Error(`No public model called ${this.repo} on Hugging Face.`);
    if (!r.ok) throw new Error(`Hugging Face API: HTTP ${r.status}`);
    const j = await r.json();
    this.files = (j.siblings || []).map((s) => s.rfilename);
    return j;
  }

  // Index every tensor across the repo's safetensors files (single or sharded).
  async tensors() {
    let shards = ['model.safetensors'];
    if (this.files && this.files.includes('model.safetensors.index.json')) {
      const idx = await this.json('model.safetensors.index.json');
      shards = [...new Set(Object.values(idx.weight_map))];
    } else if (this.files && !this.files.includes('model.safetensors')) {
      throw new Error(`${this.repo} has no safetensors weights.`);
    }
    const out = new Map();
    for (const file of shards) {
      const n = Number(new DataView(await (await this.get(file, [0, 7])).arrayBuffer()).getBigUint64(0, true));
      const header = JSON.parse(await (await this.get(file, [8, 7 + n])).text());
      for (const [name, t] of Object.entries(header)) {
        if (name === '__metadata__') continue;
        out.set(name, { file, dtype: t.dtype, shape: t.shape, start: 8 + n + t.data_offsets[0], end: 8 + n + t.data_offsets[1] });
      }
    }
    return out;
  }

  // Read many tensors with as few requests as possible: neighbours in a file are fetched
  // together, in chunks of up to `chunk` bytes. onTensor(name, fp16 bits) fires as each
  // one arrives, in file order, so callers can write and free as they go.
  async readMany(entries, onTensor, onBytes = () => {}, chunk = 96 << 20) {
    const list = [...entries].sort((a, b) => a[1].file.localeCompare(b[1].file) || a[1].start - b[1].start);
    for (let i = 0; i < list.length;) {
      const first = list[i][1];
      let j = i + 1, end = first.end;
      while (j < list.length && list[j][1].file === first.file && list[j][1].start - end < (1 << 20) && list[j][1].end - first.start <= chunk) { end = Math.max(end, list[j][1].end); j++; }
      const r = await this.get(first.file, [first.start, end - 1]);
      const buf = await readAll(r, end - first.start, onBytes);
      for (let k = i; k < j; k++) {
        const [name, t] = list[k];
        await onTensor(name, toF16(buf.slice(t.start - first.start, t.end - first.start), t.dtype));
      }
      i = j;
    }
  }
}

async function readAll(r, n, onBytes) {
  const out = new Uint8Array(n), rd = r.body.getReader();
  let o = 0;
  for (;;) {
    const { done, value } = await rd.read();
    if (done) break;
    out.set(value, o); o += value.length; onBytes(value.length);
  }
  if (o !== n) throw new Error(`short read: ${o} of ${n} bytes`);
  return out.buffer;
}

// ---- fp conversions ----
const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
export function floatToHalf(v) {
  f32[0] = v;
  const x = u32[0], s = (x >>> 16) & 0x8000, e = (x >>> 23) & 0xff, m = x & 0x7fffff;
  if (e === 0xff) return s | 0x7c00 | (m ? 0x200 : 0);
  const E = e - 127 + 15;
  if (E >= 31) return s | 0x7c00;
  if (E <= 0) {
    if (E < -10) return s;
    const mm = (m | 0x800000) >>> (1 - E);
    return s | ((mm + 0x1000 + ((mm >>> 13) & 1) - 1) >>> 13);  // round to nearest even
  }
  const h = s | (E << 10) | (m >>> 13), rem = m & 0x1fff;
  return rem > 0x1000 || (rem === 0x1000 && (h & 1)) ? h + 1 : h;
}
const HT = new Float32Array(65536);
for (let h = 0; h < 65536; h++) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  HT[h] = e === 0 ? s * 2 ** -14 * (m / 1024) : e === 31 ? (m ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + m / 1024);
}
export const halfToFloat = (h) => HT[h];

function toF16(buf, dtype) {
  if (dtype === 'F16') return new Uint16Array(buf);
  const out = new Uint16Array(buf.byteLength / (dtype === 'F32' ? 4 : 2));
  if (dtype === 'F32') {
    const src = new Float32Array(buf);
    for (let i = 0; i < src.length; i++) out[i] = floatToHalf(src[i]);
  } else if (dtype === 'BF16') {
    const src = new Uint16Array(buf);
    for (let i = 0; i < src.length; i++) { u32[0] = src[i] << 16; out[i] = floatToHalf(f32[0]); }
  } else throw new Error(`unsupported tensor type ${dtype}`);
  return out;
}
