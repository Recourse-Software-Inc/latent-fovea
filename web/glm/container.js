// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// A colibri container (JustVugg/colibri, c/tools/convert_fp8_to_int4.py): safetensors shards with no
// index file, read in place. Sources: a folder the user picked (an external SSD; File.slice reads
// only the bytes asked for, nothing is copied) or a URL base (tests, Range requests).
//
// A quantized weight is `name` (U8, flat) + `name.qs` (F32 scales): int4 grouped (group 64,
// symmetric, low nibble = even column, stored q+8) or int8 with one scale per row, told apart by the
// byte counts. Norms, the router and its bias are plain F32. tools/glm_ref.py is the reference.
export class Container {
  constructor(files) { this.files = files; this.idx = new Map(); this.config = null; }

  // files: Map<name, {size, read(offset, length) → Promise<ArrayBuffer>}>
  static async fromDirectory(dir) {
    const files = new Map();
    for await (const [name, h] of dir.entries()) {
      if (h.kind !== 'file' || !(/\.(safetensors|json)$/.test(name) || name === '.coli_usage') || name.startsWith('._')) continue;  // ._* are macOS resource forks on exFAT
      const file = await h.getFile();
      files.set(name, { size: file.size, read: (o, n) => file.slice(o, o + n).arrayBuffer(), text: () => file.text() });
    }
    return new Container(files).open();
  }
  static async fromUrl(base, names) {
    const files = new Map();
    for (const name of names) files.set(name, {
      size: null,
      read: async (o, n) => {
        let r;
        for (let k = 0; ; k++) {  // a local server can drop a connection under many parallel reads: retry
          try { r = await fetch(base + name, { headers: { Range: `bytes=${o}-${o + n - 1}` } }); break; }
          catch (e) { if (k >= 3) throw e; await new Promise((ok) => setTimeout(ok, 200 * (k + 1))); }
        }
        if (r.status !== 206 && r.status !== 200) throw new Error(`${name}: HTTP ${r.status}`);
        const b = await r.arrayBuffer();
        return r.status === 206 ? b : b.slice(o, o + n);
      },
      text: () => fetch(base + name).then((r) => r.text()),
    });
    return new Container(files).open();
  }

  async open() {
    const cfg = this.files.get('config.json');
    if (!cfg) throw new Error('No config.json in this folder: pick the folder of a converted colibri container.');
    this.config = JSON.parse(await cfg.text());
    for (const [name, f] of this.files) {
      if (!name.endsWith('.safetensors')) continue;
      const n = Number(new DataView(await f.read(0, 8)).getBigUint64(0, true));
      const h = JSON.parse(new TextDecoder().decode(await f.read(8, n)));
      for (const [k, v] of Object.entries(h)) {
        if (k === '__metadata__') continue;
        this.idx.set(k, { file: name, dtype: v.dtype, shape: v.shape, offset: 8 + n + v.data_offsets[0], bytes: v.data_offsets[1] - v.data_offsets[0] });
      }
    }
    if (!this.idx.size) throw new Error('No safetensors shards found.');
    return this;
  }
  has(name) { return this.idx.has(name); }
  async raw(name) {
    const t = this.idx.get(name);
    if (!t) throw new Error(`missing tensor ${name}`);
    return this.files.get(t.file).read(t.offset, t.bytes);
  }
  // The format of a [O, I] weight from its byte counts: 'f32', 'i8' (per row) or 'i4' (group 64).
  format(name, O, I) {
    const t = this.idx.get(name), s = this.idx.get(name + '.qs');
    if (!t) throw new Error(`missing tensor ${name}`);
    if (!s) { if (t.dtype === 'F32' && t.bytes === O * I * 4) return 'f32'; throw new Error(`${name}: unsupported ${t.dtype}`); }
    if (t.bytes === O * I && s.bytes === O * 4) return 'i8';
    if (t.bytes === O * Math.ceil(I / 2) && s.bytes === O * Math.ceil(I / 64) * 4) return 'i4';
    throw new Error(`${name}: ${t.bytes} bytes and ${s.bytes / 4} scales do not fit [${O}, ${I}]`);
  }
  // Byte ranges of one weight (data + scales), for streaming experts.
  ranges(name) {
    const t = this.idx.get(name), s = this.idx.get(name + '.qs');
    return { data: t, scales: s };
  }
}
