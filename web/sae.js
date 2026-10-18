// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// Sparse autoencoders (SAEs): a learned dictionary that rewrites one residual-stream vector as a
// sparse sum of "features": a second model of the model, useful for asking what is in one state.
// The weights stream from Hugging Face the first time a layer is opened (by HTTP Range, like a
// model) and are kept in this browser's storage at 16 bits; the encoding runs in the worker.
//   encode:  f = ReLU((x - b_dec) W_enc + b_enc)      rebuild:  x' = f W_dec + b_dec
import { Hub, halfToFloat, floatToHalf } from './hub.js';
import { writeFile, readBuf, fileSize } from './store.js';

// Known SAE sets. depth: 0 = the embedding .. L = after the last layer, as in the field.
export const SAES = {
  'gpt2-small': {
    repo: 'jbloom/GPT2-Small-SAEs-Reformatted', d: 768, n: 24576, by: 'Joseph Bloom', license: 'MIT',
    // Trained on TransformerLens's GPT-2, which centres every vector written into the residual
    // stream: its stream is ours minus each vector's mean across dimensions.
    center: true,
    hook: (depth) => (depth < 12 ? `blocks.${depth}.hook_resid_pre` : 'blocks.11.hook_resid_post'),
    neuronpedia: (depth, j) => ({ model: 'gpt2-small', source: `${depth}-res-jb`, index: j }),
  },
};
// Which set fits a model: GPT-2 small, built in or from Hugging Face (the same weights).
export function saeFor(arch, L, C, V) { return arch === 'gpt2' && L === 12 && C === 768 && V === 50257 ? 'gpt2-small' : null; }

const NAMES = ['W_enc', 'W_dec', 'b_enc', 'b_dec'];
const cache = new Map();  // `${id}:${depth}` -> loaded weights (one or two layers at a time)

export async function loadSae(id, depth, onProgress = () => {}) {
  const key = `${id}:${depth}`;
  if (cache.has(key)) return cache.get(key);
  const spec = SAES[id], hook = spec.hook(depth), dir = `sae--${id}/${hook}`;
  const t = {};
  if ((await fileSize(`${dir}/b_dec.f16`)) > 0) {  // stored earlier
    onProgress('Reading the features from this browser…');
    for (const nm of NAMES) t[nm] = Float32Array.from(new Uint16Array(await readBuf(`${dir}/${nm}.f16`)), halfToFloat);
  } else {
    const hub = new Hub(spec.repo), file = `${hook}/sae_weights.safetensors`;
    await hub.info();
    const n = Number(new DataView(await (await hub.get(file, [0, 7])).arrayBuffer()).getBigUint64(0, true));
    const header = JSON.parse(await (await hub.get(file, [8, 7 + n])).text());
    for (const nm of NAMES) {
      const h = header[nm];
      if (!h || h.dtype !== 'F32') throw new Error(`unexpected SAE tensor ${nm}`);
      onProgress(`Downloading the features for this layer · ${nm} (${Math.round((h.data_offsets[1] - h.data_offsets[0]) / 1e6)} MB)`);
      t[nm] = new Float32Array(await (await hub.get(file, [8 + n + h.data_offsets[0], 8 + n + h.data_offsets[1] - 1])).arrayBuffer());
      await writeFile(`${dir}/${nm}.f16`, Uint16Array.from(t[nm], floatToHalf));
    }
  }
  const sae = { id, depth, d: spec.d, n: spec.n, center: !!spec.center, ...t };
  if (cache.size >= 2) cache.delete(cache.keys().next().value);
  cache.set(key, sae);
  return sae;
}

// The vector as the SAE saw it in training.
function prep(sae, x) {
  if (!sae.center) return x;
  let m = 0; for (let i = 0; i < sae.d; i++) m += x[i];
  m /= sae.d;
  return Float32Array.from(x, (v) => v - m);
}
// Features of one residual-stream vector, and how faithfully they rebuild it.
export function encode(sae, x0, top = 16) {
  const { d, n, W_enc, W_dec, b_enc, b_dec } = sae, x = prep(sae, x0);
  const f = Float32Array.from(b_enc);
  for (let i = 0; i < d; i++) {
    const xi = x[i] - b_dec[i]; if (!xi) continue;
    const row = W_enc.subarray(i * n, (i + 1) * n);
    for (let j = 0; j < n; j++) f[j] += xi * row[j];
  }
  const act = [], rec = Float32Array.from(b_dec);
  for (let j = 0; j < n; j++) {
    if (f[j] <= 0) continue;
    act.push([j, f[j]]);
    const w = W_dec.subarray(j * d, (j + 1) * d);
    for (let i = 0; i < d; i++) rec[i] += f[j] * w[i];
  }
  act.sort((a, b) => b[1] - a[1]);
  let xr = 0, xx = 0, rr = 0, ee = 0;
  for (let i = 0; i < d; i++) { xr += x[i] * rec[i]; xx += x[i] * x[i]; rr += rec[i] * rec[i]; ee += (x[i] - rec[i]) ** 2; }
  return { top: act.slice(0, top), l0: act.length, cos: xr / Math.sqrt(xx * rr || 1), relErr: Math.sqrt(ee / (xx || 1)) };
}

// One feature's activation at each of several vectors (every token of the text).
export function featureOver(sae, xs, j) {
  const { d, n, W_enc, b_enc, b_dec } = sae;
  return Float32Array.from(xs, (x0) => {
    if (!x0) return NaN;
    const x = prep(sae, x0);
    let s = b_enc[j];
    for (let i = 0; i < d; i++) s += (x[i] - b_dec[i]) * W_enc[i * n + j];
    return Math.max(0, s);
  });
}
export const decoderRow = (sae, j) => sae.W_dec.subarray(j * sae.d, (j + 1) * sae.d);
