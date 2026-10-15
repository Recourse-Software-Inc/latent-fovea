// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// Runs the in-browser UMAP (web/umap.js + web/umap.simd.wasm) in Node on a prebuilt data
// set and scores it: for sample tokens, how many of their 15 nearest neighbours in the
// full space (the precomputed knn.u32) are also among their 15 nearest in the 3-D layout.
// Compares against the data set's own layout (umap-learn, offline).
//   node tools/check_umap.mjs web/data/gpt2 [out.f32]
import fs from 'fs';
import path from 'path';
import { umapLayout } from '../web/umap.js';

const dir = process.argv[2] || 'web/data/gpt2', out = process.argv[3];
const meta = JSON.parse(fs.readFileSync(path.join(dir, 'meta.json')));
const n = meta.n, d = meta.d, K = meta.k;
const vec = new Uint16Array(fs.readFileSync(path.join(dir, 'vectors.f16')).buffer.slice(0));
const knn = new Uint32Array(fs.readFileSync(path.join(dir, 'knn.u32')).buffer.slice(0));
const offline = new Float32Array(fs.readFileSync(path.join(dir, 'points.f32')).buffer.slice(0));
const wasm = fs.readFileSync(new URL('../web/umap.simd.wasm', import.meta.url));
const t0 = Date.now();
let last = '';
const Y = await umapLayout(wasm, vec, n, d, (p, t) => { const s = t.replace(/\d+%/, ''); if (s !== last) { console.log(`${((Date.now() - t0) / 1000).toFixed(1)}s ${t}`); last = s; } });
console.log(`done in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
if (out) fs.writeFileSync(out, Buffer.from(Y.buffer));

function score(L, k = 15, samples = 1500) {
  let hit = 0, tot = 0;
  for (let s = 0; s < samples; s++) {
    const i = Math.floor((s * 2654435761) % n);
    const x = L[i * 3], y = L[i * 3 + 1], z = L[i * 3 + 2];
    const best = [];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const dd = (L[j * 3] - x) ** 2 + (L[j * 3 + 1] - y) ** 2 + (L[j * 3 + 2] - z) ** 2;
      if (best.length < k) { best.push([dd, j]); best.sort((a, b) => a[0] - b[0]); }
      else if (dd < best[k - 1][0]) { best[k - 1] = [dd, j]; best.sort((a, b) => a[0] - b[0]); }
    }
    const near = new Set(best.map((b) => b[1]));
    for (let t = 0; t < Math.min(k, K); t++) { tot++; if (near.has(knn[i * K + t])) hit++; }
  }
  return hit / tot;
}
console.log(`neighbourhood preserved (15-NN in 768-d also 15-NN in 3-D):`);
console.log(`  in-browser UMAP  ${(score(Y) * 100).toFixed(1)}%`);
console.log(`  offline UMAP     ${(score(offline) * 100).toFixed(1)}%`);
