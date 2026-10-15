// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// Drives kernel/umap.c: fp16 embedding rows in, a clustered 3-D layout out (same scaling
// as the other layouts: median-centred, 98% of points inside radius 1). Used by
// layout-worker.js in the browser and by tools/check_umap.mjs in Node.

// onGraph(ex) runs once the neighbour search is done, with the kernel's exports (its kNN table).
export async function umapLayout(wasmBytes, rows16, n, d, onProgress = () => {}, onGraph = null) {
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  const ex = instance.exports, u = (p) => p >>> 0;
  if (ex.init(n, d) !== 0) throw new Error('Not enough memory to cluster this vocabulary.');
  new Uint16Array(ex.memory.buffer, u(ex.ptr_x()), n * d).set(rows16);
  const say = (p, text) => onProgress(p, text);
  say(0.02, 'Finding the main directions (PCA)…');
  if (ex.pca() !== 0) throw new Error('Not enough memory for PCA.');
  ex.knn_start();
  const slice = Math.max(64, Math.floor(4e8 / n / 64) * 64);  // ~4e8 dot products per call
  for (let s = 0; s < n; s += slice) {
    ex.knn_rows(s, Math.min(n, s + slice));
    say(0.05 + 0.6 * Math.min(1, (s + slice) / n), `Finding every token's neighbours… ${Math.round(Math.min(1, (s + slice) / n) * 100)}%`);
    await new Promise((r) => setTimeout(r, 0));
  }
  if (onGraph) onGraph(ex);
  say(0.66, 'Building the neighbour graph…');
  if (ex.graph() !== 0) throw new Error('Not enough memory for the neighbour graph.');
  ex.layout_init();
  const E = 200;
  for (let e = 0; e < E; e += 10) {
    ex.epochs(e, e + 10);
    say(0.68 + 0.32 * ((e + 10) / E), `Arranging clusters… ${Math.round(((e + 10) / E) * 100)}%`);
    await new Promise((r) => setTimeout(r, 0));
  }
  const Y = new Float32Array(ex.memory.buffer, u(ex.ptr_y()), n * 3).slice();
  return normalise(Y, n);
}

export function normalise(P, n) {
  for (let j = 0; j < 3; j++) {
    const col = Float32Array.from({ length: n }, (_, i) => P[i * 3 + j]).sort();
    const med = col[n >> 1];
    for (let i = 0; i < n; i++) P[i * 3 + j] -= med;
  }
  const r = Float32Array.from({ length: n }, (_, i) => Math.hypot(P[i * 3], P[i * 3 + 1], P[i * 3 + 2])).sort();
  const s = r[Math.floor(0.98 * (n - 1))] || 1;
  for (let i = 0; i < P.length; i++) P[i] /= s;
  return P;
}
