// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// Clusters a data set's token embeddings in the background (UMAP, kernel/umap.c) and saves
// the layout next to it as points.umap.f32, in the data set's own row order.
// In: { base, dir }  (dir: the OPFS directory to save into).  Out: progress, done { points }, error.
//
// Semantic zoom: { type: 'local', base, idx, cur } re-arranges just the rows in `idx` with
// their own UMAP, so the layout is faithful for that region (the fovea idea applied to the
// layout itself). `cur` is their current 3-D positions, used to keep the orientation and to
// measure how much of the region's true 15-NN the old and the new arrangement keep.
// Out: localProgress, local { idx, points, before, after }, localError. The rows stay loaded
// for the next request.
import { dataFetch, writeFile } from './store.js';
import { umapLayout, normalise } from './umap.js';

let rowsP = null, metaP = null, wasmP = null;
const load = (base) => {
  metaP ||= dataFetch(base + 'meta.json').then((r) => r.json());
  wasmP ||= fetch('umap.simd.wasm').then((r) => r.arrayBuffer()).then((b) => (WebAssembly.validate(b) ? b : fetch('umap.wasm').then((r) => r.arrayBuffer())));
  rowsP ||= metaP.then(async (meta) => {  // big files may be split into parts for the host
    const parts = meta.vector_parts, suf = meta.binary_suffix || '';
    if (!parts) return new Uint16Array(await (await dataFetch(base + 'vectors.f16' + suf)).arrayBuffer());
    const bufs = await Promise.all(parts.map((p) => dataFetch(base + p.file + suf).then((r) => r.arrayBuffer())));
    const out = new Uint16Array(bufs.reduce((s, b) => s + b.byteLength, 0) / 2);
    let o = 0; for (const b of bufs) { out.set(new Uint16Array(b), o); o += b.byteLength / 2; }
    return out;
  });
  return Promise.all([metaP, wasmP, rowsP]);
};

self.onmessage = async ({ data: m }) => {
  if (m.type === 'local') return local(m);
  try {
    const [meta, wasm, rows] = await load(m.base);
    const P = await umapLayout(wasm, rows, meta.n, meta.d, (p, text) => self.postMessage({ type: 'progress', p, text }));
    if (m.dir) await writeFile(`${m.dir}/points.umap.f32`, P).catch(() => {});
    self.postMessage({ type: 'done', points: P }, [P.buffer]);
  } catch (e) { self.postMessage({ type: 'error', message: e.message }); }
};

async function local({ base, idx, cur }) {
  try {
    const [meta, wasm, rows] = await load(base), d = meta.d, n = idx.length;
    const sub = new Uint16Array(n * d);
    for (let k = 0; k < n; k++) sub.set(rows.subarray(idx[k] * d, (idx[k] + 1) * d), k * d);
    // The kernel's own neighbour search (cosine, on 50 PCA dims) gives the region's true 15-NN.
    const knn = [];
    const Y = await umapLayout(wasm, sub, n, d, (p, text) => self.postMessage({ type: 'localProgress', p, text }), (ex) => {
      const KI = new Uint32Array(ex.memory.buffer, ex.ptr_knn() >>> 0, n * 29);
      for (let k = 0; k < n; k++) knn.push(Array.from(KI.subarray(k * 29, k * 29 + 15)));
    });
    const recall = (P) => {  // share of each point's 15 true neighbours among its 15 nearest in P
      let hit = 0, tot = 0;
      for (let k = 0; k < n; k++) {
        const x = P[3 * k], y = P[3 * k + 1], z = P[3 * k + 2], best = new Float64Array(15).fill(Infinity), bi = new Int32Array(15).fill(-1);
        for (let j = 0; j < n; j++) {
          if (j === k) continue;
          const dd = (P[3 * j] - x) ** 2 + (P[3 * j + 1] - y) ** 2 + (P[3 * j + 2] - z) ** 2;
          if (dd < best[14]) { let p = 14; while (p > 0 && best[p - 1] > dd) { best[p] = best[p - 1]; bi[p] = bi[p - 1]; p--; } best[p] = dd; bi[p] = j; }
        }
        const near = new Set(bi); for (const j of knn[k]) { tot++; if (near.has(j)) hit++; }
      }
      return tot ? hit / tot : 0;
    };
    self.postMessage({ type: 'local', idx, points: Y, before: recall(cur), after: recall(Y) }, [Y.buffer]);
  } catch (e) { self.postMessage({ type: 'localError', message: e.message }); }
}
