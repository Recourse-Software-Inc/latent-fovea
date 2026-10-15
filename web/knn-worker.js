// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// Exact cosine nearest neighbours on demand, for data sets built in the browser (which
// skip the precomputed knn table). A fast int8 scan of every row shortlists candidates;
// they are then rescored from the fp16 rows, so the reported neighbours and cosines are
// exact unless a true neighbour fell outside the shortlist of 64.
//
// In: { type: 'load', base } then { type: 'knn', i, k } or { type: 'pca' }.
// Out: { type: 'knn', i, idx, cos }; { type: 'pca', share, parts }: how much of the vectors'
// variation the three directions of greatest spread hold (what the PCA layout shows);
// { type: 'rank', rho }: per layout, how well near/far in the cloud agrees with near/far in the
// full space (Spearman rank correlation of distances over sampled pairs).
import { dataFetch } from './store.js';
import { halfToFloat } from './hub.js';

let n = 0, d = 0, raw = null, q8 = null, scale = null, pending = [], ready = false, wantPca = false;
const SHORT = 64;

self.onmessage = async ({ data: m }) => {
  if (m.type === 'load') {
    const meta = await (await dataFetch(m.base + 'meta.json')).json();
    n = meta.n; d = meta.d;
    const parts = meta.vector_parts;  // big files may be split into parts for the host
    if (parts) {
      const bufs = await Promise.all(parts.map((p) => dataFetch(m.base + p.file + (meta.binary_suffix || '')).then((r) => r.arrayBuffer())));
      raw = new Uint16Array(bufs.reduce((s, b) => s + b.byteLength, 0) / 2);
      let o = 0; for (const b of bufs) { raw.set(new Uint16Array(b), o); o += b.byteLength / 2; }
    } else raw = new Uint16Array(await (await dataFetch(m.base + 'vectors.f16' + (meta.binary_suffix || ''))).arrayBuffer());
    q8 = new Int8Array(n * d); scale = new Float32Array(n);
    const row = new Float32Array(d);
    for (let i = 0; i < n; i++) {  // unit-normalise, then quantise each row to int8
      let s = 0; for (let k = 0; k < d; k++) { row[k] = halfToFloat(raw[i * d + k]); s += row[k] * row[k]; }
      const inv = 1 / (Math.sqrt(s) || 1);
      let mx = 1e-12; for (let k = 0; k < d; k++) { row[k] *= inv; mx = Math.max(mx, Math.abs(row[k])); }
      scale[i] = mx / 127;
      for (let k = 0; k < d; k++) q8[i * d + k] = Math.round(row[k] / scale[i]);
    }
    ready = true;
    for (const p of pending) answer(p);
    pending = [];
    if (wantPca) pcaShare();
  } else if (m.type === 'knn') {
    if (ready) answer(m); else pending.push(m);
  } else if (m.type === 'pca') {
    if (ready) pcaShare(); else wantPca = true;
  } else if (m.type === 'rank') {
    if (ready) rankAgreement(m); else pending.push(m);
  } else if (m.type === 'axes') {
    if (ready) axesLayout(m); else pending.push(m);
  } else if (m.type === 'radial') {
    if (ready) radial(m); else pending.push(m);
  }
};

// Radial view: the exact cosine distance from token i to every token (fp16 rows, no shortlist).
function radial({ i }) {
  const a = exactRow(i, new Float32Array(d)), b = new Float32Array(d), dist = new Float32Array(n);
  for (let j = 0; j < n; j++) { exactRow(j, b); let s = 0; for (let t = 0; t < d; t++) s += a[t] * b[t]; dist[j] = 1 - s; }
  self.postMessage({ type: 'radial', i, dist }, [dist.buffer]);
}

// Semantic axes: a layout whose x and y are directions the user named by word pairs
// (x = e(a) - e(b), y = e(c) - e(d) made orthogonal to x) and whose z is the top principal
// direction of what is left. Every token's position is its true projection on those
// directions, so distances along x and y mean exactly "more a-like than b-like". Reports the
// share of the vocabulary's variation each axis holds.
function axesLayout({ pairs }) {
  const mean = new Float64Array(d), S = Math.min(n, 4000);
  for (let s = 0; s < S; s++) { const i = Math.floor(((s + 0.5) * n) / S); for (let k = 0; k < d; k++) mean[k] += halfToFloat(raw[i * d + k]) / S; }
  const rowOf = (i) => Float64Array.from({ length: d }, (_, k) => halfToFloat(raw[i * d + k]));
  const unit = (v) => { const l = Math.hypot(...v) || 1; return v.map((x) => x / l); };
  const dir = ([a, b]) => { const ra = rowOf(a), rb = rowOf(b); return unit(ra.map((x, k) => x - rb[k])); };
  const u1 = dir(pairs[0]);
  let u2 = dir(pairs[1]); { let p = 0; for (let k = 0; k < d; k++) p += u2[k] * u1[k]; u2 = unit(u2.map((x, k) => x - p * u1[k])); }
  // z: the top principal direction of the centred rows with u1 and u2 removed (sampled, power iteration).
  const Y = new Float64Array(S * d); let total = 0;
  for (let s = 0; s < S; s++) {
    const i = Math.floor(((s + 0.5) * n) / S); let p1 = 0, p2 = 0;
    for (let k = 0; k < d; k++) { const v = halfToFloat(raw[i * d + k]) - mean[k]; Y[s * d + k] = v; p1 += v * u1[k]; p2 += v * u2[k]; }
    for (let k = 0; k < d; k++) { Y[s * d + k] -= p1 * u1[k] + p2 * u2[k]; total += Y[s * d + k] ** 2; }
    total += p1 * p1 + p2 * p2;
  }
  let u3 = Float64Array.from({ length: d }, (_, k) => Math.sin(k * 1.3 + 2)), lam = 0;
  for (let it = 0; it < 40; it++) {
    const w = new Float64Array(d);
    for (let s = 0; s < S; s++) { let a = 0; for (let k = 0; k < d; k++) a += Y[s * d + k] * u3[k]; for (let k = 0; k < d; k++) w[k] += a * Y[s * d + k]; }
    lam = Math.hypot(...w); for (let k = 0; k < d; k++) u3[k] = w[k] / (lam || 1);
  }
  let v1 = 0, v2 = 0;
  for (let s = 0; s < S; s++) { let p1 = 0, p2 = 0; for (let k = 0; k < d; k++) { const v = halfToFloat(raw[Math.floor(((s + 0.5) * n) / S) * d + k]) - mean[k]; p1 += v * u1[k]; p2 += v * u2[k]; } v1 += p1 * p1; v2 += p2 * p2; }
  const P = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    let a = 0, b = 0, c = 0;
    for (let k = 0; k < d; k++) { const v = halfToFloat(raw[i * d + k]) - mean[k]; a += v * u1[k]; b += v * u2[k]; c += v * u3[k]; }
    P[3 * i] = a; P[3 * i + 1] = b; P[3 * i + 2] = c;
  }
  // Same scaling as the other layouts (median-centred, 98% within radius 1), but keep the axes'
  // relative scale: one factor for all three, so "more a-like" stays comparable across axes.
  const med = [0, 1, 2].map((j) => { const col = Float32Array.from({ length: n }, (_, i) => P[3 * i + j]).sort(); return col[n >> 1]; });
  for (let i = 0; i < n; i++) for (let j = 0; j < 3; j++) P[3 * i + j] -= med[j];
  const rad = Float32Array.from({ length: n }, (_, i) => Math.hypot(P[3 * i], P[3 * i + 1], P[3 * i + 2])).sort(), sc = rad[Math.floor(0.98 * (n - 1))] || 1;
  for (let i = 0; i < P.length; i++) P[i] /= sc;
  self.postMessage({ type: 'axes', points: P, shares: [v1 / total, v2 / total, lam / total] }, [P.buffer]);
}

// Near/far agreement, read the way the eye reads it: outward from one token. For each of 40
// sampled tokens, every other token is ranked by true (int8-scored) cosine distance; partners
// are taken at log-spaced true ranks (1st, 2nd, 3rd, 5th, 8th ... nearest, out to half the
// vocabulary) and Spearman's rank correlation between those true ranks and their distances
// in each 3-D layout is averaged over tokens. 1 = the cloud orders near-to-far exactly as the
// real space; 0 = unrelated. Random partners would ask only about far-vs-far ordering, which
// a cluster layout does not even try to keep.
function rankAgreement({ sample, layouts }) {
  const ids = Object.keys(layouts), toks = sample.filter((_, k) => k % 4 === 0);  // 40 of 150: a full scan each
  const marks = []; for (let f = 1, g = 2; f < n / 2; [f, g] = [g, f + g]) marks.push(f);  // Fibonacci ranks
  const score = new Float32Array(n), order = new Int32Array(n);
  const sum = Object.fromEntries(ids.map((id) => [id, 0]));
  for (const i of toks) {
    const qi = i * d;
    for (let j = 0; j < n; j++) { let s = 0; const qj = j * d; for (let t = 0; t < d; t++) s += q8[qi + t] * q8[qj + t]; score[j] = -s * scale[j]; order[j] = j; }
    score[i] = Infinity;
    order.sort((p, q) => score[p] - score[q]);  // nearest first
    const truthRank = marks.map((r, k) => k), x3 = (id) => marks.map((r) => { const j = order[r - 1], P = layouts[id]; return Math.hypot(P[3 * i] - P[3 * j], P[3 * i + 1] - P[3 * j + 1], P[3 * i + 2] - P[3 * j + 2]); });
    for (const id of ids) {
      const v = x3(id), o = v.map((x, k) => [x, k]).sort((p, q) => p[0] - q[0]), r3 = new Float64Array(v.length); o.forEach(([, k], pos) => { r3[k] = pos; });
      const m = (v.length - 1) / 2; let num = 0, da = 0, db = 0;
      for (let k = 0; k < v.length; k++) { const x = truthRank[k] - m, y = r3[k] - m; num += x * y; da += x * x; db += y * y; }
      sum[id] += num / Math.sqrt(da * db || 1);
    }
  }
  self.postMessage({ type: 'rank', rho: Object.fromEntries(ids.map((id) => [id, sum[id] / toks.length])) });
}

// The share of the vectors' total variation held by the top three principal directions,
// estimated on a spread sample of rows (power iteration with deflation, in double precision).
function pcaShare() {
  const S = Math.min(n, 4000), Y = new Float64Array(S * d), mean = new Float64Array(d);
  for (let s = 0; s < S; s++) { const i = Math.floor(((s + 0.5) * n) / S); for (let k = 0; k < d; k++) { Y[s * d + k] = halfToFloat(raw[i * d + k]); mean[k] += Y[s * d + k] / S; } }
  let total = 0;
  for (let s = 0; s < S; s++) for (let k = 0; k < d; k++) { Y[s * d + k] -= mean[k]; total += Y[s * d + k] ** 2; }
  const comps = [], parts = [];
  for (let c = 0; c < 3; c++) {
    let v = Float64Array.from({ length: d }, (_, k) => Math.sin(k * 1.7 + c + 1)), lam = 0;
    for (let it = 0; it < 40; it++) {
      const w = new Float64Array(d);
      for (let s = 0; s < S; s++) { let a = 0; for (let k = 0; k < d; k++) a += Y[s * d + k] * v[k]; for (let k = 0; k < d; k++) w[k] += a * Y[s * d + k]; }
      for (const u of comps) { let p = 0; for (let k = 0; k < d; k++) p += w[k] * u[k]; for (let k = 0; k < d; k++) w[k] -= p * u[k]; }
      lam = Math.hypot(...w); for (let k = 0; k < d; k++) v[k] = w[k] / (lam || 1);
    }
    comps.push(v); parts.push(lam / (total || 1));
  }
  self.postMessage({ type: 'pca', share: parts.reduce((a, b) => a + b, 0), parts });
}

function exactRow(i, out) {
  let s = 0; for (let k = 0; k < d; k++) { out[k] = halfToFloat(raw[i * d + k]); s += out[k] * out[k]; }
  const inv = 1 / (Math.sqrt(s) || 1); for (let k = 0; k < d; k++) out[k] *= inv;
  return out;
}
function answer(m) {
  if (m.type === 'rank') return rankAgreement(m);
  if (m.type === 'axes') return axesLayout(m);
  if (m.type === 'radial') return radial(m);
  const { i, k } = m;
  // 1. int8 scan of every row.
  const qi = i * d, top = new Int32Array(SHORT).fill(-1), ts = new Float32Array(SHORT).fill(-Infinity);
  for (let j = 0; j < n; j++) {
    if (j === i) continue;
    let s = 0; const qj = j * d;
    for (let t = 0; t < d; t++) s += q8[qi + t] * q8[qj + t];
    s *= scale[j];
    if (s > ts[SHORT - 1]) {
      let p = SHORT - 1;
      while (p > 0 && ts[p - 1] < s) { ts[p] = ts[p - 1]; top[p] = top[p - 1]; p--; }
      ts[p] = s; top[p] = j;
    }
  }
  // 2. Exact rescoring of the shortlist.
  const a = exactRow(i, new Float32Array(d)), b = new Float32Array(d);
  const scored = [];
  for (const j of top) {
    if (j < 0) continue;
    exactRow(j, b); let s = 0; for (let t = 0; t < d; t++) s += a[t] * b[t];
    scored.push([j, s]);
  }
  scored.sort((x, y) => y[1] - x[1]);
  const res = scored.slice(0, k);
  self.postMessage({ type: 'knn', i, idx: Uint32Array.from(res, (r) => r[0]), cos: Float32Array.from(res, (r) => r[1]) });
}
