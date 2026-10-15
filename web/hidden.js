// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// The hidden states as a space of their own: every captured token's residual stream at every
// depth, projected to 3-D by a PCA fitted on this run's states. Each vector is first normalised
// the way the model reads it (centred across its dimensions and divided by its RMS: a layer norm
// without its learned weights), so the growth of the stream's size from layer to layer, and the
// few huge dimensions some models carry, don't flatten everything onto one line.

const FIT_MAX = 1500;  // states the axes are fitted on (all states are projected)

function normalised(x) {
  const C = x.length; let m = 0;
  for (let i = 0; i < C; i++) m += x[i];
  m /= C;
  let s = 0;
  for (let i = 0; i < C; i++) s += (x[i] - m) ** 2;
  const k = 1 / Math.sqrt(s / C + 1e-6), y = new Float32Array(C);
  for (let i = 0; i < C; i++) y[i] = (x[i] - m) * k;
  return y;
}

// res: per position, Float32Array((L + 1) * C). prev: the axes of the last fit, to keep each
// axis pointing the same way (a refit must not mirror the picture). The first token is projected
// but left out of the fit (when there are others): most models use it as an attention sink and
// give it outsized values, so fitted on, it would stretch the axes toward itself.
// Returns { xyz: Float32Array(n * (L + 1) * 3), share, axes }; share is over the fitted states.
export function hiddenSpace(res, n, L, C, prev = null) {
  const D = L + 1, N = n * D, X = new Array(N), p0 = n > 1 ? 1 : 0;  // first fitted position
  const mean = new Float64Array(C);
  for (let p = 0; p < n; p++) for (let d = 0; d < D; d++) {
    const y = normalised(res[p].subarray(d * C, (d + 1) * C));
    X[p * D + d] = y;
    if (p >= p0) for (let i = 0; i < C; i++) mean[i] += y[i];
  }
  for (let i = 0; i < C; i++) mean[i] /= (n - p0) * D;
  for (const y of X) for (let i = 0; i < C; i++) y[i] -= mean[i];
  // Fit on an even spread of states (every one when there are few).
  const NF = (n - p0) * D, step = Math.max(1, NF / FIT_MAX), fit = [];
  for (let t = 0; t < NF; t += step) fit.push(X[p0 * D + Math.floor(t)]);
  let total = 0;
  for (const y of fit) for (let i = 0; i < C; i++) total += y[i] * y[i];
  // Subspace iteration for the top 3 directions: V <- orth(Xᵀ X V).
  const K = 3;
  let V = prev && prev.length === K && prev[0].length === C ? prev.map((v) => Float64Array.from(v))
    : Array.from({ length: K }, (_, k) => Float64Array.from({ length: C }, (_, i) => Math.sin(1 + i * (k + 1) * 0.7) + (i % (k + 2) ? 0 : 1)));
  orth(V);
  for (let it = 0; it < (prev ? 8 : 24); it++) {
    const W = Array.from({ length: K }, () => new Float64Array(C));
    for (const y of fit) {
      const a = new Float64Array(K);
      for (let k = 0; k < K; k++) { const v = V[k]; let s = 0; for (let i = 0; i < C; i++) s += y[i] * v[i]; a[k] = s; }
      for (let k = 0; k < K; k++) { const w = W[k], s = a[k]; for (let i = 0; i < C; i++) w[i] += y[i] * s; }
    }
    V = W; orth(V);
  }
  // Order the axes by the variation each shows, and say how much the three show together.
  const varOf = V.map((v) => { let s = 0; for (const y of fit) { let a = 0; for (let i = 0; i < C; i++) a += y[i] * v[i]; s += a * a; } return s; });
  const order = [0, 1, 2].sort((a, b) => varOf[b] - varOf[a]);
  V = order.map((k) => V[k]);
  const share = total > 0 ? order.reduce((s, k) => s + varOf[k], 0) / total : 0;
  if (prev && prev.length === K) V.forEach((v, k) => { let s = 0; for (let i = 0; i < C; i++) s += v[i] * prev[k][i]; if (s < 0) for (let i = 0; i < C; i++) v[i] = -v[i]; });
  const xyz = new Float32Array(N * 3);
  for (let j = 0; j < N; j++) for (let k = 0; k < K; k++) { const y = X[j], v = V[k]; let s = 0; for (let i = 0; i < C; i++) s += y[i] * v[i]; xyz[j * 3 + k] = s; }
  return { xyz, share, axes: V.map((v) => Float32Array.from(v)) };
}

function orth(V) {  // Gram-Schmidt, in place
  for (let k = 0; k < V.length; k++) {
    const v = V[k];
    for (let j = 0; j < k; j++) { const u = V[j]; let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * u[i]; for (let i = 0; i < v.length; i++) v[i] -= s * u[i]; }
    let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i];
    s = 1 / Math.sqrt(s || 1); for (let i = 0; i < v.length; i++) v[i] *= s;
  }
}
