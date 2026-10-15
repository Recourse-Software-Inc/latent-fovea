// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// UMAP in the browser (C -> wasm32, no libc, no imports): lays out a model's token
// embeddings as clusters, the way tools/build_data.py does offline with umap-learn
// (PCA to 50 dims, cosine kNN with n_neighbors = 30, min_dist = 0.15, 200 epochs).
//   1. pca():    top 50 principal directions (subspace iteration + Rayleigh-Ritz)
//   2. knn_*():  exact cosine nearest neighbours in that space (int16 SIMD dot products)
//   3. graph():  UMAP's fuzzy simplicial set (smooth kNN distances, fuzzy union)
//   4. epochs(): UMAP's SGD with negative sampling, initialised from PCA
// JS calls the long steps in slices so the worker can report progress.

typedef unsigned int u32;
typedef unsigned short u16;
typedef short i16;
typedef unsigned long long u64;

#define EXPORT __attribute__((visibility("default")))
#define PC 50       // PCA dims
#define PCP 56      // padded to a multiple of 8
#define KNN 29      // neighbours besides the point itself (umap-learn's n_neighbors = 30)
#define NEG 5       // negative samples per positive one
static const float A = 1.4139999f, B = 0.94881525f;  // find_ab_params(spread=1, min_dist=0.15)

extern unsigned char __heap_base;
static u32 heap_ptr;
static void *alloc(u64 bytes) {
  u64 p = ((u64)heap_ptr + 15u) & ~(u64)15u, end = p + bytes;
  if (end > 0xFFFF0000ull) return 0;
  u64 have = (u64)__builtin_wasm_memory_size(0) * 65536u;
  if (end > have && __builtin_wasm_memory_grow(0, (u32)((end - have + 65535u) / 65536u)) == (unsigned long)-1) return 0;
  heap_ptr = (u32)end;
  return (void *)(u32)p;
}

static inline float fsqrt(float x) { return __builtin_sqrtf(x); }
static float fexp(float x) {
  if (x < -87.0f) return 0.0f;
  if (x > 88.0f) x = 88.0f;
  float k = __builtin_floorf(x * 1.44269504f + 0.5f);
  float r = x - k * 0.693145751953125f - k * 1.428606765330187e-06f;
  float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f + r * (0.041666668f + r * (0.008333334f + r * 0.0013888889f)))));
  union { u32 u; float f; } s = {(u32)((int)k + 127) << 23};
  return p * s.f;
}
static float flog(float x) {
  union { float f; u32 u; } b = {x};
  int e = (int)(b.u >> 23) - 127;
  b.u = (b.u & 0x7fffff) | 0x3f800000;
  float m = b.f;
  if (m > 1.41421356f) { m *= 0.5f; e++; }
  float t = (m - 1.0f) / (m + 1.0f), t2 = t * t;
  float s = 2.0f * t * (1.0f + t2 * (0.33333333f + t2 * (0.2f + t2 * (0.14285714f + t2 * 0.11111111f))));
  return s + (float)e * 0.69314718f;
}
static inline float fpow(float x, float e) { return x > 0 ? fexp(e * flog(x)) : 0.0f; }
static inline float h2f(u16 h) {
  union { u32 u; float f; } v = {(u32)(h & 0x7fff) << 13};
  v.f *= 5.192296858534828e+33f;
  v.u |= (u32)(h & 0x8000) << 16;
  return v.f;
}
static u32 rng = 0x9e3779b9u;
static inline u32 rnd(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

// ---------- state ----------
static u32 N, D;
static u16 *X;          // [N][D] fp16 embeddings, filled by JS
static float *MEAN;     // [D]
static float *V;        // [PC][D] principal directions
static float *P;        // [N][PC] projections
static i16 *Q;          // [N][PCP] unit-normalised projections, int16
static u32 *KI;         // [N][KNN] neighbour indices
static float *KS;       // [N][KNN] cosine similarities (descending)
static u32 NE;          // edges in the symmetric graph
static u32 *EH, *ET;    // edge head / tail
static float *EW;       // edge weight
static float *EPS, *NEXT, *EPSN, *NEXTN;  // epochs per sample, next sample, same for negatives
static float *Y;        // [N][3] layout
static u32 NEPOCH = 200;

EXPORT u32 init(u32 n, u32 d) {
  heap_ptr = (u32)&__heap_base;
  N = n; D = d;
  X = alloc((u64)N * D * 2); MEAN = alloc(D * 4); V = alloc((u64)PC * D * 4);
  P = alloc((u64)N * PC * 4); Q = alloc((u64)N * PCP * 2);
  KI = alloc((u64)N * KNN * 4); KS = alloc((u64)N * KNN * 4); Y = alloc((u64)N * 12);
  return X && Y ? 0 : 1;
}
EXPORT u16 *ptr_x(void) { return X; }
EXPORT float *ptr_y(void) { return Y; }
EXPORT u32 *ptr_knn(void) { return KI; }
EXPORT float *ptr_knns(void) { return KS; }

// ---------- 1. PCA ----------
static void orthonormalise(float *Vm, u32 k) {  // modified Gram-Schmidt on k rows of length D
  for (u32 a = 0; a < k; a++) {
    float *va = Vm + (u64)a * D;
    for (u32 b = 0; b < a; b++) {
      const float *vb = Vm + (u64)b * D;
      float s = 0; for (u32 i = 0; i < D; i++) s += va[i] * vb[i];
      for (u32 i = 0; i < D; i++) va[i] -= s * vb[i];
    }
    float s = 0; for (u32 i = 0; i < D; i++) s += va[i] * va[i];
    s = 1.0f / (fsqrt(s) + 1e-20f);
    for (u32 i = 0; i < D; i++) va[i] *= s;
  }
}
// Symmetric eigen decomposition of a small k x k matrix (cyclic Jacobi). Columns of E = vectors.
static void jacobi(float *M, float *E, u32 k) {
  for (u32 i = 0; i < k * k; i++) E[i] = 0;
  for (u32 i = 0; i < k; i++) E[i * k + i] = 1;
  for (u32 sweep = 0; sweep < 60; sweep++) {
    float off = 0;
    for (u32 p = 0; p < k; p++) for (u32 q = p + 1; q < k; q++) off += M[p * k + q] * M[p * k + q];
    if (off < 1e-18f) break;
    for (u32 p = 0; p < k; p++) for (u32 q = p + 1; q < k; q++) {
      float apq = M[p * k + q];
      if (apq * apq < 1e-30f) continue;
      float theta = (M[q * k + q] - M[p * k + p]) / (2 * apq);
      float t = (theta >= 0 ? 1 : -1) / ((theta >= 0 ? theta : -theta) + fsqrt(theta * theta + 1));
      float c = 1 / fsqrt(t * t + 1), s = t * c;
      for (u32 r = 0; r < k; r++) {  // M = J^T M J
        float mrp = M[r * k + p], mrq = M[r * k + q];
        M[r * k + p] = c * mrp - s * mrq; M[r * k + q] = s * mrp + c * mrq;
      }
      for (u32 r = 0; r < k; r++) {
        float mpr = M[p * k + r], mqr = M[q * k + r];
        M[p * k + r] = c * mpr - s * mqr; M[q * k + r] = s * mpr + c * mqr;
      }
      for (u32 r = 0; r < k; r++) {
        float erp = E[r * k + p], erq = E[r * k + q];
        E[r * k + p] = c * erp - s * erq; E[r * k + q] = s * erp + c * erq;
      }
    }
  }
}
EXPORT u32 pca(void) {
  for (u32 i = 0; i < D; i++) MEAN[i] = 0;
  for (u32 r = 0; r < N; r++) for (u32 i = 0; i < D; i++) MEAN[i] += h2f(X[(u64)r * D + i]);
  for (u32 i = 0; i < D; i++) MEAN[i] /= (float)N;
  // Covariance from an even sample of rows.
  u32 step = N > 6000 ? N / 6000 : 1, m = 0;
  float *COV = alloc((u64)D * D * 4), *row = alloc(D * 4), *T = alloc((u64)PC * D * 4);
  float *Bm = alloc(PC * PC * 4), *E = alloc(PC * PC * 4);
  if (!COV || !T) return 1;
  for (u64 i = 0; i < (u64)D * D; i++) COV[i] = 0;
  for (u32 r = 0; r < N; r += step, m++) {
    for (u32 i = 0; i < D; i++) row[i] = h2f(X[(u64)r * D + i]) - MEAN[i];
    for (u32 a = 0; a < D; a++) { float ra = row[a]; float *c = COV + (u64)a * D; for (u32 b = a; b < D; b++) c[b] += ra * row[b]; }
  }
  for (u32 a = 0; a < D; a++) for (u32 b = a; b < D; b++) { float v = COV[(u64)a * D + b] / (float)m; COV[(u64)a * D + b] = v; COV[(u64)b * D + a] = v; }
  // Subspace iteration for the top PC directions.
  for (u32 j = 0; j < PC; j++) for (u32 i = 0; i < D; i++) V[(u64)j * D + i] = (float)((rnd() >> 8) & 0xffff) / 65536.0f - 0.5f;
  orthonormalise(V, PC);
  for (u32 it = 0; it < 24; it++) {
    for (u32 j = 0; j < PC; j++) {
      const float *v = V + (u64)j * D; float *t = T + (u64)j * D;
      for (u32 a = 0; a < D; a++) { const float *c = COV + (u64)a * D; float s = 0; for (u32 b = 0; b < D; b++) s += c[b] * v[b]; t[a] = s; }
    }
    for (u64 i = 0; i < (u64)PC * D; i++) V[i] = T[i];
    orthonormalise(V, PC);
  }
  // Rayleigh-Ritz: rotate the basis onto the eigenvectors, largest variance first.
  for (u32 j = 0; j < PC; j++) {
    const float *v = V + (u64)j * D; float *t = T + (u64)j * D;
    for (u32 a = 0; a < D; a++) { const float *c = COV + (u64)a * D; float s = 0; for (u32 b = 0; b < D; b++) s += c[b] * v[b]; t[a] = s; }
  }
  for (u32 p = 0; p < PC; p++) for (u32 q = 0; q < PC; q++) {
    float s = 0; for (u32 i = 0; i < D; i++) s += V[(u64)p * D + i] * T[(u64)q * D + i];
    Bm[p * PC + q] = s;
  }
  jacobi(Bm, E, PC);
  u32 ord[PC]; for (u32 j = 0; j < PC; j++) ord[j] = j;
  for (u32 a = 0; a < PC; a++) for (u32 b = a + 1; b < PC; b++) if (Bm[ord[b] * PC + ord[b]] > Bm[ord[a] * PC + ord[a]]) { u32 t = ord[a]; ord[a] = ord[b]; ord[b] = t; }
  for (u32 j = 0; j < PC; j++) {
    float *t = T + (u64)j * D;
    for (u32 i = 0; i < D; i++) { float s = 0; for (u32 q = 0; q < PC; q++) s += V[(u64)q * D + i] * E[q * PC + ord[j]]; t[i] = s; }
    u32 arg = 0; for (u32 i = 1; i < D; i++) if ((t[i] < 0 ? -t[i] : t[i]) > (t[arg] < 0 ? -t[arg] : t[arg])) arg = i;
    if (t[arg] < 0) for (u32 i = 0; i < D; i++) t[i] = -t[i];  // stable signs
  }
  for (u64 i = 0; i < (u64)PC * D; i++) V[i] = T[i];
  // Project every row; keep an int16 unit-length copy for the neighbour search.
  for (u32 r = 0; r < N; r++) {
    for (u32 i = 0; i < D; i++) row[i] = h2f(X[(u64)r * D + i]) - MEAN[i];
    float *pr = P + (u64)r * PC, nn = 0;
    for (u32 j = 0; j < PC; j++) { const float *v = V + (u64)j * D; float s = 0; for (u32 i = 0; i < D; i++) s += row[i] * v[i]; pr[j] = s; nn += s * s; }
    float inv = 16384.0f / (fsqrt(nn) + 1e-20f);
    i16 *qr = Q + (u64)r * PCP;
    for (u32 j = 0; j < PC; j++) { float v = pr[j] * inv; qr[j] = (i16)(v < 0 ? v - 0.5f : v + 0.5f); }
    for (u32 j = PC; j < PCP; j++) qr[j] = 0;
  }
  heap_ptr = (u32)COV;  // free the PCA scratch
  return 0;
}

// ---------- 2. kNN ----------
#ifdef __wasm_simd128__
#include <wasm_simd128.h>
static inline int dotq(const i16 *a, const i16 *b) {
  v128_t s = wasm_i32x4_splat(0);
  for (u32 i = 0; i < PCP; i += 8) s = wasm_i32x4_add(s, wasm_i32x4_dot_i16x8(wasm_v128_load(a + i), wasm_v128_load(b + i)));
  return wasm_i32x4_extract_lane(s, 0) + wasm_i32x4_extract_lane(s, 1) + wasm_i32x4_extract_lane(s, 2) + wasm_i32x4_extract_lane(s, 3);
}
#else
static inline int dotq(const i16 *a, const i16 *b) { int s = 0; for (u32 i = 0; i < PC; i++) s += (int)a[i] * b[i]; return s; }
#endif
EXPORT void knn_start(void) { for (u64 i = 0; i < (u64)N * KNN; i++) { KS[i] = -2.0f; KI[i] = 0; } }
// Neighbours of rows [s, e): each block of queries scans every row once.
EXPORT void knn_rows(u32 s, u32 e) {
  const float scale = 1.0f / (16384.0f * 16384.0f);
  for (u32 q0 = s; q0 < e; q0 += 32) {
    u32 q1 = q0 + 32 < e ? q0 + 32 : e;
    for (u32 j = 0; j < N; j++) {
      const i16 *rj = Q + (u64)j * PCP;
      for (u32 q = q0; q < q1; q++) {
        if (q == j) continue;
        float sim = (float)dotq(Q + (u64)q * PCP, rj) * scale;
        float *ks = KS + (u64)q * KNN;
        if (sim <= ks[KNN - 1]) continue;
        u32 *ki = KI + (u64)q * KNN, p = KNN - 1;
        while (p > 0 && ks[p - 1] < sim) { ks[p] = ks[p - 1]; ki[p] = ki[p - 1]; p--; }
        ks[p] = sim; ki[p] = j;
      }
    }
  }
}

// ---------- 3. the fuzzy graph ----------
typedef struct { u64 key; float w; u32 pad; } Ent;
static void radix(Ent *a, Ent *tmp, u32 n, u32 bytes) {
  static u32 cnt[256];
  for (u32 pass = 0; pass < bytes; pass++) {
    for (u32 i = 0; i < 256; i++) cnt[i] = 0;
    u32 sh = pass * 8;
    for (u32 i = 0; i < n; i++) cnt[(a[i].key >> sh) & 255]++;
    u32 sum = 0; for (u32 i = 0; i < 256; i++) { u32 c = cnt[i]; cnt[i] = sum; sum += c; }
    for (u32 i = 0; i < n; i++) tmp[cnt[(a[i].key >> sh) & 255]++] = a[i];
    Ent *t = a; a = tmp; tmp = t;
  }
  if (bytes & 1) for (u32 i = 0; i < n; i++) tmp[i] = a[i];  // odd pass count: result is in tmp
}
EXPORT u32 graph(void) {
  // Smooth kNN distances (umap-learn's smooth_knn_dist, local_connectivity = 1).
  float target = flog((float)(KNN + 1)) / 0.69314718f, meanAll = 0;
  for (u64 i = 0; i < (u64)N * KNN; i++) meanAll += 1.0f - KS[i];
  meanAll /= (float)N * KNN;
  u32 M = N * KNN;
  Ent *ent = alloc((u64)M * sizeof(Ent)), *tmp = alloc((u64)M * sizeof(Ent));
  if (!ent || !tmp) return 1;
  for (u32 i = 0; i < N; i++) {
    const float *ks = KS + (u64)i * KNN;
    float rho = 0, mean = 0;
    for (u32 j = 0; j < KNN; j++) { float d = 1.0f - ks[j]; if (d > 0 && rho == 0) rho = d; mean += d; }
    mean /= KNN;
    float lo = 0, hi = 1e30f, mid = 1;
    for (u32 it = 0; it < 64; it++) {
      float psum = 0;
      for (u32 j = 0; j < KNN; j++) { float d = (1.0f - ks[j]) - rho; psum += d > 0 ? fexp(-d / mid) : 1.0f; }
      float diff = psum - target; if (diff < 1e-5f && diff > -1e-5f) break;
      if (psum > target) { hi = mid; mid = (lo + hi) / 2; } else { lo = mid; mid = hi >= 1e30f ? mid * 2 : (lo + hi) / 2; }
    }
    float sigma = mid, floorv = 1e-3f * (rho > 0 ? mean : meanAll);
    if (sigma < floorv) sigma = floorv;
    for (u32 j = 0; j < KNN; j++) {
      u32 k = KI[(u64)i * KNN + j];
      float d = (1.0f - ks[j]) - rho, w = d <= 0 ? 1.0f : fexp(-d / sigma);
      u32 a = i < k ? i : k, b = i < k ? k : i;
      ent[(u64)i * KNN + j].key = (u64)a * N + b; ent[(u64)i * KNN + j].w = w;
    }
  }
  u64 maxKey = (u64)N * N; u32 bytes = 0; while (maxKey) { bytes++; maxKey >>= 8; }
  radix(ent, tmp, M, bytes);  // sorted result is back in ent
  // Fuzzy union of i->j and j->i, then every undirected edge in both directions.
  u32 U = 0;
  for (u32 i = 0; i < M;) {
    float w = ent[i].w; u32 j = i + 1;
    if (j < M && ent[j].key == ent[i].key) { w = w + ent[j].w - w * ent[j].w; j++; }
    while (j < M && ent[j].key == ent[i].key) j++;  // (can't happen, but be safe)
    tmp[U].key = ent[i].key; tmp[U].w = w; U++;
    i = j;
  }
  float wmax = 0; for (u32 i = 0; i < U; i++) if (tmp[i].w > wmax) wmax = tmp[i].w;
  heap_ptr = (u32)ent;  // reuse: ent/tmp are no longer needed once copied below
  Ent *und = tmp;
  // (und lives above ent; allocate the edge arrays after it)
  heap_ptr = (u32)(und + U);
  NE = 0;
  EH = alloc((u64)U * 2 * 4); ET = alloc((u64)U * 2 * 4); EW = alloc((u64)U * 2 * 4);
  EPS = alloc((u64)U * 2 * 4); NEXT = alloc((u64)U * 2 * 4); EPSN = alloc((u64)U * 2 * 4); NEXTN = alloc((u64)U * 2 * 4);
  if (!NEXTN) return 1;
  for (u32 i = 0; i < U; i++) {
    float w = und[i].w;
    if (w < wmax / NEPOCH) continue;  // too weak to be sampled in 200 epochs
    u32 a = (u32)(und[i].key / N), b = (u32)(und[i].key % N);
    float eps = wmax / w;
    EH[NE] = a; ET[NE] = b; EW[NE] = w; EPS[NE] = eps; NEXT[NE] = eps; EPSN[NE] = eps / NEG; NEXTN[NE] = eps / NEG; NE++;
    EH[NE] = b; ET[NE] = a; EW[NE] = w; EPS[NE] = eps; NEXT[NE] = eps; EPSN[NE] = eps / NEG; NEXTN[NE] = eps / NEG; NE++;
  }
  return 0;
}
EXPORT u32 edges(void) { return NE; }

// ---------- 4. layout ----------
// Initialise from the first three principal components, scaled into [0, 10] like umap-learn.
EXPORT void layout_init(void) {
  float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
  for (u32 i = 0; i < N; i++) for (u32 k = 0; k < 3; k++) { float v = P[(u64)i * PC + k]; if (v < lo[k]) lo[k] = v; if (v > hi[k]) hi[k] = v; }
  for (u32 i = 0; i < N; i++) for (u32 k = 0; k < 3; k++) {
    float noise = ((float)((rnd() >> 8) & 0xffff) / 65536.0f - 0.5f) * 1e-4f;
    Y[(u64)i * 3 + k] = 10.0f * (P[(u64)i * PC + k] - lo[k]) / (hi[k] - lo[k] + 1e-12f) + noise;
  }
}
static inline float clip(float x) { return x > 4 ? 4 : x < -4 ? -4 : x; }
// Run epochs [e0, e1) of NEPOCH.
EXPORT void epochs(u32 e0, u32 e1) {
  for (u32 ep = e0; ep < e1; ep++) {
    float alpha = 1.0f - (float)ep / NEPOCH, n = (float)ep;
    for (u32 i = 0; i < NE; i++) {
      if (NEXT[i] > n) continue;
      float *cur = Y + (u64)EH[i] * 3, *oth = Y + (u64)ET[i] * 3;
      float d0 = cur[0] - oth[0], d1 = cur[1] - oth[1], d2v = cur[2] - oth[2], dist2 = d0 * d0 + d1 * d1 + d2v * d2v;
      float g = 0.0f;
      if (dist2 > 0) { float pb = fexp(B * flog(dist2)); g = (-2.0f * A * B * pb / dist2) / (A * pb + 1.0f); }  // d^(b-1) = d^b / d
      float gd;
      gd = clip(g * d0) * alpha; cur[0] += gd; oth[0] -= gd;
      gd = clip(g * d1) * alpha; cur[1] += gd; oth[1] -= gd;
      gd = clip(g * d2v) * alpha; cur[2] += gd; oth[2] -= gd;
      NEXT[i] += EPS[i];
      float fneg = (n - NEXTN[i]) / EPSN[i];
      u32 nneg = fneg > 0 ? (u32)fneg : 0;
      for (u32 p = 0; p < nneg; p++) {
        u32 k = rnd() % N;
        if (k == EH[i]) continue;
        float *o = Y + (u64)k * 3;
        float e0v = cur[0] - o[0], e1v = cur[1] - o[1], e2v = cur[2] - o[2], dd = e0v * e0v + e1v * e1v + e2v * e2v;
        float gr = dd > 0 ? (2.0f * B) / ((0.001f + dd) * (A * fpow(dd, B) + 1.0f)) : 0.0f;
        if (gr > 0) { cur[0] += clip(gr * e0v) * alpha; cur[1] += clip(gr * e1v) * alpha; cur[2] += clip(gr * e2v) * alpha; }
        else { cur[0] += 4 * alpha; cur[1] += 4 * alpha; cur[2] += 4 * alpha; }
      }
      NEXTN[i] += nneg * EPSN[i];
    }
  }
}
