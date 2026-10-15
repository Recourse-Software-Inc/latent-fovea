// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// General decoder-only transformer (C -> wasm32, no libc, no imports), one token at a
// time with a KV cache. One kernel for the families Latent Fovea loads from Hugging Face:
//   gpt2         LayerNorm, learned positions, GELU (tanh) MLP
//   llama/qwen2  RMSNorm, rotary positions, SwiGLU MLP, grouped-query attention, qkv bias (qwen2)
//   qwen3        as llama, plus RMSNorm on each head's q and k
//   gpt_neox     LayerNorm, partial rotary, GELU (erf) MLP, parallel attention + MLP
// Weights stay fp16 in memory ([out][in], one row per output) and are widened on the fly,
// so a model takes 2 bytes per parameter. Activations are f32.
//
// It records what the visualizer shows, for the most recent position:
//   each head's attention over the keys, each head's write strength into the residual
//   stream, every MLP activation, the residual stream after every block, each head's and
//   each MLP's output vector (for direct logit attribution), and (on request) a logit-lens
//   readout at any depth. Token indices are tile rows of vectors.f16.

typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned long long u64;

#define EXPORT __attribute__((visibility("default")))
#define LMAX 64
#define MAXT 256
#define TOPK 10

// ---------- config (written by JS into CFG before init) ----------
enum { V_, C_, L_, NH_, NKV_, HD_, F_, NCTX_, NORM_, NORM_BIAS_, POS_, ROT_, MLP_, QKV_BIAS_, O_BIAS_, MLP_BIAS_,
       QK_NORM_, PARALLEL_, TIED_, STREAM_, NCFG };
// NORM_: 0 LayerNorm, 1 RMSNorm.  POS_: 0 learned, 1 rotary.  MLP_: 0 GELU-tanh, 1 GELU-erf, 2 SwiGLU.
// STREAM_: 1 = weights streamed: every layer shares one buffer (ptr_layer_buf), which the caller
// fills with layer l's file before step_layer(l). Memory then holds one layer instead of all.
static u32 CFG[NCFG];
static float EPS;
static float INV_FREQ[256];  // rotary inverse frequencies (ROT / 2 of them), computed in JS

// Per-layer weight slots, in this order in each layer file. Absent ones have size 0.
enum { S_LN1W, S_LN1B, S_WQ, S_BQ, S_WK, S_BK, S_WV, S_BV, S_QN, S_KN, S_WO, S_BO,
       S_LN2W, S_LN2B, S_W1, S_B1, S_W3, S_W2, S_B2, NSLOT };
// W1 is the up projection (GELU) or the gate (SwiGLU); W3 is SwiGLU's up projection.

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

// ---------- math ----------
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
static float fsin(float x) {  // |error| < 2e-7 after reduction to [-pi, pi]
  const float TWO_PI = 6.28318530718f, PI = 3.14159265359f;
  x -= TWO_PI * __builtin_floorf((x + PI) / TWO_PI);
  float s = 1.0f;
  if (x > PI * 0.5f) x = PI - x; else if (x < -PI * 0.5f) x = -PI - x;
  float x2 = x * x;
  return s * x * (1.0f + x2 * (-0.16666667f + x2 * (0.008333331f + x2 * (-0.00019840874f + x2 * (2.7525562e-06f - x2 * 2.3889859e-08f)))));
}
static inline float fcos(float x) { return fsin(x + 1.57079632679f); }
static inline float ftanh(float x) { float e = fexp(2.0f * x); return 1.0f - 2.0f / (e + 1.0f); }
static float ferf(float x) {  // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
  float s = x < 0 ? -1.0f : 1.0f; x = x < 0 ? -x : x;
  float t = 1.0f / (1.0f + 0.3275911f * x);
  float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * fexp(-x * x);
  return s * y;
}
static inline float gelu_tanh(float x) { return 0.5f * x * (1.0f + ftanh(0.7978845608f * (x + 0.044715f * x * x * x))); }
static inline float gelu_erf(float x) { return 0.5f * x * (1.0f + ferf(x * 0.70710678f)); }
static inline float silu(float x) { return x / (1.0f + fexp(-x)); }

// fp16 bits -> f32, exact for normals, subnormals and zero (weights hold no inf/NaN).
static inline float h2f(u16 h) {
  union { u32 u; float f; } v = {(u32)(h & 0x7fff) << 13};
  v.f *= 5.192296858534828e+33f;  // 2^112
  v.u |= (u32)(h & 0x8000) << 16;
  return v.f;
}

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
static inline v128_t h2f4(v128_t h32) {  // four fp16 (in u32 lanes) -> four f32
  v128_t m = wasm_i32x4_shl(wasm_v128_and(h32, wasm_i32x4_splat(0x7fff)), 13);
  v128_t f = wasm_f32x4_mul(m, wasm_f32x4_splat(5.192296858534828e+33f));
  return wasm_v128_or(f, wasm_i32x4_shl(wasm_v128_and(h32, wasm_i32x4_splat(0x8000)), 16));
}
static float dot16(const u16 *w, const float *x, u32 n) {  // n % 8 == 0
  v128_t a0 = wasm_f32x4_splat(0), a1 = wasm_f32x4_splat(0);
  for (u32 i = 0; i < n; i += 8) {
    v128_t h = wasm_v128_load(w + i);
    a0 = wasm_f32x4_add(a0, wasm_f32x4_mul(h2f4(wasm_u32x4_extend_low_u16x8(h)), wasm_v128_load(x + i)));
    a1 = wasm_f32x4_add(a1, wasm_f32x4_mul(h2f4(wasm_u32x4_extend_high_u16x8(h)), wasm_v128_load(x + i + 4)));
  }
  a0 = wasm_f32x4_add(a0, a1);
  return wasm_f32x4_extract_lane(a0, 0) + wasm_f32x4_extract_lane(a0, 1) + wasm_f32x4_extract_lane(a0, 2) + wasm_f32x4_extract_lane(a0, 3);
}
#else
static float dot16(const u16 *w, const float *x, u32 n) {
  float s0 = 0, s1 = 0;
  for (u32 i = 0; i < n; i += 2) { s0 += h2f(w[i]) * x[i]; s1 += h2f(w[i + 1]) * x[i + 1]; }
  return s0 + s1;
}
#endif
// out[o] = b[o] + W[o] . x   for W [O][I] fp16
static void matvec(const u16 *W, const u16 *b, const float *x, float *out, u32 O, u32 I) {
  for (u32 o = 0; o < O; o++) out[o] = dot16(W + (u64)o * I, x, I) + (b ? h2f(b[o]) : 0.0f);
}

// ---------- state ----------
static u32 V, C, L, NH, NKV, HD, F, QD, KD;
static u16 *LW[LMAX][NSLOT];
static u32 SLOTN[NSLOT];
static u16 *WTE, *LMHEAD, *WPE, *LNFW, *LNFB;
static u16 *stage;
static u32 stageCap;
static float *KC, *VC;       // [L][MAXT][KD]
static float *ATT;           // [L][NH][MAXT], probabilities for the latest query
static float *HEADN;         // [L][NH]
static float *MLPA;          // [L][F]
static float *RES;           // [(L + 1)][C]
static float LENS_P[TOPK], LENS_H;
static u32 LENS_I[TOPK];
static float *x, *xn, *x2, *q, *k, *v, *y, *tmp, *contrib, *hb, *hb2, *LOGITS;
static float *HOUT;          // [L][NH][C] each head's output vector (after projection)
static float *MOUT;          // [L][C] each MLP's output vector (with its bias)
static float *ATTR;          // attribution: L x NH heads, L MLPs, embedding, rest, logit, 1/scale
static float *UG, *GAIN, *DIRV;  // [C] unembedding row x final-norm weight; [F] per-neuron gain; [C] scratch
static float *UMEAN;             // [C] the average word's unembedding row (computed on first use)
static float *QVEC;              // [C] a residual-stream vector to read out on request (lens_q)
// ---------- interventions ----------
// Scale any head's output, any layer's whole MLP output, or single neurons' activations, inside
// the forward pass, so every later layer (and the context kept for later tokens) sees the changed
// model. 0 knocks a part out; 1 leaves it alone. Recordings show the changed values.
#define NKO_MAX 256
static float *HSCALE, *MSCALE, NKO_S[NKO_MAX];  // [L][NH], [L]
static u32 NKO_L[NKO_MAX], NKO_I[NKO_MAX], nko;
EXPORT void iv_clear(void) { for (u32 j = 0; j < L * NH; j++) HSCALE[j] = 1; for (u32 l = 0; l < L; l++) MSCALE[l] = 1; nko = 0; }
EXPORT void iv_head(u32 l, u32 h, float s) { if (l < L && h < NH) HSCALE[l * NH + h] = s; }
EXPORT void iv_mlp(u32 l, float s) { if (l < L) MSCALE[l] = s; }
EXPORT u32 iv_neuron(u32 l, u32 i, float s) { if (nko >= NKO_MAX || i >= F) return 1; NKO_L[nko] = l; NKO_I[nko] = i; NKO_S[nko] = s; nko++; return 0; }
static u32 umean_ready;
static float NV_P[2 * TOPK];     // a neuron's most promoted, then most suppressed, words
static u32 NV_I[2 * TOPK];
static u32 LAST_POS, CUR_POS;
static u16 *LBUF;            // streaming: the one layer buffer every layer points into

static void slot_sizes(void) {
  u32 nb = CFG[NORM_BIAS_], qb = CFG[QKV_BIAS_], ob = CFG[O_BIAS_], mb = CFG[MLP_BIAS_], sw = CFG[MLP_] == 2;
  u32 s[NSLOT] = {
    C, nb ? C : 0, QD * C, qb ? QD : 0, KD * C, qb ? KD : 0, KD * C, qb ? KD : 0,
    CFG[QK_NORM_] ? HD : 0, CFG[QK_NORM_] ? HD : 0, C * QD, ob ? C : 0,
    C, nb ? C : 0, F * C, mb ? F : 0, sw ? F * C : 0, C * F, mb ? C : 0,
  };
  for (u32 i = 0; i < NSLOT; i++) SLOTN[i] = s[i];
}

EXPORT u32 *ptr_cfg(void) { return CFG; }
EXPORT float *ptr_eps(void) { return &EPS; }
EXPORT float *ptr_inv_freq(void) { return INV_FREQ; }
EXPORT u32 layer_halves(void) { u32 n = 0; for (u32 i = 0; i < NSLOT; i++) n += SLOTN[i]; return n; }
EXPORT u32 slot_halves(u32 s) { return s < NSLOT ? SLOTN[s] : 0; }

// Returns 0 on success, 1 out of memory, 2 unsupported shape.
EXPORT u32 init(void) {
  heap_ptr = (u32)&__heap_base;
  V = CFG[V_]; C = CFG[C_]; L = CFG[L_]; NH = CFG[NH_]; NKV = CFG[NKV_]; HD = CFG[HD_]; F = CFG[F_];
  QD = NH * HD; KD = NKV * HD;
  if (L < 1 || L > LMAX || NKV < 1 || NH % NKV || C % 8 || HD % 8 || F % 8 || CFG[ROT_] > HD || CFG[ROT_] > 512) return 2;
  slot_sizes();
  if (CFG[STREAM_]) {  // one shared layer buffer, slots at their offsets in the layer file
    LBUF = alloc((u64)layer_halves() * 2);
    if (!LBUF) return 1;
    for (u32 l = 0; l < L; l++) { u64 at = 0; for (u32 s = 0; s < NSLOT; s++) { LW[l][s] = SLOTN[s] ? LBUF + at : 0; at += SLOTN[s]; } }
  } else for (u32 l = 0; l < L; l++) for (u32 s = 0; s < NSLOT; s++) {
    LW[l][s] = SLOTN[s] ? alloc((u64)SLOTN[s] * 2) : 0;
    if (SLOTN[s] && !LW[l][s]) return 1;
  }
  WTE = alloc((u64)V * C * 2);
  LMHEAD = CFG[TIED_] ? WTE : alloc((u64)V * C * 2);
  WPE = CFG[POS_] == 0 ? alloc((u64)CFG[NCTX_] * C * 2) : 0;
  LNFW = alloc(C * 2); LNFB = CFG[NORM_BIAS_] ? alloc(C * 2) : 0;
  KC = alloc((u64)L * MAXT * KD * 4); VC = alloc((u64)L * MAXT * KD * 4);
  ATT = alloc((u64)L * NH * MAXT * 4); HEADN = alloc(L * NH * 4);
  MLPA = alloc((u64)L * F * 4); RES = alloc((u64)(L + 1) * C * 4);
  x = alloc(C * 4); xn = alloc(C * 4); x2 = alloc(C * 4); tmp = alloc(C * 4); contrib = alloc(C * 4);
  q = alloc(QD * 4); k = alloc(KD * 4); v = alloc(KD * 4); y = alloc(QD * 4);
  hb = alloc(F * 4); hb2 = alloc(F * 4); LOGITS = alloc((u64)V * 4);
  HOUT = alloc((u64)L * NH * C * 4); MOUT = alloc((u64)L * C * 4); ATTR = alloc((L * NH + L + 4) * 4);
  UG = alloc(C * 4); GAIN = alloc(F * 4); DIRV = alloc(C * 4); UMEAN = alloc(C * 4); umean_ready = 0; QVEC = alloc(C * 4);
  HSCALE = alloc(L * NH * 4); MSCALE = alloc(L * 4);
  if (!HSCALE || !MSCALE) return 1;
  iv_clear();
  stageCap = 4u << 20; stage = alloc(stageCap * 2);
  return stage && LOGITS && WTE && LMHEAD && HOUT && UMEAN ? 0 : 1;
}
EXPORT u32 max_t(void) { return MAXT; }
EXPORT u16 *ptr_layer_buf(void) { return LBUF; }
EXPORT u32 stage_halves(void) { return stageCap; }
EXPORT u16 *ptr_stage(void) { return stage; }
EXPORT float *ptr_att(void) { return ATT; }
EXPORT float *ptr_headn(void) { return HEADN; }
EXPORT float *ptr_mlp(void) { return MLPA; }
EXPORT float *ptr_res(void) { return RES; }
EXPORT float *ptr_mout(void) { return MOUT; }
// The size of what token pos offers to query head h at layer l: the norm of its value vector.
EXPORT float vnorm(u32 l, u32 pos, u32 h) {
  const float *v = VC + ((u64)l * MAXT + pos) * KD + (h / (NH / NKV)) * HD; float s = 0;
  for (u32 i = 0; i < HD; i++) s += v[i] * v[i];
  return __builtin_sqrtf(s);
}
EXPORT float *ptr_lens_p(void) { return LENS_P; }
EXPORT u32 *ptr_lens_i(void) { return LENS_I; }
EXPORT float lens_entropy(void) { return LENS_H; }

// Copy `n` fp16 values from the stage buffer. which: 0..L-1 = layer file (slots in order),
// 100 = misc (wpe if learned, final norm weight, final norm bias if any),
// 200 = token embedding rows, 201 = unembedding rows (untied models). offset in halves.
static void copy16(u16 *dst, u32 n) { for (u32 i = 0; i < n; i++) dst[i] = stage[i]; }
EXPORT u32 load(u32 which, u32 offset, u32 n) {
  if (which < L) {
    for (u32 i = 0; i < n;) {  // the layer file is its slots concatenated
      u32 at = offset + i, s = 0, base = 0;
      while (s < NSLOT && at >= base + SLOTN[s]) { base += SLOTN[s]; s++; }
      if (s == NSLOT) return 1;
      u32 m = SLOTN[s] - (at - base); if (m > n - i) m = n - i;
      u16 *dst = LW[which][s] + (at - base);
      for (u32 j = 0; j < m; j++) dst[j] = stage[i + j];
      i += m;
    }
    return 0;
  }
  if (which == 200) { copy16(WTE + offset, n); return 0; }
  if (which == 201) { if (CFG[TIED_]) return 1; copy16(LMHEAD + offset, n); return 0; }
  if (which != 100) return 1;
  u32 np = CFG[POS_] == 0 ? CFG[NCTX_] * C : 0;
  for (u32 i = 0; i < n; i++) {
    u32 at = offset + i;
    if (at < np) WPE[at] = stage[i];
    else if (at < np + C) LNFW[at - np] = stage[i];
    else if (LNFB && at < np + 2 * C) LNFB[at - np - C] = stage[i];
  }
  return 0;
}

// ---------- blocks ----------
static void norm(const float *in, const u16 *w, const u16 *b, float *out, u32 n) {
  if (CFG[NORM_] == 1) {  // RMSNorm
    float ss = 0;
    for (u32 i = 0; i < n; i++) ss += in[i] * in[i];
    float r = 1.0f / fsqrt(ss / n + EPS);
    for (u32 i = 0; i < n; i++) out[i] = in[i] * r * h2f(w[i]);
    return;
  }
  float m = 0, var = 0;
  for (u32 i = 0; i < n; i++) m += in[i];
  m /= n;
  for (u32 i = 0; i < n; i++) { float d = in[i] - m; var += d * d; }
  float r = 1.0f / fsqrt(var / n + EPS);
  for (u32 i = 0; i < n; i++) out[i] = (in[i] - m) * r * h2f(w[i]) + (b ? h2f(b[i]) : 0.0f);
}
static void rmsnorm_head(float *h, const u16 *w) {  // qwen3's per-head q/k norm (always RMS)
  float ss = 0;
  for (u32 i = 0; i < HD; i++) ss += h[i] * h[i];
  float r = 1.0f / fsqrt(ss / HD + EPS);
  for (u32 i = 0; i < HD; i++) h[i] = h[i] * r * h2f(w[i]);
}
static void rope(float *h, u32 pos) {  // rotate_half pairing over the first ROT dims of the head
  u32 half = CFG[ROT_] / 2;
  for (u32 i = 0; i < half; i++) {
    float a = (float)pos * INV_FREQ[i], c = fcos(a), s = fsin(a);
    float x1 = h[i], x2v = h[i + half];
    h[i] = x1 * c - x2v * s;
    h[i + half] = x2v * c + x1 * s;
  }
}

// Attention for layer l on input xin (already normed); adds its output into acc.
static void attention(u32 l, const float *xin, u32 pos, float *acc) {
  u16 **w = LW[l];
  matvec(w[S_WQ], w[S_BQ], xin, q, QD, C);
  matvec(w[S_WK], w[S_BK], xin, k, KD, C);
  matvec(w[S_WV], w[S_BV], xin, v, KD, C);
  for (u32 h = 0; h < NH; h++) {
    if (CFG[QK_NORM_]) rmsnorm_head(q + h * HD, w[S_QN]);
    if (CFG[POS_] == 1) rope(q + h * HD, pos);
  }
  for (u32 h = 0; h < NKV; h++) {
    if (CFG[QK_NORM_]) rmsnorm_head(k + h * HD, w[S_KN]);
    if (CFG[POS_] == 1) rope(k + h * HD, pos);
  }
  float *kc = KC + ((u64)l * MAXT + pos) * KD, *vc = VC + ((u64)l * MAXT + pos) * KD;
  for (u32 i = 0; i < KD; i++) { kc[i] = k[i]; vc[i] = v[i]; }
  float scale = 1.0f / fsqrt((float)HD);
  u32 group = NH / NKV;
  for (u32 h = 0; h < NH; h++) {
    const float *qh = q + h * HD;
    u32 kvh = h / group;
    float *att = ATT + ((u64)l * NH + h) * MAXT, mx = -1e30f;
    for (u32 t = 0; t <= pos; t++) {
      const float *kt = KC + ((u64)l * MAXT + t) * KD + kvh * HD;
      float s = 0;
      for (u32 i = 0; i < HD; i++) s += qh[i] * kt[i];
      s *= scale; att[t] = s; if (s > mx) mx = s;
    }
    float sum = 0;
    for (u32 t = 0; t <= pos; t++) { att[t] = fexp(att[t] - mx); sum += att[t]; }
    float inv = 1.0f / sum;
    for (u32 t = 0; t <= pos; t++) att[t] *= inv;
    float *yh = y + h * HD;
    for (u32 i = 0; i < HD; i++) yh[i] = 0;
    for (u32 t = 0; t <= pos; t++) {
      const float *vt = VC + ((u64)l * MAXT + t) * KD + kvh * HD;
      float a = att[t];
      for (u32 i = 0; i < HD; i++) yh[i] += a * vt[i];
    }
  }
  // Output projection, one head at a time so each head's write can be measured.
  const u16 *wo = w[S_WO], *bo = w[S_BO];
  for (u32 o = 0; o < C; o++) tmp[o] = bo ? h2f(bo[o]) : 0.0f;
  for (u32 h = 0; h < NH; h++) {
    float nn = 0, *hout = HOUT + ((u64)l * NH + h) * C, hs = HSCALE[l * NH + h];
    for (u32 o = 0; o < C; o++) {
      float s = dot16(wo + (u64)o * QD + h * HD, y + h * HD, HD) * hs;
      tmp[o] += s; nn += s * s; hout[o] = s;
    }
    HEADN[l * NH + h] = fsqrt(nn);
  }
  for (u32 o = 0; o < C; o++) acc[o] += tmp[o];
}

static void mlp(u32 l, const float *xin, float *acc) {
  u16 **w = LW[l];
  float *act = MLPA + (u64)l * F;
  matvec(w[S_W1], w[S_B1], xin, hb, F, C);
  if (CFG[MLP_] == 2) {
    matvec(w[S_W3], 0, xin, hb2, F, C);
    for (u32 i = 0; i < F; i++) hb[i] = silu(hb[i]) * hb2[i];
  } else if (CFG[MLP_] == 1) for (u32 i = 0; i < F; i++) hb[i] = gelu_erf(hb[i]);
  else for (u32 i = 0; i < F; i++) hb[i] = gelu_tanh(hb[i]);
  for (u32 j = 0; j < nko; j++) if (NKO_L[j] == l) hb[NKO_I[j]] *= NKO_S[j];
  for (u32 i = 0; i < F; i++) act[i] = hb[i];
  matvec(w[S_W2], w[S_B2], hb, tmp, C, F);
  if (MSCALE[l] != 1.0f) for (u32 o = 0; o < C; o++) tmp[o] *= MSCALE[l];
  for (u32 o = 0; o < C; o++) { acc[o] += tmp[o]; MOUT[(u64)l * C + o] = tmp[o]; }
}

// Run one position in three parts, so a caller streaming weights can load each layer in
// between: step_begin (the embedding), step_layer(l) for every layer in order, step_end.
// `row` is the token's tile row.
EXPORT u32 step_begin(u32 row, u32 pos) {
  if (pos >= MAXT || (CFG[POS_] == 0 && pos >= CFG[NCTX_])) return 1;
  const u16 *te = WTE + (u64)row * C;
  for (u32 i = 0; i < C; i++) x[i] = h2f(te[i]) + (WPE ? h2f(WPE[(u64)pos * C + i]) : 0.0f);
  for (u32 i = 0; i < C; i++) RES[i] = x[i];
  CUR_POS = pos;
  return 0;
}
EXPORT void step_layer(u32 l) {
  u32 pos = CUR_POS;
  {
    u16 **w = LW[l];
    if (CFG[PARALLEL_]) {  // x + attn(ln1(x)) + mlp(ln2(x))
      norm(x, w[S_LN1W], w[S_LN1B], xn, C);
      norm(x, w[S_LN2W], w[S_LN2B], x2, C);
      for (u32 i = 0; i < C; i++) contrib[i] = x[i];
      attention(l, xn, pos, contrib);
      mlp(l, x2, contrib);
      for (u32 i = 0; i < C; i++) x[i] = contrib[i];
    } else {
      norm(x, w[S_LN1W], w[S_LN1B], xn, C);
      attention(l, xn, pos, x);
      norm(x, w[S_LN2W], w[S_LN2B], xn, C);
      mlp(l, xn, x);
    }
    for (u32 i = 0; i < C; i++) RES[(u64)(l + 1) * C + i] = x[i];
  }
}
EXPORT void step_end(void) { LAST_POS = CUR_POS; }
EXPORT u32 step(u32 row, u32 pos) {  // all layers resident
  if (step_begin(row, pos)) return 1;
  for (u32 l = 0; l < L; l++) step_layer(l);
  step_end();
  return 0;
}

// Logit lens at a depth (0 = embedding .. L = after the last block): final norm, then the
// unembedding; keeps the top TOPK and the entropy of the full distribution.
static void lens_on(const float *vec);
EXPORT void lens(u32 depth) { lens_on(RES + (u64)depth * C); }
// Logit lens on any residual-stream vector written into QVEC (a stored depth of an earlier token).
EXPORT void lens_q(void) { lens_on(QVEC); }
EXPORT float *ptr_qvec(void) { return QVEC; }
// After a logit lens: the probability of any word in that readout (LOGITS holds exp(logit - max)).
static float LENS_SUM;
EXPORT float prob_of(u32 row) { return row < V && LENS_SUM > 0 ? LOGITS[row] / LENS_SUM : 0; }
static void lens_on(const float *vec) {
  norm(vec, LNFW, LNFB, xn, C);
  float mx = -1e30f;
  for (u32 r = 0; r < V; r++) { float s = dot16(LMHEAD + (u64)r * C, xn, C); LOGITS[r] = s; if (s > mx) mx = s; }
  float sum = 0;
  for (u32 r = 0; r < V; r++) { float e = fexp(LOGITS[r] - mx); LOGITS[r] = e; sum += e; }
  float inv = 1.0f / sum, H = 0;
  LENS_SUM = sum;
  for (u32 j = 0; j < TOPK; j++) { LENS_P[j] = -1; LENS_I[j] = 0; }
  for (u32 r = 0; r < V; r++) {
    float p = LOGITS[r] * inv;
    if (p > 1e-12f) H -= p * flog(p);
    if (p > LENS_P[TOPK - 1]) {
      u32 j = TOPK - 1;
      while (j > 0 && LENS_P[j - 1] < p) { LENS_P[j] = LENS_P[j - 1]; LENS_I[j] = LENS_I[j - 1]; j--; }
      LENS_P[j] = p; LENS_I[j] = r;
    }
  }
  LENS_H = H;
}

// ---------- attribution ----------
// The residual stream is a sum: the embedding, every head's write, every MLP's write, and
// the biases. Holding the final norm's scale at this token's value, the logit of a word
// splits exactly into one term per part (direct logit attribution). Only the direct path is
// counted: a part that matters through later layers shows up in them, not here.
// LayerNorm also subtracts each vector's mean; RMSNorm doesn't.
// Every word's logit shares a large common offset that cancels in the softmax, so the score
// attributed is the word's logit minus the average word's: what actually sets the odds.
static float set_ug(u32 row) {  // UG = (unembedding row - average row) x final-norm weight; returns sum(UG)
  if (!umean_ready) {
    for (u32 i = 0; i < C; i++) {  // in double: a few huge residual dimensions amplify any rounding
      double s = 0;
      for (u32 r = 0; r < V; r++) s += h2f(LMHEAD[(u64)r * C + i]);
      UMEAN[i] = (float)(s / V);
    }
    umean_ready = 1;
  }
  const u16 *u = LMHEAD + (u64)row * C;
  float sug = 0;
  for (u32 i = 0; i < C; i++) { UG[i] = (h2f(u[i]) - UMEAN[i]) * h2f(LNFW[i]); sug += UG[i]; }
  return sug;
}
static float part(const float *c, float sug, float inv) {  // one vector's share of the logit
  float s = 0, m = 0;
  for (u32 i = 0; i < C; i++) { s += UG[i] * c[i]; m += c[i]; }
  return CFG[NORM_] == 1 ? inv * s : inv * (s - sug * m / C);
}
static float final_inv(const float *fin, float *mean) {  // 1/scale of the final norm here
  float m = 0, ss = 0;
  if (CFG[NORM_] == 1) { for (u32 i = 0; i < C; i++) ss += fin[i] * fin[i]; *mean = 0; return 1.0f / fsqrt(ss / C + EPS); }
  for (u32 i = 0; i < C; i++) m += fin[i];
  m /= C;
  for (u32 i = 0; i < C; i++) { float d = fin[i] - m; ss += d * d; }
  *mean = m;
  return 1.0f / fsqrt(ss / C + EPS);
}

// Attribute word `row`'s score (logit minus the average word's) at the most recent position.
// ATTR holds, in order: L x NH heads, L MLPs, the embedding (with learned positions if any),
// the rest (biases), the score, and 1/scale.
EXPORT u32 attribute(u32 row) {
  if (row >= V) return 1;
  const float *fin = RES + (u64)L * C;
  float m, inv = final_inv(fin, &m), sug = set_ug(row), sum = 0;
  for (u32 j = 0; j < L * NH; j++) { ATTR[j] = part(HOUT + (u64)j * C, sug, inv); sum += ATTR[j]; }
  for (u32 l = 0; l < L; l++) { ATTR[L * NH + l] = part(MOUT + (u64)l * C, sug, inv); sum += ATTR[L * NH + l]; }
  float emb = part(RES, sug, inv), logit = 0;
  const u16 *u = LMHEAD + (u64)row * C;
  for (u32 i = 0; i < C; i++) logit += (h2f(u[i]) - UMEAN[i]) * ((fin[i] - m) * inv * h2f(LNFW[i]) + (LNFB ? h2f(LNFB[i]) : 0.0f));
  float *t = ATTR + L * NH + L;
  t[0] = emb; t[1] = logit - sum - emb; t[2] = logit; t[3] = inv;
  return 0;
}
EXPORT float *ptr_attr(void) { return ATTR; }

// Each neuron of layer l: its direct effect on word `row` per unit of activation, before the
// final norm's 1/scale. A neuron's effect at a position = activation x gain x 1/scale there.
// W2 is [C][F] (one row per output), so neuron i's output direction is column i.
EXPORT u32 neuron_gain(u32 l, u32 row) {
  if (l >= L || row >= V) return 1;
  float sug = set_ug(row);
  const u16 *w2 = LW[l][S_W2];
  for (u32 i = 0; i < F; i++) { GAIN[i] = 0; hb2[i] = 0; }  // hb2: column sums, for LayerNorm's mean
  for (u32 o = 0; o < C; o++) {
    const u16 *r = w2 + (u64)o * F;
    float g = UG[o];
    for (u32 i = 0; i < F; i++) { float w = h2f(r[i]); GAIN[i] += g * w; hb2[i] += w; }
  }
  if (CFG[NORM_] != 1) for (u32 i = 0; i < F; i++) GAIN[i] -= sug * hb2[i] / C;
  return 0;
}
EXPORT float *ptr_gain(void) { return GAIN; }

static u32 rank_dirv(void);
// Neuron i of layer l: the words its output direction pushes up most and down most,
// through the final norm's weight (per unit activation, before 1/scale).
EXPORT u32 neuron_vocab(u32 l, u32 i) {
  if (l >= L || i >= F) return 1;
  const u16 *w2 = LW[l][S_W2];
  float m = 0;
  for (u32 o = 0; o < C; o++) { DIRV[o] = h2f(w2[(u64)o * F + i]); m += DIRV[o]; }
  m = CFG[NORM_] == 1 ? 0 : m / C;
  for (u32 o = 0; o < C; o++) DIRV[o] = (DIRV[o] - m) * h2f(LNFW[o]);
  return rank_dirv();
}
// Any direction written into QVEC (an SAE feature's, say): the words it pushes up and down most.
EXPORT u32 vocab_q(void) {
  float m = 0;
  for (u32 o = 0; o < C; o++) m += QVEC[o];
  m = CFG[NORM_] == 1 ? 0 : m / C;
  for (u32 o = 0; o < C; o++) DIRV[o] = (QVEC[o] - m) * h2f(LNFW[o]);
  return rank_dirv();
}
static u32 rank_dirv(void) {  // top and bottom TOPK words along DIRV
  for (u32 j = 0; j < TOPK; j++) { NV_P[j] = -1e30f; NV_P[TOPK + j] = 1e30f; NV_I[j] = NV_I[TOPK + j] = 0; }
  for (u32 r = 0; r < V; r++) {
    float s = dot16(LMHEAD + (u64)r * C, DIRV, C);
    if (s > NV_P[TOPK - 1]) {
      u32 j = TOPK - 1;
      while (j > 0 && NV_P[j - 1] < s) { NV_P[j] = NV_P[j - 1]; NV_I[j] = NV_I[j - 1]; j--; }
      NV_P[j] = s; NV_I[j] = r;
    }
    if (s < NV_P[2 * TOPK - 1]) {
      u32 j = 2 * TOPK - 1;
      while (j > TOPK && NV_P[j - 1] > s) { NV_P[j] = NV_P[j - 1]; NV_I[j] = NV_I[j - 1]; j--; }
      NV_P[j] = s; NV_I[j] = r;
    }
  }
  return 0;
}
EXPORT float *ptr_nv_p(void) { return NV_P; }
EXPORT u32 *ptr_nv_i(void) { return NV_I; }
