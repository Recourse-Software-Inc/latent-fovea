// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// MDLM forward pass (C -> wasm32, no libc, no imports).
//
// MDLM is a masked diffusion language model with a DiT backbone: bidirectional
// attention with rotary positions, LayerNorms without bias, and adaLN modulation.
// mdlm-owt runs without time conditioning, so the modulations are constants
// precomputed by tools/build_data.py (mods.f32).
//
// One call to forward() runs every position at once (there is no KV cache: each
// denoising step sees the whole sequence). Weights are streamed once per call and
// applied to all positions, tile by tile. It records, for the visualizer:
//   attention probabilities for every layer x head x (query, key),
//   each head's write strength per position, the MLP activations,
//   the residual stream after every block, and for masked positions the model's
//   belief over the vocabulary (top-k, entropy; full probabilities for sampling).
// Token indices are tile rows (the viewer's point order), as in gpt2.c.

typedef unsigned int u32;
typedef int i32;
typedef unsigned short u16;

#define EXPORT __attribute__((visibility("default")))
#define LMAX 12
#define NH 12
#define C 768
#define HS 64
#define F 3072
#define PMAX 128
#define TOPK 8
#define BLOCK_FLOATS (C + C * 3 * C + C * C + C + C * F + F + F * C + C)
#define OT 256  // output tile width for the batched matmul

extern unsigned char __heap_base;
static u32 heap_ptr;
static void *alloc(u32 bytes) {
  u32 p = (heap_ptr + 15u) & ~15u, end = p + bytes;
  u32 have = (u32)__builtin_wasm_memory_size(0) * 65536u;
  if (end > have && __builtin_wasm_memory_grow(0, (end - have + 65535u) / 65536u) == (unsigned long)-1) return 0;
  heap_ptr = end;
  return (void *)p;
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
  return 2.0f * t * (1.0f + t2 * (0.33333333f + t2 * (0.2f + t2 * (0.14285714f + t2 * 0.11111111f)))) + (float)e * 0.69314718f;
}
// sin/cos for rotary angles (range-reduced, ~1e-7)
static void fsincos(float x, float *s, float *c) {
  const float TAU = 6.28318530718f, PI = 3.14159265359f;
  x = x - TAU * __builtin_floorf(x / TAU + 0.5f);  // [-pi, pi)
  float sgn = 1.0f, y = x;
  if (y > PI * 0.5f) { y = PI - y; sgn = -1.0f; } else if (y < -PI * 0.5f) { y = -PI - y; sgn = -1.0f; }
  float y2 = y * y;
  *s = y * (1.0f - y2 / 6.0f * (1.0f - y2 / 20.0f * (1.0f - y2 / 42.0f * (1.0f - y2 / 72.0f * (1.0f - y2 / 110.0f)))));
  float cc = 1.0f - y2 / 2.0f * (1.0f - y2 / 12.0f * (1.0f - y2 / 30.0f * (1.0f - y2 / 56.0f * (1.0f - y2 / 90.0f * (1.0f - y2 / 132.0f)))));
  *c = sgn * cc;
}
static inline float ftanh(float x) { float e = fexp(2.0f * x); return 1.0f - 2.0f / (e + 1.0f); }
static inline float gelu(float x) { return 0.5f * x * (1.0f + ftanh(0.7978845608f * (x + 0.044715f * x * x * x))); }
static float half_to_float(u16 h) {
  u32 s = (u32)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, bits;
  if (e == 0) {
    if (m == 0) bits = s;
    else { e = 1; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3ff; bits = s | ((e + 112) << 23) | (m << 13); }
  } else if (e == 31) bits = s | 0x7f800000u | (m << 13);
  else bits = s | ((e + 112) << 23) | (m << 13);
  union { u32 u; float f; } v = {bits};
  return v.f;
}

// ---------- state ----------
static u32 L, V, MASKROW;
static float *BLK[LMAX], *MODS, *FINAL, *EMB, *UNEMB;
static u16 *stage;
static u32 stageCap;

static u32 P;                       // positions in the current sequence
static u32 ROWS[PMAX];              // current tokens (tile rows; MASKROW = masked)
static float *X, *XN, *QKV, *Y, *HB, *TMP;
static float *ATT;                  // [L][NH][P][P]
static float *HEADN;                // [L][P][NH]
static float *MLP;                  // [L][P][F]
static float *RES;                  // [L+1][P][C]
static float *PROBS;                // [PMAX][V] for masked positions
static float *LSCR;                 // [V] scratch for the lens
static float TOPP[PMAX * TOPK], ENT[PMAX];
static u32 TOPI[PMAX * TOPK];
static float LENS_P[(LMAX + 1) * TOPK], LENS_H[LMAX + 1];
static u32 LENS_I[(LMAX + 1) * TOPK];
static float invf[HS / 2];

EXPORT u32 init(u32 vocab, u32 nlayer, u32 maskRow) {
  heap_ptr = (u32)&__heap_base;
  if (nlayer < 1 || nlayer > LMAX) return 2;
  L = nlayer; V = vocab; MASKROW = maskRow;
  for (u32 l = 0; l < L; l++) BLK[l] = alloc(BLOCK_FLOATS * 4);
  MODS = alloc((L * 6 + 2) * C * 4);
  FINAL = alloc((C + V) * 4);
  EMB = alloc(V * C * 4); UNEMB = alloc(V * C * 4);
  X = alloc(PMAX * C * 4); XN = alloc(PMAX * C * 4); QKV = alloc(PMAX * 3 * C * 4);
  Y = alloc(PMAX * C * 4); HB = alloc(PMAX * F * 4); TMP = alloc(PMAX * C * 4);
  ATT = alloc(L * NH * PMAX * PMAX * 4); HEADN = alloc(L * PMAX * NH * 4);
  MLP = alloc(L * PMAX * F * 4); RES = alloc((L + 1) * PMAX * C * 4);
  PROBS = alloc(PMAX * V * 4); LSCR = alloc(V * 4);
  stageCap = 8u << 20; stage = alloc(stageCap * 2);
  for (int i = 0; i < HS / 2; i++) invf[i] = fexp(-flog(10000.0f) * (2.0f * i) / HS);
  return stage ? 0 : 1;
}
EXPORT u32 max_p(void) { return PMAX; }
EXPORT u32 top_k(void) { return TOPK; }
EXPORT u32 stage_floats(void) { return stageCap; }
EXPORT u16 *ptr_stage(void) { return stage; }
EXPORT float *ptr_stage_f32(void) { return (float *)stage; }
EXPORT u32 *ptr_rows(void) { return ROWS; }
EXPORT float *ptr_att(void) { return ATT; }
EXPORT float *ptr_headn(void) { return HEADN; }
EXPORT float *ptr_mlp(void) { return MLP; }
EXPORT float *ptr_res(void) { return RES; }
EXPORT float *ptr_topp(void) { return TOPP; }
EXPORT u32 *ptr_topi(void) { return TOPI; }
EXPORT float *ptr_ent(void) { return ENT; }
EXPORT float *ptr_lens_p(void) { return LENS_P; }
EXPORT u32 *ptr_lens_i(void) { return LENS_I; }
EXPORT float *ptr_lens_h(void) { return LENS_H; }

// which: 0..L-1 block (fp16), 100 mods (f32), 101 final (f32), 200 embed (fp16), 201 unembed (fp16).
// f32 sources are staged as raw float bytes; `offset`/`n` count elements of the destination.
EXPORT u32 load(u32 which, u32 offset, u32 n) {
  if (which == 100 || which == 101) {
    float *dst = which == 100 ? MODS : FINAL, *src = (float *)stage;
    for (u32 i = 0; i < n; i++) dst[offset + i] = src[i];
    return 0;
  }
  float *dst = which < L ? BLK[which] : which == 200 ? EMB : which == 201 ? UNEMB : 0;
  if (!dst) return 1;
  for (u32 i = 0; i < n; i++) dst[offset + i] = half_to_float(stage[i]);
  return 0;
}

// ---------- blocks ----------
static void layernorm_w(const float *in, const float *w, float *out) {
  float m = 0, v = 0;
  for (int i = 0; i < C; i++) m += in[i];
  m /= C;
  for (int i = 0; i < C; i++) { float d = in[i] - m; v += d * d; }
  float r = 1.0f / fsqrt(v / C + 1e-5f);
  for (int i = 0; i < C; i++) out[i] = (in[i] - m) * r * w[i];
}
static void modulate(float *x, const float *shift, const float *scale) {
  for (int i = 0; i < C; i++) x[i] = x[i] * (1.0f + scale[i]) + shift[i];
}
// OUT[p][o] = bias[o] + sum_i IN[p][i] * W[i][o] for p < P. Weights are read once,
// in output tiles of OT columns, and applied to every position while in cache.
static void matmat(const float *IN, const float *W, const float *bias, float *OUT, u32 I, u32 O) {
  for (u32 p = 0; p < P; p++)
    for (u32 o = 0; o < O; o++) OUT[p * O + o] = bias ? bias[o] : 0.0f;
  for (u32 o0 = 0; o0 < O; o0 += OT) {
    u32 ow = O - o0 < OT ? O - o0 : OT;
    for (u32 i = 0; i < I; i++) {
      const float *row = W + i * O + o0;
      for (u32 p = 0; p < P; p++) {
        float a = IN[p * I + i];
        float *out = OUT + p * O + o0;
        for (u32 o = 0; o < ow; o++) out[o] += a * row[o];
      }
    }
  }
}

static void readout(const float *resid, const float *normw, const float *shift, const float *scale, float *probs,
                    float *tp, u32 *ti, float *ent) {
  float z[C];
  layernorm_w(resid, normw, z);
  modulate(z, shift, scale);
  const float *bias = FINAL + C;
  float mx = -1e30f;
  for (u32 v = 0; v < V; v++) {
    const float *r = UNEMB + v * C;
    float s = bias[v];
    for (int i = 0; i < C; i++) s += z[i] * r[i];
    probs[v] = s;
    if (v != MASKROW && s > mx) mx = s;
  }
  float sum = 0;
  for (u32 v = 0; v < V; v++) { float e = v == MASKROW ? 0.0f : fexp(probs[v] - mx); probs[v] = e; sum += e; }
  float inv = 1.0f / sum, H = 0;
  for (int j = 0; j < TOPK; j++) { tp[j] = -1; ti[j] = 0; }
  for (u32 v = 0; v < V; v++) {
    float p = probs[v] * inv;
    probs[v] = p;
    if (p > 1e-12f) H -= p * flog(p);
    if (p > tp[TOPK - 1]) {
      int j = TOPK - 1;
      while (j > 0 && tp[j - 1] < p) { tp[j] = tp[j - 1]; ti[j] = ti[j - 1]; j--; }
      tp[j] = p; ti[j] = v;
    }
  }
  *ent = H;
}

// Run the whole sequence (ROWS[0..n)) once. Beliefs are computed for masked positions.
EXPORT u32 forward(u32 n) {
  if (n < 1 || n > PMAX) return 1;
  P = n;
  for (u32 p = 0; p < P; p++) {
    const float *e = EMB + ROWS[p] * C;
    for (int i = 0; i < C; i++) { X[p * C + i] = e[i]; RES[p * C + i] = e[i]; }
  }
  for (u32 l = 0; l < L; l++) {
    const float *w = BLK[l];
    const float *n1 = w, *qkvT = n1 + C, *outT = qkvT + C * 3 * C, *n2 = outT + C * C;
    const float *fc1 = n2 + C, *fc1b = fc1 + C * F, *fc2 = fc1b + F, *fc2b = fc2 + F * C;
    const float *md = MODS + l * 6 * C;
    const float *sh_a = md, *sc_a = md + C, *g_a = md + 2 * C, *sh_m = md + 3 * C, *sc_m = md + 4 * C, *g_m = md + 5 * C;

    for (u32 p = 0; p < P; p++) { layernorm_w(X + p * C, n1, XN + p * C); modulate(XN + p * C, sh_a, sc_a); }
    matmat(XN, qkvT, 0, QKV, C, 3 * C);
    // rotary on q and k: pairs (i, i + 32) within each head, angle = pos * invf[i]
    for (u32 p = 0; p < P; p++)
      for (int i = 0; i < HS / 2; i++) {
        float s, c;
        fsincos((float)p * invf[i], &s, &c);
        for (int h = 0; h < NH; h++)
          for (int part = 0; part < 2; part++) {
            float *v = QKV + p * 3 * C + part * C + h * HS;
            float x1 = v[i], x2 = v[i + HS / 2];
            v[i] = x1 * c - x2 * s;
            v[i + HS / 2] = x1 * s + x2 * c;
          }
      }
    for (u32 p = 0; p < P; p++)
      for (int i = 0; i < C; i++) TMP[p * C + i] = 0;
    for (int h = 0; h < NH; h++) {
      for (u32 q = 0; q < P; q++) {
        const float *qv = QKV + q * 3 * C + h * HS;
        float *att = ATT + (((l * NH + h) * PMAX) + q) * PMAX;
        float mx = -1e30f;
        for (u32 t = 0; t < P; t++) {
          const float *k = QKV + t * 3 * C + C + h * HS;
          float s = 0;
          for (int i = 0; i < HS; i++) s += qv[i] * k[i];
          s *= 0.125f;
          att[t] = s;
          if (s > mx) mx = s;
        }
        float sum = 0;
        for (u32 t = 0; t < P; t++) { att[t] = fexp(att[t] - mx); sum += att[t]; }
        float inv = 1.0f / sum;
        float *yh = Y + q * C + h * HS;
        for (int i = 0; i < HS; i++) yh[i] = 0;
        for (u32 t = 0; t < P; t++) {
          att[t] *= inv;
          const float *v = QKV + t * 3 * C + 2 * C + h * HS;
          float a = att[t];
          for (int i = 0; i < HS; i++) yh[i] += a * v[i];
        }
      }
      // this head's slice of the output projection, per position (write strength)
      for (u32 p = 0; p < P; p++) {
        float contrib[C];
        for (int o = 0; o < C; o++) contrib[o] = 0;
        const float *yh = Y + p * C + h * HS;
        for (int i = 0; i < HS; i++) {
          float a = yh[i];
          const float *r = outT + (u32)(h * HS + i) * C;
          for (int o = 0; o < C; o++) contrib[o] += a * r[o];
        }
        float nn = 0;
        for (int o = 0; o < C; o++) { TMP[p * C + o] += contrib[o]; nn += contrib[o] * contrib[o]; }
        HEADN[(l * PMAX + p) * NH + h] = fsqrt(nn);
      }
    }
    for (u32 p = 0; p < P; p++)
      for (int i = 0; i < C; i++) X[p * C + i] += g_a[i] * TMP[p * C + i];

    for (u32 p = 0; p < P; p++) { layernorm_w(X + p * C, n2, XN + p * C); modulate(XN + p * C, sh_m, sc_m); }
    matmat(XN, fc1, fc1b, HB, C, F);
    for (u32 p = 0; p < P; p++) {
      float *hb = HB + p * F, *cap = MLP + (l * PMAX + p) * F;
      for (int i = 0; i < F; i++) { hb[i] = gelu(hb[i]); cap[i] = hb[i]; }
    }
    matmat(HB, fc2, fc2b, TMP, F, C);
    for (u32 p = 0; p < P; p++)
      for (int i = 0; i < C; i++) {
        X[p * C + i] += g_m[i] * TMP[p * C + i];
        RES[((l + 1) * PMAX + p) * C + i] = X[p * C + i];
      }
  }
  // Readout for every masked position at once: each unembedding row is read once.
  const float *fm = MODS + L * 6 * C, *bias = FINAL + C;
  u32 M = 0, mpos[PMAX];
  for (u32 p = 0; p < P; p++) {
    if (ROWS[p] != MASKROW) { ENT[p] = 0; continue; }
    layernorm_w(X + p * C, FINAL, XN + M * C);
    modulate(XN + M * C, fm, fm + C);
    mpos[M++] = p;
  }
  for (u32 v = 0; v < V; v++) {
    const float *r = UNEMB + v * C;
    for (u32 m = 0; m < M; m++) {
      const float *z = XN + m * C;
      float s = bias[v];
      for (int i = 0; i < C; i++) s += z[i] * r[i];
      PROBS[mpos[m] * V + v] = s;
    }
  }
  for (u32 m = 0; m < M; m++) {
    u32 p = mpos[m];
    float *pr = PROBS + p * V, *tp = TOPP + p * TOPK, mx = -1e30f, sum = 0, H = 0;
    u32 *ti = TOPI + p * TOPK;
    for (u32 v = 0; v < V; v++) if (v != MASKROW && pr[v] > mx) mx = pr[v];
    for (u32 v = 0; v < V; v++) { float e = v == MASKROW ? 0.0f : fexp(pr[v] - mx); pr[v] = e; sum += e; }
    float inv = 1.0f / sum;
    for (int j = 0; j < TOPK; j++) { tp[j] = -1; ti[j] = 0; }
    for (u32 v = 0; v < V; v++) {
      float q = pr[v] * inv;
      pr[v] = q;
      if (q > 1e-12f) H -= q * flog(q);
      if (q > tp[TOPK - 1]) {
        int j = TOPK - 1;
        while (j > 0 && tp[j - 1] < q) { tp[j] = tp[j - 1]; ti[j] = ti[j - 1]; j--; }
        tp[j] = q; ti[j] = v;
      }
    }
    ENT[p] = H;
  }
  return 0;
}

EXPORT void set_row(u32 p, u32 row) { if (p < PMAX) ROWS[p] = row; }

// Draw a token for masked position p from its last belief: u in [0,1) (inverse CDF),
// or the most likely token when greedy.
EXPORT u32 sample(u32 p, float u, u32 greedy) {
  if (greedy) return TOPI[p * TOPK];
  const float *pr = PROBS + p * V;
  float acc = 0;
  for (u32 v = 0; v < V; v++) { acc += pr[v]; if (acc > u) return v; }
  return TOPI[p * TOPK];
}

// Logit lens for one position: the final readout applied to the residual after each block.
EXPORT void lens(u32 p) {
  const float *fm = MODS + L * 6 * C;
  for (u32 d = 0; d <= L; d++)
    readout(RES + (d * PMAX + p) * C, FINAL, fm, fm + C, LSCR, LENS_P + d * TOPK, LENS_I + d * TOPK, LENS_H + d);
}
