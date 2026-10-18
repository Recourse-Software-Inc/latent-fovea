// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// GPT-2 small forward pass (C -> wasm32, no libc, no imports), one token at a time
// with a KV cache, recording what the visualizer shows:
//   attention probabilities for every layer x head x (query, key) pair,
//   each head's write strength into the residual stream,
//   the 3072 MLP activations of every layer,
//   the residual stream after every block,
//   a logit-lens readout (top-k next tokens) at every depth,
//   each head's and each MLP's output vector, for direct logit attribution.
//
// Token indices are *tile rows* of vectors.f16 (the viewer's point order), so
// the embedding table is shared with the embedding explorer.

typedef unsigned int u32;
typedef unsigned short u16;

#define EXPORT __attribute__((visibility("default")))
#define LMAX 12   // compiled capacity; the model's own layer count is set at init
#define NH 12
#define C 768
#define HS 64
#define F 3072
#define MAXT 256
#define TOPK 10
#define FC2_OFF (C + C + C * 3 * C + 3 * C + C * C + C + C + C + C * F + F)  // fc2 weights in a layer block
#define LAYER_FLOATS (C + C + C * 3 * C + 3 * C + C * C + C + C + C + C * F + F + F * C + C)

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
static float flog(float x) {  // natural log, ~1e-7 relative
  union { float f; u32 u; } b = {x};
  int e = (int)(b.u >> 23) - 127;
  b.u = (b.u & 0x7fffff) | 0x3f800000;  // mantissa in [1, 2)
  float m = b.f;
  if (m > 1.41421356f) { m *= 0.5f; e++; }
  float t = (m - 1.0f) / (m + 1.0f), t2 = t * t;
  float s = 2.0f * t * (1.0f + t2 * (0.33333333f + t2 * (0.2f + t2 * (0.14285714f + t2 * 0.11111111f))));
  return s + (float)e * 0.69314718f;
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

// ---------- weights ----------
static u32 L;                 // layers in the loaded model (<= LMAX)
static float *LW[LMAX];           // per-layer block, LAYER_LAYOUT order
static float *WPE, *LNF_W, *LNF_B, *WTE;
static u32 V, NCTX;
static u16 *stage;
static u32 stageCap;

static float *KC, *VC;         // [L][MAXT][C]
static float *ATT;             // [L][NH][MAXT][MAXT] probabilities (row = query)
static float HEADN[LMAX * NH];    // |head output after projection| for the last token
static float *MLP;             // [L][F] GELU activations, last token
static float *PRE;             // [L][F] each neuron's input to GELU, last token (any input steer included)
static float LN2R[LMAX];          // 1/scale of each MLP's LayerNorm, last token
static float *NL;              // neuron_links / neuron_readers output (see there)
static float RES[(LMAX + 1) * C]; // residual stream after embed and each block, last token
static float LENS_P[(LMAX + 2) * TOPK];  // one slot per depth, plus LMAX + 1 for lens_q
static u32 LENS_I[(LMAX + 2) * TOPK];
static float LENS_H[LMAX + 2];    // entropy of each depth's next-token distribution
static float QVEC[C];             // a residual-stream vector to read out on request (lens_q)
static float *LOGITS;          // [V] scratch
static float HOUT[LMAX * NH * C]; // each head's output vector (after projection), last token
static float MOUT[LMAX * C];      // each MLP's output vector (with its bias), last token
static float ATTR[LMAX * NH + LMAX + 4];  // attribution: heads, MLPs, embedding, rest, logit, 1/scale
static float UG[C];               // unembedding row x final-norm weight, for the word attributed
static float GAIN[F];             // per-neuron direct effect per unit activation (before 1/scale)
static float NV_P[2 * TOPK];      // a neuron's most promoted, then most suppressed, words
static u32 NV_I[2 * TOPK];
static float DIRV[C];             // scratch: a neuron's output direction through the final norm
// ---------- interventions ----------
// Scale any head's output, any layer's whole MLP output, or single neurons' activations, inside
// the forward pass, so every later layer (and the context kept for later tokens) sees the changed
// model. 0 knocks a part out; 1 leaves it alone. Recordings show the changed values.
#define NKO_MAX 256
static float HSCALE[LMAX * NH], MSCALE[LMAX], NKO_S[NKO_MAX], NKO_A[NKO_MAX], NKO_P[NKO_MAX];
static u32 NKO_L[NKO_MAX], NKO_I[NKO_MAX], nko;
EXPORT void iv_clear(void) { for (u32 j = 0; j < LMAX * NH; j++) HSCALE[j] = 1; for (u32 l = 0; l < LMAX; l++) MSCALE[l] = 1; nko = 0; }
EXPORT void iv_head(u32 l, u32 h, float s) { if (l < LMAX && h < NH) HSCALE[l * NH + h] = s; }
EXPORT void iv_mlp(u32 l, float s) { if (l < LMAX) MSCALE[l] = s; }
EXPORT u32 iv_neuron(u32 l, u32 i, float s) { if (nko >= NKO_MAX || i >= F) return 1; NKO_L[nko] = l; NKO_I[nko] = i; NKO_S[nko] = s; NKO_A[nko] = 0; NKO_P[nko] = 0; nko++; return 0; }
// Steering: activation -> activation * s + add, at every position (add turns a silent neuron on).
EXPORT u32 iv_neuron2(u32 l, u32 i, float s, float add) { if (iv_neuron(l, i, s)) return 1; NKO_A[nko - 1] = add; return 0; }
// Steering the input instead: input -> input + pre before GELU, so the neuron is pushed the way
// the network itself pushes it (GELU's output can't go below -0.17).
EXPORT u32 iv_neuron3(u32 l, u32 i, float s, float add, float pre) { if (iv_neuron2(l, i, s, add)) return 1; NKO_P[nko - 1] = pre; return 0; }
static float UMEAN[C];            // the average word's unembedding row (computed on first use)
static u32 umean_ready;
static float x[C], xn[C], qkv[3 * C], y[C], hbuf[F], tmp[C];

EXPORT u32 init(u32 vocab, u32 nctx, u32 nlayer) {
  heap_ptr = (u32)&__heap_base;
  V = vocab; NCTX = nctx;
  if (nlayer < 1 || nlayer > LMAX) return 2;
  L = nlayer;
  iv_clear();
  for (u32 l = 0; l < L; l++) LW[l] = alloc(LAYER_FLOATS * 4);
  WPE = alloc(nctx * C * 4); LNF_W = alloc(C * 4); LNF_B = alloc(C * 4);
  WTE = alloc(V * C * 4);
  KC = alloc(L * MAXT * C * 4); VC = alloc(L * MAXT * C * 4);
  ATT = alloc(L * NH * MAXT * MAXT * 4);
  MLP = alloc(L * F * 4); PRE = alloc(L * F * 4); LOGITS = alloc(V * 4);
  NL = alloc((L * F + L * NH + 4) * 4);
  stageCap = 8u << 20; stage = alloc(stageCap * 2);
  return stage ? 0 : 1;
}
EXPORT u32 max_t(void) { return MAXT; }
EXPORT u32 stage_floats(void) { return stageCap; }
EXPORT u16 *ptr_stage(void) { return stage; }
EXPORT float *ptr_att(void) { return ATT; }
EXPORT float *ptr_headn(void) { return HEADN; }
EXPORT float *ptr_mlp(void) { return MLP; }
EXPORT float *ptr_pre(void) { return PRE; }
EXPORT float *ptr_ln2r(void) { return LN2R; }
EXPORT float *ptr_nl(void) { return NL; }
EXPORT float *ptr_res(void) { return RES; }
EXPORT float *ptr_mout(void) { return MOUT; }
// The size of what token pos offers to head h at layer l: the norm of its value vector.
EXPORT float vnorm(u32 l, u32 pos, u32 h) {
  const float *v = VC + ((u32)l * MAXT + pos) * C + h * HS; float s = 0;
  for (int i = 0; i < HS; i++) s += v[i] * v[i];
  return __builtin_sqrtf(s);
}
EXPORT float *ptr_lens_p(void) { return LENS_P; }
EXPORT u32 *ptr_lens_i(void) { return LENS_I; }
EXPORT float *ptr_lens_h(void) { return LENS_H; }
EXPORT float *ptr_qvec(void) { return QVEC; }
// After a logit lens: the probability of any word in that readout (LOGITS holds exp(logit - max)).
static float LENS_SUM;
EXPORT float prob_of(u32 row) { return row < V && LENS_SUM > 0 ? LOGITS[row] / LENS_SUM : 0; }

// Decode `n` fp16 values from the stage buffer into a destination.
// which: 0..L-1 = layer block, 100 = misc (wpe, ln_f), 200 = wte rows.
// misc.f16 is wpe, then ln_f.weight, then ln_f.bias; `offset` indexes that file.
EXPORT u32 load(u32 which, u32 offset, u32 n) {
  float *dst;
  if (which < L) dst = LW[which];
  else if (which == 200) dst = WTE;
  else if (which != 100) return 1;
  if (which == 100) {
    for (u32 i = 0; i < n; i++) {
      u32 k = offset + i;
      float v = half_to_float(stage[i]);
      if (k < NCTX * C) WPE[k] = v;
      else if (k < NCTX * C + C) LNF_W[k - NCTX * C] = v;
      else LNF_B[k - NCTX * C - C] = v;
    }
    return 0;
  }
  for (u32 i = 0; i < n; i++) dst[offset + i] = half_to_float(stage[i]);
  return 0;
}

// ---------- blocks ----------
static float layernorm(const float *in, const float *w, const float *b, float *out) {  // returns 1/scale
  float m = 0, v = 0;
  for (int i = 0; i < C; i++) m += in[i];
  m /= C;
  for (int i = 0; i < C; i++) { float d = in[i] - m; v += d * d; }
  float r = 1.0f / fsqrt(v / C + 1e-5f);
  for (int i = 0; i < C; i++) out[i] = (in[i] - m) * r * w[i] + b[i];
  return r;
}
// out[o] = b[o] + sum_i in[i] * W[i*O + o]   (Conv1D layout: [in][out])
static void matvec(const float *in, const float *W, const float *b, float *out, int I, int O) {
  for (int o = 0; o < O; o++) out[o] = b[o];
  for (int i = 0; i < I; i++) {
    float a = in[i];
    const float *row = W + (u32)i * O;
    for (int o = 0; o < O; o++) out[o] += a * row[o];
  }
}

static void lens(int depth, const float *resid, int k) {
  layernorm(resid, LNF_W, LNF_B, xn);
  float mx = -1e30f;
  for (u32 v = 0; v < V; v++) {
    const float *r = WTE + v * C;
    float s = 0;
    for (int i = 0; i < C; i++) s += xn[i] * r[i];
    LOGITS[v] = s;
    if (s > mx) mx = s;
  }
  float sum = 0;
  for (u32 v = 0; v < V; v++) { float e = fexp(LOGITS[v] - mx); LOGITS[v] = e; sum += e; }
  float inv = 1.0f / sum, H = 0;
  LENS_SUM = sum;
  float *tp = LENS_P + depth * TOPK;
  u32 *ti = LENS_I + depth * TOPK;
  for (int j = 0; j < TOPK; j++) { tp[j] = -1; ti[j] = 0; }
  for (u32 v = 0; v < V; v++) {
    float p = LOGITS[v] * inv;
    if (p > 1e-12f) H -= p * flog(p);
    if (p > tp[k - 1]) {
      int j = k - 1;
      while (j > 0 && tp[j - 1] < p) { tp[j] = tp[j - 1]; ti[j] = ti[j - 1]; j--; }
      tp[j] = p; ti[j] = v;
    }
  }
  LENS_H[depth] = H;
}

// Logit lens on any residual-stream vector written into QVEC (a stored depth of an earlier
// token, say); the readout lands in slot LMAX + 1.
EXPORT u32 lens_q(void) { lens(LMAX + 1, QVEC, TOPK); return LMAX + 1; }

// Run one position. `row` is the token's tile row; lensMode: 0 none, 1 final only, 2 every depth.
EXPORT u32 step(u32 row, u32 pos, u32 lensMode) {
  if (pos >= MAXT || pos >= NCTX) return 1;
  const float *te = WTE + row * C, *pe = WPE + pos * C;
  for (int i = 0; i < C; i++) x[i] = te[i] + pe[i];
  for (int i = 0; i < C; i++) RES[i] = x[i];
  for (u32 l = 0; l < L; l++) {
    float *w = LW[l];
    const float *ln1w = w, *ln1b = ln1w + C, *attw = ln1b + C, *attb = attw + C * 3 * C;
    const float *projw = attb + 3 * C, *projb = projw + C * C, *ln2w = projb + C, *ln2b = ln2w + C;
    const float *fcw = ln2b + C, *fcb = fcw + C * F, *fc2w = fcb + F, *fc2b = fc2w + F * C;

    layernorm(x, ln1w, ln1b, xn);
    matvec(xn, attw, attb, qkv, C, 3 * C);
    float *kc = KC + ((u32)l * MAXT + pos) * C, *vc = VC + ((u32)l * MAXT + pos) * C;
    for (int i = 0; i < C; i++) { kc[i] = qkv[C + i]; vc[i] = qkv[2 * C + i]; }
    for (int i = 0; i < C; i++) tmp[i] = 0;
    for (int h = 0; h < NH; h++) {
      const float *q = qkv + h * HS;
      float *att = ATT + ((((u32)l * NH + h) * MAXT) + pos) * MAXT;
      float mx = -1e30f;
      for (u32 t = 0; t <= pos; t++) {
        const float *k = KC + ((u32)l * MAXT + t) * C + h * HS;
        float s = 0;
        for (int i = 0; i < HS; i++) s += q[i] * k[i];
        s *= 0.125f;  // 1/sqrt(64)
        att[t] = s;
        if (s > mx) mx = s;
      }
      float sum = 0;
      for (u32 t = 0; t <= pos; t++) { att[t] = fexp(att[t] - mx); sum += att[t]; }
      float inv = 1.0f / sum;
      for (u32 t = 0; t <= pos; t++) att[t] *= inv;
      float *yh = y + h * HS;
      for (int i = 0; i < HS; i++) yh[i] = 0;
      for (u32 t = 0; t <= pos; t++) {
        const float *v = VC + ((u32)l * MAXT + t) * C + h * HS;
        float a = att[t];
        for (int i = 0; i < HS; i++) yh[i] += a * v[i];
      }
      // This head's contribution through its slice of the output projection.
      float contrib[C];
      for (int o = 0; o < C; o++) contrib[o] = 0;
      for (int i = 0; i < HS; i++) {
        float a = yh[i];
        const float *r = projw + (u32)(h * HS + i) * C;
        for (int o = 0; o < C; o++) contrib[o] += a * r[o];
      }
      float nn = 0, *hout = HOUT + ((u32)l * NH + h) * C, hs = HSCALE[l * NH + h];
      if (hs != 1.0f) for (int o = 0; o < C; o++) contrib[o] *= hs;
      for (int o = 0; o < C; o++) { tmp[o] += contrib[o]; nn += contrib[o] * contrib[o]; hout[o] = contrib[o]; }
      HEADN[l * NH + h] = fsqrt(nn);
    }
    for (int i = 0; i < C; i++) x[i] += tmp[i] + projb[i];

    LN2R[l] = layernorm(x, ln2w, ln2b, xn);
    matvec(xn, fcw, fcb, hbuf, C, F);
    float *act = MLP + (u32)l * F, *pre = PRE + (u32)l * F;
    for (u32 j = 0; j < nko; j++) if (NKO_L[j] == l) hbuf[NKO_I[j]] += NKO_P[j];
    for (int i = 0; i < F; i++) { pre[i] = hbuf[i]; hbuf[i] = gelu(hbuf[i]); }
    for (u32 j = 0; j < nko; j++) if (NKO_L[j] == l) hbuf[NKO_I[j]] = hbuf[NKO_I[j]] * NKO_S[j] + NKO_A[j];
    for (int i = 0; i < F; i++) act[i] = hbuf[i];
    matvec(hbuf, fc2w, fc2b, tmp, F, C);
    if (MSCALE[l] != 1.0f) for (int i = 0; i < C; i++) tmp[i] *= MSCALE[l];
    for (int i = 0; i < C; i++) { x[i] += tmp[i]; MOUT[l * C + i] = tmp[i]; }
    for (int i = 0; i < C; i++) RES[(l + 1) * C + i] = x[i];
  }
  if (lensMode == 2) for (u32 d = 0; d <= L; d++) lens(d, RES + d * C, TOPK);
  else if (lensMode == 1) lens(L, RES + L * C, TOPK);
  return 0;
}

// ---------- attribution ----------
// The residual stream is a sum: the embedding, every head's write, every MLP's write, and
// the biases. Holding the final LayerNorm's scale at this token's value, the logit of a word
// splits exactly into one term per part (direct logit attribution). Only the direct path is
// counted: a part that matters through later layers shows up in them, not here.
// Every word's logit shares a large common offset that cancels in the softmax, so the score
// attributed is the word's logit minus the average word's: what actually sets the odds.
static float part(const float *c, float sug, float inv) {  // one vector's share of the logit
  float s = 0, m = 0;
  for (int i = 0; i < C; i++) { s += UG[i] * c[i]; m += c[i]; }
  return inv * (s - sug * m / C);
}
static float set_ug(u32 row) {  // UG = (u - average u) * gamma for word `row`; returns sum(UG)
  if (!umean_ready) {
    for (int i = 0; i < C; i++) {  // in double: GPT-2's few huge residual dimensions amplify any rounding
      double s = 0;
      for (u32 r = 0; r < V; r++) s += WTE[r * C + i];
      UMEAN[i] = (float)(s / V);
    }
    umean_ready = 1;
  }
  const float *u = WTE + row * C;
  float sug = 0;
  for (int i = 0; i < C; i++) { UG[i] = (u[i] - UMEAN[i]) * LNF_W[i]; sug += UG[i]; }
  return sug;
}

// Attribute word `row`'s score (logit minus the average word's) at the most recent position.
// ATTR holds, in order: L x NH heads, L MLPs, the embedding (token + position), the rest
// (biases), the score, 1/scale.
EXPORT u32 attribute(u32 row) {
  if (row >= V) return 1;
  const float *fin = RES + L * C;
  float m = 0, v = 0;
  for (int i = 0; i < C; i++) m += fin[i];
  m /= C;
  for (int i = 0; i < C; i++) { float d = fin[i] - m; v += d * d; }
  float inv = 1.0f / fsqrt(v / C + 1e-5f), sug = set_ug(row), sum = 0;
  for (u32 j = 0; j < L * NH; j++) { ATTR[j] = part(HOUT + j * C, sug, inv); sum += ATTR[j]; }
  for (u32 l = 0; l < L; l++) { ATTR[L * NH + l] = part(MOUT + l * C, sug, inv); sum += ATTR[L * NH + l]; }
  float emb = part(RES, sug, inv), logit = 0;
  const float *u = WTE + row * C;
  for (int i = 0; i < C; i++) logit += (u[i] - UMEAN[i]) * ((fin[i] - m) * inv * LNF_W[i] + LNF_B[i]);
  float *t = ATTR + L * NH + L;
  t[0] = emb; t[1] = logit - sum - emb; t[2] = logit; t[3] = inv;
  return 0;
}
EXPORT float *ptr_attr(void) { return ATTR; }

// Each neuron of layer l: its direct effect on word `row` per unit of activation, before the
// final norm's 1/scale. A neuron's effect at a position = activation x gain x 1/scale there.
EXPORT u32 neuron_gain(u32 l, u32 row) {
  if (l >= L || row >= V) return 1;
  float sug = set_ug(row);
  const float *w2 = LW[l] + FC2_OFF;  // fc2 [F][C]: row i is neuron i's output
  for (int i = 0; i < F; i++) {
    const float *w = w2 + (u32)i * C;
    float s = 0, m = 0;
    for (int o = 0; o < C; o++) { s += UG[o] * w[o]; m += w[o]; }
    GAIN[i] = s - sug * m / C;
  }
  return 0;
}
EXPORT float *ptr_gain(void) { return GAIN; }

static u32 rank_dirv(void);
// Neuron i of layer l: the words its output direction pushes up most and down most,
// through the final norm's weight (per unit activation, before 1/scale).
EXPORT u32 neuron_vocab(u32 l, u32 i) {
  if (l >= L || i >= F) return 1;
  const float *w = LW[l] + FC2_OFF + i * C;
  float m = 0;
  for (int o = 0; o < C; o++) m += w[o];
  m /= C;
  for (int o = 0; o < C; o++) DIRV[o] = (w[o] - m) * LNF_W[o];
  return rank_dirv();
}
// Any direction written into QVEC (an SAE feature's, say): the words it pushes up and down most.
EXPORT u32 vocab_q(void) {
  float m = 0;
  for (int o = 0; o < C; o++) m += QVEC[o];
  m /= C;
  for (int o = 0; o < C; o++) DIRV[o] = (QVEC[o] - m) * LNF_W[o];
  return rank_dirv();
}
static u32 rank_dirv(void) {  // top and bottom TOPK words along DIRV
  for (int j = 0; j < TOPK; j++) { NV_P[j] = -1e30f; NV_P[TOPK + j] = 1e30f; NV_I[j] = NV_I[TOPK + j] = 0; }
  for (u32 r = 0; r < V; r++) {
    const float *u = WTE + r * C;
    float s = 0;
    for (int o = 0; o < C; o++) s += u[o] * DIRV[o];
    if (s > NV_P[TOPK - 1]) {
      int j = TOPK - 1;
      while (j > 0 && NV_P[j - 1] < s) { NV_P[j] = NV_P[j - 1]; NV_I[j] = NV_I[j - 1]; j--; }
      NV_P[j] = s; NV_I[j] = r;
    }
    if (s < NV_P[2 * TOPK - 1]) {
      int j = 2 * TOPK - 1;
      while (j > TOPK && NV_P[j - 1] > s) { NV_P[j] = NV_P[j - 1]; NV_I[j] = NV_I[j - 1]; j--; }
      NV_P[j] = s; NV_I[j] = r;
    }
  }
  return 0;
}
EXPORT float *ptr_nv_p(void) { return NV_P; }
EXPORT u32 *ptr_nv_i(void) { return NV_I; }

// ---------- neuron to neuron ----------
// A neuron's input is its LayerNorm'd residual stream dotted with its input weights, plus a bias.
// The residual stream is a sum of parts (the embedding, every earlier head and neuron, biases), and
// LayerNorm at a fixed scale is linear in each part once that part is centred, so the input splits
// exactly into one term per part:   input_j = sum over parts of r * (c - mean c) . (gamma * w_j) + rest
// with r the norm's 1/scale at this token.
static float G[C];
static float set_g(u32 l, u32 j) {  // G = gamma2 * neuron j's input weights; returns sum(G)
  const float *w = LW[l], *ln2w = w + C + C + C * 3 * C + 3 * C + C * C + C, *fcw = ln2w + C + C;
  float sg = 0;
  for (int c = 0; c < C; c++) { G[c] = ln2w[c] * fcw[(u32)c * F + j]; sg += G[c]; }
  return sg;
}
static float dotg(const float *c, float sg) { float s = 0, m = 0; for (int i = 0; i < C; i++) { s += G[i] * c[i]; m += c[i]; } return s - sg * m / C; }

// Into neuron j of layer l. NL holds, in order: L x F neurons, L x NH heads, the embedding, the
// neuron's whole input, r. dyn = 1: each part's exact share of the input at the most recent
// position (earlier neurons as activation x weight); the input minus their sum is the biases.
// dyn = 0: weights only, per unit of each earlier neuron's activation, before r (heads and the
// embedding are then 0, since they have no fixed output).
EXPORT u32 neuron_links(u32 l, u32 j, u32 dyn) {
  if (l >= L || j >= F) return 1;
  float sg = set_g(l, j), r = dyn ? LN2R[l] : 1.0f;
  for (u32 k = 0; k < L; k++) {
    float *out = NL + k * F;
    if (k >= l) { for (int i = 0; i < F; i++) out[i] = 0; continue; }
    const float *w2 = LW[k] + FC2_OFF, *a = MLP + k * F;
    for (int i = 0; i < F; i++) { float v = dotg(w2 + (u32)i * C, sg); out[i] = dyn ? v * a[i] * r * MSCALE[k] : v; }  // a knocked-out MLP (MSCALE 0) writes nothing
  }
  float *hd = NL + L * F;
  for (u32 hl = 0; hl < L; hl++) for (int h = 0; h < NH; h++)
    hd[hl * NH + h] = dyn && hl <= l ? r * dotg(HOUT + (hl * NH + h) * C, sg) : 0;
  float *t = hd + L * NH;
  float steer = 0;  // an input steer on this neuron is the reader's own change, not a bias: leave it out
  for (u32 q = 0; q < nko; q++) if (NKO_L[q] == l && NKO_I[q] == j) steer += NKO_P[q];
  t[0] = dyn ? r * dotg(RES, sg) : 0; t[1] = dyn ? PRE[l * F + j] - steer : 0; t[2] = r; t[3] = 0;
  return 0;
}
// Out of neuron i of layer l: how strongly each later neuron reads its output, per unit of its
// activation, from the weights alone (before each reader's r). NL holds L x F (0 for layers <= l).
EXPORT u32 neuron_readers(u32 l, u32 i) {
  if (l >= L || i >= F) return 1;
  const float *wo = LW[l] + FC2_OFF + i * C;
  float m = 0;
  for (int c = 0; c < C; c++) m += wo[c];
  m /= C;
  for (u32 k = 0; k < L; k++) {
    float *out = NL + k * F;
    for (int jj = 0; jj < F; jj++) out[jj] = 0;
    if (k <= l) continue;
    const float *w = LW[k], *ln2w = w + C + C + C * 3 * C + 3 * C + C * C + C, *fcw = ln2w + C + C;
    for (int c = 0; c < C; c++) {
      float a = (wo[c] - m) * ln2w[c];
      const float *row = fcw + (u32)c * F;
      for (int jj = 0; jj < F; jj++) out[jj] += a * row[jj];
    }
  }
  return 0;
}
