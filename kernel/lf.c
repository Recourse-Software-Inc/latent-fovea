// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// latent-fovea kernel (C -> wasm32, no libc, no imports).
//
// Holds everything that scales with the model: the projected point cloud,
// the tile index, the streamed embedding rows and every per-frame pass.
// JS draws what this produces and moves bytes between the network and here.
//
// Coordinates: points arrive in tile order (grouped by grid cell), so one cell
// is one contiguous byte range of vectors.f16 on the server.

typedef unsigned int u32;
typedef int i32;
typedef unsigned short u16;
typedef unsigned char u8;

#define EXPORT __attribute__((visibility("default")))
#define FMAX 2048
#define QMAX 64
#define DMAX 1024

extern unsigned char __heap_base;
static u32 heap_ptr;

static void *alloc(u32 bytes) {
  u32 p = (heap_ptr + 15u) & ~15u, end = p + bytes;
  u32 have = (u32)__builtin_wasm_memory_size(0) * 65536u;
  if (end > have && __builtin_wasm_memory_grow(0, (end - have + 65535u) / 65536u) == (unsigned long)-1) return 0;
  heap_ptr = end;
  return (void *)p;
}

static inline float fsqrt(float x) { return __builtin_sqrtf(x); }
static inline float smoothstep(float a, float b, float x) {
  float t = (x - a) / (b - a);
  t = t < 0 ? 0 : (t > 1 ? 1 : t);
  return t * t * (3.0f - 2.0f * t);
}
static inline u32 hash32(u32 x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
  return x;
}
static inline float half_to_float(u16 h) {
  u32 s = (u32)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, bits;
  if (e == 0) {
    if (m == 0) bits = s;
    else {  // subnormal
      e = 1;
      while (!(m & 0x400)) { m <<= 1; e--; }
      m &= 0x3ff;
      bits = s | ((e + 112) << 23) | (m << 13);
    }
  } else if (e == 31) bits = s | 0x7f800000u | (m << 13);
  else bits = s | ((e + 112) << 23) | (m << 13);
  union { u32 u; float f; } v = {bits};
  return v.f;
}

// ---------- state ----------
static u32 N, D, NC;           // points, dims, cells
static float *P;               // N*3 world positions
static u8 *CAT;                // N categories
static u32 *CELLS;             // NC+1 prefix offsets
static u32 *CELL_OF;           // N -> cell
static float *CELL_C;          // NC*4 centre xyz + radius
static u8 *CSTATE;             // 0 absent, 1 requested, 2 resident
static u32 *CLAST;             // last frame a resident cell was needed
static float *POOL;            // SLOTS*D decoded rows
static i32 *ROWSLOT;           // N -> slot or -1
static i32 *FREE;              // free slot stack
static u32 SLOTS, nfree, slotCap, slotsUsed, clockHand;
static u8 *ATTR;               // N*2 per-point draw attrs: alpha, size
static float MAT[16];          // view-projection, column-major
static float pxScale;          // pixels per world unit at w = 1
static u32 fidx[FMAX];
static float fscr[FMAX * 2], fdisp[FMAX * 2], fz[FMAX];
static u32 fcount;
static float lpc[2 * DMAX], lensK;
static int lpcInit;
static u32 req[QMAX];
static float reqd[QMAX];
static u32 nreq;
static float stats[32];
static u32 knnI[64];
static float knnS[64];
static float scratch[DMAX];

EXPORT u32 init(u32 n, u32 d, u32 ncells, u32 maxPoolBytes) {
  heap_ptr = (u32)&__heap_base;
  N = n; D = d > DMAX ? DMAX : d; NC = ncells;
  P = alloc(N * 12); CAT = alloc(N); CELLS = alloc((NC + 1) * 4); CELL_OF = alloc(N * 4);
  CELL_C = alloc(NC * 16); CSTATE = alloc(NC); CLAST = alloc(NC * 4);
  ROWSLOT = alloc(N * 4); ATTR = alloc(N * 2);
  SLOTS = maxPoolBytes / (D * 4);
  POOL = alloc(SLOTS * D * 4); FREE = alloc(SLOTS * 4);
  if (!FREE) return 1;
  slotCap = SLOTS; lpcInit = 0; lensK = 0; clockHand = 0;
  return 0;
}
EXPORT float *ptr_P(void) { return P; }
EXPORT u8 *ptr_cat(void) { return CAT; }
EXPORT u32 *ptr_cells(void) { return CELLS; }
EXPORT u8 *ptr_attr(void) { return ATTR; }
EXPORT float *ptr_mat(void) { return MAT; }
EXPORT float *ptr_stats(void) { return stats; }
EXPORT u32 *ptr_fidx(void) { return fidx; }
EXPORT float *ptr_fdisp(void) { return fdisp; }
EXPORT float *ptr_fz(void) { return fz; }
EXPORT u32 *ptr_req(void) { return req; }
EXPORT u32 *ptr_knn(void) { return knnI; }
EXPORT float *ptr_knns(void) { return knnS; }
EXPORT u8 *ptr_cstate(void) { return CSTATE; }

// Call after JS has filled P, CAT and CELLS.
EXPORT void prepare(void) {
  for (u32 c = 0; c < NC; c++) {
    u32 s = CELLS[c], e = CELLS[c + 1];
    float cx = 0, cy = 0, cz = 0, r = 0;
    for (u32 i = s; i < e; i++) { CELL_OF[i] = c; cx += P[3 * i]; cy += P[3 * i + 1]; cz += P[3 * i + 2]; }
    if (e > s) { float k = 1.0f / (e - s); cx *= k; cy *= k; cz *= k; }
    for (u32 i = s; i < e; i++) {
      float dx = P[3 * i] - cx, dy = P[3 * i + 1] - cy, dz = P[3 * i + 2] - cz, q = dx * dx + dy * dy + dz * dz;
      if (q > r) r = q;
    }
    CELL_C[4 * c] = cx; CELL_C[4 * c + 1] = cy; CELL_C[4 * c + 2] = cz; CELL_C[4 * c + 3] = fsqrt(r);
    CSTATE[c] = 0; CLAST[c] = 0;
  }
  for (u32 i = 0; i < N; i++) ROWSLOT[i] = -1;
  nfree = SLOTS;
  for (u32 s = 0; s < SLOTS; s++) FREE[s] = (i32)(SLOTS - 1 - s);
  slotsUsed = 0;
}

EXPORT void set_cache(u32 bytes) { u32 s = bytes / (D * 4); slotCap = s > SLOTS ? SLOTS : s; }
EXPORT void set_view(float pxPerUnit) { pxScale = pxPerUnit; }

static void evict_cell(u32 c) {
  for (u32 i = CELLS[c]; i < CELLS[c + 1]; i++)
    if (ROWSLOT[i] >= 0) { FREE[nfree++] = ROWSLOT[i]; ROWSLOT[i] = -1; slotsUsed--; }
  CSTATE[c] = 0;
}

EXPORT void flush_cache(void) {
  for (u32 c = 0; c < NC; c++) if (CSTATE[c] == 2) evict_cell(c);
  lpcInit = 0;
}

// Scratch for an incoming tile; JS copies fp16 bytes here then calls load_cell.
static u16 *inbuf;
static u32 inCap;
EXPORT u16 *in_buffer(u32 bytes) {
  if (bytes > inCap) { inbuf = alloc(bytes); inCap = inbuf ? bytes : 0; }
  return inbuf;
}

// Decode a streamed tile into cache slots, evicting stale cells if needed.
EXPORT u32 load_cell(u32 c, u32 frame) {
  u32 s = CELLS[c], e = CELLS[c + 1], need = e - s;
  if (CSTATE[c] == 2) return 1;
  for (u32 scanned = 0; (slotsUsed + need > slotCap || nfree < need) && scanned < 2 * NC; scanned++) {
    u32 v = clockHand; clockHand = (clockHand + 1) % NC;
    if (CSTATE[v] == 2 && frame - CLAST[v] > 2) evict_cell(v);
  }
  if (nfree < need || slotsUsed + need > slotCap) { CSTATE[c] = 0; return 0; }
  for (u32 i = s; i < e; i++) {
    i32 slot = FREE[--nfree];
    ROWSLOT[i] = slot; slotsUsed++;
    float *dst = POOL + (u32)slot * D;
    const u16 *src = inbuf + (i - s) * D;
    for (u32 k = 0; k < D; k++) dst[k] = half_to_float(src[k]);
  }
  CSTATE[c] = 2; CLAST[c] = frame;
  return 1;
}
EXPORT void mark_requested(u32 c) { if (CSTATE[c] == 0) CSTATE[c] = 1; }
EXPORT void mark_failed(u32 c) { if (CSTATE[c] == 1) CSTATE[c] = 0; }
EXPORT float *row(u32 i) { return ROWSLOT[i] >= 0 ? POOL + (u32)ROWSLOT[i] * D : 0; }

// ---------- per frame ----------
// flags: 1 foveated, 2 lens
EXPORT void frame(float w, float h, float gx, float gy, float r0, float r1, u32 flags, float lensMix, u32 frameNo) {
  const int fov = flags & 1, lens = (flags & 2) && fov;
  const float hw = w * 0.5f, hh = h * 0.5f, r0s = r0 * r0, r1s = r1 * r1;
  static u32 prevF = 0;
  float keepF = prevF > FMAX ? (float)FMAX / prevF : 1.0f;
  u32 fcand = 0, drawn = 0, para = 0, pending = 0;
  fcount = 0;
  float nearest = 1e30f;
  i32 focus = -1;
  const float *m = MAT;

  for (u32 i = 0; i < N; i++) {
    float x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
    float cw = m[3] * x + m[7] * y + m[11] * z + m[15];
    u8 *a = ATTR + 2 * i;
    if (cw <= 0.05f) { a[0] = 0; a[1] = 0; continue; }
    float sx = (m[0] * x + m[4] * y + m[8] * z + m[12]) / cw * hw + hw;
    float sy = hh - (m[1] * x + m[5] * y + m[9] * z + m[13]) / cw * hh;
    float dx = sx - gx, dy = sy - gy, e2 = dx * dx + dy * dy;
    if (e2 < nearest && e2 < r0s) { nearest = e2; focus = (i32)i; }
    if (!fov) { a[0] = 200; a[1] = 110; drawn++; continue; }
    if (e2 < r0s) {
      if (ROWSLOT[i] >= 0) {
        fcand++;
        if (fcount < FMAX && (keepF >= 1.0f || (hash32(i) & 0xffff) < (u32)(keepF * 65535.0f))) {
          fidx[fcount] = i; fscr[2 * fcount] = sx; fscr[2 * fcount + 1] = sy;
          fz[fcount] = (m[2] * x + m[6] * y + m[10] * z + m[14]) / cw;
          fcount++;
          a[0] = 0; a[1] = 0;  // drawn by the fovea pass
          continue;
        }
      } else pending++;
      a[0] = 170; a[1] = 90; drawn++;  // in the fovea but not streamed yet
      continue;
    }
    if (e2 < r1s) {
      float t = smoothstep(r0, r1, fsqrt(e2));
      float keep = 1.0f - 0.8f * t;
      if ((hash32(i) & 0xffff) > (u32)(keep * 65535.0f)) { a[0] = 26; a[1] = 40; continue; }
      a[0] = (u8)(210.0f - 120.0f * t); a[1] = (u8)(95.0f - 35.0f * t);
      para++; drawn++;
      continue;
    }
    a[0] = 34; a[1] = 44;  // periphery: faint haze, never labelled, never streamed
  }
  prevF = fcand;

  // Tile scheduling: cells under the fovea, nearest the gaze first.
  nreq = 0;
  u32 want = 0, residentWant = 0;
  if (fov) {
    for (u32 c = 0; c < NC; c++) {
      if (CELLS[c + 1] == CELLS[c]) continue;
      float x = CELL_C[4 * c], y = CELL_C[4 * c + 1], z = CELL_C[4 * c + 2];
      float cw = m[3] * x + m[7] * y + m[11] * z + m[15];
      if (cw <= 0.05f) continue;
      float sx = (m[0] * x + m[4] * y + m[8] * z + m[12]) / cw * hw + hw;
      float sy = hh - (m[1] * x + m[5] * y + m[9] * z + m[13]) / cw * hh;
      float rp = CELL_C[4 * c + 3] * pxScale / cw;
      float dx = sx - gx, dy = sy - gy, dist = fsqrt(dx * dx + dy * dy) - rp;
      if (dist > r0 * 1.15f) continue;
      want++;
      if (CSTATE[c] == 2) { CLAST[c] = frameNo; residentWant++; continue; }
      if (CSTATE[c] == 1) continue;
      float pri = (dist < 0 ? 0 : dist) + cw * 2.0f;  // near the gaze, then near the eye
      if (nreq < QMAX) { req[nreq] = c; reqd[nreq] = pri; nreq++; }
      else if (pri < reqd[QMAX - 1]) { req[QMAX - 1] = c; reqd[QMAX - 1] = pri; }
      else continue;
      for (i32 j = (i32)nreq - 1; j > 0 && reqd[j - 1] > reqd[j]; j--) {
        float td = reqd[j]; reqd[j] = reqd[j - 1]; reqd[j - 1] = td;
        u32 tc = req[j]; req[j] = req[j - 1]; req[j - 1] = tc;
      }
    }
  }

  // Fovea: local PCA on the full 768-d rows actually being looked at.
  float lensBytes = 0, lensShare = -1;  // share of the fovea rows' variation its 2-D local PCA shows
  if (lens && fcount >= 8) {
    float mean[DMAX];
    for (u32 k = 0; k < D; k++) mean[k] = 0;
    for (u32 f = 0; f < fcount; f++) { const float *r = POOL + (u32)ROWSLOT[fidx[f]] * D; for (u32 k = 0; k < D; k++) mean[k] += r[k]; }
    float inv = 1.0f / fcount;
    for (u32 k = 0; k < D; k++) mean[k] *= inv;
    if (!lpcInit) {
      for (u32 k = 0; k < 2 * D; k++) lpc[k] = (float)((hash32(k + 17) & 0xffff) / 65535.0) - 0.5f;
      lpcInit = 1;
    }
    float w1[DMAX], w2[DMAX];
    for (int it = 0; it < 2; it++) {
      for (u32 k = 0; k < D; k++) { w1[k] = 0; w2[k] = 0; }
      for (u32 f = 0; f < fcount; f++) {
        const float *r = POOL + (u32)ROWSLOT[fidx[f]] * D;
        float a1 = 0, a2 = 0;
        for (u32 k = 0; k < D; k++) { float y = r[k] - mean[k]; a1 += y * lpc[k]; a2 += y * lpc[D + k]; }
        for (u32 k = 0; k < D; k++) { float y = r[k] - mean[k]; w1[k] += a1 * y; w2[k] += a2 * y; }
      }
      float n1 = 0, dp = 0, n2 = 0;
      for (u32 k = 0; k < D; k++) n1 += w1[k] * w1[k];
      n1 = 1.0f / fsqrt(n1 + 1e-20f);
      for (u32 k = 0; k < D; k++) { w1[k] *= n1; dp += w1[k] * w2[k]; }
      for (u32 k = 0; k < D; k++) { w2[k] -= dp * w1[k]; n2 += w2[k] * w2[k]; }
      n2 = 1.0f / fsqrt(n2 + 1e-20f);
      for (u32 k = 0; k < D; k++) { lpc[k] = w1[k]; lpc[D + k] = w2[k] * n2; }
    }
    float var = 0, var2 = 0, tot = 0;
    for (u32 f = 0; f < fcount; f++) {
      const float *r = POOL + (u32)ROWSLOT[fidx[f]] * D;
      float a1 = 0, a2 = 0;
      for (u32 k = 0; k < D; k++) { float y = r[k] - mean[k]; a1 += y * lpc[k]; a2 += y * lpc[D + k]; tot += y * y; }
      fdisp[2 * f] = a1; fdisp[2 * f + 1] = a2;
      var += a1 * a1; var2 += a2 * a2;
    }
    lensShare = tot > 0 ? (var + var2) / tot : -1;
    float sd = fsqrt(var * inv) + 1e-6f, k = r0 * 0.8f / (2.6f * sd);
    lensK = lensK <= 0 ? k : lensK * 0.85f + k * 0.15f;
    const float R = r0 * 0.93f, R0 = R * 0.7f, R1 = R - R0;
    for (u32 f = 0; f < fcount; f++) {
      float lx = fdisp[2 * f] * lensK, ly = -fdisp[2 * f + 1] * lensK, len = fsqrt(lx * lx + ly * ly);
      if (len > R0) { float t = (len - R0) / R1, nl = R0 + R1 * (t / (1.0f + t)); lx *= nl / len; ly *= nl / len; }
      fdisp[2 * f] = fscr[2 * f] + (gx + lx - fscr[2 * f]) * lensMix;
      fdisp[2 * f + 1] = fscr[2 * f + 1] + (gy + ly - fscr[2 * f + 1]) * lensMix;
    }
    lensBytes = (float)fcount * D * 4 * 4;  // bytes of full-D rows read by the lens
  } else {
    for (u32 f = 0; f < 2 * fcount; f++) fdisp[f] = fscr[f];
  }
  if (fcount) {  // focus = foveal point nearest the gaze as displayed
    float best = 1e30f;
    for (u32 f = 0; f < fcount; f++) {
      float dx = fdisp[2 * f] - gx, dy = fdisp[2 * f + 1] - gy, q = dx * dx + dy * dy;
      if (q < best) { best = q; focus = (i32)fidx[f]; }
    }
  }

  stats[0] = (float)fcount; stats[1] = (float)fcand; stats[2] = (float)para; stats[3] = (float)pending;
  stats[4] = (float)focus; stats[5] = (float)nreq; stats[6] = (float)want; stats[7] = (float)residentWant;
  stats[8] = (float)slotsUsed * D * 4; stats[9] = (float)slotCap * D * 4; stats[10] = lensBytes;
  stats[11] = (float)drawn; stats[12] = lensShare;
}

// Exact cosine top-k over the rows currently resident (the streamed working set).
EXPORT u32 knn_resident(u32 q, u32 k) {
  if (k > 64) k = 64;
  if (ROWSLOT[q] < 0) return 0;
  const float *qr = POOL + (u32)ROWSLOT[q] * D;
  float qn = 0;
  for (u32 j = 0; j < D; j++) { scratch[j] = qr[j]; qn += qr[j] * qr[j]; }
  qn = 1.0f / fsqrt(qn + 1e-20f);
  for (u32 j = 0; j < k; j++) { knnS[j] = -2.0f; knnI[j] = 0; }
  u32 found = 0;
  for (u32 i = 0; i < N; i++) {
    if (ROWSLOT[i] < 0 || i == q) continue;
    const float *r = POOL + (u32)ROWSLOT[i] * D;
    float dot = 0, nn = 0;
    for (u32 j = 0; j < D; j++) { dot += r[j] * scratch[j]; nn += r[j] * r[j]; }
    float s = dot * qn / fsqrt(nn + 1e-20f);
    if (s > knnS[k - 1]) {
      i32 j = (i32)k - 1;
      while (j > 0 && knnS[j - 1] < s) { knnS[j] = knnS[j - 1]; knnI[j] = knnI[j - 1]; j--; }
      knnS[j] = s; knnI[j] = i;
      found++;
    }
  }
  return found < k ? found : k;
}
