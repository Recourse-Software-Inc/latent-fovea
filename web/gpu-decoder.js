// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// The general decoder (kernel/dec.c) on the GPU, through WebGPU. Same families, same maths and
// the same recordings, with the same interface as decoder.js, so the worker and every view work
// unchanged. Weights stay fp16 in GPU buffers (read with unpack2x16float, so no shader-f16 needed)
// and live outside WebAssembly's 4 GB, which is what lets bigger models run when the GPU has the
// memory. kernel/dec.c is the reference: every kernel below mirrors one of its steps.
//
// A token is one compute pass (embedding, every layer, the recordings), then one copy of what the
// views read back to the CPU. Readouts on request (logit lens, attribution, a neuron's effect)
// are their own small passes, so those methods are async; the worker awaits them.

const MAXT = 256, TOPK = 10, TMAX = 64;  // TMAX: target words attributed in one pass
const SLOTS = ['LN1W', 'LN1B', 'WQ', 'BQ', 'WK', 'BK', 'WV', 'BV', 'QN', 'KN', 'WO', 'BO', 'LN2W', 'LN2B', 'W1', 'B1', 'W3', 'W2', 'B2'];

// Halves per weight slot, as kernel/dec.c's slot_sizes().
function slotSizes(cfg) {
  const { C, F, NH, NKV, HD } = cfg, QD = NH * HD, KD = NKV * HD;
  const nb = cfg.NORM_BIAS, qb = cfg.QKV_BIAS, ob = cfg.O_BIAS, mb = cfg.MLP_BIAS, sw = cfg.MLP === 2, qk = cfg.QK_NORM;
  return [C, nb ? C : 0, QD * C, qb ? QD : 0, KD * C, qb ? KD : 0, KD * C, qb ? KD : 0, qk ? HD : 0, qk ? HD : 0,
    C * QD, ob ? C : 0, C, nb ? C : 0, F * C, mb ? F : 0, sw ? F * C : 0, C * F, mb ? C : 0];
}

// ---------- WGSL ----------
// Every kernel takes its fixed parameters in a 16-slot uniform P (u32, or f32 bits), and the
// position and token row of the step in S. hN(i) reads the i-th fp16 of an array of u32 pairs.
const HEAD = /* wgsl */`
struct Prm { u: array<vec4<u32>, 4> }
@group(0) @binding(0) var<uniform> P: Prm;
fn pu(i: u32) -> u32 { return P.u[i / 4u][i % 4u]; }
fn pf(i: u32) -> f32 { return bitcast<f32>(pu(i)); }
fn pick(w: vec2<f32>, i: u32) -> f32 { return select(w.x, w.y, (i & 1u) == 1u); }
`;
const STEP = `struct Stp { v: vec4<u32> }\n@group(0) @binding(1) var<uniform> S: Stp;\n`;
// Weights are read through wq(i) (element i) and wq2(k) (elements 2k, 2k + 1), so one kernel
// text serves both formats: fp16 pairs in u32s, or 8-bit integers (four per u32) with one f32
// scale per 64 weights (WS, binding 9). $W(n) marks where the weight array is declared.
const WF16 = (b) => `@group(0) @binding(${b}) var<storage, read> W: array<u32>;
fn wq(i: u32) -> f32 { return pick(unpack2x16float(W[i >> 1u]), i); }
fn wq2(k: u32) -> vec2<f32> { return unpack2x16float(W[k]); }`;
const WQ8 = (b) => `@group(0) @binding(${b}) var<storage, read> W: array<u32>;
@group(0) @binding(9) var<storage, read> WS: array<f32>;
fn wq(i: u32) -> f32 { return f32(bitcast<i32>(W[i >> 2u] << (24u - (i & 3u) * 8u)) >> 24u) * WS[i >> 6u]; }
fn wq2(k: u32) -> vec2<f32> { return vec2<f32>(wq(2u * k), wq(2u * k + 1u)); }
fn q4(w: u32) -> vec4<f32> { return vec4<f32>(f32(bitcast<i32>(w << 24u) >> 24u), f32(bitcast<i32>(w << 16u) >> 24u), f32(bitcast<i32>(w << 8u) >> 24u), f32(bitcast<i32>(w) >> 24u)); }`;

const K = {
  // Y[yOff + pos*posStride + r] = W[r] . X[xOff ..] (+ B[r]), one 64-thread workgroup per row.
  mv: HEAD + STEP + `
$W(2)
@group(0) @binding(3) var<storage, read> B: array<u32>;
@group(0) @binding(4) var<storage, read> X: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let r = wg.x + wg.y * nw.x;
  if (r >= pu(0)) { return; }
  let I2 = pu(1) / 2u; let base = r * I2; let xo = pu(2);
  var s = 0.0;
  for (var k = li; k < I2; k += 64u) { let w = unpack2x16float(W[base + k]); s += w.x * X[xo + 2u * k] + w.y * X[xo + 2u * k + 1u]; }
  red[li] = s; workgroupBarrier();
  for (var st = 32u; st > 0u; st >>= 1u) { if (li < st) { red[li] += red[li + st]; } workgroupBarrier(); }
  if (li == 0u) {
    var v = red[0];
    if (pu(4) == 1u) { v += pick(unpack2x16float(B[r >> 1u]), r); }
    Y[pu(3) + S.v.x * pu(5) + r] = v;
  }
}`,
  // LayerNorm (kind 0) or RMSNorm (1) of IN[inOff + n*S.pos*posStride ..] into OUT[outOff ..].
  norm: HEAD + STEP + `
@group(0) @binding(2) var<storage, read> IN: array<f32>;
@group(0) @binding(3) var<storage, read> NW: array<u32>;
@group(0) @binding(4) var<storage, read> NB: array<u32>;
@group(0) @binding(5) var<storage, read_write> OUT: array<f32>;
var<workgroup> red: array<f32, 256>;
fn total(li: u32) -> f32 {
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) { if (li < st) { red[li] += red[li + st]; } workgroupBarrier(); }
  let t = red[0]; workgroupBarrier(); return t;
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  let n = pu(0); let io = pu(1) + S.v.x * pu(6); let oo = pu(2); let rms = pu(3) == 1u; let eps = pf(5);
  var mean = 0.0;
  if (!rms) { var s = 0.0; for (var i = li; i < n; i += 256u) { s += IN[io + i]; } red[li] = s; mean = total(li) / f32(n); }
  var ss = 0.0;
  for (var i = li; i < n; i += 256u) { let d = IN[io + i] - mean; ss += d * d; }
  red[li] = ss; let r = 1.0 / sqrt(total(li) / f32(n) + eps);
  for (var i = li; i < n; i += 256u) {
    var v = (IN[io + i] - mean) * r * pick(unpack2x16float(NW[i >> 1u]), i);
    if (pu(4) == 1u) { v += pick(unpack2x16float(NB[i >> 1u]), i); }
    OUT[oo + i] = v;
  }
}`,
  // Per head (one workgroup each): qwen3's RMSNorm of the head, then rotary positions from a table.
  head: HEAD + STEP + `
@group(0) @binding(2) var<storage, read_write> V: array<f32>;
@group(0) @binding(3) var<storage, read> NW: array<u32>;
@group(0) @binding(4) var<storage, read> ROPE: array<f32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let hd = pu(0); let o = pu(5) + S.v.x * pu(6) + wg.x * hd;
  if (pu(2) == 1u) {
    var ss = 0.0;
    for (var i = li; i < hd; i += 64u) { ss += V[o + i] * V[o + i]; }
    red[li] = ss; workgroupBarrier();
    for (var st = 32u; st > 0u; st >>= 1u) { if (li < st) { red[li] += red[li + st]; } workgroupBarrier(); }
    let r = 1.0 / sqrt(red[0] / f32(hd) + pf(4));
    workgroupBarrier();
    for (var i = li; i < hd; i += 64u) { V[o + i] = V[o + i] * r * pick(unpack2x16float(NW[i >> 1u]), i); }
    storageBarrier();
  }
  if (pu(3) == 1u) {
    let half = pu(1) / 2u; let t = S.v.x * 256u;
    for (var i = li; i < half; i += 64u) {
      let c = ROPE[(t + i) * 2u]; let s = ROPE[(t + i) * 2u + 1u];
      let x1 = V[o + i]; let x2 = V[o + i + half];
      V[o + i] = x1 * c - x2 * s; V[o + i + half] = x2 * c + x1 * s;
    }
  }
}`,
  // Attention for one head per workgroup: scores over the cached keys, softmax, weighted values.
  attn: HEAD + STEP + `
@group(0) @binding(2) var<storage, read> Q: array<f32>;
@group(0) @binding(3) var<storage, read> KC: array<f32>;
@group(0) @binding(4) var<storage, read> VC: array<f32>;
@group(0) @binding(5) var<storage, read_write> ATT: array<f32>;
@group(0) @binding(6) var<storage, read_write> Y: array<f32>;
var<workgroup> sc: array<f32, 256>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let h = wg.x; let hd = pu(0); let kd = pu(1); let kv = (h / pu(2)) * hd; let kco = pu(3); let pos = S.v.x;
  let scale = pf(5);
  var mx = -1e30;
  for (var t = li; t <= pos; t += 64u) {
    var s = 0.0; let kb = kco + t * kd + kv;
    for (var i = 0u; i < hd; i++) { s += Q[h * hd + i] * KC[kb + i]; }
    s *= scale; sc[t] = s; mx = max(mx, s);
  }
  red[li] = mx; workgroupBarrier();
  for (var st = 32u; st > 0u; st >>= 1u) { if (li < st) { red[li] = max(red[li], red[li + st]); } workgroupBarrier(); }
  let m = red[0]; workgroupBarrier();
  var sum = 0.0;
  for (var t = li; t <= pos; t += 64u) { let e = exp(sc[t] - m); sc[t] = e; sum += e; }
  red[li] = sum; workgroupBarrier();
  for (var st = 32u; st > 0u; st >>= 1u) { if (li < st) { red[li] += red[li + st]; } workgroupBarrier(); }
  let inv = 1.0 / red[0]; workgroupBarrier();
  for (var t = li; t <= pos; t += 64u) { let a = sc[t] * inv; sc[t] = a; ATT[pu(4) + h * 256u + t] = a; }
  workgroupBarrier();
  for (var i = li; i < hd; i += 64u) {
    var acc = 0.0;
    for (var t = 0u; t <= pos; t++) { acc += sc[t] * VC[kco + t * kd + kv + i]; }
    Y[h * hd + i] = acc;
  }
}`,
  // Each head's output vector, one thread per (output, head): HOUT[l][h][o] = Wo[o, head h] . y_h x scale.
  oproj: HEAD + `
$W(2)
@group(0) @binding(3) var<storage, read> Y: array<f32>;
@group(0) @binding(4) var<storage, read> HS: array<f32>;
@group(0) @binding(5) var<storage, read_write> HOUT: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let c = pu(0); let qd = pu(1); let hd = pu(2); let nh = pu(3);
  if (g.x >= c * nh) { return; }
  let o = g.x / nh; let h = g.x % nh;
  let base = (o * qd + h * hd) / 2u;
  var s = 0.0;
  for (var k = 0u; k < hd / 2u; k++) { let w = unpack2x16float(W[base + k]); s += w.x * Y[h * hd + 2u * k] + w.y * Y[h * hd + 2u * k + 1u]; }
  HOUT[pu(4) + h * c + o] = s * HS[pu(5) + h];
}`,
  // The heads' outputs added up (plus the projection bias) into the running state.
  osum: HEAD + `
@group(0) @binding(2) var<storage, read> BO: array<u32>;
@group(0) @binding(3) var<storage, read> HOUT: array<f32>;
@group(0) @binding(4) var<storage, read_write> ACC: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let c = pu(0); let o = g.x;
  if (o >= c) { return; }
  var t = 0.0;
  if (pu(3) == 1u) { t = pick(unpack2x16float(BO[o >> 1u]), o); }
  for (var h = 0u; h < pu(1); h++) { t += HOUT[pu(2) + h * c + o]; }
  ACC[o] += t;
}`,
  // How much each head wrote: the length of its output vector.
  headn: HEAD + `
@group(0) @binding(2) var<storage, read> HOUT: array<f32>;
@group(0) @binding(3) var<storage, read_write> HN: array<f32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let c = pu(0); let b = pu(1) + wg.x * c;
  var ss = 0.0;
  for (var i = li; i < c; i += 64u) { ss += HOUT[b + i] * HOUT[b + i]; }
  red[li] = ss; workgroupBarrier();
  for (var st = 32u; st > 0u; st >>= 1u) { if (li < st) { red[li] += red[li + st]; } workgroupBarrier(); }
  if (li == 0u) { HN[pu(2) + wg.x] = sqrt(red[0]); }
}`,
  // The MLP's activation (GELU tanh / erf, or SwiGLU), knocked-out neurons, and its recording.
  act: HEAD + `
@group(0) @binding(2) var<storage, read_write> H1: array<f32>;
@group(0) @binding(3) var<storage, read> H3: array<f32>;
@group(0) @binding(4) var<storage, read> KO: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read_write> MLPA: array<f32>;
fn erf(x0: f32) -> f32 {
  let s = sign(x0); let x = abs(x0); let t = 1.0 / (1.0 + 0.3275911 * x);
  return s * (1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x));
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let f = pu(0); let i = g.x;
  if (i >= f) { return; }
  let x = H1[i]; var v: f32;
  let kind = pu(1);
  if (kind == 2u) { v = x / (1.0 + exp(-x)) * H3[i]; }
  else if (kind == 1u) { v = 0.5 * x * (1.0 + erf(x * 0.70710678)); }
  else { let u = 0.7978845608 * (x + 0.044715 * x * x * x); v = 0.5 * x * (1.0 + (1.0 - 2.0 / (exp(2.0 * u) + 1.0))); }
  let n = KO[0].x;
  for (var j = 1u; j <= n; j++) { let e = KO[j]; if (e.x == pu(2) && e.y == i) { v *= bitcast<f32>(e.z); } }
  H1[i] = v; MLPA[pu(4) + i] = v;
}`,
  // The MLP's output, scaled by any intervention, added into the running state and recorded.
  mout: HEAD + `
@group(0) @binding(2) var<storage, read> TMP: array<f32>;
@group(0) @binding(3) var<storage, read> MS: array<f32>;
@group(0) @binding(4) var<storage, read_write> ACC: array<f32>;
@group(0) @binding(5) var<storage, read_write> MOUT: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let o = g.x;
  if (o >= pu(0)) { return; }
  let t = TMP[o] * MS[pu(1)];
  ACC[o] += t; MOUT[pu(2) + o] = t;
}`,
  // DST[dstOff + i] = SRC[srcOff + i]
  copy: HEAD + `
@group(0) @binding(2) var<storage, read> SRC: array<f32>;
@group(0) @binding(3) var<storage, read_write> DST: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= pu(0)) { return; }
  DST[pu(2) + g.x] = SRC[pu(1) + g.x];
}`,
  // The token's embedding row (plus a learned position, if the model has them).
  embed: HEAD + STEP + `
$W(2)
@group(0) @binding(3) var<storage, read> WPE: array<u32>;
@group(0) @binding(4) var<storage, read_write> X: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let c = pu(0); let i = g.x;
  if (i >= c) { return; }
  var v = wq((S.v.y - pu(1)) * c + i);
  if (pu(2) == 1u) { let b = S.v.x * c + i; v += pick(unpack2x16float(WPE[b >> 1u]), b); }
  X[i] = v;
}`,
  // The size of what this position offers each head: the norm of its value vector, every layer.
  vnorm: HEAD + STEP + `
@group(0) @binding(2) var<storage, read> VC: array<f32>;
@group(0) @binding(3) var<storage, read_write> OUT: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let L = pu(0); let nh = pu(1); let hd = pu(3); let kd = pu(4);
  if (g.x >= L * nh) { return; }
  let l = g.x / nh; let h = g.x % nh;
  let b = (l * 256u + S.v.x) * kd + (h / pu(2)) * hd;
  var s = 0.0;
  for (var i = 0u; i < hd; i++) { s += VC[b + i] * VC[b + i]; }
  OUT[g.x] = sqrt(s);
}`,
  // Column sums of the unembedding over blocks of 256 rows (for the average word): short enough
  // to be accurate in f32; the CPU adds the blocks up in double precision.
  colsum: HEAD + `
$W(2)
@group(0) @binding(3) var<storage, read_write> OUT: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let c = pu(0); let i = wg.x * 64u + li; let blk = wg.y;
  if (i >= c) { return; }
  let r0 = blk * 256u; let r1 = min(r0 + 256u, pu(1));
  var s = 0.0;
  for (var r = r0; r < r1; r++) { s += wq(r * c + i); }
  OUT[(pu(2) + blk) * c + i] = s;
}`,
  // A word's unembedding row minus the average word's (UD), and that times the final norm weight (UG).
  ug: HEAD + `
$W(2)
@group(0) @binding(3) var<storage, read> UM: array<f32>;
@group(0) @binding(4) var<storage, read> NW: array<u32>;
@group(0) @binding(5) var<storage, read_write> UD: array<f32>;
@group(0) @binding(6) var<storage, read_write> UG: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let c = pu(0); let i = g.x;
  if (i >= c) { return; }
  let d = wq(pu(1) * c + i) - UM[i];
  UD[i] = d; UG[i] = d * pick(unpack2x16float(NW[i >> 1u]), i);
}`,
  // Per neuron (two per thread): UG . its output column, and the column's sum.
  gain: HEAD + `
$W(2)
@group(0) @binding(3) var<storage, read> UG: array<f32>;
@group(0) @binding(4) var<storage, read_write> OUT: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let c = pu(0); let f2 = pu(1) / 2u; let k = g.x;
  if (k >= f2) { return; }
  var gs = vec2<f32>(0.0); var cs = vec2<f32>(0.0);
  for (var o = 0u; o < c; o++) { let w = wq2(o * f2 + k); gs += UG[o] * w; cs += w; }
  OUT[2u * k] = gs.x; OUT[2u * k + 1u] = gs.y; OUT[pu(1) + 2u * k] = cs.x; OUT[pu(1) + 2u * k + 1u] = cs.y;
}`,
  // The final norm of several stored depths at once: workgroup k normalises RES[DL[k]] into XK[k].
  normk: HEAD + `
@group(0) @binding(2) var<storage, read> RES: array<f32>;
@group(0) @binding(3) var<storage, read> DL: array<u32>;
@group(0) @binding(4) var<storage, read> NW: array<u32>;
@group(0) @binding(5) var<storage, read> NB: array<u32>;
@group(0) @binding(6) var<storage, read_write> XK: array<f32>;
var<workgroup> red: array<f32, 256>;
fn total(li: u32) -> f32 {
  workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) { if (li < st) { red[li] += red[li + st]; } workgroupBarrier(); }
  let t = red[0]; workgroupBarrier(); return t;
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let n = pu(0); let io = DL[wg.x] * n; let oo = wg.x * n; let rms = pu(3) == 1u; let eps = pf(5);
  var mean = 0.0;
  if (!rms) { var s = 0.0; for (var i = li; i < n; i += 256u) { s += RES[io + i]; } red[li] = s; mean = total(li) / f32(n); }
  var ss = 0.0;
  for (var i = li; i < n; i += 256u) { let d = RES[io + i] - mean; ss += d * d; }
  red[li] = ss; let r = 1.0 / sqrt(total(li) / f32(n) + eps);
  for (var i = li; i < n; i += 256u) {
    var v = (RES[io + i] - mean) * r * pick(unpack2x16float(NW[i >> 1u]), i);
    if (pu(4) == 1u) { v += pick(unpack2x16float(NB[i >> 1u]), i); }
    XK[oo + i] = v;
  }
}`,
  // Y[k * yStride + yOff + r] = W[r] . X[k] for K vectors at once: each weight row is read once.
  mvk: HEAD + `
$W(2)
@group(0) @binding(3) var<storage, read> X: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 2080>;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let r = wg.x + wg.y * nw.x;
  if (r >= pu(0)) { return; }
  let I = pu(1); let I2 = I / 2u; let base = r * I2; let K = pu(2);
  var acc: array<f32, 65>;
  for (var k = li; k < I2; k += 32u) {
    let w = unpack2x16float(W[base + k]);
    for (var j = 0u; j < K; j++) { acc[j] += w.x * X[j * I + 2u * k] + w.y * X[j * I + 2u * k + 1u]; }
  }
  for (var j = 0u; j < K; j++) { red[li * 65u + j] = acc[j]; }
  workgroupBarrier();
  for (var j = li; j < K; j += 32u) {
    var s = 0.0;
    for (var t = 0u; t < 32u; t++) { s += red[t * 65u + j]; }
    Y[j * pu(3) + pu(4) + r] = s;
  }
}`,
  // Per readout k (one workgroup): the maximum, the softmax's sum, the entropy, and each thread's
  // ten best candidates; the CPU merges 256 x 10 candidates instead of scanning the vocabulary.
  stats: HEAD + `
@group(0) @binding(2) var<storage, read> LG: array<f32>;
@group(0) @binding(3) var<storage, read_write> ST: array<f32>;
@group(0) @binding(4) var<storage, read_write> CAND: array<f32>;
var<workgroup> red: array<f32, 256>;
var<workgroup> red2: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let V = pu(0); let b = wg.x * V;
  var mx = -1e30;
  for (var r = li; r < V; r += 256u) { mx = max(mx, LG[b + r]); }
  red[li] = mx; workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) { if (li < st) { red[li] = max(red[li], red[li + st]); } workgroupBarrier(); }
  let m = red[0]; workgroupBarrier();
  var s = 0.0; var sl = 0.0;
  var tv: array<f32, 10>; var ti: array<u32, 10>;
  for (var j = 0u; j < 10u; j++) { tv[j] = -1e30; ti[j] = 0u; }
  for (var r = li; r < V; r += 256u) {
    let l = LG[b + r] - m; let e = exp(l); s += e; sl += e * l;
    if (l > tv[9]) {
      var j = 9u;
      loop { if (j == 0u || tv[j - 1u] >= l) { break; } tv[j] = tv[j - 1u]; ti[j] = ti[j - 1u]; j--; }
      tv[j] = l; ti[j] = r;
    }
  }
  red[li] = s; red2[li] = sl; workgroupBarrier();
  for (var st = 128u; st > 0u; st >>= 1u) { if (li < st) { red[li] += red[li + st]; red2[li] += red2[li + st]; } workgroupBarrier(); }
  if (li == 0u) { ST[wg.x * 4u] = m; ST[wg.x * 4u + 1u] = red[0]; ST[wg.x * 4u + 2u] = log(red[0]) - red2[0] / red[0]; }
  let cb = (wg.x * 256u + li) * 20u;
  for (var j = 0u; j < 10u; j++) { CAND[cb + 2u * j] = tv[j]; CAND[cb + 2u * j + 1u] = bitcast<f32>(ti[j]); }
}`,
  // UD/UG for several target words: entries of RW are (slot, row in this chunk).
  ugk: HEAD + `
$W(2)
@group(0) @binding(3) var<storage, read> UM: array<f32>;
@group(0) @binding(4) var<storage, read> NW: array<u32>;
@group(0) @binding(5) var<storage, read> RW: array<u32>;
@group(0) @binding(6) var<storage, read_write> UD: array<f32>;
@group(0) @binding(7) var<storage, read_write> UG: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let c = pu(0); let e = pu(1) + wg.y; let slot = RW[2u * e]; let row = RW[2u * e + 1u];
  let i = wg.x * 64u + li;
  if (i >= c) { return; }
  let d = wq(row * c + i) - UM[i];
  UD[slot * c + i] = d; UG[slot * c + i] = d * pick(unpack2x16float(NW[i >> 1u]), i);
}`,
  // (UG_t . v, sum v) for a run of vectors (x) and every target (y).
  partsk: HEAD + `
@group(0) @binding(2) var<storage, read> UG: array<f32>;
@group(0) @binding(3) var<storage, read> SRC: array<f32>;
@group(0) @binding(4) var<storage, read_write> OUT: array<f32>;
var<workgroup> r1: array<f32, 64>;
var<workgroup> r2: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let c = pu(0); let b = pu(1) + wg.x * c; let t = wg.y; let ub = t * c;
  var s = 0.0; var m = 0.0;
  for (var i = li; i < c; i += 64u) { let v = SRC[b + i]; s += UG[ub + i] * v; m += v; }
  r1[li] = s; r2[li] = m; workgroupBarrier();
  for (var st = 32u; st > 0u; st >>= 1u) { if (li < st) { r1[li] += r1[li + st]; r2[li] += r2[li + st]; } workgroupBarrier(); }
  if (li == 0u) { let o = 2u * (t * pu(3) + pu(2) + wg.x); OUT[o] = r1[0]; OUT[o + 1u] = r2[0]; }
}`,
  // One neuron's output direction: column i of W2.
  col: HEAD + `
$W(2)
@group(0) @binding(3) var<storage, read_write> OUT: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let o = g.x;
  if (o >= pu(0)) { return; }
  OUT[o] = wq(o * pu(1) + pu(2));
}`,
};

// 8-bit versions of the three hot loops: four weights per u32 read, one scale per 16 words.
const K8 = {
  mv: HEAD + STEP + '$W(2)' + `
@group(0) @binding(3) var<storage, read> B: array<u32>;
@group(0) @binding(4) var<storage, read> X: array<f32>;
@group(0) @binding(5) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let r = wg.x + wg.y * nw.x;
  if (r >= pu(0)) { return; }
  let I4 = pu(1) / 4u; let base = r * I4; let xo = pu(2);
  var s = 0.0;
  for (var k = li; k < I4; k += 64u) {
    let j = xo + 4u * k;
    s += WS[(base + k) >> 4u] * dot(q4(W[base + k]), vec4<f32>(X[j], X[j + 1u], X[j + 2u], X[j + 3u]));
  }
  red[li] = s; workgroupBarrier();
  for (var st = 32u; st > 0u; st >>= 1u) { if (li < st) { red[li] += red[li + st]; } workgroupBarrier(); }
  if (li == 0u) {
    var v = red[0];
    if (pu(4) == 1u) { v += pick(unpack2x16float(B[r >> 1u]), r); }
    Y[pu(3) + S.v.x * pu(5) + r] = v;
  }
}`,
  mvk: HEAD + '$W(2)' + `
@group(0) @binding(3) var<storage, read> X: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 2080>;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let r = wg.x + wg.y * nw.x;
  if (r >= pu(0)) { return; }
  let I = pu(1); let I4 = I / 4u; let base = r * I4; let K = pu(2);
  var acc: array<f32, 65>;
  for (var k = li; k < I4; k += 32u) {
    let w = q4(W[base + k]) * WS[(base + k) >> 4u];
    for (var j = 0u; j < K; j++) { let x = j * I + 4u * k; acc[j] += dot(w, vec4<f32>(X[x], X[x + 1u], X[x + 2u], X[x + 3u])); }
  }
  for (var j = 0u; j < K; j++) { red[li * 65u + j] = acc[j]; }
  workgroupBarrier();
  for (var j = li; j < K; j += 32u) {
    var s = 0.0;
    for (var t = 0u; t < 32u; t++) { s += red[t * 65u + j]; }
    Y[j * pu(3) + pu(4) + r] = s;
  }
}`,
  oproj: HEAD + '$W(2)' + `
@group(0) @binding(3) var<storage, read> Y: array<f32>;
@group(0) @binding(4) var<storage, read> HS: array<f32>;
@group(0) @binding(5) var<storage, read_write> HOUT: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let c = pu(0); let qd = pu(1); let hd = pu(2); let nh = pu(3);
  if (g.x >= c * nh) { return; }
  let o = g.x / nh; let h = g.x % nh;
  let base = (o * qd + h * hd) / 4u;
  var s = 0.0;
  for (var k = 0u; k < hd / 4u; k++) {
    let y = h * hd + 4u * k;
    s += WS[(base + k) >> 4u] * dot(q4(W[base + k]), vec4<f32>(Y[y], Y[y + 1u], Y[y + 2u], Y[y + 3u]));
  }
  HOUT[pu(4) + h * c + o] = s * HS[pu(5) + h];
}`,
};
// fp16 -> 8-bit, 64 weights per scale (absolute maximum / 127), with the error it makes:
// 16 threads per group, four groups per pass; each workgroup adds up its squared error and the
// weights' squared size, so the load can report how far the 8-bit weights are from the real ones.
const QUANT = HEAD + `
@group(0) @binding(2) var<storage, read> SRC: array<u32>;
@group(0) @binding(3) var<storage, read_write> Q: array<u32>;
@group(0) @binding(4) var<storage, read_write> QS: array<f32>;
@group(0) @binding(5) var<storage, read_write> ERR: array<f32>;
var<workgroup> mx: array<f32, 64>;
var<workgroup> r1: array<f32, 64>;
var<workgroup> r2: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let ng = pu(0); let n = pu(1); let iters = pu(2); let g0 = pu(4);
  let sub = li / 16u; let lane = li % 16u;
  var se = 0.0; var sw = 0.0;
  for (var it = 0u; it < iters; it++) {
    let g = (it * nw.x + wg.x) * 4u + sub;
    let e = g * 64u + lane * 4u;
    var v = vec4<f32>(0.0);
    if (g < ng) {
      let a = unpack2x16float(SRC[e >> 1u]); let b = unpack2x16float(SRC[(e >> 1u) + 1u]);
      v = vec4<f32>(a.x, a.y, b.x, b.y) * vec4<f32>(select(0.0, 1.0, e < n), select(0.0, 1.0, e + 1u < n), select(0.0, 1.0, e + 2u < n), select(0.0, 1.0, e + 3u < n));
    }
    mx[li] = max(max(abs(v.x), abs(v.y)), max(abs(v.z), abs(v.w)));
    workgroupBarrier();
    for (var st = 8u; st > 0u; st >>= 1u) { if (lane < st) { mx[li] = max(mx[li], mx[li + st]); } workgroupBarrier(); }
    let m = mx[sub * 16u];
    workgroupBarrier();
    let sc = m / 127.0; let inv = select(0.0, 127.0 / m, m > 0.0);
    let q = clamp(round(v * inv), vec4<f32>(-127.0), vec4<f32>(127.0));
    if (g < ng) {
      let qi = vec4<i32>(q);
      Q[(g0 + g) * 16u + lane] = (u32(qi.x) & 0xffu) | ((u32(qi.y) & 0xffu) << 8u) | ((u32(qi.z) & 0xffu) << 16u) | ((u32(qi.w) & 0xffu) << 24u);
      if (lane == 0u) { QS[g0 + g] = sc; }
      let d = v - q * sc; se += dot(d, d); sw += dot(v, v);
    }
  }
  r1[li] = se; r2[li] = sw; workgroupBarrier();
  for (var st = 32u; st > 0u; st >>= 1u) { if (li < st) { r1[li] += r1[li + st]; r2[li] += r2[li + st]; } workgroupBarrier(); }
  if (li == 0u) { ERR[pu(3) + 2u * wg.x] = r1[0]; ERR[pu(3) + 2u * wg.x + 1u] = r2[0]; }
}`;
const kernelSrc = (name, q8) => ((q8 && K8[name]) || K[name]).replace('$W(2)', (q8 ? WQ8 : WF16)(2));

// ---------- the decoder ----------
export class GpuDecoder {
  // Fails (throws) when there is no WebGPU or no usable adapter: the caller falls back to the CPU.
  static async create() {
    if (typeof navigator === 'undefined' || !navigator.gpu) throw new Error('WebGPU is not available in this browser');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('no GPU adapter');
    const l = adapter.limits;
    const device = await adapter.requestDevice({
      requiredLimits: { maxStorageBufferBindingSize: l.maxStorageBufferBindingSize, maxBufferSize: l.maxBufferSize, maxStorageBuffersPerShaderStage: Math.min(8, l.maxStorageBuffersPerShaderStage) },
    });
    const d = new GpuDecoder();
    const info = adapter.info || {};
    Object.assign(d, { device, gpu: true, simd: false, limits: device.limits, gpuName: [info.vendor, info.architecture || info.description].filter(Boolean).join(' ') || 'GPU' });
    device.lost.then((e) => { d.lost = e.message || 'the GPU stopped responding'; });
    return d;
  }

  // ----- buffers -----
  buf(bytes, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST) {
    const b = this.device.createBuffer({ size: Math.max(16, Math.ceil(bytes / 4) * 4), usage });
    this.bytes = (this.bytes || 0) + b.size;
    return b;
  }
  put(b, data, at = 0) { this.device.queue.writeBuffer(b, at, data.buffer, data.byteOffset, Math.ceil(data.byteLength / 4) * 4); }
  f16(halves) {  // an fp16 array (Uint16Array) as a GPU buffer, padded to whole u32s
    let h = halves;
    if (h.length & 1) { h = new Uint16Array(h.length + 1); h.set(halves); }
    const b = this.buf(h.byteLength); this.put(b, h); return b;
  }
  uni(vals) {  // 16-slot parameter block; {f: x} stores a float
    const a = new Uint32Array(16), f = new Float32Array(a.buffer);
    vals.forEach((v, i) => { if (v && typeof v === 'object') f[i] = v.f; else a[i] = v >>> 0; });
    const b = this.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(b, 0, a); return b;
  }
  pipe(name, q8 = false) {
    this.pipes ||= {};
    const key = q8 ? name + '/q8' : name, code = name === 'quant' ? QUANT : kernelSrc(name, q8);
    return (this.pipes[key] ||= this.device.createComputePipeline({ layout: 'auto', compute: { module: this.device.createShaderModule({ code }), entryPoint: 'main' } }));
  }
  // A dispatch, ready to encode: pipeline, bind group (uniform first, then S if the kernel uses
  // it, then the buffers), and workgroup counts.
  job(name, params, bufs, groups) {
    if (!this.building) return this.job0(name, params, bufs, groups);  // later jobs reuse validated pipelines
    const dev = this.device;
    dev.pushErrorScope('validation');
    const j = this.job0(name, params, bufs, groups);
    (this.checks ||= []).push(dev.popErrorScope().then((e) => e && `${name}: ${e.message}`));
    return j;
  }
  // Every pipeline and binding is validated as it is built; a mistake fails the load (and the
  // model runs on the CPU) instead of silently computing nothing.
  async validated() {
    const errs = (await Promise.all(this.checks || [])).filter(Boolean);
    this.checks = [];
    if (errs.length) throw new Error(`GPU setup failed: ${errs[0]}`);
  }
  // A weight is a tensor { b, s }: fp16 pairs (s null) or 8-bit with scales s; a kernel given an
  // 8-bit tensor runs its 8-bit version, with the scales at binding 9.
  job0(name, params, bufs, groups) {
    const t = bufs.find((b) => b && b.b), q8 = !!(t && t.s);
    const p = this.pipe(name, q8), step = (K[name] || QUANT).includes('var<uniform> S');
    const entries = [{ binding: 0, resource: { buffer: this.uni(params) } }];
    if (step) entries.push({ binding: 1, resource: { buffer: this.S } });
    bufs.forEach((b, i) => entries.push({ binding: i + 2, resource: { buffer: b && b.b ? b.b : b } }));  // buffers always start at binding 2
    if (q8) entries.push({ binding: 9, resource: { buffer: t.s } });
    let gx, gy;
    if (Array.isArray(groups)) [gx, gy] = groups;  // an explicit grid
    else { const n = typeof groups === 'number' ? groups : 1; gx = Math.min(n, 65535); gy = Math.ceil(n / gx); }
    return { p, g: this.device.createBindGroup({ layout: p.getBindGroupLayout(0), entries }), gx, gy };
  }
  run(pass, j) { pass.setPipeline(j.p); pass.setBindGroup(0, j.g); pass.dispatchWorkgroups(j.gx, j.gy); }
  // A table bigger than one binding (the word tables) is kept as chunks of whole rows.
  async chunked(halves, rowLen, kind) {
    const maxB = Math.min(this.limits.maxStorageBufferBindingSize, this.limits.maxBufferSize);
    const per = Math.max(1, Math.floor(maxB / (rowLen * 2))), rows = halves.length / rowLen, out = [];
    for (let r0 = 0; r0 < rows; r0 += per) {
      const n = Math.min(per, rows - r0);
      out.push({ r0, n, t: await this.weight(halves.subarray(r0 * rowLen, (r0 + n) * rowLen), kind) });
    }
    return out;
  }
  // A weight matrix as a tensor: fp16 as stored, or (8-bit mode) quantized on the GPU.
  async weight(halves, kind) { return this.q8 ? this.quantize(halves, kind) : { b: this.f16(halves), s: null }; }
  // fp16 -> 8-bit in slices of 16M weights through one staging buffer; the error is read back
  // per tensor and added to this.qerr[kind] = [squared error, squared size].
  async quantize(halves, kind) {
    const n = halves.length, ng = Math.ceil(n / 64), SL = 1 << 24, dev = this.device;
    const Q = this.buf(ng * 64), QS = this.buf(ng * 4);
    this.qtemp ||= this.buf(SL * 2, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.qerrBuf ||= this.buf(64 * 2048 * 4);
    let slot = 0;
    const jobs = [];
    for (let at = 0; at < n; at += SL) {
      const m = Math.min(SL, n - at), sg = Math.ceil(m / 64);
      let h = halves.subarray(at, at + m);
      if (m & 1) { const pad = new Uint16Array(m + 1); pad.set(h); h = pad; }
      dev.queue.writeBuffer(this.qtemp, 0, h.buffer, h.byteOffset, Math.ceil(h.byteLength / 4) * 4);
      const nwg = Math.min(1024, Math.ceil(sg / 4));
      const j = this.job('quant', [sg, m, Math.ceil(sg / 4 / nwg), slot * 2048, at / 64], [this.qtemp, Q, QS, this.qerrBuf], nwg);
      this.submit([j]);
      jobs.push(nwg); slot++;
    }
    const e = await this.read(this.qerrBuf, slot * 2048 * 4);
    const acc = (this.qerr ||= {})[kind] ||= [0, 0];
    jobs.forEach((nwg, k) => { for (let i = 0; i < nwg; i++) { acc[0] += e[k * 2048 + 2 * i]; acc[1] += e[k * 2048 + 2 * i + 1]; } });
    this.qbytes = (this.qbytes || 0) + ng * 68;
    return { b: Q, s: QS };
  }
  // How far the 8-bit weights are from the published ones: relative RMS error, overall and per kind.
  quantError() {
    if (!this.qerr) return null;
    let se = 0, sw = 0; const by = {};
    for (const [k, [a, b]] of Object.entries(this.qerr)) { se += a; sw += b; by[k] = Math.sqrt(a / (b || 1)); }
    return { all: Math.sqrt(se / (sw || 1)), by };
  }
  async read(src, bytes, at = 0) {  // GPU -> CPU copy of bytes [at, at + bytes) of src
    const st = this.device.createBuffer({ size: Math.ceil(bytes / 4) * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const e = this.device.createCommandEncoder();
    e.copyBufferToBuffer(src, at, st, 0, st.size);
    this.device.queue.submit([e.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(st.getMappedRange().slice(0));
    st.unmap(); st.destroy();
    return out;
  }
  async readMany(parts) {  // several [buffer, floats] regions in one copy and one map
    const total = parts.reduce((t, [, n]) => t + n, 0) * 4, dev = this.device;
    const st = dev.createBuffer({ size: total, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const e = dev.createCommandEncoder();
    let at = 0;
    for (const [b, n] of parts) { e.copyBufferToBuffer(b, 0, st, at, n * 4); at += n * 4; }
    dev.queue.submit([e.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const all = new Float32Array(st.getMappedRange().slice(0));
    st.unmap(); st.destroy();
    let o = 0;
    return parts.map(([, n]) => { const v = all.subarray(o, o + n); o += n; return v; });
  }
  check() { if (this.lost) throw new Error(`GPU: ${this.lost}`); }

  // ----- loading -----
  // opts.q8: keep the weight matrices as 8-bit integers (about half the GPU memory).
  async load(getBuf, manifest, ids, onProgress = () => {}, opts = {}) {
    const cfg = manifest.cfg, dev = this.device;
    this.q8 = !!opts.q8;
    const { V, C, L, NH, NKV, HD, F } = cfg, QD = NH * HD, KD = NKV * HD;
    if (L > 64 || NH % NKV || C % 8 || HD % 8 || F % 8 || (cfg.ROT || 0) > HD) throw new Error('This model is built in a way this tool can\'t run yet.');
    Object.assign(this, { m: manifest, cfg, L, H: NH, C, F, V, NKV, HD, QD, KD, maxT: MAXT });
    dev.pushErrorScope('out-of-memory');
    const B = (n) => this.buf(n * 4);
    this.S = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const none = this.buf(16);
    // Working vectors and recordings (f32).
    const w = this.w = {
      x: B(C), xn: B(C), x2: B(C), contrib: B(C), q: B(QD), y: B(QD), tmp: B(C), h1: B(F), h3: B(F),
      KC: B(L * MAXT * KD), VC: B(L * MAXT * KD), ATT: B(L * NH * MAXT), HEADN: B(L * NH), MLPA: B(L * F),
      RES: B((L + 1) * C), HOUT: B(L * NH * C), MOUT: B(L * C), VN: B(L * NH), HS: B(L * NH), MS: B(L),
      KO: this.buf(16 * 257), LOGITS: B(V), XNL: B(C), QVEC: B(C), UM: B(C), UD: B(C), UG: B(C),
      GAIN: B(2 * F), COL: B(C),
      DL: this.buf(4 * (L + 1)), XK: B((L + 1) * C), LG: B((L + 1) * V), ST: B(4 * (L + 1)), CAND: B((L + 1) * 256 * 20),
      RW: this.buf(8 * TMAX), UDK: B(TMAX * C), UGK: B(TMAX * C), PARTSK: B(TMAX * 2 * (L * NH + L + 1)),
    };
    this.setInterventions({});
    if (this.q8) {  // the quantizer must compile before any weight goes through it
      dev.pushErrorScope('validation'); this.pipe('quant');
      const e = await dev.popErrorScope();
      if (e) throw new Error(`GPU setup failed: quant: ${e.message}`);
    }
    // Rotary table: cos and sin of pos x inverse frequency, worked out in double precision.
    const half = (cfg.ROT || 0) / 2, rope = new Float32Array(MAXT * 256 * 2);
    for (let p = 0; p < MAXT; p++) for (let i = 0; i < half; i++) { const a = p * cfg.INV_FREQ[i]; rope[(p * 256 + i) * 2] = Math.cos(a); rope[(p * 256 + i) * 2 + 1] = Math.sin(a); }
    w.ROPE = this.buf(rope.byteLength); this.put(w.ROPE, rope);
    // Weights.
    const sizes = slotSizes(cfg), files = manifest.layer_files, total = files.length + 3;
    this.lw = [];
    for (let l = 0; l < L; l++) {
      onProgress(l / total, files[l]);
      const all = new Uint16Array(await getBuf('model/' + files[l])), s = {};
      let at = 0;
      const MATS = new Set(['WQ', 'WK', 'WV', 'WO', 'W1', 'W3', 'W2']);  // the matrices; norms and biases stay fp16
      for (let i = 0; i < SLOTS.length; i++) {
        const name = SLOTS[i], n = sizes[i], h = all.subarray(at, at + n);
        s[name] = !n ? null : MATS.has(name) ? await this.weight(h, name) : this.f16(h);
        at += n;
      }
      if (at !== all.length) throw new Error(`layer ${l} of this model has the wrong size; remove it and download it again`);
      this.lw.push(s);
      await dev.queue.onSubmittedWorkDone();  // let the upload drain before reading the next layer
    }
    onProgress(L / total, 'misc');
    const misc = new Uint16Array(await getBuf('model/misc.f16')), np = cfg.POS === 0 ? cfg.NCTX * C : 0;
    this.wpe = np ? this.f16(misc.subarray(0, np)) : none;
    this.lnfwH = misc.slice(np, np + C); this.lnfbH = cfg.NORM_BIAS ? misc.slice(np + C, np + 2 * C) : null;
    this.lnfw = this.f16(this.lnfwH); this.lnfb = this.lnfbH ? this.f16(this.lnfbH) : none;
    onProgress((L + 1) / total, 'vectors.f16');
    this.wte = await this.chunked(new Uint16Array(await getBuf('vectors.f16')), C, 'embedding');
    onProgress((L + 2) / total, 'lmhead');
    this.lm = cfg.TIED ? this.wte : await this.chunked(new Uint16Array(await getBuf('model/lmhead.f16')), C, 'unembedding');
    const err = await dev.popErrorScope();
    if (err) throw new Error(`not enough GPU memory for this model (${Math.round(this.bytes / 1e6)} MB needed)`);
    this.none = none;
    this.building = true;
    this.build();
    // Compile every on-request kernel now too, so a mistake in any of them also shows at load.
    for (const k of ['normk', 'mvk', 'stats', 'ugk', 'partsk', 'colsum', 'ug', 'gain', 'col']) {
      dev.pushErrorScope('validation'); this.pipe(k, this.q8);
      this.checks.push(dev.popErrorScope().then((e) => e && `${k}: ${e.message}`));
    }
    await this.validated();
    this.building = false;
    onProgress(1, 'ready');
    let maxId = 0;
    for (let i = 0; i < ids.length; i++) if (ids[i] > maxId) maxId = ids[i];
    this.rowOf = new Int32Array(maxId + 1).fill(-1);
    ids.forEach((id, row) => { this.rowOf[id] = row; });
    this.ids = ids;
  }

  // Every dispatch of a step, built once: only the position and token row change (in S).
  build() {
    const { cfg, C, L, H: NH, F, QD, KD, HD, w, none } = this, eps = { f: cfg.EPS }, rms = cfg.NORM === 1 ? 1 : 0;
    const nb = cfg.NORM_BIAS ? 1 : 0, mvG = (n) => n;
    const norm = (src, inOff, posStride, nw, nbuf, dst) => this.job('norm', [C, inOff, 0, rms, nbuf === none ? 0 : 1, eps, posStride], [src, nw, nbuf, dst], 1);
    const mv = (W, Bb, X, xOff, Y, yOff, posStride, O, I) => this.job('mv', [O, I, xOff, yOff, Bb ? 1 : 0, posStride], [W, Bb || none, X, Y], mvG(O));
    const copy = (src, sOff, dst, dOff, n) => this.job('copy', [n, sOff, dOff], [src, dst], Math.ceil(n / 64));
    this.embed = this.wte.map((ch) => this.job('embed', [C, ch.r0, cfg.POS === 0 ? 1 : 0], [ch.t, this.wpe, w.x], Math.ceil(C / 64)));
    const seq = [copy(w.x, 0, w.RES, 0, C)];
    for (let l = 0; l < L; l++) {
      const s = this.lw[l], kco = l * MAXT * KD, acc = cfg.PARALLEL ? w.contrib : w.x;
      const attn = [
        mv(s.WQ, s.BQ, w.xn, 0, w.q, 0, 0, QD, C),
        mv(s.WK, s.BK, w.xn, 0, w.KC, kco, KD, KD, C),
        mv(s.WV, s.BV, w.xn, 0, w.VC, kco, KD, KD, C),
      ];
      if (cfg.QK_NORM || cfg.POS === 1) {
        attn.push(this.job('head', [HD, cfg.ROT || 0, cfg.QK_NORM ? 1 : 0, cfg.POS === 1 ? 1 : 0, eps, 0, 0], [w.q, s.QN || none, w.ROPE], NH));
        attn.push(this.job('head', [HD, cfg.ROT || 0, cfg.QK_NORM ? 1 : 0, cfg.POS === 1 ? 1 : 0, eps, kco, KD], [w.KC, s.KN || none, w.ROPE], cfg.NKV));
      }
      attn.push(this.job('attn', [HD, KD, NH / cfg.NKV, kco, l * NH * MAXT, { f: 1 / Math.sqrt(HD) }], [w.q, w.KC, w.VC, w.ATT, w.y], NH));
      attn.push(this.job('oproj', [C, QD, HD, NH, l * NH * C, l * NH], [s.WO, w.y, w.HS, w.HOUT], Math.ceil(C * NH / 64)));
      attn.push(this.job('osum', [C, NH, l * NH * C, s.BO ? 1 : 0], [s.BO || none, w.HOUT, acc], Math.ceil(C / 64)));
      attn.push(this.job('headn', [C, l * NH * C, l * NH], [w.HOUT, w.HEADN], NH));
      const mlp = (xin) => [
        mv(s.W1, s.B1, xin, 0, w.h1, 0, 0, F, C),
        ...(cfg.MLP === 2 ? [mv(s.W3, null, xin, 0, w.h3, 0, 0, F, C)] : []),
        this.job('act', [F, cfg.MLP, l, 0, l * F], [w.h1, cfg.MLP === 2 ? w.h3 : none, w.KO, w.MLPA], Math.ceil(F / 64)),
        mv(s.W2, s.B2, w.h1, 0, w.tmp, 0, 0, C, F),
        this.job('mout', [C, l, l * C], [w.tmp, w.MS, acc, w.MOUT], Math.ceil(C / 64)),
      ];
      if (cfg.PARALLEL) {
        seq.push(norm(w.x, 0, 0, s.LN1W, s.LN1B || none, w.xn), norm(w.x, 0, 0, s.LN2W, s.LN2B || none, w.x2), copy(w.x, 0, w.contrib, 0, C),
          ...attn, ...mlp(w.x2), copy(w.contrib, 0, w.x, 0, C));
      } else {
        seq.push(norm(w.x, 0, 0, s.LN1W, s.LN1B || none, w.xn), ...attn, norm(w.x, 0, 0, s.LN2W, s.LN2B || none, w.xn), ...mlp(w.xn));
      }
      seq.push(copy(w.x, 0, w.RES, (l + 1) * C, C));
    }
    seq.push(this.job('vnorm', [L, NH, NH / cfg.NKV, HD, KD], [w.VC, w.VN], Math.ceil(L * NH / 64)));
    this.seq = seq;
    // The logit lens: final norm of a residual vector, then the unembedding, chunk by chunk.
    this.lensRes = this.job('norm', [C, 0, 0, rms, nb, eps, C], [w.RES, this.lnfw, this.lnfb, w.XNL], 1);  // S.x = depth here
    this.lensQ = this.job('norm', [C, 0, 0, rms, nb, eps, 0], [w.QVEC, this.lnfw, this.lnfb, w.XNL], 1);
    this.lensMv = this.lm.map((ch) => mv(ch.t, null, w.XNL, 0, w.LOGITS, ch.r0, 0, ch.n, C));
    this.rankMv = this.lm.map((ch) => mv(ch.t, null, w.QVEC, 0, w.LOGITS, ch.r0, 0, ch.n, C));
  }

  // ----- a step -----
  async step(row, pos) {
    this.check();
    if (pos >= MAXT || (this.cfg.POS === 0 && pos >= this.cfg.NCTX)) throw new Error(`position ${pos} is past the ${MAXT}-token window`);
    const { device: dev, w, L, H, F, C } = this;
    dev.queue.writeBuffer(this.S, 0, new Uint32Array([pos, row, 0, 0]));
    const e = dev.createCommandEncoder(), pass = e.beginComputePass();
    const ch = this.wte.findIndex((c) => row >= c.r0 && row < c.r0 + c.n);
    this.run(pass, this.embed[ch]);
    for (const j of this.seq) this.run(pass, j);
    pass.end();
    // One readback of everything the views use.
    const parts = [[w.ATT, L * H * MAXT], [w.HEADN, L * H], [w.MLPA, L * F], [w.RES, (L + 1) * C], [w.MOUT, L * C], [w.VN, L * H]];
    const total = parts.reduce((s, [, n]) => s + n, 0) * 4;
    const st = dev.createBuffer({ size: total, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    let at = 0;
    for (const [b, n] of parts) { e.copyBufferToBuffer(b, 0, st, at, n * 4); at += n * 4; }
    dev.queue.submit([e.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const all = new Float32Array(st.getMappedRange().slice(0));
    st.unmap(); st.destroy();
    this.check();
    let o = 0;
    const take = (n) => { const v = all.subarray(o, o + n); o += n; return v; };
    this.rec = { att: take(L * H * MAXT), headN: take(L * H), mlp: take(L * F), res: take((L + 1) * C), mout: take(L * C), vn: take(L * H) };
    this.pos = pos; this.finalLens = null;
  }
  attRow(l, h, q) { return this.rec.att.subarray((l * this.H + h) * MAXT, (l * this.H + h) * MAXT + q + 1); }
  headNorms() { return this.rec.headN; }
  mlp() { return this.rec.mlp; }
  residual(depth) { return this.rec.res.subarray(depth * this.C, (depth + 1) * this.C); }
  mlpOut(l) { return this.rec.mout.subarray(l * this.C, (l + 1) * this.C); }
  valueNorms() { return this.rec.vn.slice(); }
  // On the GPU the readout is cheap: every depth gets one.
  lensDepths() { return Array.from({ length: this.L + 1 }, (_, d) => d); }

  // ----- readouts -----
  async logits(normJob, mvJobs, depth = 0) {
    this.check();
    const dev = this.device;
    if (normJob) dev.queue.writeBuffer(this.S, 0, new Uint32Array([depth, 0, 0, 0]));
    const e = dev.createCommandEncoder(), pass = e.beginComputePass();
    if (normJob) this.run(pass, normJob);
    for (const j of mvJobs) this.run(pass, j);
    pass.end(); dev.queue.submit([e.finish()]);
    return this.read(this.w.LOGITS, this.V * 4);
  }
  // Top 10, entropy and the softmax's normaliser, as the C kernel's lens_on().
  static top(lg) {
    let mx = -Infinity;
    for (let r = 0; r < lg.length; r++) if (lg[r] > mx) mx = lg[r];
    let sum = 0;
    for (let r = 0; r < lg.length; r++) sum += Math.exp(lg[r] - mx);
    const P = new Float64Array(TOPK).fill(-1), I = new Uint32Array(TOPK);
    let H = 0;
    for (let r = 0; r < lg.length; r++) {
      const p = Math.exp(lg[r] - mx) / sum;
      if (p > 1e-12) H -= p * Math.log(p);
      if (p > P[TOPK - 1]) { let j = TOPK - 1; while (j > 0 && P[j - 1] < p) { P[j] = P[j - 1]; I[j] = I[j - 1]; j--; } P[j] = p; I[j] = r; }
    }
    return { rows: Array.from(I), probs: Array.from(P, (p) => Math.fround(p)), entropy: H, mx, sum };
  }
  // The logit lens at several depths of the current token in one pass: the unembedding is read
  // once for all of them, and the top 10, the entropy and the normaliser are found on the GPU.
  async lensMany(depths) {
    this.check();
    const { w, C, V, cfg } = this, K = depths.length;
    if (!K) return [];
    this.put(w.DL, Uint32Array.from(depths));
    const jobs = [this.job('normk', [C, 0, 0, cfg.NORM === 1 ? 1 : 0, cfg.NORM_BIAS ? 1 : 0, { f: cfg.EPS }], [w.RES, w.DL, this.lnfw, this.lnfb, w.XK], [K, 1])];
    for (const ch of this.lm) jobs.push(this.job('mvk', [ch.n, C, K, V, ch.r0], [ch.t, w.XK, w.LG], ch.n));
    jobs.push(this.job('stats', [V], [w.LG, w.ST, w.CAND], [K, 1]));
    this.submit(jobs);
    const [st, cand] = await this.readMany([[w.ST, 4 * K], [w.CAND, K * 256 * 20]]);
    return depths.map((d, k) => {
      const m = st[4 * k], S = st[4 * k + 1], H = st[4 * k + 2], c = cand.subarray(k * 5120, (k + 1) * 5120), iv = new Uint32Array(c.buffer, c.byteOffset, c.length);
      const best = [];
      for (let j = 0; j < 2560; j++) if (c[2 * j] > -1e29) best.push([c[2 * j], iv[2 * j + 1]]);
      best.sort((a, b) => b[0] - a[0]);
      const top = best.slice(0, TOPK);
      if (d === this.L) this.finalLens = { k, m, S };
      return { rows: top.map((x) => x[1]), probs: top.map((x) => Math.fround(Math.exp(x[0]) / S)), entropy: H };
    });
  }
  async lens(depth) { return (await this.lensMany([depth]))[0]; }
  async lensOf(vec) {
    this.put(this.w.QVEC, Float32Array.from(vec));
    const t = GpuDecoder.top(await this.logits(this.lensQ, this.lensMv));
    return { rows: t.rows, probs: t.probs, entropy: t.entropy };
  }
  // After a lens that included the last depth: the probability of any word there.
  async probOf(row) {
    const f = this.finalLens; if (!f || row >= this.V) return null;
    const v = await this.read(this.w.LG, 4, (f.k * this.V + row) * 4);
    return Math.exp(v[0] - f.m) / f.S;
  }
  // The words a direction pushes up and down most, through the final norm's weight.
  async rank(dir) {
    const C = this.C, rms = this.cfg.NORM === 1;
    let m = 0; if (!rms) { for (let o = 0; o < C; o++) m += dir[o]; m /= C; }
    const q = new Float32Array(C);
    for (let o = 0; o < C; o++) q[o] = (dir[o] - m) * halfToFloat(this.lnfwH[o]);
    this.put(this.w.QVEC, q);
    const s = await this.logits(null, this.rankMv), idx = Array.from(s.keys());
    const up = idx.slice().sort((a, b) => s[b] - s[a]).slice(0, TOPK), down = idx.sort((a, b) => s[a] - s[b]).slice(0, TOPK);
    return { up: up.map((r) => ({ row: r, score: s[r] })), down: down.map((r) => ({ row: r, score: s[r] })) };
  }
  vocabOf(vec) { return this.rank(vec); }
  async neuronVocab(l, i) {
    if (l >= this.L || i >= this.F) return null;
    const j = this.job('col', [this.C, this.F, i], [this.lw[l].W2, this.w.COL], Math.ceil(this.C / 64));
    this.submit([j]);
    return this.rank(await this.read(this.w.COL, this.C * 4));
  }
  submit(jobs) {
    const e = this.device.createCommandEncoder(), pass = e.beginComputePass();
    for (const j of jobs) this.run(pass, j);
    pass.end(); this.device.queue.submit([e.finish()]);
  }

  // ----- attribution (see kernel/dec.c: the score is the word's logit minus the average word's) -----
  async umean() {  // the average word's unembedding row, once; in double, as the CPU kernel does
    if (this.umReady) return;
    const C = this.C, nb = this.lm.reduce((t, ch) => t + Math.ceil(ch.n / 256), 0), part = this.buf(nb * C * 4);
    let b0 = 0;
    this.submit(this.lm.map((ch) => { const n = Math.ceil(ch.n / 256), j = this.job('colsum', [C, ch.n, b0], [ch.t, part], [Math.ceil(C / 64), n]); b0 += n; return j; }));
    const blocks = await this.read(part, nb * C * 4), sum = new Float64Array(C);
    part.destroy();
    for (let b = 0; b < nb; b++) for (let i = 0; i < C; i++) sum[i] += blocks[b * C + i];
    this.put(this.w.UM, Float32Array.from(sum, (x) => x / this.V));
    this.umReady = true;
  }
  async setUg(row) {  // UG on the GPU; returns UD (row - average) on the CPU
    await this.umean();
    const ch = this.lm.find((c) => row >= c.r0 && row < c.r0 + c.n);
    this.submit([this.job('ug', [this.C, row - ch.r0], [ch.t, this.w.UM, this.lnfw, this.w.UD, this.w.UG], Math.ceil(this.C / 64))]);
    return this.read(this.w.UD, this.C * 4);
  }
  finalInv() {
    const C = this.C, fin = this.residual(this.L);
    let m = 0, ss = 0;
    if (this.cfg.NORM === 1) { for (let i = 0; i < C; i++) ss += fin[i] * fin[i]; return { m: 0, inv: 1 / Math.sqrt(ss / C + this.cfg.EPS) }; }
    for (let i = 0; i < C; i++) m += fin[i];
    m /= C;
    for (let i = 0; i < C; i++) ss += (fin[i] - m) ** 2;
    return { m, inv: 1 / Math.sqrt(ss / C + this.cfg.EPS) };
  }
  // Attribution toward several words at once: one pass, one readback (see attribute()).
  async attributeMany(rows) {
    const out = [];
    for (let i = 0; i < rows.length; i += TMAX) out.push(...await this.attributeBatch(rows.slice(i, i + TMAX)));
    return out;
  }
  async attributeBatch(rows) {
    await this.umean();
    const { C, L, H, w } = this, T = rows.length, NV = L * H + L + 1;
    const ent = [], jobs = [];
    for (const ch of this.lm) {
      const at = ent.length / 2;
      rows.forEach((r, t) => { if (r >= ch.r0 && r < ch.r0 + ch.n) ent.push(t, r - ch.r0); });
      const n = ent.length / 2 - at;
      if (n) jobs.push([ch, at, n]);
    }
    this.put(w.RW, Uint32Array.from(ent));
    this.submit([
      ...jobs.map(([ch, at, n]) => this.job('ugk', [C, at], [ch.t, w.UM, this.lnfw, w.RW, w.UDK, w.UGK], [Math.ceil(C / 64), n])),
      this.job('partsk', [C, 0, 0, NV], [w.UGK, w.HOUT, w.PARTSK], [L * H, T]),
      this.job('partsk', [C, 0, L * H, NV], [w.UGK, w.MOUT, w.PARTSK], [L, T]),
      this.job('partsk', [C, 0, L * H + L, NV], [w.UGK, w.RES, w.PARTSK], [1, T]),
    ]);
    const [udAll, pAll] = await this.readMany([[w.UDK, T * C], [w.PARTSK, T * 2 * NV]]);
    const { m, inv } = this.finalInv(), rms = this.cfg.NORM === 1, fin = this.residual(L);
    return rows.map((row, t) => {
      if (row >= this.V) return null;
      const ud = udAll.subarray(t * C, (t + 1) * C), p = pAll.subarray(t * 2 * NV, (t + 1) * 2 * NV);
      let sug = 0;
      for (let i = 0; i < C; i++) sug += ud[i] * halfToFloat(this.lnfwH[i]);
      const part = (j) => (rms ? inv * p[2 * j] : inv * (p[2 * j] - sug * p[2 * j + 1] / C));
      const heads = new Float32Array(L * H), mlps = new Float32Array(L);
      let sum = 0;
      for (let j = 0; j < L * H; j++) { heads[j] = part(j); sum += heads[j]; }
      for (let l = 0; l < L; l++) { mlps[l] = part(L * H + l); sum += mlps[l]; }
      const emb = part(L * H + L);
      let logit = 0;
      for (let i = 0; i < C; i++) logit += ud[i] * ((fin[i] - m) * inv * halfToFloat(this.lnfwH[i]) + (this.lnfbH ? halfToFloat(this.lnfbH[i]) : 0));
      return { heads, mlps, emb, rest: logit - sum - emb, logit, inv };
    });
  }
  async attribute(row) { return (await this.attributeMany([row]))[0]; }
  async neuronGain(l, row) {
    if (l >= this.L || row >= this.V) return null;
    const { C, F } = this, ud = await this.setUg(row);
    this.submit([this.job('gain', [C, F], [this.lw[l].W2, this.w.UG, this.w.GAIN], Math.ceil(F / 2 / 64))]);
    const g = await this.read(this.w.GAIN, 2 * F * 4), gain = g.slice(0, F);
    if (this.cfg.NORM !== 1) {
      let sug = 0;
      for (let i = 0; i < C; i++) sug += ud[i] * halfToFloat(this.lnfwH[i]);
      for (let i = 0; i < F; i++) gain[i] -= sug * g[F + i] / C;
    }
    return gain;
  }

  // ----- interventions: { heads: [[l, h, s]], mlps: [[l, s]], neurons: [[l, i, s]] } -----
  setInterventions(spec = {}) {
    const { L, H } = this, hs = new Float32Array(L * H).fill(1), ms = new Float32Array(L).fill(1);
    for (const [l, h, s] of spec.heads || []) if (l < L && h < H) hs[l * H + h] = s;
    for (const [l, s] of spec.mlps || []) if (l < L) ms[l] = s;
    const ko = new Uint32Array(4 * 257), kf = new Float32Array(ko.buffer), list = (spec.neurons || []).slice(0, 256);
    ko[0] = list.length;
    list.forEach(([l, i, s], j) => { ko[4 * (j + 1)] = l; ko[4 * (j + 1) + 1] = i; kf[4 * (j + 1) + 2] = s; });
    this.put(this.w.HS, hs); this.put(this.w.MS, ms); this.put(this.w.KO, ko);
    return true;
  }
  close() { try { this.device.destroy(); } catch {} }
}

function halfToFloat(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}
