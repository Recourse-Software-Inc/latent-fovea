// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// GLM-5.2 on WebGPU, colibri-style: the dense part (attention, shared experts, dense layers,
// lm_head) resident on the GPU in the container's own int4/int8 bytes; the routed experts streamed
// from the container (an external SSD, read in place) into a GPU slot pool with a per-layer LRU, as
// the router asks for them. tools/glm_ref.py is the reference this must match.
//
// Per token the CPU drives each layer: attention (absorbed MLA over a latent cache of
// kv_lora + rope floats per position), then the router's scores come back (E floats), the top-k is
// picked here exactly as the reference does (sigmoid + correction bias, normalised, scaled), missing
// experts are read and uploaded, and the experts and the shared expert run, added into the residual.
import { Container } from './container.js';

const WG = 64;
const fmtFn = { // weight element (row r, column c) of a [O, I] matrix in each format
  i4: `fn wat(r: u32, c: u32) -> f32 {
    let i = r * U.I + c; let word = W[i >> 3u]; let q = f32((word >> ((i & 7u) * 4u)) & 15u) - 8.0;
    return q * SC[r * ((U.I + 63u) / 64u) + c / 64u]; }`,
  i8: `fn wat(r: u32, c: u32) -> f32 {
    let i = r * U.I + c; let word = W[i >> 2u]; let q = f32(i32(word << (24u - (i & 3u) * 8u)) >> 24u);
    return q * SC[r]; }`,
  // f32 has no scales, but must still use the binding or the auto layout drops it
  f32: `fn wat(r: u32, c: u32) -> f32 { return bitcast<f32>(W[r * U.I + c]) + 0.0 * SC[0]; }`,
};
const HEAD = `struct Uni { O: u32, I: u32, xo: u32, yo: u32, acc: u32, alpha: f32, a: u32, b: u32 };
@group(0) @binding(0) var<uniform> U: Uni;`;
// y[yo + o] (=|+=) alpha * sum_i W[o, i] x[xo + i]; one workgroup per output row.
const MV = (fmt) => `${HEAD}
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> SC: array<f32>;
@group(0) @binding(3) var<storage, read> X: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
${fmtFn[fmt]}
var<workgroup> red: array<f32, ${WG}>;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let o = g.x + g.y * n.x;
  if (o >= U.O) { return; }
  var s = 0.0;
  for (var c = l.x; c < U.I; c += ${WG}u) { s += wat(o, c) * X[U.xo + c]; }
  red[l.x] = s; workgroupBarrier();
  for (var k = ${WG / 2}u; k > 0u; k >>= 1u) { if (l.x < k) { red[l.x] += red[l.x + k]; } workgroupBarrier(); }
  if (l.x == 0u) { let v = U.alpha * red[0]; if (U.acc == 1u) { Y[U.yo + o] += v; } else { Y[U.yo + o] = v; } }
}`;
// The same for int4 and int8, faster: each 32-bit word is read once and all its weights used
// (8 nibbles or 4 bytes against 8 or 4 inputs); 8 rows per workgroup, 32 threads per row.
const MVQ = (fmt) => `${HEAD}
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> SC: array<f32>;
@group(0) @binding(3) var<storage, read> X: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let lane = l.x & 31u; let o = (g.x + g.y * n.x) * 8u + (l.x >> 5u);
  var s = 0.0;
  if (o < U.O) {
${fmt === 'i4' ? `    let wpr = U.I / 8u; let ng = (U.I + 63u) / 64u;
    for (var k = lane; k < wpr; k += 32u) {
      let w = W[o * wpr + k]; let x = U.xo + k * 8u;
      var d = 0.0;
      for (var j = 0u; j < 8u; j++) { d += (f32((w >> (j * 4u)) & 15u) - 8.0) * X[x + j]; }
      s += d * SC[o * ng + k / 8u];
    }` : `    let wpr = U.I / 4u;
    for (var k = lane; k < wpr; k += 32u) {
      let w = W[o * wpr + k]; let x = U.xo + k * 4u;
      s += f32(i32(w << 24u) >> 24u) * X[x] + f32(i32(w << 16u) >> 24u) * X[x + 1u] + f32(i32(w << 8u) >> 24u) * X[x + 2u] + f32(i32(w) >> 24u) * X[x + 3u];
    }
    s *= SC[min(o, U.O - 1u)];`}
  }
  red[l.x] = s; workgroupBarrier();
  for (var k = 16u; k > 0u; k >>= 1u) { if (lane < k) { red[l.x] += red[l.x + k]; } workgroupBarrier(); }
  if (lane == 0u && o < U.O) { let v = U.alpha * red[l.x]; if (U.acc == 1u) { Y[U.yo + o] += v; } else { Y[U.yo + o] = v; } }
}`;
// Y[yo..] = X[xo..] / rms * w, over I values (one workgroup).
const RMS = `${HEAD}
@group(0) @binding(1) var<storage, read> Wt: array<f32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_id) l: vec3<u32>) {
  var s = 0.0;
  for (var i = l.x; i < U.I; i += 256u) { let v = X[U.xo + i]; s += v * v; }
  red[l.x] = s; workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) { if (l.x < k) { red[l.x] += red[l.x + k]; } workgroupBarrier(); }
  let r = 1.0 / sqrt(red[0] / f32(U.I) + U.alpha);
  for (var i = l.x; i < U.I; i += 256u) { Y[U.yo + i] = X[U.xo + i] * r * Wt[i]; }
}`;
// Rotary: interleaved pairs in, split halves out. O = groups (heads), I = rope dim, a = pos,
// xo/yo = input/output offsets, b = input stride between groups (output stride is I).
const ROPE = `${HEAD}
@group(0) @binding(1) var<storage, read> F: array<f32>;
@group(0) @binding(2) var<storage, read> X: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let half = U.I / 2u; let h = id.x / half; let j = id.x % half;
  if (h >= U.O) { return; }
  let a = X[U.xo + h * U.b + 2u * j]; let b = X[U.xo + h * U.b + 2u * j + 1u];
  let ang = f32(U.a) * F[j]; let c = cos(ang); let s = sin(ang);
  Y[U.yo + h * U.I + j] = a * c - b * s; Y[U.yo + h * U.I + half + j] = a * s + b * c;
}`;
// MLA absorb: QA[h, j] = sum_d Q[h*QS + d] * Wkvb[h*(NOPE+VD) + d, j], d < NOPE. O = H*KL threads.
const QABS = (fmt) => `${HEAD}
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> SC: array<f32>;
@group(0) @binding(3) var<storage, read> X: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
${fmtFn[fmt]}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let KL = U.I; let NOPE = U.a & 0xffffu; let RV = U.a >> 16u; let QS = U.b;
  let h = id.x / KL; let j = id.x % KL;
  if (h >= U.O) { return; }
  var s = 0.0;
  for (var d = 0u; d < NOPE; d++) { s += X[h * QS + d] * wat(h * RV + d, j); }
  Y[h * KL + j] = s;
}`;
// Scores: S[h, t] = (QA[h]·Lc[t] + QR[h]·Rc[t]) * scale, t < T. Thread per (h, t). a = T, b = R.
const SCORE = `${HEAD}
@group(0) @binding(1) var<storage, read> QA: array<f32>;
@group(0) @binding(2) var<storage, read> QR: array<f32>;
@group(0) @binding(3) var<storage, read> LC: array<f32>;
@group(0) @binding(4) var<storage, read> RC: array<f32>;
@group(0) @binding(5) var<storage, read_write> S: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let T = U.a; let R = U.b; let KL = U.I; let h = id.x / T; let t = id.x % T;
  if (h >= U.O) { return; }
  var s = 0.0;
  for (var j = 0u; j < KL; j++) { s += QA[h * KL + j] * LC[U.xo + t * KL + j]; }
  for (var j = 0u; j < R; j++) { s += QR[h * R + j] * RC[U.yo + t * R + j]; }
  S[h * T + t] = s * U.alpha;
}`;
// Softmax over t < T per head (in place, kept for the views), then CL[h, j] = sum_t p_t Lc[t, j].
const SOFTCL = `${HEAD}
@group(0) @binding(1) var<storage, read> LC: array<f32>;
@group(0) @binding(2) var<storage, read_write> S: array<f32>;
@group(0) @binding(3) var<storage, read_write> CL: array<f32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64) fn main(@builtin(workgroup_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let h = g.x; let T = U.a; let KL = U.I;
  var m = -3.4e38;
  for (var t = l.x; t < T; t += 64u) { m = max(m, S[h * T + t]); }
  red[l.x] = m; workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) { if (l.x < k) { red[l.x] = max(red[l.x], red[l.x + k]); } workgroupBarrier(); }
  let mx = red[0]; workgroupBarrier();
  var s = 0.0;
  for (var t = l.x; t < T; t += 64u) { let e = exp(S[h * T + t] - mx); S[h * T + t] = e; s += e; }
  red[l.x] = s; workgroupBarrier();
  for (var k = 32u; k > 0u; k >>= 1u) { if (l.x < k) { red[l.x] += red[l.x + k]; } workgroupBarrier(); }
  let inv = 1.0 / red[0]; workgroupBarrier();
  for (var t = l.x; t < T; t += 64u) { S[h * T + t] *= inv; }
  storageBarrier(); workgroupBarrier();
  for (var j = l.x; j < KL; j += 64u) {
    var c = 0.0;
    for (var t = 0u; t < T; t++) { c += S[h * T + t] * LC[U.xo + t * KL + j]; }
    CL[h * KL + j] = c;
  }
}`;
// Context: CTX[h*VD + v] = sum_j Wkvb[h*(NOPE+VD) + NOPE + v, j] * CL[h, j]. Thread per (h, v).
const CTX = (fmt) => `${HEAD}
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> SC: array<f32>;
@group(0) @binding(3) var<storage, read> X: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
${fmtFn[fmt]}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let KL = U.I; let NOPE = U.a & 0xffffu; let RV = U.a >> 16u; let VD = U.b;
  let h = id.x / VD; let v = id.x % VD;
  if (h >= U.O) { return; }
  var s = 0.0;
  for (var j = 0u; j < KL; j++) { s += wat(h * RV + NOPE + v, j) * X[h * KL + j]; }
  Y[h * VD + v] = s;
}`;
// G[i] = silu(G[i]) * G[I + i] (gate and up computed into one buffer).
const SILU = `${HEAD}
@group(0) @binding(1) var<storage, read_write> G: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x; if (i >= U.I) { return; }
  let g = G[U.xo + i]; G[U.xo + i] = g / (1.0 + exp(-g)) * G[U.xo + U.I + i];
}`;
// A[yo + t] = mean over heads of S[h*T + t] (the attention this token paid, per layer).
const HAVG = `${HEAD}
@group(0) @binding(1) var<storage, read> S: array<f32>;
@group(0) @binding(2) var<storage, read_write> A: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let t = id.x; let T = U.a; if (t >= T) { return; }
  var s = 0.0; for (var h = 0u; h < U.O; h++) { s += S[h * T + t]; }
  A[U.yo + t] = s / f32(U.O);
}`;
// Gram matrix of a router (f32 rows): G[i*O + j] = row i · row j.
const GRAM = `${HEAD}
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read_write> G: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
  let t = id.x + id.y * n.x * 64u; if (t >= U.O * U.O) { return; }
  let i = t / U.O; let j = t % U.O; var s = 0.0;
  for (var k = 0u; k < U.I; k++) { s += bitcast<f32>(W[i * U.I + k]) * bitcast<f32>(W[j * U.I + k]); }
  G[t] = s;
}`;
const COPY = `${HEAD}
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> Y: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < U.I) { Y[U.yo + id.x] = X[U.xo + id.x]; }
}`;

const align = (n, a = 256) => Math.ceil(n / a) * a;

export class GlmGpu {
  // opts: { expertBudget (bytes for the expert pool), onProgress }
  static async open(container, opts = {}) {
    const g = new GlmGpu(); await g.init(container, opts); return g;
  }

  async init(ct, opts) {
    this.ct = ct;
    const c = ct.config;
    const rp = c.rope_parameters || {};
    Object.assign(this, {
      C: c.hidden_size, H: c.num_attention_heads, QL: c.q_lora_rank, KL: c.kv_lora_rank,
      NOPE: c.qk_nope_head_dim, R: c.qk_rope_head_dim, VD: c.v_head_dim, L: c.num_hidden_layers,
      V: c.vocab_size, FD: c.intermediate_size, FE: c.moe_intermediate_size, E: c.n_routed_experts,
      K: c.num_experts_per_tok, NS: c.n_shared_experts || 1, dense: c.first_k_dense_replace || 0,
      eps: c.rms_norm_eps, theta: rp.rope_theta || c.rope_theta || 10000, scale: c.routed_scaling_factor || 1,
      maxT: Math.min(opts.maxT || 2048, c.max_position_embeddings || 2048),
    });
    if ((c.n_group || 1) !== 1) throw new Error('Group-limited routing (n_group > 1) is not supported.');
    const ad = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    if (!ad) throw new Error('WebGPU is not available.');
    this.dev = await ad.requestDevice({ requiredLimits: {
      maxBufferSize: ad.limits.maxBufferSize, maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize,
      maxStorageBuffersPerShaderStage: Math.min(8, ad.limits.maxStorageBuffersPerShaderStage) } });
    this.maxBind = Math.min(ad.limits.maxStorageBufferBindingSize, ad.limits.maxBufferSize);
    this.pipes = new Map(); this.bytes = 0;
    const p = opts.onProgress || (() => {});
    const { C, H, QL, KL, NOPE, R, VD, L, FD, FE, NS, E } = this;
    const RV = NOPE + VD;
    // ---- resident weights
    const names = []; const W = (n, O, I) => names.push([n, O, I]);
    for (let l = 0; l < L; l++) {
      const a = `model.layers.${l}.self_attn.`, m = `model.layers.${l}.mlp.`;
      W(a + 'q_a_proj.weight', QL, C); W(a + 'q_b_proj.weight', H * (NOPE + R), QL);
      W(a + 'kv_a_proj_with_mqa.weight', KL + R, C); W(a + 'kv_b_proj.weight', H * RV, KL); W(a + 'o_proj.weight', C, H * VD);
      if (l < this.dense) { W(m + 'gate_proj.weight', FD, C); W(m + 'up_proj.weight', FD, C); W(m + 'down_proj.weight', C, FD); }
      else {
        W(m + 'gate.weight', E, C);
        W(m + 'shared_experts.gate_proj.weight', FE * NS, C); W(m + 'shared_experts.up_proj.weight', FE * NS, C); W(m + 'shared_experts.down_proj.weight', C, FE * NS);
      }
    }
    W('lm_head.weight', this.V, C);
    this.w = new Map(); let done = 0, next = 0;
    const loadOne = async () => {  // eight at a time: a drive (or server) serves overlapping reads faster
      while (next < names.length) { const [n, O, I] = names[next++]; this.w.set(n, await this.upload(n, O, I)); p({ stage: 'weights', done: ++done, total: names.length }); }
    };
    await Promise.all(Array.from({ length: 8 }, loadOne));
    this.vec = new Map();
    const vecs = ['model.norm.weight'];
    for (let l = 0; l < L; l++) {
      const a = `model.layers.${l}.`;
      vecs.push(a + 'input_layernorm.weight', a + 'post_attention_layernorm.weight', a + 'self_attn.q_a_layernorm.weight', a + 'self_attn.kv_a_layernorm.weight');
    }
    for (const n of vecs) this.vec.set(n, this.f32buf(new Float32Array(await ct.raw(n))));
    this.bias = []; this.embedFmt = ct.format('model.embed_tokens.weight', this.V, C);
    for (let l = this.dense; l < L; l++) this.bias[l] = new Float32Array(await ct.raw(`model.layers.${l}.mlp.gate.e_score_correction_bias`));
    // ---- state
    const f = (n) => this.f32buf(new Float32Array(n));
    this.X = f(C); this.HN = f(Math.max(C, QL)); this.QR = f(QL); this.QN = f(QL); this.Q = f(H * (NOPE + R)); this.QP = f(H * R);
    this.CKV = f(KL + R); this.QA = f(H * KL); this.CL = f(H * KL); this.CTX = f(H * VD);
    this.S = f(H * this.maxT); this.G = f(2 * Math.max(FD, FE * NS)); this.RL = f(2 * E); this.LOG = f(this.V);  // RL: this layer's router scores, and the next one's (lookahead)
    this.LC = []; this.RC = [];
    for (let l = 0; l < L; l++) { this.LC.push(f(this.maxT * KL)); this.RC.push(f(this.maxT * R)); }
    this.RES = f((L + 1) * C); this.ATT = f(L * this.maxT); this.LN = f(C);
    this.FREQ = this.f32buf(Float32Array.from({ length: R / 2 }, (_, j) => this.theta ** (-2 * j / R)));
    // ---- expert pool: slots of [gate | up | down | scales…], each piece 256-aligned
    const ex = (n) => `model.layers.${this.dense}.mlp.experts.0.${n}_proj.weight`;
    this.exFmt = ['gate', 'up', 'down'].map((n, k) => ct.format(ex(n), k < 2 ? FE : C, k < 2 ? C : FE));
    const pieces = ['gate', 'up', 'down'].flatMap((n) => [ct.ranges(ex(n)).data.bytes, ct.ranges(ex(n)).scales.bytes]);
    this.slotLayout = []; let off = 0;
    for (const b of pieces) { this.slotLayout.push(off); off += align(b); }
    this.slotBytes = off;
    // One pool per MoE layer, equal shares (as colibri does): a shared LRU smaller than one token's
    // working set (75 layers x 8 experts) evicts each next layer's experts just before they are needed,
    // so nothing ever hits. A layer needs at least K slots (one token can use K experts at once).
    const budget = opts.expertBudget || 16e9, nMoE = L - this.dense;
    this.perLayer = Math.min(E, Math.max(this.K, Math.floor(budget / this.slotBytes / nMoE)));
    this.nSlots = this.perLayer * nMoE;
    const perBuf = Math.max(1, Math.floor(this.maxBind / this.slotBytes));
    this.pool = [];
    for (let s = 0; s < this.nSlots; s += perBuf) {
      const n = Math.min(perBuf, this.nSlots - s);
      this.pool.push(this.dev.createBuffer({ size: n * this.slotBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC }));  // COPY_SRC: evicted experts are copied out to RAM
    }
    this.perBuf = perBuf; this.bytes += this.nSlots * this.slotBytes;
    this.slotOf = new Map(); this.keyOf = new Array(this.nSlots).fill(-1); this.lastUse = new Float64Array(this.nSlots); this.clock = 0;
    this.stats = { hits: 0, misses: 0, readBytes: 0, readMs: 0, ramHits: 0, ramBytes: 0, ramSaved: 0,
      prefetched: 0, prefetchUsed: 0, prefetchBytes: 0, prefetchWaitMs: 0, predicted: 0, predictedRight: 0 };
    // Parafovea prefetch (opts.prefetchAhead: how many of the next layer's predicted experts to read
    // early; 0 = off). K + 2 fills the two spare slots of a 10-slot pool with the next-ranked guesses.
    this.prefetchAhead = opts.prefetchAhead ?? 0;
    this.pending = new Map(); this.pendingSlot = new Set(); this.prefetched = new Set(); this.predicted = []; this.tokenClock = 0;
    // RAM tier: per-layer pools of experts evicted from the GPU (opts.ramBudget bytes; 0 = off)
    this.ramPer = Math.floor((opts.ramBudget ?? 0) / this.slotBytes / (L - this.dense));
    this.ram = new Map(); this.ramLru = new Map();
    this.uni = this.dev.createBuffer({ size: 256 * 4096, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.readRL = this.dev.createBuffer({ size: 2 * E * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.pos = 0;
  }

  f32buf(a) {
    const b = this.dev.createBuffer({ size: align(Math.max(a.byteLength, 16), 16), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.dev.queue.writeBuffer(b, 0, a); this.bytes += b.size; return b;
  }
  // A [O, I] weight as the container stores it, split into row chunks that fit one binding.
  async upload(name, O, I) {
    const fmt = this.ct.format(name, O, I);
    const rowB = fmt === 'i4' ? Math.ceil(I / 2) : fmt === 'i8' ? I : I * 4;
    const ng = fmt === 'i4' ? Math.ceil(I / 64) : 1, rowS = fmt === 'f32' ? 0 : ng * 4;
    const rowsPer = Math.max(1, Math.floor(this.maxBind / Math.max(rowB, rowS || 1)));
    const t = this.ct.ranges(name), chunks = [];
    for (let r0 = 0; r0 < O; r0 += rowsPer) {
      const n = Math.min(rowsPer, O - r0);
      const d = await this.ct.files.get(t.data.file).read(t.data.offset + r0 * rowB, n * rowB);
      const buf = this.raw(d);
      const sb = fmt === 'f32' ? this.raw(new ArrayBuffer(16)) : this.raw(await this.ct.files.get(t.scales.file).read(t.scales.offset + r0 * rowS, n * rowS));
      chunks.push({ r0, n, buf, sb });
    }
    return { fmt, O, I, chunks };
  }
  raw(ab) {  // a GPU buffer holding these bytes, padded to whole 32-bit words
    const n = align(Math.max(ab.byteLength, 16), 16), b = this.dev.createBuffer({ size: n, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const pad = new Uint8Array(align(ab.byteLength, 4)); pad.set(new Uint8Array(ab));
    this.dev.queue.writeBuffer(b, 0, pad); this.bytes += n; return b;
  }
  pipe(key, src) {
    if (!this.pipes.has(key)) this.pipes.set(key, this.dev.createComputePipeline({ layout: 'auto', compute: { module: this.dev.createShaderModule({ code: src }), entryPoint: 'main' } }));
    return this.pipes.get(key);
  }

  // ---- recording: ops are collected, then written and submitted together
  begin() { this.ops = []; }
  op(key, src, u, bufs, groups) {
    const a = new ArrayBuffer(32), dv = new DataView(a);
    [u.O || 0, u.I || 0, u.xo || 0, u.yo || 0, u.acc || 0].forEach((v, k) => dv.setUint32(k * 4, v, true));
    dv.setFloat32(20, u.alpha ?? 1, true); dv.setUint32(24, u.a || 0, true); dv.setUint32(28, u.b || 0, true);
    this.ops.push({ p: this.pipe(key, src), a, bufs, groups });
  }
  async submit(after) {
    if (this.ops.length > 4096) throw new Error('too many operations in one submission');
    const up = new Uint8Array(Math.max(1, this.ops.length) * 256);
    this.ops.forEach((o, k) => up.set(new Uint8Array(o.a), k * 256));
    this.dev.queue.writeBuffer(this.uni, 0, up);
    this.dev.pushErrorScope('validation');
    const enc = this.dev.createCommandEncoder(), pass = enc.beginComputePass();
    this.ops.forEach((o, k) => {
      const entries = [{ binding: 0, resource: { buffer: this.uni, offset: k * 256, size: 32 } },
        ...o.bufs.map((b, j) => ({ binding: j + 1, resource: b instanceof GPUBuffer ? { buffer: b } : b }))];
      pass.setPipeline(o.p);
      pass.setBindGroup(0, this.dev.createBindGroup({ layout: o.p.getBindGroupLayout(0), entries }));
      const [x, y] = o.groups > 65535 ? [65535, Math.ceil(o.groups / 65535)] : [o.groups, 1];
      pass.dispatchWorkgroups(x, y);
    });
    pass.end();
    const encErr = await this.dev.popErrorScope();
    if (encErr) throw new Error('GPU: ' + encErr.message);
    if (after) after(enc);
    this.dev.pushErrorScope('validation');
    this.dev.queue.submit([enc.finish()]); this.ops = [];
    const err = await this.dev.popErrorScope();
    if (err) throw new Error('GPU: ' + err.message);
  }

  // y = W x (or y += alpha W x), over a weight's row chunks
  mv(w, x, y, { xo = 0, yo = 0, acc = 0, alpha = 1 } = {}) {
    for (const c of w.chunks) this.mvRaw(w.fmt, c.n, w.I, c.buf, c.sb, x, y, { xo, yo: yo + c.r0, acc, alpha });
  }
  mvSlot(fmt, O, I, data, scales, x, y, o) { this.mvRaw(fmt, O, I, data, scales, x, y, o); }
  mvRaw(fmt, O, I, data, scales, x, y, o) {
    const fast = (fmt === 'i4' && I % 8 === 0) || (fmt === 'i8' && I % 4 === 0);  // whole words per row
    if (fast) this.op('mvq' + fmt, MVQ(fmt), { O, I, ...o }, [data, scales, x, y], Math.ceil(O / 8));
    else this.op('mv' + fmt, MV(fmt), { O, I, ...o }, [data, scales, x, y], O);
  }
  rms(w, x, y, I, xo = 0, yo = 0) { this.op('rms', RMS, { I, xo, yo, alpha: this.eps }, [w, x, y], 1); }

  // ---- one token
  async embedRow(tok) {
    const { C } = this, t = this.ct.ranges('model.embed_tokens.weight'), f = this.embedFmt;
    const rowB = f === 'i4' ? Math.ceil(C / 2) : f === 'i8' ? C : C * 4;
    const d = new Uint8Array(await this.ct.files.get(t.data.file).read(t.data.offset + tok * rowB, rowB));
    const x = new Float32Array(C);
    if (f === 'f32') x.set(new Float32Array(d.buffer));
    else {
      const ng = f === 'i4' ? Math.ceil(C / 64) : 1;
      const s = new Float32Array(await this.ct.files.get(t.scales.file).read(t.scales.offset + tok * ng * 4, ng * 4));
      for (let i = 0; i < C; i++) x[i] = f === 'i8' ? ((d[i] << 24) >> 24) * s[0] : (((d[i >> 1] >> ((i & 1) * 4)) & 15) - 8) * s[Math.floor(i / 64)];
    }
    return x;
  }

  // Runs token `tok` at position `pos`; returns { logits, experts[l] = [[e, weight]…], probs } and
  // keeps residuals (RES) and attention (S, last layer) on the GPU for the views.
  // draft: null (exact), 'drop' or 'subst': a provisional pass from what is already on the GPU, with
  // no drive reads. Chosen experts not resident are left out ('drop', the rest rescaled to the same
  // total weight) or replaced by the resident expert of the same layer whose router row is closest
  // ('subst': the periphery at lower fidelity). Leaves the cache order alone; the exact pass for the
  // same position afterwards overwrites everything it wrote.
  async step(tok, pos, draft = null) {
    if (pos >= this.maxT) throw new Error(`Context is limited to ${this.maxT} tokens here.`);
    if (draft === 'subst' && !this.cos) await this.routerCosines();
    const { C, H, QL, KL, NOPE, R, VD, L, FD, FE, NS, E, K } = this, RV = NOPE + VD, T = pos + 1;
    this.dev.queue.writeBuffer(this.X, 0, await this.embedRow(tok));
    if (!draft) { this.tokenClock = this.clock + 1; this.predicted = []; }  // slots used from here on belong to this token
    const draftInfo = { missing: 0, substituted: 0 };
    const chosen = [], scores = [];
    this.begin();
    this.op('copy', COPY, { I: C, xo: 0, yo: 0 }, [this.X, this.RES], Math.ceil(C / 64));
    for (let l = 0; l < L; l++) {
      const a = `model.layers.${l}.`, w = (n) => this.w.get(a + n), v = (n) => this.vec.get(a + n);
      // attention
      this.rms(v('input_layernorm.weight'), this.X, this.HN, C);
      this.mv(w('self_attn.q_a_proj.weight'), this.HN, this.QR);
      this.rms(v('self_attn.q_a_layernorm.weight'), this.QR, this.QN, QL);  // not in place: a buffer can't be read and written in one dispatch
      this.mv(w('self_attn.q_b_proj.weight'), this.QN, this.Q);
      this.op('rope', ROPE, { O: H, I: R, a: pos, b: NOPE + R, xo: NOPE, yo: 0 }, [this.FREQ, this.Q, this.QP], Math.ceil(H * R / 2 / 64));
      this.mv(w('self_attn.kv_a_proj_with_mqa.weight'), this.HN, this.CKV);
      this.rms(v('self_attn.kv_a_layernorm.weight'), this.CKV, this.LC[l], KL, 0, pos * KL);
      this.op('rope', ROPE, { O: 1, I: R, a: pos, b: R, xo: KL, yo: pos * R }, [this.FREQ, this.CKV, this.RC[l]], 1);
      const kvb = w('self_attn.kv_b_proj.weight'), c0 = kvb.chunks[0];
      if (kvb.chunks.length > 1) throw new Error('kv_b larger than one binding');
      this.op('qabs' + kvb.fmt, QABS(kvb.fmt), { O: H, I: KL, a: NOPE | (RV << 16), b: NOPE + R }, [c0.buf, c0.sb, this.Q, this.QA], Math.ceil(H * KL / 64));
      this.op('score', SCORE, { O: H, I: KL, a: T, b: R, alpha: 1 / Math.sqrt(NOPE + R) }, [this.QA, this.QP, this.LC[l], this.RC[l], this.S], Math.ceil(H * T / 64));
      this.op('softcl', SOFTCL, { I: KL, a: T }, [this.LC[l], this.S, this.CL], H);
      this.op('havg', HAVG, { O: H, a: T, yo: l * this.maxT }, [this.S, this.ATT], Math.ceil(T / 64));
      this.op('ctx' + kvb.fmt, CTX(kvb.fmt), { O: H, I: KL, a: NOPE | (RV << 16), b: VD }, [c0.buf, c0.sb, this.CL, this.CTX], Math.ceil(H * VD / 64));
      this.mv(w('self_attn.o_proj.weight'), this.CTX, this.X, { acc: 1 });
      // feed-forward
      this.rms(v('post_attention_layernorm.weight'), this.X, this.HN, C);
      if (l < this.dense) {
        this.mv(w('mlp.gate_proj.weight'), this.HN, this.G); this.mv(w('mlp.up_proj.weight'), this.HN, this.G, { yo: FD });
        this.op('silu', SILU, { I: FD }, [this.G], Math.ceil(FD / 64));
        this.mv(w('mlp.down_proj.weight'), this.G, this.X, { acc: 1 });
        chosen.push([]);
      } else {
        this.mv(w('mlp.gate.weight'), this.HN, this.RL);
        // lookahead: the next layer's router on this layer's state predicts what it will ask for
        const ahead = !draft && this.prefetchAhead > 0 && l + 1 < L;
        if (ahead) this.mv(this.w.get(`model.layers.${l + 1}.mlp.gate.weight`), this.HN, this.RL, { yo: E });
        await this.submit((enc) => enc.copyBufferToBuffer(this.RL, 0, this.readRL, 0, (ahead ? 2 : 1) * E * 4));
        await this.readRL.mapAsync(GPUMapMode.READ);
        const both = new Float32Array(this.readRL.getMappedRange().slice(0)); this.readRL.unmap();
        const { order, pick, sig } = this.route(both.subarray(0, E), l);
        if (!draft && this.predicted[l]) this.scorePrediction(this.predicted[l], order);
        if (ahead) {
          const nx = this.route(both.subarray(E, 2 * E), l + 1, this.prefetchAhead).order;
          this.predicted[l + 1] = nx.slice(0, K);
          this.prefetch(l + 1, nx);  // starts reading now, alongside this layer's own misses
        }
        let slots, use = pick;
        if (draft) ({ slots, use } = this.draftSlots(l, pick, draft, draftInfo));
        else { slots = await this.ensure(l, order); pick.forEach((pk, k) => pk.push(this.lastTier[k])); }  // [expert, weight, tier]
        chosen.push(use); scores[l] = sig;  // (sig: the router's scores, for the views)
        this.begin();
        const FS = FE * NS;
        this.mv(w('mlp.shared_experts.gate_proj.weight'), this.HN, this.G); this.mv(w('mlp.shared_experts.up_proj.weight'), this.HN, this.G, { yo: FS });
        this.op('silu', SILU, { I: FS }, [this.G], Math.ceil(FS / 64));
        this.mv(w('mlp.shared_experts.down_proj.weight'), this.G, this.X, { acc: 1 });
        use.forEach(([, wt], k) => {
          const [gd, gs, ud, us, dd, ds] = this.slotViews(slots[k]);
          this.mvSlot(this.exFmt[0], FE, C, gd, gs, this.HN, this.G, {});
          this.mvSlot(this.exFmt[1], FE, C, ud, us, this.HN, this.G, { yo: FE });
          this.op('silu', SILU, { I: FE }, [this.G], Math.ceil(FE / 64));
          this.mvSlot(this.exFmt[2], C, FE, dd, ds, this.G, this.X, { acc: 1, alpha: wt });
        });
      }
      this.op('copy', COPY, { I: C, xo: 0, yo: (l + 1) * C }, [this.X, this.RES], Math.ceil(C / 64));
    }
    this.rms(this.vec.get('model.norm.weight'), this.X, this.HN, C);
    this.mv(this.w.get('lm_head.weight'), this.HN, this.LOG);
    const rd = this.dev.createBuffer({ size: this.V * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    await this.submit((enc) => enc.copyBufferToBuffer(this.LOG, 0, rd, 0, this.V * 4));
    await rd.mapAsync(GPUMapMode.READ);
    const logits = new Float32Array(rd.getMappedRange().slice(0)); rd.unmap(); rd.destroy();
    this.pos = pos + 1;
    return { logits, experts: chosen, scores, draft: draft ? draftInfo : null };
  }
  // A whole prompt at once, layer by layer (colibri's batch-union): each layer reads every expert
  // its tokens chose once, then runs it for all of them; the same arithmetic as step() per token.
  // Returns the last token's logits and every token's experts; onLayer(l) reports progress.
  async prefill(ids, pos0 = 0, onLayer = () => {}) {
    const n = ids.length;
    if (pos0 + n > this.maxT) throw new Error(`Context is limited to ${this.maxT} tokens here.`);
    if (n === 1) { const r = await this.step(ids[0], pos0); return { logits: r.logits, experts: r.experts.map((ch) => [ch]) }; }
    const { C, H, QL, KL, NOPE, R, VD, L, FD, FE, NS, E, K } = this, RV = NOPE + VD, FS = FE * NS;
    if (!this.XS || this.XS.size < n * C * 4) {
      for (const b of [this.XS, this.HNS, this.RLS]) b?.destroy();
      const f = (m) => this.f32buf(new Float32Array(m));
      this.XS = f(n * C); this.HNS = f(n * C); this.RLS = f(n * E);
      this.readRLS = this.dev.createBuffer({ size: n * E * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }
    const x0 = new Float32Array(n * C);
    for (let t = 0; t < n; t++) x0.set(await this.embedRow(ids[t]), t * C);
    this.dev.queue.writeBuffer(this.XS, 0, x0);
    const last = n - 1, experts = [], lastScores = [];
    this.begin();
    this.op('copy', COPY, { I: C, xo: last * C, yo: 0 }, [this.XS, this.RES], Math.ceil(C / 64));
    for (let l = 0; l < L; l++) {
      const a = `model.layers.${l}.`, w = (nm) => this.w.get(a + nm), v = (nm) => this.vec.get(a + nm);
      const kvb = w('self_attn.kv_b_proj.weight'), c0 = kvb.chunks[0];
      for (let t = 0; t < n; t++) {  // attention, in order: token t reads the latents of tokens < t
        const pos = pos0 + t, T = pos + 1, xo = t * C;
        this.rms(v('input_layernorm.weight'), this.XS, this.HN, C, xo, 0);
        this.mv(w('self_attn.q_a_proj.weight'), this.HN, this.QR);
        this.rms(v('self_attn.q_a_layernorm.weight'), this.QR, this.QN, QL);
        this.mv(w('self_attn.q_b_proj.weight'), this.QN, this.Q);
        this.op('rope', ROPE, { O: H, I: R, a: pos, b: NOPE + R, xo: NOPE, yo: 0 }, [this.FREQ, this.Q, this.QP], Math.ceil(H * R / 2 / 64));
        this.mv(w('self_attn.kv_a_proj_with_mqa.weight'), this.HN, this.CKV);
        this.rms(v('self_attn.kv_a_layernorm.weight'), this.CKV, this.LC[l], KL, 0, pos * KL);
        this.op('rope', ROPE, { O: 1, I: R, a: pos, b: R, xo: KL, yo: pos * R }, [this.FREQ, this.CKV, this.RC[l]], 1);
        this.op('qabs' + kvb.fmt, QABS(kvb.fmt), { O: H, I: KL, a: NOPE | (RV << 16), b: NOPE + R }, [c0.buf, c0.sb, this.Q, this.QA], Math.ceil(H * KL / 64));
        this.op('score', SCORE, { O: H, I: KL, a: T, b: R, alpha: 1 / Math.sqrt(NOPE + R) }, [this.QA, this.QP, this.LC[l], this.RC[l], this.S], Math.ceil(H * T / 64));
        this.op('softcl', SOFTCL, { I: KL, a: T }, [this.LC[l], this.S, this.CL], H);
        if (t === last) this.op('havg', HAVG, { O: H, a: T, yo: l * this.maxT }, [this.S, this.ATT], Math.ceil(T / 64));
        this.op('ctx' + kvb.fmt, CTX(kvb.fmt), { O: H, I: KL, a: NOPE | (RV << 16), b: VD }, [c0.buf, c0.sb, this.CL, this.CTX], Math.ceil(H * VD / 64));
        this.mv(w('self_attn.o_proj.weight'), this.CTX, this.XS, { yo: xo, acc: 1 });
        this.rms(v('post_attention_layernorm.weight'), this.XS, this.HNS, C, xo, xo);
        if (this.ops.length > 3500) { await this.submit(); this.begin(); }  // long prompts: one submission holds 4096
      }
      if (l < this.dense) {
        for (let t = 0; t < n; t++) {
          this.mv(w('mlp.gate_proj.weight'), this.HNS, this.G, { xo: t * C }); this.mv(w('mlp.up_proj.weight'), this.HNS, this.G, { xo: t * C, yo: FD });
          this.op('silu', SILU, { I: FD }, [this.G], Math.ceil(FD / 64));
          this.mv(w('mlp.down_proj.weight'), this.G, this.XS, { yo: t * C, acc: 1 });
        }
        experts.push(ids.map(() => []));
      } else {
        for (let t = 0; t < n; t++) this.mv(w('mlp.gate.weight'), this.HNS, this.RLS, { xo: t * C, yo: t * E });
        await this.submit((enc) => enc.copyBufferToBuffer(this.RLS, 0, this.readRLS, 0, n * E * 4));
        await this.readRLS.mapAsync(GPUMapMode.READ);
        const lg = new Float32Array(this.readRLS.getMappedRange().slice(0)); this.readRLS.unmap();
        const bias = this.bias[l], users = new Map(), picks = [];
        lastScores[l] = Array.from(lg.subarray(last * E, (last + 1) * E), (z) => 1 / (1 + Math.exp(-z)));  // the last token's, for the views
        for (let t = 0; t < n; t++) {
          const sig = Array.from(lg.subarray(t * E, (t + 1) * E), (z) => 1 / (1 + Math.exp(-z)));
          const order = sig.map((s2, e) => [s2 + bias[e], e]).sort((x, y) => y[0] - x[0] || x[1] - y[1]).slice(0, K).map(([, e]) => e);
          let sum = 0; for (const e of order) sum += sig[e];
          picks.push(order.map((e) => [e, sig[e] / sum * this.scale]));
          for (const e of order) { if (!users.has(e)) users.set(e, []); users.get(e).push([t, sig[e] / sum * this.scale]); }
        }
        this.begin();
        for (let t = 0; t < n; t++) {
          this.mv(w('mlp.shared_experts.gate_proj.weight'), this.HNS, this.G, { xo: t * C }); this.mv(w('mlp.shared_experts.up_proj.weight'), this.HNS, this.G, { xo: t * C, yo: FS });
          this.op('silu', SILU, { I: FS }, [this.G], Math.ceil(FS / 64));
          this.mv(w('mlp.shared_experts.down_proj.weight'), this.G, this.XS, { yo: t * C, acc: 1 });
        }
        // the union of chosen experts, at most one layer pool at a time: load, run for every token, submit
        const union = [...users.keys()], tiers = new Map();
        for (let i = 0; i < union.length; i += this.perLayer) {
          const chunk = union.slice(i, i + this.perLayer), slots = await this.ensure(l, chunk);
          chunk.forEach((e, k) => tiers.set(e, this.lastTier[k]));
          chunk.forEach((e, k) => {
            const [gd, gs, ud, us, dd, ds] = this.slotViews(slots[k]);
            for (const [t, wt] of users.get(e)) {
              this.mvSlot(this.exFmt[0], FE, C, gd, gs, this.HNS, this.G, { xo: t * C });
              this.mvSlot(this.exFmt[1], FE, C, ud, us, this.HNS, this.G, { xo: t * C, yo: FE });
              this.op('silu', SILU, { I: FE }, [this.G], Math.ceil(FE / 64));
              this.mvSlot(this.exFmt[2], C, FE, dd, ds, this.G, this.XS, { yo: t * C, acc: 1, alpha: wt });
            }
          });
          await this.submit(); this.begin();  // before the next chunk may reuse these slots
        }
        experts.push(picks.map((pk) => pk.map(([e, wt]) => [e, wt, tiers.get(e)])));
      }
      this.op('copy', COPY, { I: C, xo: last * C, yo: (l + 1) * C }, [this.XS, this.RES], Math.ceil(C / 64));
      onLayer(l);
    }
    this.op('copy', COPY, { I: C, xo: last * C, yo: 0 }, [this.XS, this.X], Math.ceil(C / 64));
    this.rms(this.vec.get('model.norm.weight'), this.XS, this.HN, C, last * C, 0);
    this.mv(this.w.get('lm_head.weight'), this.HN, this.LOG);
    const rd = this.dev.createBuffer({ size: this.V * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    await this.submit((enc) => enc.copyBufferToBuffer(this.LOG, 0, rd, 0, this.V * 4));
    await rd.mapAsync(GPUMapMode.READ);
    const logits = new Float32Array(rd.getMappedRange().slice(0)); rd.unmap(); rd.destroy();
    this.pos = pos0 + n;
    return { logits, experts, scores: lastScores };  // experts[l][t] = [[e, weight, tier]…]; scores: the last token's, per layer
  }
  slotViews(s) {
    const buf = this.pool[Math.floor(s / this.perBuf)], base = (s % this.perBuf) * this.slotBytes;
    const sizes = this.slotSizes();
    return this.slotLayout.map((o, k) => ({ buffer: buf, offset: base + o, size: align(sizes[k], 4) }));
  }
  slotSizes() {
    if (!this._sizes) { const ex = (n) => this.ct.ranges(`model.layers.${this.dense}.mlp.experts.0.${n}_proj.weight`); this._sizes = ['gate', 'up', 'down'].flatMap((n) => [ex(n).data.bytes, ex(n).scales.bytes]); }
    return this._sizes;
  }
  // A draft's experts for one layer: the resident ones (not mid-read), and for the rest a resident
  // stand-in of the same layer ('subst', by router-row cosine) or nothing ('drop', weights rescaled).
  draftSlots(l, pick, mode, info) {
    const lo = (l - this.dense) * this.perLayer, slots = [], use = [];
    const resident = (e) => { const s2 = this.slotOf.get(l * this.E + e); return s2 !== undefined && !this.pendingSlot.has(s2) ? s2 : -1; };
    let total = 0, kept = 0;
    for (const [e, wt] of pick) {
      total += wt;
      const s2 = resident(e);
      if (s2 >= 0) { slots.push(s2); use.push([e, wt, 'gpu']); kept += wt; continue; }
      info.missing++;
      if (mode !== 'subst') continue;
      let best = -1, bc = -2;
      const cos = this.cos[l - this.dense];
      for (let i = lo; i < lo + this.perLayer; i++) {
        const k = this.keyOf[i]; if (k < 0 || this.pendingSlot.has(i) || use.some(([x]) => l * this.E + x === k)) continue;
        const c = cos[e * this.E + (k % this.E)]; if (c > bc) { bc = c; best = i; }
      }
      if (best >= 0) { slots.push(best); use.push([this.keyOf[best] % this.E, wt, 'sub']); kept += wt; info.substituted++; }
    }
    if (mode === 'drop' && kept > 0 && kept < total) for (const u of use) u[1] *= total / kept;
    return { slots, use };
  }
  // Cosine between every two experts' router rows, per MoE layer (for 'subst' drafts), on the GPU.
  async routerCosines() {
    const { E, C } = this, out = [];
    const G = this.f32buf(new Float32Array(E * E));
    for (let l = this.dense; l < this.L; l++) {
      const c = this.w.get(`model.layers.${l}.mlp.gate.weight`).chunks[0];
      this.begin(); this.op('gram', GRAM, { O: E, I: C }, [c.buf, G], Math.ceil(E * E / 64)); await this.submit();
      const g = await this.read(G, E * E), n = new Float32Array(E);
      for (let i = 0; i < E; i++) n[i] = Math.sqrt(g[i * E + i]) || 1;
      for (let i = 0; i < E; i++) for (let j = 0; j < E; j++) g[i * E + j] /= n[i] * n[j];
      out.push(g);
    }
    G.destroy(); this.cos = out;
  }
  // The router's choice for one layer: sigmoid scores plus the balance bias pick the top n (K by
  // default); the chosen experts' weights are their scores, normalised and scaled.
  route(logits, l, n = this.K) {
    const sig = Array.from(logits, (z) => 1 / (1 + Math.exp(-z))), bias = this.bias[l];
    const order = sig.map((s2, e) => [s2 + bias[e], e]).sort((x, y) => y[0] - x[0] || x[1] - y[1]).slice(0, n).map(([, e]) => e);
    const top = order.slice(0, this.K); let sum = 0; for (const e of top) sum += sig[e];
    return { order, sig, pick: top.map((e) => [e, sig[e] / sum * this.scale]) };
  }
  scorePrediction(pred, actual) {
    const a = new Set(actual); let n = 0; for (const e of pred) if (a.has(e)) n++;
    this.stats.predicted += pred.length; this.stats.predictedRight += n;
  }
  // Parafovea: start reading experts a layer is likely to ask for, into its pool, without waiting.
  // Never evicts an expert this token has used, nor one being read; an ensure() that needs an
  // expert in flight waits for that read instead of starting another.
  prefetch(l, es) {
    const lo = (l - this.dense) * this.perLayer, want = new Set(es.map((e) => l * this.E + e));
    for (const e of es) {
      const key = l * this.E + e;
      if (this.slotOf.has(key) || this.pending.has(key)) continue;
      let s = -1, best = Infinity;
      for (let i = lo; i < lo + this.perLayer; i++) {
        const k = this.keyOf[i];
        if (this.lastUse[i] >= this.tokenClock || this.pendingSlot.has(i) || want.has(k)) continue;
        if (this.lastUse[i] < best) { best = this.lastUse[i]; s = i; }
      }
      if (s < 0) break;  // nothing this token can spare
      if (this.keyOf[s] >= 0) this.slotOf.delete(this.keyOf[s]);
      this.keyOf[s] = key; this.slotOf.set(key, s); this.lastUse[s] = ++this.clock;
      this.pendingSlot.add(s); this.prefetched.add(key); this.stats.prefetched++;
      const p = `model.layers.${l}.mlp.experts.${e}.`;
      const parts = ['gate', 'up', 'down'].flatMap((n) => { const r = this.ct.ranges(p + n + '_proj.weight'); return [r.data, r.scales]; });
      const job = Promise.all(parts.map((r) => this.ct.files.get(r.file).read(r.offset, r.bytes))).then((bufs) => {
        const v = this.slotViews(s);
        bufs.forEach((b, j) => { this.stats.prefetchBytes += b.byteLength; const pad = new Uint8Array(align(b.byteLength, 4)); pad.set(new Uint8Array(b)); this.dev.queue.writeBuffer(v[j].buffer, v[j].offset, pad); });
      }).finally(() => { this.pending.delete(key); this.pendingSlot.delete(s); });
      this.pending.set(key, job);
    }
  }
  // Three tiers, as a fovea: GPU (the experts in use), RAM (recently used), the drive (the rest).
  // Exclusive: an expert evicted from the GPU is copied back out of its slot into RAM (one GPU copy,
  // cheap on unified memory), and one promoted from RAM leaves it. Both tiers are split per layer,
  // so the sweep through 75 layers never evicts the next layer's experts. Misses in both: the drive.
  async ensure(l, es) {
    const out = [], miss = [], tier = []; this.lastTier = tier;
    // wait for this layer's reads still in flight: those it needs, and any holding slots it may need
    const inflight = [...this.pending].filter(([k]) => Math.floor(k / this.E) === l).map(([, p]) => p);
    if (inflight.length) { const tw = performance.now(); await Promise.all(inflight); this.stats.prefetchWaitMs += performance.now() - tw; }
    for (const e of es) {
      const key = l * this.E + e, s = this.slotOf.get(key);
      if (s === undefined) { miss.push([e, key, out.length]); out.push(-1); tier.push(this.ram.has(key) ? 'ram' : 'ssd'); }
      else if (this.prefetched.has(key)) { this.prefetched.delete(key); out.push(s); this.stats.prefetchUsed++; tier.push('pre'); }
      else { out.push(s); this.stats.hits++; tier.push('gpu'); }
    }
    const used = new Set(out.filter((s) => s >= 0));
    for (const s of used) this.lastUse[s] = ++this.clock;
    if (!miss.length) return out;
    // choose the slots, and save the experts leaving them into RAM before they are overwritten
    const lo = (l - this.dense) * this.perLayer, leaving = [];
    for (const m of miss) {
      let s = -1, best = Infinity;
      for (let i = lo; i < lo + this.perLayer; i++) if (!used.has(i) && !this.pendingSlot.has(i) && this.lastUse[i] < best) { best = this.lastUse[i]; s = i; }
      used.add(s); this.lastUse[s] = ++this.clock;
      if (this.keyOf[s] >= 0) { this.slotOf.delete(this.keyOf[s]); this.prefetched.delete(this.keyOf[s]); leaving.push([this.keyOf[s], s]); }
      this.keyOf[s] = m[1]; this.slotOf.set(m[1], s); out[m[2]] = s;
    }
    const saved = this.ramPer > 0 && leaving.length ? this.saveToRam(leaving) : null;  // copy queued before the writes below
    const t0 = performance.now();
    await Promise.all(miss.map(async ([e, key, k]) => {
      const v = this.slotViews(out[k]);
      let bufs = this.ram.get(key);
      if (bufs) {  // from RAM: no drive read
        this.ram.delete(key); this.stats.ramHits++; this.stats.ramBytes += bufs.reduce((t, b) => t + b.byteLength, 0);
      } else {
        const p = `model.layers.${l}.mlp.experts.${e}.`;
        const parts = ['gate', 'up', 'down'].flatMap((n) => { const r = this.ct.ranges(p + n + '_proj.weight'); return [r.data, r.scales]; });
        const tr = performance.now();
        bufs = await Promise.all(parts.map((r) => this.ct.files.get(r.file).read(r.offset, r.bytes)));
        this.stats.misses++; for (const b of bufs) this.stats.readBytes += b.byteLength;
        this.lastReadMs = performance.now() - tr;
      }
      bufs.forEach((b, j) => { const pad = new Uint8Array(align(b.byteLength, 4)); pad.set(new Uint8Array(b)); this.dev.queue.writeBuffer(v[j].buffer, v[j].offset, pad); });
    }));
    this.stats.readMs += performance.now() - t0;
    if (saved) await saved;
    return out;
  }
  // Copy the experts leaving GPU slots into RAM (the RAM pool of their layer, least recently used out).
  async saveToRam(leaving) {
    const n = leaving.length, sz = this.slotBytes;
    if (!this.staging || this.staging.size < n * sz) {
      this.staging?.destroy();
      this.staging = this.dev.createBuffer({ size: Math.max(n, this.perLayer) * sz, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    }
    const st = this.staging, enc = this.dev.createCommandEncoder();
    leaving.forEach(([, s], i) => { const v = this.slotViews(s)[0]; enc.copyBufferToBuffer(v.buffer, v.offset, st, i * sz, sz); });
    this.dev.pushErrorScope('validation');
    this.dev.queue.submit([enc.finish()]);  // ordered before the writeBuffer calls that refill these slots
    const err = this.dev.popErrorScope();
    this.saving = (this.saving || Promise.resolve()).then(async () => {
      const e = await err; if (e) throw new Error('GPU: saving experts to RAM: ' + e.message);
      await st.mapAsync(GPUMapMode.READ, 0, n * sz);
      const all = st.getMappedRange(0, n * sz), sizes = this.slotSizes();
      leaving.forEach(([key], i) => {
        const bufs = this.slotLayout.map((o, j) => all.slice(i * sz + o, i * sz + o + sizes[j]));
        const l = Math.floor(key / this.E);
        const mine = (this.ramLru.get(l) || []).filter((k) => this.ram.has(k));  // promoted ones have left
        mine.push(key); this.ram.set(key, bufs); this.stats.ramSaved++;
        while (mine.length > this.ramPer) this.ram.delete(mine.shift());
        this.ramLru.set(l, mine);
      });
      st.unmap();
    });
    return this.saving;
  }
  ramBytes() { let t = 0; for (const b of this.ram.values()) for (const x of b) t += x.byteLength; return t; }
  async read(buf, n, off = 0) {
    const rd = this.dev.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.dev.createCommandEncoder(); enc.copyBufferToBuffer(buf, off * 4, rd, 0, n * 4); this.dev.queue.submit([enc.finish()]);
    await rd.mapAsync(GPUMapMode.READ); const r = new Float32Array(rd.getMappedRange().slice(0)); rd.unmap(); rd.destroy(); return r;
  }
  // Head-averaged attention of the last token, per layer: [L][T].
  async attention(T) {
    const all = await this.read(this.ATT, this.L * this.maxT), out = [];
    for (let l = 0; l < this.L; l++) out.push(all.slice(l * this.maxT, l * this.maxT + T));
    return out;
  }
  // Logit lens: the final norm and lm_head applied to the residual after layer d (d = 0: embedding).
  // Returns the top k {id, p} and the entropy per depth asked for. One lm_head pass per depth.
  async lens(depths, k = 5) {
    const out = [];
    for (const d of depths) {
      this.begin();
      this.rms(this.vec.get('model.norm.weight'), this.RES, this.LN, this.C, d * this.C, 0);
      this.mv(this.w.get('lm_head.weight'), this.LN, this.LOG);
      await this.submit();
      out.push(topk(await this.read(this.LOG, this.V), k));
    }
    return out;
  }
  // Fill the expert pool with the most-routed experts from a colibri routing history (.coli_usage:
  // lines "layer expert count"; -1/-2 lines are a header), most used first, until the pool is full.
  async prewarm(text, onProgress = () => {}) {
    const rows = [];
    for (const line of text.split('\n')) {
      const [l, e, n] = line.trim().split(/\s+/).map(Number);
      if (l >= this.dense && l < this.L && e >= 0 && e < this.E && n > 0) rows.push([n, l, e]);
    }
    // Each layer has its own pool, so warm each with that layer's own most-routed experts.
    rows.sort((a, b) => b[0] - a[0]);
    const per = new Map(), take = [];
    for (const r of rows) { const n = per.get(r[1]) || 0; if (n < this.perLayer) { per.set(r[1], n + 1); take.push(r); } }
    for (let i = 0; i < take.length; i += 8) {
      const batch = take.slice(i, i + 8);
      for (const [, l, e] of batch) await this.ensure(l, [e]);
      onProgress(Math.min(i + 8, take.length), take.length);
    }
    return take.length;
  }
  async residuals() {
    const n = (this.L + 1) * this.C, rd = this.dev.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.dev.createCommandEncoder(); enc.copyBufferToBuffer(this.RES, 0, rd, 0, n * 4); this.dev.queue.submit([enc.finish()]);
    await rd.mapAsync(GPUMapMode.READ); const r = new Float32Array(rd.getMappedRange().slice(0)); rd.unmap(); rd.destroy(); return r;
  }
}
function topk(logits, k) {
  let m = -Infinity; for (const v of logits) if (v > m) m = v;
  let z = 0; for (const v of logits) z += Math.exp(v - m);
  const lz = Math.log(z), best = [];
  for (let i = 0; i < logits.length; i++) {
    if (best.length < k || logits[i] > best[best.length - 1][1]) {
      best.push([i, logits[i]]); best.sort((a, b) => b[1] - a[1]); if (best.length > k) best.pop();
    }
  }
  let H = 0; for (const v of logits) { const lp = v - m - lz; H -= Math.exp(lp) * lp; }
  return { top: best.map(([id, v]) => ({ id, p: Math.exp(v - m - lz) })), entropy: H };
}
export { Container, topk };
