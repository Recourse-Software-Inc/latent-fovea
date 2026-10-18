// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// Runs GPT-2 off the main thread. Messages in: load, reset, step, gain, nvocab, lens, attr, intervene, pinA, diff,
// sae, saeFeat, hidden.
// Out: progress, ready, captured, gain, nvocab, lens, attr, error.
import { Forward } from './forward.js';
import { MDLM } from './mdlm.js';
import { dataFetch, syncReader, LOCAL } from './store.js';
import { Decoder } from './decoder.js';
import { GpuDecoder } from './gpu-decoder.js';
import { saeFor, loadSae, encode, featureOver, decoderRow } from './sae.js';
import { hiddenSpace } from './hidden.js';
import { traceGraph } from './ngraph.js';

let fwd = null;
let md = null, ds = null;  // masked-diffusion model and its sampler state
const post = (m, t) => self.postMessage(m, t || []);
// Bump when a kernel's exports change: browsers cache .wasm for an hour, and new code must
// never meet an old kernel.
const KERNEL_V = 'nrn1';
let saeId = null;  // the SAE set for this model, if there is one
// Models whose weights are bigger than this are streamed from the browser's storage, one layer at
// a time, instead of held in memory (wasm32 has 4 GB for everything).
const STREAM_OVER = 3.2e9;
// Foveated readouts: every captured position's residual stream at every depth, and its row,
// so any depth of any token can be read out, or attributed, later, when someone looks there.
const resStore = [], rowStore = [];
let ivSpec = {};  // the interventions in force, so a probe can lift and restore them
let resA = null;  // a pinned reference run's residual streams, to compare the current run with
let lastPos = -1;  // the position whose head and MLP outputs the kernel holds right now
let hidAxes = null;  // the hidden-state space's last axes, so a refit keeps its orientation

// MDLM's ddpm_cache sampler. Step i of `steps` goes from t to s = t - dt; each still-masked
// slot unmasks with probability (t - s) / t and draws its token from the model's belief.
// The model is only re-run after something changed (it does not depend on t). The step
// after the last removes the remaining noise: one more pass, then the most likely token.
function diffusionStep(i) {
  const { steps, eps } = ds, dt = (1 - eps) / steps;
  const t = 1 - i * dt, s = Math.max(t - dt, 0);
  const cleanup = i >= steps;
  let snap = null, ms = 0;
  const masked = () => ds.rows.map((r, p) => (r === md.maskRow ? p : -1)).filter((p) => p >= 0);
  if (!ds.valid && masked().length) {
    const t0 = performance.now();
    md.forward(ds.rows);
    ms = performance.now() - t0;
    ds.valid = true; ds.snapId++;
    snap = capture();
  }
  const beliefs = {};
  for (const p of masked()) beliefs[p] = md.belief(p);
  const changes = [];
  for (const p of masked()) {
    const go = cleanup || ds.rng() < (t - s) / t;
    if (!go) continue;
    const row = md.sample(p, ds.rng(), cleanup || ds.greedy);
    ds.rows[p] = row; changes.push([p, row]);
  }
  // Logit lens for the slots that just locked in, while this pass's residuals are still here.
  const lens = {};
  for (const [p] of changes.slice(0, 4)) lens[p] = md.lens(p);
  if (changes.length) ds.valid = false;
  const out = { type: 'dstep', i, t, s, cleanup, snapId: ds.snapId, forwarded: !!snap, snap, beliefs, changes, lens, rows: ds.rows.slice(), ms, left: masked().length };
  return [out, snap ? [snap.att.buffer, snap.headN.buffer, snap.mlp.buffer] : []];
}
function capture() {  // everything the view replays, for every position, quantized
  const { L, H, F } = md, n = ds.rows.length;
  const att = new Uint8Array(L * H * n * n), headN = new Float32Array(n * L * H), mlp = new Int8Array(L * n * F);
  for (let l = 0; l < L; l++) for (let h = 0; h < H; h++) for (let q = 0; q < n; q++) {
    const row = md.attRow(l, h, q), o = ((l * H + h) * n + q) * n;
    for (let k = 0; k < n; k++) att[o + k] = Math.round(row[k] * 255);
  }
  for (let p = 0; p < n; p++) {
    headN.set(md.headNorms(p), p * L * H);
    const a = md.mlp(p);
    for (let l = 0; l < L; l++) for (let i = 0; i < F; i++) mlp[(l * n + p) * F + i] = Math.max(-127, Math.min(127, Math.round(a[l * F + i] * 25)));
  }
  return { n, att, headN, mlp };
}
function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

// On the GPU every call is asynchronous, so messages are handled one at a time: a readout must
// never interleave with a step. (The CPU kernels are synchronous and keep the old behaviour.)
let chain = Promise.resolve();
self.onmessage = (ev) => { if (fwd && fwd.gpu) chain = chain.then(() => handle(ev)); else handle(ev); };
async function handle({ data: m }) {
  try {
    if (m.type === 'load') {
      const base = m.data;
      const meta = await dataFetch(base + 'meta.json').then((r) => r.json());
      const bin = (name) => (/\.(json|txt)$/.test(name) ? name : name + (meta.binary_suffix || ''));
      const one = async (name) => {
        const r = await dataFetch(base + bin(name));
        if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
        return r.arrayBuffer();
      };
      const get = async (name) => {  // large files may be split into parts for the host
        const parts = name === 'vectors.f16' ? meta.vector_parts : meta.parts && meta.parts[name];
        if (!parts) return one(name);
        const bufs = await Promise.all(parts.map((p) => one(p.file)));
        const out = new Uint8Array(bufs.reduce((s, b) => s + b.byteLength, 0));
        let o = 0; for (const b of bufs) { out.set(new Uint8Array(b), o); o += b.byteLength; }
        return out.buffer;
      };
      const [manifest, ids] = await Promise.all([
        dataFetch(base + 'model/manifest.json').then((r) => { if (!r.ok) throw new Error('This data set has no model weights (model/manifest.json).'); return r.json(); }),
        get('ids.u32').then((b) => new Uint32Array(b)),
      ]);
      if (manifest.arch === 'mdlm') {
        const [simd, plain] = await Promise.all([fetch('mdlm.simd.wasm').then((r) => r.arrayBuffer()), fetch('mdlm.wasm').then((r) => r.arrayBuffer())]);
        md = await MDLM.create(simd, plain);
        await md.load(get, manifest, ids, (p, name) => post({ type: 'progress', p, name }));
        post({ type: 'ready', arch: 'mdlm', simd: md.simd, L: md.L, H: md.H, F: md.F, C: md.C, maxT: md.maxP, rowOf: md.rowOf, maskRow: md.maskRow });
        return;
      }
      const general = manifest.kernel === 'dec';  // models built in the browser use the general kernel
      const progress = (p, name) => post({ type: 'progress', p, name });
      // The GPU first, for those models, when the browser has one (WebGPU) and it holds the model;
      // otherwise, or when asked, the CPU kernel exactly as before.
      let gpuNote = '';
      // 16-bit weights first; if the GPU can't hold them, 8-bit (about half the memory); asked
      // for ('gpu8'), 8-bit only.
      if (general && m.backend !== 'cpu' && self.navigator && navigator.gpu) {
        for (const q8 of m.backend === 'gpu8' ? [true] : [false, true]) {
          let g = null;
          try {
            g = await GpuDecoder.create();
            await g.load(get, manifest, ids, (p, name) => progress(p, q8 ? `${name} · 8-bit` : name), { q8 });
            fwd = g; gpuNote = ''; break;
          } catch (e) { if (g) g.close(); fwd = null; gpuNote = `the GPU couldn't run it (${e.message}), so it runs on the CPU`; }
        }
      }
      if (!fwd) {
        const [simd, plain] = await Promise.all([fetch(`${general ? 'dec' : 'gpt2'}.simd.wasm?v=${KERNEL_V}`).then((r) => r.arrayBuffer()), fetch(`${general ? 'dec' : 'gpt2'}.wasm?v=${KERNEL_V}`).then((r) => r.arrayBuffer())]);
        fwd = await (general ? Decoder : Forward).create(simd, plain);
        const stream = general && base.startsWith(LOCAL) && (m.stream || (meta.forward_bytes || 0) > STREAM_OVER);
        const opts = stream ? { openLayer: (name) => syncReader(base.slice(LOCAL.length) + name) } : {};
        await fwd.load(get, manifest, ids, progress, opts);
      }
      saeId = saeFor(general ? manifest.arch : 'gpt2', fwd.L, fwd.C, ids.length);
      post({ type: 'ready', simd: fwd.simd, L: fwd.L, H: fwd.H, F: fwd.F, C: fwd.C, maxT: fwd.maxT, rowOf: fwd.rowOf, streaming: !!fwd.streaming, sae: saeId,
        backend: fwd.gpu ? 'gpu' : 'cpu', gpuName: fwd.gpuName || '', gpuNote, canGpu: general && !!(self.navigator && navigator.gpu),
        q8: !!fwd.q8, qerr: fwd.quantError ? fwd.quantError() : null,
        caps: { links: !!(fwd.links && !fwd.streaming), pre: !!fwd.pre, act: fwd.act || 'gelu_tanh' } });
    } else if (m.type === 'step') {
      const t0 = performance.now();
      const general = !(fwd instanceof Forward);
      await fwd.step(m.row, m.pos, general ? undefined : m.lensMode);
      if (m.pos === 0) { rowStore.length = 0; resStore.length = 0; }  // a new run: older, longer runs' rows and states are gone
      lastPos = m.pos; rowStore[m.pos] = m.row;
      const ms = performance.now() - t0;
      const L = fwd.L, H = fwd.H, n = m.pos + 1;
      const att = new Float32Array(L * H * n);  // this query's attention over keys 0..pos, every head
      for (let l = 0; l < L; l++) for (let h = 0; h < H; h++) att.set(fwd.attRow(l, h, m.pos), (l * H + h) * n);
      const lens = [];
      if (general) {  // lens on demand; big vocabularies get an even spread of depths
        const want = new Set(m.lensMode === 2 ? fwd.lensDepths() : m.lensMode ? [L] : []);
        if (fwd.lensMany) {  // the GPU reads out every depth in one pass
          const ds = [...want].sort((a, b) => a - b), got = await fwd.lensMany(ds);
          for (let d = 0; d <= L; d++) { const k = ds.indexOf(d); lens.push(k >= 0 ? got[k] : null); }
        } else for (let d = 0; d <= L; d++) lens.push(want.has(d) ? await fwd.lens(d) : null);
      } else if (m.lensMode) for (let d = 0; d <= L; d++) lens.push(m.lensMode === 2 || d === L ? fwd.lens(d) : null);
      const headN = fwd.headNorms().slice(), mlp = fwd.mlp().slice();
      // Each neuron's input before its activation function (and SwiGLU's value input): its
      // negative pole and sign regimes. Kept for every token on models small enough; else the
      // page keeps only the latest.
      const preV = fwd.pre ? await fwd.pre() : null, preUV = fwd.preU ? await fwd.preU() : null;
      const pre = preV ? preV.slice() : null, preU = preUV ? preUV.slice() : null;
      const vn = fwd.valueNorms(m.pos) || new Float32Array(0);  // what this token offers each head, for the flow view
      const res = new Float32Array(L + 1);  // residual stream norm at every depth
      const full = new Float32Array((L + 1) * fwd.C);
      for (let d = 0; d <= L; d++) { const r = fwd.residual(d); full.set(r, d * fwd.C); let s = 0; for (let i = 0; i < r.length; i++) s += r[i] * r[i]; res[d] = Math.sqrt(s); }
      resStore[m.pos] = full;
      // How much each layer wrote, split: the MLP's own output, and the rest of the layer's
      // change (attention, with its bias). wr[2l] = attention, wr[2l+1] = MLP.
      const wr = new Float32Array(2 * L).fill(NaN), C = fwd.C;
      for (let l = 0; l < L; l++) {
        const mo = fwd.mlpOut(l); let a = 0, b = 0;
        for (let i = 0; i < C; i++) {
          const dl = full[(l + 1) * C + i] - full[l * C + i];
          if (mo) { b += mo[i] * mo[i]; a += (dl - mo[i]) ** 2; } else a += dl * dl;
        }
        wr[2 * l] = Math.sqrt(a); if (mo) wr[2 * l + 1] = Math.sqrt(b);  // no MLP output: the whole write, unsplit
      }
      // Attribution toward every word the logit lens shows (top 5 at each depth, top 10 at the
      // end), so the view can explain any guess without asking again.
      const want = new Set();
      lens.forEach((lz, d) => { if (lz) lz.rows.slice(0, d === L ? 10 : 5).forEach((r) => want.add(r)); });
      if (m.next != null) want.add(m.next);  // in the prompt, the word that actually comes next
      const targets = Uint32Array.from(want), per = L * H + L + 4, attr = new Float32Array(targets.length * per);
      const many = fwd.attributeMany ? await fwd.attributeMany(Array.from(targets)) : null;  // the GPU does them all at once
      for (let j = 0; j < targets.length; j++) {
        const a = many ? many[j] : await fwd.attribute(targets[j]); if (!a) continue;
        const o = j * per;
        attr.set(a.heads, o); attr.set(a.mlps, o + L * H);
        attr[o + L * H + L] = a.emb; attr[o + L * H + L + 1] = a.rest; attr[o + L * H + L + 2] = a.logit; attr[o + L * H + L + 3] = a.inv;
      }
      // Interventions: the probability now given to the word the unchanged model wrote next.
      const watch = m.watch != null && lens[L] ? { row: m.watch, p: await fwd.probOf(m.watch) } : null;
      post({ type: 'captured', id: m.id, run: m.run, pos: m.pos, row: m.row, ms, att, lens, headN, mlp, pre, preU, res, wr, vn, targets, attr, watch },
        [att.buffer, headN.buffer, mlp.buffer, res.buffer, wr.buffer, vn.buffer, targets.buffer, attr.buffer, ...(pre ? [pre.buffer] : []), ...(preU ? [preU.buffer] : [])]);
    } else if (m.type === 'gain') {  // per-neuron effect of layer l on word row (see neuronGain)
      const gain = fwd && await fwd.neuronGain(m.l, m.row);
      if (gain) post({ type: 'gain', l: m.l, row: m.row, gain }, [gain.buffer]);
    } else if (m.type === 'intervene') {  // knock out or scale heads, MLPs, neurons from the next step on
      if (fwd) fwd.setInterventions(m.spec);
      ivSpec = m.spec || {};
      lastPos = -1;  // the kernel's held outputs are from the old model
    } else if (m.type === 'sae') {  // the features of one token's state at one depth
      const full = resStore[m.pos], C = fwd && fwd.C;
      if (!saeId || !full) return;
      try {
        const sae = await loadSae(saeId, m.depth, (text) => post({ type: 'saeProgress', depth: m.depth, text }));
        const r = encode(sae, full.subarray(m.depth * C, (m.depth + 1) * C));
        post({ type: 'sae', run: m.run, depth: m.depth, pos: m.pos, ...r });
      } catch (e) { post({ type: 'saeProgress', depth: m.depth, text: `Could not load the features: ${e.message}` }); }
    } else if (m.type === 'saeFeat') {  // one feature: its activation at every token, and the words it writes
      const C = fwd && fwd.C;
      if (!saeId) return;
      // this run's states, taken before waiting on the dictionary (a new run may start meanwhile)
      const states = Array.from({ length: m.n }, (_, p) => resStore[p] && resStore[p].subarray(m.depth * C, (m.depth + 1) * C));
      let sae; try { sae = await loadSae(saeId, m.depth, (text) => post({ type: 'saeProgress', depth: m.depth, text })); } catch (e) { return; }
      const acts = featureOver(sae, states, m.j);
      const v = await fwd.vocabOf(decoderRow(sae, m.j));
      post({ type: 'saeFeat', run: m.run, depth: m.depth, j: m.j, acts, ...(v || {}) }, [acts.buffer]);
    } else if (m.type === 'hidden') {  // every captured token's state at every depth, projected to 3-D
      let n = 0; while (n < m.n && resStore[n]) n++;
      if (!fwd || !n) return;
      if (m.fresh) hidAxes = null;
      const t0 = performance.now(), h = hiddenSpace(resStore, n, fwd.L, fwd.C, hidAxes);
      hidAxes = h.axes;
      post({ type: 'hidden', run: m.run, id: m.id, n, xyz: h.xyz, share: h.share, ms: performance.now() - t0 }, [h.xyz.buffer]);
    } else if (m.type === 'pinA') {  // keep this run's residual streams as the reference (later runs store new arrays)
      resA = m.clear ? null : resStore.slice();
    } else if (m.type === 'diff') {  // how far this run's state has moved from the reference's, at every depth
      const a = resA && resA[m.pos], b = resStore[m.pos], C = fwd && fwd.C;
      if (!a || !b || a.length !== b.length) return;
      const d = new Float32Array(a.length / C);
      for (let k = 0; k < d.length; k++) {
        let ab = 0, aa = 0, bb = 0;
        for (let i = k * C; i < (k + 1) * C; i++) { ab += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; }
        d[k] = Math.max(0, 1 - ab / Math.sqrt(aa * bb || 1));  // cosine distance: 0 = the same direction
      }
      post({ type: 'diff', run: m.run, pos: m.pos, d }, [d.buffer]);
    } else if (m.type === 'lens') {  // the logit lens at any depth of any captured token
      const full = resStore[m.pos], C = fwd && fwd.C;
      const lz = full && await fwd.lensOf(full.subarray(m.depth * C, (m.depth + 1) * C));
      if (lz) post({ type: 'lens', run: m.run, pos: m.pos, depth: m.depth, lens: lz });
    } else if (m.type === 'attr') {  // attribution toward any word at any captured token
      if (!fwd || rowStore[m.pos] == null) return;
      if (lastPos !== m.pos) { await fwd.step(rowStore[m.pos], m.pos, 0); lastPos = m.pos; }  // replays that position exactly
      const a = await fwd.attribute(m.row); if (!a) return;
      const L = fwd.L, H = fwd.H, rec = new Float32Array(L * H + L + 4);
      rec.set(a.heads); rec.set(a.mlps, L * H);
      rec[L * H + L] = a.emb; rec[L * H + L + 1] = a.rest; rec[L * H + L + 2] = a.logit; rec[L * H + L + 3] = a.inv;
      post({ type: 'attr', run: m.run, pos: m.pos, row: m.row, rec }, [rec.buffer]);
    } else if (m.type === 'links') {  // into neuron j of layer l at token pos: each earlier part's share
      if (!fwd || rowStore[m.pos] == null) return;
      if (!fwd.links) { post({ type: 'links', run: m.run, l: m.l, j: m.j, pos: m.pos, none: 'Run the model on the CPU to see what feeds a neuron.' }); return; }
      if (lastPos !== m.pos) { await fwd.step(rowStore[m.pos], m.pos, 0); lastPos = m.pos; }
      const lk = fwd.links(m.l, m.j, true);
      if (!lk) { post({ type: 'links', run: m.run, l: m.l, j: m.j, pos: m.pos, none: 'Not available while the weights are streamed.' }); return; }
      post({ type: 'links', run: m.run, l: m.l, j: m.j, pos: m.pos, neurons: lk.neurons, heads: lk.heads, emb: lk.emb, total: lk.total, r: lk.r,
        up: lk.up ? { neurons: lk.up.neurons, heads: lk.up.heads, emb: lk.up.emb, total: lk.up.total } : null, act: fwd.act });
    } else if (m.type === 'readers') {  // which later neurons read neuron i of layer l (weights)
      if (!fwd || !fwd.readers) { post({ type: 'readers', run: m.run, l: m.l, i: m.i, pos: m.pos, none: true }); return; }
      if (rowStore[m.pos] == null) return;
      if (lastPos !== m.pos) { await fwd.step(rowStore[m.pos], m.pos, 0); lastPos = m.pos; }
      const rd = fwd.readers(m.l, m.i), r = fwd.ln2r ? fwd.ln2r() : null, F = fwd.F;
      // weights alone overrate late readers, whose norms divide their input by more: scale each
      // reader by its layer's 1/scale at this token, so readers in different layers compare
      if (rd && r) for (const arr of [rd.into, rd.intoUp]) if (arr) for (let k = 0; k < fwd.L; k++) for (let j = 0; j < F; j++) arr[k * F + j] *= r[k];
      post({ type: 'readers', run: m.run, l: m.l, i: m.i, pos: m.pos, into: rd ? rd.into : null, intoUp: rd ? rd.intoUp : null, scaled: !!r, none: !rd });
    } else if (m.type === 'graph') {  // the neurons behind one word at one token, and how they feed each other
      if (!fwd || rowStore[m.pos] == null) return;
      if (!fwd.links) { post({ type: 'graph', pos: m.pos, row: m.row, runId: m.runId, ivVer: m.ivVer, error: 'Tracing needs the CPU kernel: choose CPU under Run on.' }); return; }
      if (lastPos !== m.pos) { await fwd.step(rowStore[m.pos], m.pos, 2); lastPos = m.pos; }
      const g = await traceGraph(fwd, m.row, { spec: ivSpec, onProgress: (k, n) => post({ type: 'graphProgress', k, n, runId: m.runId }) });
      post({ type: 'graph', pos: m.pos, row: m.row, runId: m.runId, ivVer: m.ivVer, ...g });
    } else if (m.type === 'probe') {  // a neuron on a text of the reader's own: its activation at every token
      if (!fwd || !fwd.pre) return;
      // The probe reads the unchanged model: interventions are lifted while it runs, then restored.
      const F = fwd.F, idx = m.l * F + m.i, acts = [], pres = [], preUs = [], lead = m.lead || 0;
      fwd.setInterventions({});
      for (let p = 0; p < m.rows.length && p < fwd.maxT; p++) {
        await fwd.step(m.rows[p], p, 0);
        if (p < lead) continue;  // a leading <|endoftext|>: the position the model parks attention on
        acts.push(fwd.mlp()[idx]); pres.push(fwd.pre()[idx]); const u = fwd.preU && fwd.preU(); preUs.push(u ? u[idx] : null);
      }
      fwd.setInterventions(ivSpec);
      // put the run on screen back: the cache now holds the probe's tokens (every position this
      // worker has run, even ones the page hasn't heard about yet)
      for (let p = 0; p < rowStore.length; p++) if (rowStore[p] != null) await fwd.step(rowStore[p], p, 0);
      lastPos = rowStore.length - 1;
      post({ type: 'probe', l: m.l, i: m.i, id: m.id, rows: m.rows.slice(lead), acts, pres, preUs, kind: m.kind || 'test', key: m.key });
    } else if (m.type === 'nvocab') {  // the words neuron i of layer l pushes up and down
      const v = fwd && await fwd.neuronVocab(m.l, m.i);
      if (v) post({ type: 'nvocab', l: m.l, i: m.i, ...v });
    } else if (m.type === 'dreset') {
      ds = { rows: m.rows.slice(), steps: m.steps, greedy: m.greedy, rng: mulberry32(m.seed), valid: false, snapId: 0, eps: 1e-5 };
    } else if (m.type === 'dstep') {
      post(...diffusionStep(m.i));
    } else if (m.type === 'dlens') {
      if (ds && m.snapId === ds.snapId) post({ type: 'dlens', p: m.p, snapId: m.snapId, lens: md.lens(m.p) });
    }
  } catch (err) {
    post({ type: 'error', run: m.run, message: err.message, id: m.id });
  }
}
