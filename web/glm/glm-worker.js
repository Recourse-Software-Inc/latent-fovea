// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// Runs GLM-5.2 (web/glm/glm-gpu.js) off the page's thread and streams one record per token.
//   in:  { type: 'open', dir } | { type: 'open', url, names }   (a picked folder, or a URL for tests)
//        { type: 'generate', prompt, max, temperature, lensDepths } | { type: 'stop' } | { type: 'prewarm' }
//   out: { type: 'progress', … } | { type: 'ready', … } | { type: 'token', … } | { type: 'done', … } | { type: 'error', message }
import { Container, GlmGpu } from './glm-gpu.js';
import { HFTokenizer } from '../bpe.js';

let g = null, tok = null, ct = null, stop = false, eos = new Set(), usage = null;
const post = (m) => postMessage(m);

async function open(m) {
  ct = m.dir ? await Container.fromDirectory(m.dir) : await Container.fromUrl(m.url, m.names);
  if (ct.config.model_type !== 'glm_moe_dsa') throw new Error(`This folder holds a ${ct.config.model_type || 'unknown'} model, not GLM-5.2 (glm_moe_dsa).`);
  if (ct.files.has('tokenizer.json')) tok = new HFTokenizer(JSON.parse(await ct.files.get('tokenizer.json').text()));
  const ids = [].concat(ct.config.eos_token_id || []);
  eos = new Set(ids);
  // The DSA indexer's weights are not in colibri's container: attention is dense, which is exact
  // only up to index_topk positions, so the context stops there.
  const maxT = Math.min(m.maxT || 2048, ct.has('model.layers.0.self_attn.indexer.wk.weight') ? 1 << 20 : ct.config.index_topk || 2048);
  g = await GlmGpu.open(ct, { expertBudget: m.expertBudget, ramBudget: m.ramBudget, prefetchAhead: m.prefetchAhead, maxT, onProgress: (p) => post({ type: 'progress', ...p }) });
  usage = ct.files.has('.coli_usage') ? await ct.files.get('.coli_usage').text() : null;
  const c = ct.config;
  post({ type: 'ready', ramPer: g.ramPer, L: g.L, E: g.E, K: g.K, dense: g.dense, H: g.H, C: g.C, V: g.V, maxT, nSlots: g.nSlots, slotBytes: g.slotBytes,
    gpuBytes: g.bytes, hasTokenizer: !!tok, hasUsage: !!usage, params: c.num_hidden_layers, name: 'GLM-5.2' });
}

async function generate(m) {
  stop = false;
  const ids = tok ? tok.encode(m.prompt) : m.ids;
  const max = m.max || 64, depths = m.lensDepths || [];
  g.pos = 0; const t0 = performance.now(); let produced = 0;
  // the question: all of it at once, layer by layer, each expert read once for every token that chose it
  let s0 = { ...g.stats }, ts = performance.now();
  const pf = await g.prefill(ids, 0, (l) => post({ type: 'prefill', layer: l, L: g.L, n: ids.length }));
  let logits = pf.logits, experts = pf.experts.map((byTok) => byTok[ids.length - 1]), scores = pf.scores, pos = ids.length - 1, id = ids[pos];
  for (;;) {
    const T = pos + 1;
    const att = await g.attention(T);
    const lens = depths.length ? (await g.lens(depths)).map((r) => ({ ...r, top: r.top.map((t) => ({ ...t, text: tok ? tok.decode([t.id]) : String(t.id) })) })) : null;
    const next = sample(logits, m.temperature || 0), s1 = { ...g.stats };
    post({ type: 'token', pos, id, text: eos.has(id) ? '' : tok ? tok.decode([id]) : String(id), prompt: pos < ids.length, next,
      nextText: tok && !eos.has(next) ? tok.decode([next]) : null,  // a stop token ends the turn; it isn't shown
      experts, scores: scores.map((x) => (x ? Array.from(x, (v) => +v.toFixed(5)) : null)), att, lens, depths, prefilled: pos === ids.length - 1 ? ids.length : 0,
      ms: performance.now() - ts, hits: s1.hits - s0.hits, ramHits: s1.ramHits - s0.ramHits, preHits: s1.prefetchUsed - s0.prefetchUsed, prefetched: s1.prefetched - s0.prefetched, predicted: s1.predicted - s0.predicted, predictedRight: s1.predictedRight - s0.predictedRight, prefetchBytes: s1.prefetchBytes - s0.prefetchBytes, misses: s1.misses - s0.misses, ramBytes: g.ramBytes(), readBytes: s1.readBytes - s0.readBytes, readMs: s1.readMs - s0.readMs });
    produced++;
    if (stop || eos.has(next) || produced >= max || pos + 1 >= g.maxT) break;
    // Preview: the next few words drafted from what is already on the GPU (missing experts stood in
    // for by their nearest resident neighbour, no drive reads), shown provisionally until the exact
    // tokens arrive. Their cache entries are overwritten by the exact passes for those positions.
    if (m.draftAhead > 0) {
      const td = performance.now(), words = [];
      let x = next;
      for (let k = 0; k < m.draftAhead && pos + 1 + k < g.maxT && !stop; k++) {
        const r = await g.step(x, pos + 1 + k, 'subst');
        const y = sample(r.logits, 0);
        if (eos.has(y)) break;
        words.push(tok ? tok.decode([y]) : String(y)); x = y;
      }
      post({ type: 'drafts', pos: pos + 1, words, ms: performance.now() - td });
    }
    s0 = { ...g.stats }; ts = performance.now(); pos++; id = next;
    ({ logits, experts, scores } = await g.step(id, pos));
  }
  post({ type: 'done', tokens: produced, ms: performance.now() - t0, stats: g.stats });
}

function sample(logits, temp) {
  let am = 0; for (let i = 1; i < logits.length; i++) if (logits[i] > logits[am]) am = i;
  if (!temp) return am;
  let z = 0; const p = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i++) z += (p[i] = Math.exp((logits[i] - logits[am]) / temp));
  let r = Math.random() * z;
  for (let i = 0; i < p.length; i++) if ((r -= p[i]) <= 0) return i;
  return am;
}

let chain = Promise.resolve();
onmessage = ({ data: m }) => {
  if (m.type === 'stop') { stop = true; return; }
  chain = chain.then(async () => {
    try {
      if (m.type === 'open') await open(m);
      else if (m.type === 'generate') await generate(m);
      else if (m.type === 'prewarm') {
        if (!usage) throw new Error('No .coli_usage routing history in this folder.');
        const n = await g.prewarm(usage, (done, total) => post({ type: 'progress', stage: 'prewarm', done, total }));
        for (const k in g.stats) g.stats[k] = 0;
        post({ type: 'prewarmed', n, resident: g.slotOf.size });
      }
    } catch (e) { post({ type: 'error', message: e.message }); }
  });
};
