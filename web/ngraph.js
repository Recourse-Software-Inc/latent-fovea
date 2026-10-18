// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// A neuron-level attribution graph for one word at the most recent position: the model's own MLP
// neurons, how they feed each other, and how much of the word each route carries. No replacement
// model and no error nodes: every number comes from the network itself.
//
// Each neuron's input splits exactly into one share per earlier part (kernel neuron_links). A share
// becomes an edge by passing it through the receiving neuron's activation function, linearised at
// this token (its slope there), and weighting it by how much that neuron moves the word:
//   influence(n) = direct effect per unit activation + sum over readers m of slope(m) x weight(n->m) x influence(m)
//   edge(n -> m) = share of m's input from n x slope(m) x influence(m)        (in units of the word's score)
// Attention heads and the embedding end the walk: they bring in what other tokens contributed.
// The nonlinearity is linearised, so edges are a first-order account; knocking a group out (Rerun)
// is the exact test.

const SQRT2PI = Math.sqrt(2 / Math.PI);
function slope(act, g, u) {  // d activation / d input(s): { g: d/d gate-or-input, u: d/d value }
  if (act === 'swiglu') {
    const s = 1 / (1 + Math.exp(-g)), si = g * s;
    return { g: s * (1 + g * (1 - s)) * u, u: si };
  }
  if (act === 'gelu_erf') {  // x Phi(x): Phi(x) + x phi(x)
    const phi = Math.exp(-0.5 * g * g) / Math.sqrt(2 * Math.PI);
    return { g: 0.5 * (1 + erf(g / Math.SQRT2)) + g * phi, u: 0 };
  }
  const k = SQRT2PI * (g + 0.044715 * g * g * g), t = Math.tanh(k);  // GELU, tanh form
  return { g: 0.5 * (1 + t) + 0.5 * g * (1 - t * t) * SQRT2PI * (1 + 3 * 0.044715 * g * g), u: 0 };
}
function erf(x) {  // Abramowitz-Stegun 7.1.26, 1.5e-7
  const s = Math.sign(x), a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
  return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a));
}

// fwd: a CPU kernel wrapper (Forward or Decoder) that has just run the position to explain.
// row: the word (tile row). Returns { nodes, edges, word score, coverage } or { error }.
export async function traceGraph(fwd, row, { seeds = 8, parents = 4, maxNodes = 36, onProgress, spec = {} } = {}) {
  const L = fwd.L, H = fwd.H, F = fwd.F, act = fwd.act || 'gelu_tanh';
  if (!fwd.links || !fwd.pre) return { error: 'This kernel can\'t trace neurons; run the model on the CPU.' };
  const a = await fwd.attribute(row);
  if (!a) return { error: 'No attribution for this word.' };
  const acts = fwd.mlp().slice(), pre = fwd.pre() ? fwd.pre().slice() : null, preU = fwd.preU ? (fwd.preU() || null) : null;
  if (!pre) return { error: 'This kernel doesn\'t record neuron inputs.' };
  const preUc = preU ? preU.slice() : null;
  const gains = [];
  for (let l = 0; l < L; l++) { const g = await fwd.neuronGain(l, row); if (!g) return { error: 'No neuron gains for this model.' }; gains.push(g); }
  // Interventions in force: a knocked-out or scaled neuron passes on its input times its scale,
  // and a scaled MLP writes its neurons' outputs times its scale.
  const nScale = new Map((spec.neurons || []).map(([l, i, s]) => [`${l}:${i}`, s])), mScale = new Map((spec.mlps || []).map(([l, s]) => [l, s]));
  const ms = (l) => (mScale.has(l) ? mScale.get(l) : 1);
  const direct = (l, i) => acts[l * F + i] * gains[l][i] * a.inv * ms(l);     // direct effect on the word's score
  const perUnit = (l, i) => gains[l][i] * a.inv * ms(l);
  // seeds: the neurons with the largest direct effect, either sign
  const all = [];
  for (let l = 0; l < L; l++) for (let i = 0; i < F; i++) { const d = direct(l, i); if (d) all.push([Math.abs(d), l, i]); }
  all.sort((p, q) => q[0] - p[0]);
  const nodes = new Map(), key = (l, i) => `${l}:${i}`;
  const add = (l, i) => { const k = key(l, i); if (!nodes.has(k)) nodes.set(k, { kind: 'neuron', key0: k, l, i, act: acts[l * F + i], pre: pre[l * F + i], preU: preUc ? preUc[l * F + i] : null, direct: direct(l, i), infl: perUnit(l, i), total: 0, done: false }); return nodes.get(k); };
  for (const [, l, i] of all.slice(0, seeds)) add(l, i);
  const edges = [], terminal = new Map();
  const term = (k, label, l, h) => { if (!terminal.has(k)) terminal.set(k, { kind: k.startsWith('h') ? 'head' : 'emb', key: k, label, l, h, total: 0 }); return terminal.get(k); };
  // Readers come before the neurons they read, so walk from the top layer down: by the time a
  // neuron is expanded, every reader already in the graph has passed its influence back to it.
  let expanded = 0;
  for (;;) {
    let next = null;
    for (const n of nodes.values()) if (!n.done && (!next || n.l > next.l || (n.l === next.l && Math.abs(n.direct) > Math.abs(next.direct)))) next = n;
    if (!next) break;
    next.done = true; expanded++;
    if (onProgress) onProgress(expanded, nodes.size);
    const lk = fwd.links(next.l, next.i, true);
    if (!lk) return { error: 'Tracing isn\'t available while the weights are streamed; reload without streaming.' };
    const sl = slope(act, next.pre, next.preU || 0), ks = nScale.has(next.key0) ? nScale.get(next.key0) : 1;
    sl.g *= ks; sl.u *= ks;
    // shares of this neuron's input from each earlier part, as effects on the word
    const cand = [];
    const push = (share, upShare, src) => { const e = (share * sl.g + (upShare || 0) * sl.u) * next.infl; if (e) cand.push([e, src]); };
    for (let l = 0; l < next.l; l++) for (let i = 0; i < F; i++) {
      const s = lk.neurons[l * F + i], su = lk.up ? lk.up.neurons[l * F + i] : 0;
      if (s || su) push(s, su, { kind: 'neuron', l, i });
    }
    for (let l = 0; l <= next.l; l++) for (let h = 0; h < H; h++) {
      const s = lk.heads[l * H + h], su = lk.up ? lk.up.heads[l * H + h] : 0;
      if (s || su) push(s, su, { kind: 'head', l, h });
    }
    push(lk.emb, lk.up ? lk.up.emb : 0, { kind: 'emb' });
    // the node's total = its direct effect plus what it sends on (already summed into infl x act)
    next.total = next.act * next.infl;
    cand.sort((p, q) => Math.abs(q[0]) - Math.abs(p[0]));
    let taken = 0;
    for (const [e, src] of cand) {
      if (taken >= parents) break;
      if (src.kind === 'neuron') {
        const known = nodes.has(key(src.l, src.i));
        if (!known && nodes.size >= maxNodes) continue;
        const p = add(src.l, src.i);
        // pass influence back: d word / d act(p) through this route = weight x slope x infl(next)
        const per = p.act ? e / p.act : 0;
        p.infl += per;
        edges.push({ from: key(src.l, src.i), to: key(next.l, next.i), w: e });
      } else {
        const t = src.kind === 'head' ? term(`h${src.l}.${src.h}`, `L${src.l} H${src.h}`, src.l, src.h) : term('emb', 'embedding', 0, -1);
        t.total += e;
        edges.push({ from: t.key, to: key(next.l, next.i), w: e });
      }
      taken++;
    }
  }
  const out = [...nodes.values()].map((n) => ({ ...n, key: key(n.l, n.i), key0: undefined, done: undefined }));
  const covered = out.reduce((s, n) => s + n.direct, 0);
  return { nodes: out, heads: [...terminal.values()], edges, score: a.logit, mlpsTotal: Array.from(a.mlps).reduce((s, x) => s + x, 0), covered, act };
}
