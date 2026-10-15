// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// Turns a Hugging Face model into a Latent Fovea data set, in the browser: reads the
// config and safetensors headers, streams the tensors it needs one at a time (converted
// to fp16), builds the 3-D atlas of the token embeddings (PCA), and writes everything to
// OPFS in the same layout tools/build_data.py produces. meta.json is written last.
//
// In: { type: 'build', repo, token }.  Out: progress { stage, p, text }, done { base }, error.
import { Hub, halfToFloat } from './hub.js';
import { writeFile, localId, localBase, dataFetch, removeLocal, readBuf, removeFile, fileSize } from './store.js';

const CATEGORIES = ['word', 'subword', 'number', 'punctuation', 'whitespace', 'other'];
const GRID = 16, K = 16;
const post = (m) => self.postMessage(m);

// ---- model families: how each maps onto the general kernel (kernel/dec.c) ----
// Slot order must match dec.c: one layer file is its slots concatenated, fp16, [out][in].
const SLOTS = ['LN1W', 'LN1B', 'WQ', 'BQ', 'WK', 'BK', 'WV', 'BV', 'QN', 'KN', 'WO', 'BO', 'LN2W', 'LN2B', 'W1', 'B1', 'W3', 'W2', 'B2'];
export const FAMILIES = ['gpt2', 'llama', 'qwen2', 'qwen3', 'gpt_neox'];
const MAX_BYTES = 3.4e9;  // wasm32 gives the kernel 4 GB; leave room for caches and the page

function invFreq(theta, rot, scaling) {
  const f = Array.from({ length: rot / 2 }, (_, i) => 1 / theta ** ((2 * i) / rot));
  const type = scaling && (scaling.rope_type || scaling.type);
  if (!type || type === 'default' || type === 'dynamic') return f;  // dynamic only differs past the trained length
  if (type === 'linear') return f.map((x) => x / scaling.factor);
  if (type === 'llama3') {
    const { factor, low_freq_factor: lo, high_freq_factor: hi, original_max_position_embeddings: old } = scaling;
    return f.map((x) => {
      const wl = (2 * Math.PI) / x;
      if (wl < old / hi) return x;
      if (wl > old / lo) return x / factor;
      const s = (old / wl - lo) / (hi - lo);
      return (1 - s) * (x / factor) + s * x;
    });
  }
  throw new Error(`rotary scaling “${type}” is not supported yet`);
}

// Returns { cfg (kernel), names: tensors to fetch, layer(l, get) -> slot arrays, embed, lmHead, misc(get) }.
function family(hf, T) {
  const arch = hf.model_type;
  const has = (n) => T.has(n);
  const pick = (...names) => names.find(has);
  if (arch === 'gpt2') {
    const C = hf.n_embd, NH = hf.n_head, HD = C / NH, F = hf.n_inner || 4 * C;
    const pre = has('h.0.ln_1.weight') ? '' : 'transformer.';
    const act = hf.activation_function || 'gelu_new';
    const cfg = { C, L: hf.n_layer, NH, NKV: NH, HD, F, NCTX: hf.n_positions, NORM: 0, NORM_BIAS: 1, POS: 0, ROT: 0,
      MLP: act === 'gelu' ? 1 : 0, QKV_BIAS: 1, O_BIAS: 1, MLP_BIAS: 1, QK_NORM: 0, PARALLEL: 0, TIED: 1, EPS: hf.layer_norm_epsilon ?? 1e-5 };
    const L = (l, n) => `${pre}h.${l}.${n}`;
    return {
      cfg, embed: pre + 'wte.weight', lmHead: null,
      miscNames: [pre + 'wpe.weight', pre + 'ln_f.weight', pre + 'ln_f.bias'],
      layerNames: (l) => ['ln_1.weight', 'ln_1.bias', 'attn.c_attn.weight', 'attn.c_attn.bias', 'attn.c_proj.weight', 'attn.c_proj.bias',
        'ln_2.weight', 'ln_2.bias', 'mlp.c_fc.weight', 'mlp.c_fc.bias', 'mlp.c_proj.weight', 'mlp.c_proj.bias'].map((n) => L(l, n)),
      layer: (l, g) => {  // GPT-2 stores Conv1D weights as [in][out]
        const qkv = g(L(l, 'attn.c_attn.weight')), qb = g(L(l, 'attn.c_attn.bias'));
        return { LN1W: g(L(l, 'ln_1.weight')), LN1B: g(L(l, 'ln_1.bias')),
          WQ: tcols(qkv, C, 3 * C, 0, C), WK: tcols(qkv, C, 3 * C, C, C), WV: tcols(qkv, C, 3 * C, 2 * C, C),
          BQ: qb.slice(0, C), BK: qb.slice(C, 2 * C), BV: qb.slice(2 * C),
          WO: tcols(g(L(l, 'attn.c_proj.weight')), C, C, 0, C), BO: g(L(l, 'attn.c_proj.bias')),
          LN2W: g(L(l, 'ln_2.weight')), LN2B: g(L(l, 'ln_2.bias')),
          W1: tcols(g(L(l, 'mlp.c_fc.weight')), C, F, 0, F), B1: g(L(l, 'mlp.c_fc.bias')),
          W2: tcols(g(L(l, 'mlp.c_proj.weight')), F, C, 0, C), B2: g(L(l, 'mlp.c_proj.bias')) };
      },
    };
  }
  if (arch === 'llama' || arch === 'qwen2' || arch === 'qwen3') {
    const C = hf.hidden_size, NH = hf.num_attention_heads, NKV = hf.num_key_value_heads || NH, HD = hf.head_dim || C / NH;
    if ((hf.hidden_act || 'silu') !== 'silu') throw new Error(`activation ${hf.hidden_act} is not supported yet`);
    const rp = hf.rope_parameters || hf.rope_scaling || null;
    const theta = hf.rope_theta ?? rp?.rope_theta ?? 10000;
    const bias = arch === 'qwen2' ? 1 : hf.attention_bias ? 1 : 0;
    const cfg = { C, L: hf.num_hidden_layers, NH, NKV, HD, F: hf.intermediate_size, NCTX: 0, NORM: 1, NORM_BIAS: 0, POS: 1, ROT: HD,
      MLP: 2, QKV_BIAS: bias, O_BIAS: arch === 'llama' && hf.attention_bias ? 1 : 0, MLP_BIAS: hf.mlp_bias ? 1 : 0,
      QK_NORM: arch === 'qwen3' ? 1 : 0, PARALLEL: 0, TIED: hf.tie_word_embeddings ?? !has('lm_head.weight') ? 1 : 0,
      EPS: hf.rms_norm_eps ?? 1e-6, INV_FREQ: invFreq(theta, HD, rp && (rp.rope_type || rp.type) ? rp : null) };
    if (cfg.TIED && has('lm_head.weight') && hf.tie_word_embeddings === false) cfg.TIED = 0;
    const L = (l, n) => `model.layers.${l}.${n}`;
    const names = (l) => ['input_layernorm.weight', 'self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight', 'self_attn.o_proj.weight',
      'post_attention_layernorm.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight',
      ...(cfg.QKV_BIAS ? ['self_attn.q_proj.bias', 'self_attn.k_proj.bias', 'self_attn.v_proj.bias'] : []),
      ...(cfg.O_BIAS ? ['self_attn.o_proj.bias'] : []), ...(cfg.QK_NORM ? ['self_attn.q_norm.weight', 'self_attn.k_norm.weight'] : []),
      ...(cfg.MLP_BIAS ? ['mlp.gate_proj.bias', 'mlp.up_proj.bias', 'mlp.down_proj.bias'] : [])].map((n) => L(l, n));
    if (cfg.MLP_BIAS) throw new Error('MLP biases with SwiGLU are not supported yet');
    return {
      cfg, embed: 'model.embed_tokens.weight', lmHead: cfg.TIED ? null : 'lm_head.weight',
      miscNames: ['model.norm.weight'], layerNames: names,
      layer: (l, g) => ({ LN1W: g(L(l, 'input_layernorm.weight')), WQ: g(L(l, 'self_attn.q_proj.weight')), WK: g(L(l, 'self_attn.k_proj.weight')),
        WV: g(L(l, 'self_attn.v_proj.weight')), WO: g(L(l, 'self_attn.o_proj.weight')),
        ...(cfg.QKV_BIAS ? { BQ: g(L(l, 'self_attn.q_proj.bias')), BK: g(L(l, 'self_attn.k_proj.bias')), BV: g(L(l, 'self_attn.v_proj.bias')) } : {}),
        ...(cfg.O_BIAS ? { BO: g(L(l, 'self_attn.o_proj.bias')) } : {}),
        ...(cfg.QK_NORM ? { QN: g(L(l, 'self_attn.q_norm.weight')), KN: g(L(l, 'self_attn.k_norm.weight')) } : {}),
        LN2W: g(L(l, 'post_attention_layernorm.weight')), W1: g(L(l, 'mlp.gate_proj.weight')), W3: g(L(l, 'mlp.up_proj.weight')),
        W2: g(L(l, 'mlp.down_proj.weight')) }),
    };
  }
  if (arch === 'gpt_neox') {
    const C = hf.hidden_size, NH = hf.num_attention_heads, HD = C / NH;
    const rot = Math.floor(HD * (hf.rotary_pct ?? hf.partial_rotary_factor ?? 1));
    const act = hf.hidden_act || 'gelu';
    if (act !== 'gelu' && act !== 'gelu_new') throw new Error(`activation ${act} is not supported yet`);
    const rp = hf.rope_parameters || hf.rope_scaling || null;
    const cfg = { C, L: hf.num_hidden_layers, NH, NKV: NH, HD, F: hf.intermediate_size, NCTX: 0, NORM: 0, NORM_BIAS: 1, POS: 1, ROT: rot,
      MLP: act === 'gelu' ? 1 : 0, QKV_BIAS: 1, O_BIAS: 1, MLP_BIAS: 1, QK_NORM: 0, PARALLEL: hf.use_parallel_residual ?? true ? 1 : 0,
      TIED: hf.tie_word_embeddings ? 1 : 0, EPS: hf.layer_norm_eps ?? 1e-5,
      INV_FREQ: invFreq(hf.rotary_emb_base ?? hf.rope_theta ?? rp?.rope_theta ?? 10000, rot, rp && (rp.rope_type || rp.type) ? rp : null) };
    const L = (l, n) => `gpt_neox.layers.${l}.${n}`;
    // query_key_value interleaves per head: [head][q|k|v][HD] rows.
    const part = (w, which, cols) => {
      const out = new Uint16Array(NH * HD * cols);
      for (let h = 0; h < NH; h++) out.set(w.subarray((h * 3 + which) * HD * cols, (h * 3 + which + 1) * HD * cols), h * HD * cols);
      return out;
    };
    return {
      cfg, embed: 'gpt_neox.embed_in.weight', lmHead: cfg.TIED ? null : 'embed_out.weight',
      miscNames: ['gpt_neox.final_layer_norm.weight', 'gpt_neox.final_layer_norm.bias'],
      layerNames: (l) => ['input_layernorm.weight', 'input_layernorm.bias', 'attention.query_key_value.weight', 'attention.query_key_value.bias',
        'attention.dense.weight', 'attention.dense.bias', 'post_attention_layernorm.weight', 'post_attention_layernorm.bias',
        'mlp.dense_h_to_4h.weight', 'mlp.dense_h_to_4h.bias', 'mlp.dense_4h_to_h.weight', 'mlp.dense_4h_to_h.bias'].map((n) => L(l, n)),
      layer: (l, g) => {
        const w = g(L(l, 'attention.query_key_value.weight')), b = g(L(l, 'attention.query_key_value.bias'));
        return { LN1W: g(L(l, 'input_layernorm.weight')), LN1B: g(L(l, 'input_layernorm.bias')),
          WQ: part(w, 0, C), WK: part(w, 1, C), WV: part(w, 2, C), BQ: part(b, 0, 1), BK: part(b, 1, 1), BV: part(b, 2, 1),
          WO: g(L(l, 'attention.dense.weight')), BO: g(L(l, 'attention.dense.bias')),
          LN2W: g(L(l, 'post_attention_layernorm.weight')), LN2B: g(L(l, 'post_attention_layernorm.bias')),
          W1: g(L(l, 'mlp.dense_h_to_4h.weight')), B1: g(L(l, 'mlp.dense_h_to_4h.bias')),
          W2: g(L(l, 'mlp.dense_4h_to_h.weight')), B2: g(L(l, 'mlp.dense_4h_to_h.bias')) };
      },
    };
  }
  throw new Error(`“${arch}” models are not supported yet. Supported: GPT-2, Llama, Qwen2/3, GPT-NeoX (Pythia).`);
}
// Columns [c0, c0 + n) of a row-major [rows][cols] matrix, transposed to [n][rows].
function tcols(src, rows, cols, c0, n) {
  const out = new Uint16Array(n * rows);
  for (let r = 0; r < rows; r++) { const o = r * cols + c0; for (let c = 0; c < n; c++) out[c * rows + r] = src[o + c]; }
  return out;
}

self.onmessage = async ({ data: m }) => {
  if (m.type !== 'build') return;
  try { post({ type: 'done', base: await build(m.repo, m.token) }); }
  catch (e) { post({ type: 'error', message: e.message }); }
};

async function build(repo, token) {
  const hub = new Hub(repo, { token });
  const dir = localId(repo);
  const say = (stage, p, text) => post({ type: 'progress', stage, p, text });
  say('info', 0, `Reading ${repo} on Hugging Face…`);
  const info = await hub.info();
  if (info.sha) hub.revision = info.sha;  // pin every file to one revision
  // Resume: tensors already saved for this same revision are kept (see below).
  let partial = null;
  try { const r = await dataFetch(`opfs:${dir}/partial.json`); if (r.ok) partial = await r.json(); } catch {}
  if (!partial || partial.sha !== info.sha) {
    await removeLocal(dir).catch(() => {});
    await writeFile(`${dir}/partial.json`, JSON.stringify({ sha: info.sha, started: new Date().toISOString() }));
  }
  const hf = await hub.json('config.json');
  if (hf.quantization_config) throw new Error(`${repo} has compressed (quantized) weights, which this tool can't read yet.`);
  const T = await hub.tensors();
  const fam = family(hf, T);
  const { cfg } = fam;
  const [n, d] = T.get(fam.embed).shape;
  cfg.V = n;
  if (d !== cfg.C) throw new Error(`embedding width ${d} does not match hidden size ${cfg.C}`);
  const params = [...T.values()].reduce((s, t) => s + t.shape.reduce((a, b) => a * b, 1), 0);
  // Held in memory: every weight + KV cache + logits. Too much, and the model is streamed instead:
  // its word tables and one layer in memory, the other layers read from storage as a pass needs them.
  const extra = cfg.L * 256 * cfg.NKV * cfg.HD * 8 + n * 4 + cfg.L * cfg.NH * cfg.C * 4;
  const tables = n * d * (fam.lmHead ? 2 : 1), bytes = params * 2 + extra;
  const streamed = (tables + (params - tables) / cfg.L) * 2 + extra;
  if (bytes > MAX_BYTES && streamed > MAX_BYTES) throw new Error(`${repo} is too large to run in a browser, even with its weights streamed from storage (its word tables and one layer need about ${(streamed / 1e9).toFixed(1)} GB; the limit is about ${(MAX_BYTES / 1e9).toFixed(1)} GB).`);

  // Tokenizer: kept whole for the browser tokenizer; byte-level BPE only for now.
  const tj = await hub.json('tokenizer.json');
  const bpe = tj.model?.type === 'BPE' || (!tj.model?.type && Array.isArray(tj.model?.merges));  // older files omit the type
  if (!bpe || !JSON.stringify(tj.decoder || tj.pre_tokenizer || {}).includes('ByteLevel'))
    throw new Error(`${repo} uses a tokenizer this tool can't read yet. Models in the GPT-2, Llama 3, Qwen, SmolLM and Pythia families work.`);
  const vocab = { ...tj.model.vocab };
  for (const a of tj.added_tokens || []) vocab[a.content] = a.id;

  // Every tensor we need, each saved to its own file as it arrives: a download that is
  // interrupted resumes where it stopped, and memory stays bounded whatever order the
  // weights file is in. Layer files are assembled from them at the end.
  const need = [fam.embed, ...(fam.lmHead ? [fam.lmHead] : []), ...fam.miscNames];
  for (let l = 0; l < cfg.L; l++) need.push(...fam.layerNames(l));
  const tpath = (name) => `${dir}/t/${name}.f16`;
  const elems = (t) => t.shape.reduce((a, b) => a * b, 1);
  const want = new Map();
  let keptBytes = 0;
  for (const name of need) {
    const t = T.get(name);
    if (!t) throw new Error(`tensor ${name} not found in ${repo}`);
    if (partial && (await fileSize(tpath(name))) === elems(t) * 2) { keptBytes += t.end - t.start; continue; }
    want.set(name, t);
  }
  const total = [...want.values()].reduce((s, t) => s + (t.end - t.start), 0);
  let got = 0;
  const resumed = keptBytes ? ` · resumed, ${(keptBytes / 1e6).toFixed(0)} MB already stored` : '';
  const tick = (k) => { got += k; say('download', got / total, `Downloading weights · ${(got / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB${resumed}`); };
  const w = (name, arr) => writeFile(`${dir}/${name}`, arr);
  await hub.readMany(want, (name, f16) => writeFile(tpath(name), f16), tick);

  say('atlas', 0, 'Putting the model together…');
  const read = async (name) => new Uint16Array(await readBuf(tpath(name)));
  const files = Array.from({ length: cfg.L }, (_, l) => `layer${String(l).padStart(2, '0')}.f16`);
  for (let l = 0; l < cfg.L; l++) {
    const got16 = new Map();
    for (const nm of fam.layerNames(l)) got16.set(nm, await read(nm));
    const slots = fam.layer(l, (nm) => got16.get(nm));
    await w('model/' + files[l], concat(SLOTS.map((k) => slots[k]).filter(Boolean)));
    say('atlas', 0, `Putting the model together… layer ${l + 1} of ${cfg.L}`);
  }
  const misc = [];
  for (const nm of fam.miscNames) misc.push(await read(nm));
  await w('model/misc.f16', concat(misc));
  await w('model/tokenizer.json', JSON.stringify(tj));
  const held = new Map([[fam.embed, await read(fam.embed)]]);
  if (fam.lmHead) held.set(fam.lmHead, await read(fam.lmHead));

  // ---- the token embedding and its atlas ----
  const wte = held.get(fam.embed);
  say('atlas', 0, 'Reading the vocabulary…');
  const tokens = decodeVocab(vocab, n);
  const cats = Uint8Array.from(tokens, category);
  say('atlas', 0.2, 'Mapping the vocabulary…');
  const P = pca3(wte, n, d);
  say('atlas', 0.6, 'Preparing the map…');
  const { order, cells, lo, hi } = tiles(P, n);
  const reorder = (src) => { const out = new Uint16Array(n * d); for (let r = 0; r < n; r++) out.set(src.subarray(order[r] * d, (order[r] + 1) * d), r * d); return out; };
  const vecs = reorder(wte), pts = new Float32Array(n * 3), norms = new Float32Array(n);
  for (let r = 0; r < n; r++) {
    const i = order[r];
    pts.set(P.subarray(i * 3, i * 3 + 3), r * 3);
    let s = 0; for (let k = 0; k < d; k++) { const v = halfToFloat(wte[i * d + k]); s += v * v; }
    norms[r] = Math.sqrt(s);
  }
  if (fam.lmHead) await w('model/lmhead.f16', reorder(held.get(fam.lmHead)));
  const catsT = Uint8Array.from(order, (i) => cats[i]);
  const counts = new Array(CATEGORIES.length).fill(0); for (const c of cats) counts[c]++;
  await w('ids.u32', order); await w('cat.u8', catsT); await w('points.f32', pts); await w('norms.f32', norms);
  await w('cells.u32', cells); await w('vectors.f16', vecs);
  await w('tokens.json', JSON.stringify(Array.from(order, (i) => tokens[i])));

  const manifest = { kernel: 'dec', arch: hf.model_type, cfg, n_layer: cfg.L, n_head: cfg.NH, d_model: cfg.C, d_mlp: cfg.F,
    dtype: 'f16', slots: SLOTS, layer_files: files, source: 'huggingface', tokenizer: 'tokenizer.json' };
  await w('model/manifest.json', JSON.stringify(manifest));

  const shown = info.safetensors?.total || params;
  const [author, name] = repo.split('/');
  const meta = {
    schema: 'latent-fovea/1', id: dir,
    model: {
      name: `${name} (${fmtParams(shown)})`, author, license: info.cardData?.license || 'see model card',
      params: fmtParams(shown), family: 'causal-lm', blurb: '', repo, arch: hf.model_type,
      source: `https://huggingface.co/${repo}`, tensor: fam.embed, tensor_label: `token embedding (${fam.embed.split('.').slice(-2, -1)[0]})`,
    },
    n, d, k: K, grid: GRID, cells: GRID ** 3, vector_dtype: 'f16', row_bytes: d * 2,
    projection: { method: 'pca', where: 'browser' }, bounds: [lo, hi],
    categories: CATEGORIES, category_counts: counts,
    knn: 'lazy',  // neighbours are computed on demand, not precomputed
    forward: 'model/manifest.json', forward_bytes: total, built_at: new Date().toISOString(), built_in: 'browser',
  };
  meta.model.revision = info.sha;
  await removeFile(`${dir}/t`).catch(() => {});
  await w('meta.json', JSON.stringify(meta));
  await removeFile(`${dir}/partial.json`).catch(() => {});
  say('done', 1, 'Ready');
  return localBase(repo);
}

const fmtParams = (p) => (p >= 1e9 ? `${(p / 1e9).toFixed(1)}B` : `${Math.round(p / 1e6)}M`);
function concat(arrs) {
  const out = new Uint16Array(arrs.reduce((s, a) => s + a.length, 0));
  let o = 0; for (const a of arrs) { out.set(a, o); o += a.length; }
  return out;
}

// ---- vocabulary ----
function byteDecoder() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice(); let k = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + k); k++; }
  return new Map(cs.map((c, i) => [String.fromCharCode(c), bs[i]]));
}
function decodeVocab(vocab, n) {
  const dec = byteDecoder(), utf8 = new TextDecoder('utf-8', { fatal: false }), out = new Array(n).fill('');
  for (const [tok, i] of Object.entries(vocab)) {
    if (i >= n) continue;
    const bytes = [...tok].map((c) => dec.get(c));
    out[i] = bytes.some((b) => b === undefined) ? tok : utf8.decode(new Uint8Array(bytes));
  }
  return out;
}
// Same rules as build_data.category().
function category(s) {
  if (s === '' || /^\s+$/u.test(s)) return 4;
  const core = s.startsWith(' ') ? s.slice(1) : s;
  const digits = (t) => /^\p{Nd}+$/u.test(t);
  if (digits(core) || (digits(core.replace(/[.,]/g, '')) && /^\p{Nd}/u.test(core))) return 2;
  if (/^[A-Za-z]+$/.test(core)) return s.startsWith(' ') ? 0 : 1;
  if (/^[\x00-\x7f]*$/.test(core) && !/[A-Za-z0-9]/.test(core)) return 3;
  return 5;
}

// ---- 3-D layout: the top three principal components, scaled like build_data ----
function pca3(X16, n, d) {
  const mean = new Float64Array(d);
  for (let i = 0; i < n; i++) for (let k = 0; k < d; k++) mean[k] += halfToFloat(X16[i * d + k]);
  for (let k = 0; k < d; k++) mean[k] /= n;
  // Covariance from an even sample of rows (enough for three directions, and fast).
  const step = Math.max(1, Math.floor(n / 4096)), cov = new Float64Array(d * d), row = new Float64Array(d);
  let m = 0;
  for (let i = 0; i < n; i += step, m++) {
    for (let k = 0; k < d; k++) row[k] = halfToFloat(X16[i * d + k]) - mean[k];
    for (let a = 0; a < d; a++) { const ra = row[a], o = a * d; if (ra === 0) continue; for (let b = a; b < d; b++) cov[o + b] += ra * row[b]; }
  }
  for (let a = 0; a < d; a++) for (let b = a; b < d; b++) cov[b * d + a] = cov[a * d + b] /= m;
  // Subspace iteration for the top 3 eigenvectors.
  let V = Array.from({ length: 3 }, (_, j) => Float64Array.from({ length: d }, (_, k) => Math.sin(1 + k * (j + 1) * 0.7)));
  for (let it = 0; it < 200; it++) {
    V = V.map((v) => { const o = new Float64Array(d); for (let a = 0; a < d; a++) { let s = 0; const r = a * d; for (let b = 0; b < d; b++) s += cov[r + b] * v[b]; o[a] = s; } return o; });
    for (let j = 0; j < 3; j++) {  // Gram-Schmidt
      for (let q = 0; q < j; q++) { let s = 0; for (let k = 0; k < d; k++) s += V[j][k] * V[q][k]; for (let k = 0; k < d; k++) V[j][k] -= s * V[q][k]; }
      let s = 0; for (let k = 0; k < d; k++) s += V[j][k] * V[j][k]; s = Math.sqrt(s) || 1; for (let k = 0; k < d; k++) V[j][k] /= s;
    }
  }
  for (const v of V) { let a = 0; for (let k = 1; k < d; k++) if (Math.abs(v[k]) > Math.abs(v[a])) a = k; if (v[a] < 0) for (let k = 0; k < d; k++) v[k] = -v[k]; }  // stable signs
  const P = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) for (let j = 0; j < 3; j++) {
    let s = 0; const v = V[j];
    for (let k = 0; k < d; k++) s += (halfToFloat(X16[i * d + k]) - mean[k]) * v[k];
    P[i * 3 + j] = s;
  }
  // Centre on the median, scale so 98% of points sit inside radius 1.
  for (let j = 0; j < 3; j++) {
    const col = Float32Array.from({ length: n }, (_, i) => P[i * 3 + j]).sort(), med = col[n >> 1];
    for (let i = 0; i < n; i++) P[i * 3 + j] -= med;
  }
  const r = Float32Array.from({ length: n }, (_, i) => Math.hypot(P[i * 3], P[i * 3 + 1], P[i * 3 + 2])).sort();
  const s = r[Math.floor(0.98 * (n - 1))] || 1;
  for (let i = 0; i < P.length; i++) P[i] /= s;
  return P;
}

// ---- tiles: a 16^3 grid over the layout, cells in Morton order ----
function spread(v) {
  v = (v | (v << 16)) & 0x030000ff; v = (v | (v << 8)) & 0x0300f00f;
  v = (v | (v << 4)) & 0x030c30c3; v = (v | (v << 2)) & 0x09249249;
  return v >>> 0;
}
function tiles(P, n) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) for (let j = 0; j < 3; j++) { lo[j] = Math.min(lo[j], P[i * 3 + j]); hi[j] = Math.max(hi[j], P[i * 3 + j]); }
  const cell = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    const g = [0, 1, 2].map((j) => Math.min(GRID - 1, Math.max(0, Math.floor((P[i * 3 + j] - lo[j]) / (hi[j] - lo[j] + 1e-9) * GRID))));
    cell[i] = (spread(g[0]) | (spread(g[1]) << 1) | (spread(g[2]) << 2)) >>> 0;
  }
  const order = Uint32Array.from({ length: n }, (_, i) => i).sort((a, b) => cell[a] - cell[b] || a - b);
  const cells = new Uint32Array(GRID ** 3 + 1);
  for (let i = 0; i < n; i++) cells[cell[i] + 1]++;
  for (let c = 1; c < cells.length; c++) cells[c] += cells[c - 1];
  return { order, cells, lo, hi };
}
