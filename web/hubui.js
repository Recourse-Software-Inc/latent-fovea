// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// The model picker: a dropdown under the model button. At the top, the models ready to use (built
// in, or downloaded earlier); below, a scrolling search of Hugging Face for open-weights models
// this tool can load, saying which will work and why not. It checks a model before downloading,
// manages what is stored, and keeps an optional Hugging Face token for gated models (sent only
// to huggingface.co).
import { HF } from './hub.js';
import { listLocal, localId, listStored, removeStored, canExport, exportModel, findExports, importModel } from './store.js';

export const FAMILIES = {
  gpt2: 'GPT-2', llama: 'Llama · SmolLM', qwen2: 'Qwen2 · 2.5', qwen3: 'Qwen3', gpt_neox: 'Pythia · GPT-NeoX',
};
const LIMIT_BYTES = 3.2e9;  // fp16 weights the kernel can hold (wasm32: 4 GB in all); bigger ones stream
const STREAM_LIMIT = 3.3e9;  // what a streamed model may keep in memory: word tables, one layer, context, recordings
const STREAM_GBPS = 2;       // reading weights back from the browser's storage, GB per second (a desktop SSD)
const STREAM_PARAMS = 9e9;   // for model lists, which lack the sizes streamedBytes needs: about where streaming stops fitting
const MS_PER_MPARAM = 0.8;  // measured on one desktop CPU core (0.5B: ~350 ms, 1.5B: ~1.3 s); varies by machine
const DTYPE_BYTES = { F32: 4, F16: 2, BF16: 2, F64: 8, I64: 8, I32: 4, I8: 1, U8: 1, BOOL: 1 };
// Models too big for the CPU path can run on the GPU (WebGPU), where weights live in GPU memory:
// at 16 bits per weight, or at 8 (about 1.06 bytes per weight with the scales) when 16 doesn't
// fit. Browsers don't say how much memory a GPU has, so this is a planning figure for the weights
// (a 16 GB Mac or a 12 GB graphics card); a model that turns out not to fit at 16 bits loads at
// 8, and one that doesn't fit at all runs on the CPU, and the status line says which.
const GPU_BUDGET = 9e9, Q8_BYTES = 1.0625;
let gpuProbe = null;
export function gpuInfo() {  // { name } when this browser can run models on the GPU, else null
  gpuProbe ||= (async () => {
    try {
      const a = navigator.gpu && await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!a) return null;
      const i = a.info || {};
      return { name: [i.vendor, i.architecture || i.description].filter(Boolean).join(' ') || 'GPU' };
    } catch { return null; }
  })();
  return gpuProbe;
}

const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmtN = (p) => (p >= 1e9 ? `${(p / 1e9).toFixed(1)}B` : p >= 1e6 ? `${Math.round(p / 1e6)}M` : `${Math.round(p / 1e3)}K`);
const fmtB = (b) => (b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${Math.round(b / 1e6)} MB`);
const GPU_MS_PER_MPARAM = 0.055;  // measured on an Apple M-series GPU, every readout included: Qwen2.5-3B ~170 ms a token
const speed = (p, gpu) => {
  const ms = gpu ? Math.max(10, (p / 1e6) * GPU_MS_PER_MPARAM)
    : Math.max(20, (p / 1e6) * MS_PER_MPARAM) + (p * 2 > LIMIT_BYTES ? (p * 2) / (STREAM_GBPS * 1e6) : 0);  // + reading the weights each token
  const s = 1000 / ms; return s >= 1.5 ? `~${Math.round(s)} tokens / s` : `~${(1 / s).toFixed(1)} s / token`;
};
// Memory a model needs when its weights are streamed from storage: the word tables (input, and
// output if separate), one layer's weights, the context cache and what the view records.
function streamedBytes(cfg, params) {
  const V = cfg.vocab_size || 0, C = cfg.hidden_size || cfg.n_embd || 0, L = cfg.num_hidden_layers || cfg.n_layer || 1;
  const NH = cfg.num_attention_heads || cfg.n_head || 1, NKV = cfg.num_key_value_heads || NH, HD = cfg.head_dim || C / NH;
  const tables = V * C * ((cfg.tie_word_embeddings ?? cfg.model_type === 'gpt2') ? 1 : 2);
  const layer = Math.max(0, params - tables) / L;
  return (tables + layer) * 2 + L * 256 * NKV * HD * 8 + L * NH * C * 4 + V * 4 + 64e6;
}
const fmtD = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e3 ? `${Math.round(n / 1e3)}K` : String(n));

export function getToken() { try { return localStorage.getItem('lf-hf-token') || ''; } catch { return ''; } }
function setToken(t) { try { if (t) localStorage.setItem('lf-hf-token', t); else localStorage.removeItem('lf-hf-token'); } catch {} }

// Llama 2-style models (Llama 2, CodeLlama, the original LLaMA, TinyLlama, Mistral, …) use a
// SentencePiece tokenizer; their <s> start token gives them away. Byte-level BPE models use <|…|>
// tokens or none. A token may be written as text or as an object ({ "content": "<s>", … }).
const tokText = (t) => (t && typeof t === 'object' ? t.content : t) ?? '';
const sentencePiece = (cfg) => /^<s>$|^<unk>$/.test(tokText(cfg.tokenizer_config?.bos_token) || tokText(cfg.tokenizer_config?.unk_token));

// What the Hub says about a model -> can we load it? { status, label, reason }
export function compat(m, verified = {}, gpu = null) {
  const cfg = m.config || {}, st = m.safetensors;
  const params = st?.total || 0;
  const arch = cfg.model_type;
  if (!FAMILIES[arch]) return { status: 'no', label: 'Unsupported', reason: arch ? `a model type it can't run yet (${arch})` : 'not a standard model repository' };
  if (cfg.quantization_config) return { status: 'no', label: 'Unsupported', reason: 'compressed (quantized) weights, not readable yet' };
  if (sentencePiece(cfg)) return { status: 'no', label: 'Unsupported', reason: 'a tokenizer it can\'t read yet (Llama 2 style)' };
  if (!st) return { status: 'no', label: 'Unsupported', reason: 'no weights in a format it can read' };
  const streams = params * 2 > LIMIT_BYTES;
  const known = cfg.hidden_size && cfg.vocab_size;  // the full config.json (the pre-download check fetches it)
  const onGpu = gpu && streams && params * 2 <= GPU_BUDGET, onGpu8 = gpu && streams && !onGpu && params * Q8_BYTES <= GPU_BUDGET;
  if (!onGpu && !onGpu8 && streams && (known ? streamedBytes(cfg, params) > STREAM_LIMIT : params > STREAM_PARAMS)) return { status: 'big', label: 'Too big', reason: `${fmtN(params)} parameters; even streamed from storage, its word tables and one layer need more memory than a browser gives a page` };
  if (m.gated && !getToken()) return { status: 'token', label: 'Needs HF token', reason: 'gated: accept its licence on Hugging Face and add a token' };
  if (onGpu8) return { status: 'gpu', label: 'GPU, 8-bit', reason: `too big for the CPU path and, at 16 bits, for a typical GPU; with its weights stored as 8-bit integers it needs about ${fmtB(params * Q8_BYTES)} of GPU memory: a close copy of the model (its weights within about 0.6%), much faster than streaming` };
  if (onGpu) return { status: 'gpu', label: 'Runs on your GPU', reason: `too big for the CPU path, but your GPU can hold it (about ${fmtB(params * 2)} of GPU memory): the full model, much faster than streaming` };
  if (verified[m.id]) return { status: 'verified', label: 'Verified', reason: `checked against the reference implementation ${verified[m.id].date || ''}`.trim() };
  if (streams) return { status: 'stream', label: 'Streams', reason: 'too big to hold in memory, so its weights are read from storage one layer at a time: exact, but slower' };
  return { status: 'ok', label: 'Should load', reason: 'a model type it runs, and it fits' };
}
export const downloadBytes = (m) => Object.entries(m.safetensors?.parameters || {}).reduce((s, [t, n]) => s + n * (DTYPE_BYTES[t] || 2), 0);

const EXPAND = ['safetensors', 'config', 'gated', 'downloads', 'likes', 'cardData', 'trendingScore'].map((e) => `expand[]=${e}`).join('&');
// Search results are cached in this browser, so opening the picker doesn't query Hugging Face
// every time. Within CACHE_FRESH the cached list is used as is; after that it is still shown at
// once, and checked in the background: the browser revalidates each request with its ETag, so an
// unchanged list costs a 304 with no body, and the list is redrawn only if something changed.
const CACHE_FRESH = 60 * 60 * 1000, CACHE_KEEP = 12, CACHE_PREFIX = 'lf-hub1:';
// Only what the list, its badges and compat() read; keeps each cached search small.
const slim = (m) => ({
  id: m.id, gated: m.gated, downloads: m.downloads, likes: m.likes, trendingScore: m.trendingScore,
  cardData: m.cardData?.license ? { license: m.cardData.license } : undefined,
  safetensors: m.safetensors ? { total: m.safetensors.total, parameters: m.safetensors.parameters } : undefined,
  config: m.config ? { model_type: m.config.model_type, quantization_config: m.config.quantization_config ? true : undefined,
    tokenizer_config: m.config.tokenizer_config ? { bos_token: m.config.tokenizer_config.bos_token, unk_token: m.config.tokenizer_config.unk_token } : undefined } : undefined,
});
const cacheKey = ({ q, family, sort, all, big }) => CACHE_PREFIX + JSON.stringify([q, family, sort, !!all, !!getToken(), !!big]);
function cacheGet(key) { try { return JSON.parse(localStorage.getItem(key)); } catch { return null; } }
function cachePut(key, models) {
  try {
    localStorage.setItem(key, JSON.stringify({ t: Date.now(), models }));
    const keys = Object.keys(localStorage).filter((k) => k.startsWith(CACHE_PREFIX));  // keep the most recent searches only
    if (keys.length > CACHE_KEEP) keys.map((k) => [cacheGet(k)?.t || 0, k]).sort((a, b) => a[0] - b[0]).slice(0, keys.length - CACHE_KEEP).forEach(([, k]) => localStorage.removeItem(k));
  } catch {}  // storage full or blocked: the list still works, just uncached
}
async function search({ q, family, sort, all, big }) {
  const base = `${HF}/api/models?pipeline_tag=text-generation&sort=${sort}&limit=${all ? 60 : 30}&${EXPAND}` + (q ? `&search=${encodeURIComponent(q)}` : '');
  const headers = getToken() ? { Authorization: `Bearer ${getToken()}` } : {};
  const get = async (url) => { const r = await fetch(url, { headers, cache: 'no-cache' }); if (!r.ok) throw new Error(`Hugging Face API: HTTP ${r.status}`); return (await r.json()).map(slim); };
  if (all) return get(base);
  const fams = family ? [family] : Object.keys(FAMILIES);
  const lists = await Promise.all(fams.map((f) => get(`${base}&filter=${f}&num_parameters=min:0,max:${big ? '9B' : '2B'}`)));  // with a GPU (8-bit if need be), bigger models are in reach
  const seen = new Set(), out = [];
  for (const m of lists.flat()) if (!seen.has(m.id)) { seen.add(m.id); out.push(m); }
  const key = sort === 'likes' ? 'likes' : sort === 'trendingScore' ? 'trendingScore' : 'downloads';
  return out.sort((a, b) => (b[key] || 0) - (a[key] || 0));
}

const KIND = { 'causal-lm': 'writes word by word', 'masked-diffusion': 'diffusion: fills in the whole answer', kdstore: 'attention without a decoder' };

export class ModelBrowser {
  // builtins: the registry's models; onPick(entry): open one of them; onLoad(repo): open a Hugging
  // Face model (downloading it first if needed); anchor: the button the dropdown hangs from.
  constructor({ onLoad, onPick, currentId, currentRepo, builtins = [], anchor = null }) {
    Object.assign(this, { onLoad, onPick, currentId, currentRepo, builtins, anchor });
    this.dlg = document.getElementById('hubDlg');
    this.state = { q: '', family: '', sort: 'downloads', all: false, tab: 'find' };
    this.verified = {};
    fetch('verified.json').then((r) => (r.ok ? r.json() : {})).then((v) => { this.verified = v.models || {}; }).catch(() => {});
    gpuInfo().then((g) => { this.gpu = g; this.state.big = !!g; if (g) { this.loaded = false; if (this.dlg.open) this.refresh(); } });
    this.bind();
  }
  $(sel) { return this.dlg.querySelector(sel); }
  open(tab = 'find') {
    this.place();
    this.setTab(tab);
    if (!this.dlg.open) this.dlg.showModal();
    this.anchor?.setAttribute('aria-expanded', 'true');
    if (tab === 'find') this.renderPicks();
    if (tab === 'find' && !this.loaded) this.refresh();
  }
  toggle() { if (this.dlg.open) this.dlg.close(); else this.open('find'); }
  // Hang the dropdown from the button, kept on screen.
  place() {
    if (!this.anchor) return;
    const r = this.anchor.getBoundingClientRect(), w = Math.min(620, innerWidth - 24);
    this.dlg.style.setProperty('--drop-top', `${Math.round(r.bottom + 8)}px`);
    this.dlg.style.setProperty('--drop-left', `${Math.round(Math.max(12, Math.min(r.left, innerWidth - w - 12)))}px`);
  }
  // The models ready to use: built in, then downloaded to this browser. The one open is marked.
  async renderPicks() {
    const box = this.$('#hubPicks');
    const local = await listLocal().catch(() => []);
    const row = (attrs, name, sub, meta, badge, cls) => `<li><button type="button" class="hub-row" ${attrs}>
      <span class="hub-name"><b>${esc(name)}</b><span>${esc(sub)}</span></span>
      <span class="hub-meta">${esc(meta)}</span>
      <span class="hub-badge ${cls}">${badge}</span></button></li>`;
    const cur = (on) => (on ? ['✓ Open now', 'current'] : null);
    const head = this.$('#hubStoreLine');
    if (head) listStored().then((all) => {
      const b = all.reduce((t, m) => t + m.bytes, 0);
      head.innerHTML = b ? `${fmtB(b)} stored in this browser · <button type="button" class="linkish" data-tab-go="stored">Manage storage</button>` : '';
    }).catch(() => {});
    const isStore = (m) => m.kind === 'kdstore' || m.family === 'kdstore';
    const isExperts = (m) => m.family === 'experts';
    const experts = this.builtins.filter(isExperts);
    const builtin = (m) => {
      const [b, c] = cur(m.id === this.currentId) || (m.url ? ['Own page ↗', 'builtin'] : ['Built in', 'builtin']);
      return row(`data-pick="${esc(m.id)}"`, m.name.replace(/\s*\(.*\)$/, ''), m.author || '',
        [m.params && `${m.params} params`, KIND[m.family] || '', m.license].filter(Boolean).join(' · '), b, c);
    };
    // KnowDrive stores: retrieval stores read as one attention layer. Opening one loads its map
    // (open_bytes); the vectors stream from the site as you look, like a model's vocabulary.
    const store = (m) => {
      const [b, c] = cur(m.id === this.currentId) || (m.url ? ['Own page ↗', 'builtin'] : ['View', 'stream']);
      const size = m.bytes ? `${fmtB(m.open_bytes || 0)} to open · ${fmtB(m.bytes)} in all, streamed as you look` : 'streamed as you look';
      return row(`data-pick="${esc(m.id)}"`, m.name, [m.author || 'KnowDrive', m.host].filter(Boolean).join(' · '),
        [m.atoms ? `${m.atoms.toLocaleString()} atoms${m.files ? ` from ${m.files.toLocaleString()} files` : ''}` : m.params, size, m.license].filter(Boolean).join(' · '), b, c);
    };
    const stores = this.builtins.filter(isStore);
    box.innerHTML = this.builtins.filter((m) => !isStore(m) && !isExperts(m)).map(builtin).join('') + local.map((m) => {
      const [b, c] = cur(m.repo === this.currentRepo) || ['Downloaded', 'ok'];
      return row(`data-repo="${esc(m.repo)}"`, m.name, m.repo.split('/')[0], `${fmtB(m.bytes)} in this browser · loads without downloading`, b, c);
    }).join('') + `<li class="hub-sec hub-sec-kd">Streamed from your drive <span>models far larger than memory: experts read from an SSD as they are needed</span></li>`
      + `<li><a class="hub-row" href="glm.html"><span class="hub-name"><b>GLM-5.2</b><span>Z.ai · via colibri's container</span></span>`
      + `<span class="hub-meta">744B params, 40B per token · mixture of 19,200 experts · ~419 GB on an external SSD · MIT</span>`
      + `<span class="hub-badge stream">Open ↗</span></a></li>`
      + experts.map((m) => row(`data-pick="${esc(m.id)}"`, m.name, 'the routing, as a space', `${m.params} · each placed by its row of the router · lights up as GLM-5.2 runs`, ...(cur(m.id === this.currentId) || ['View', 'stream']))).join('')
      + (stores.length ? `<li class="hub-sec hub-sec-kd">KnowDrive stores <span>retrieval as attention: the keys a question is scored against</span></li>` + stores.map(store).join('') : '');
  }
  bind() {
    const d = this.dlg;
    d.addEventListener('click', (e) => { if (e.target === d) d.close(); });  // click on the backdrop
    d.addEventListener('close', () => this.anchor?.setAttribute('aria-expanded', 'false'));
    addEventListener('resize', () => { if (d.open) this.place(); });
    this.$('#hubPicks').addEventListener('click', (e) => {
      const p = e.target.closest('[data-pick]'), r = e.target.closest('[data-repo]');
      if (p) { d.close(); if (p.dataset.pick !== this.currentId) this.onPick(p.dataset.pick); }
      if (r) { d.close(); if (r.dataset.repo !== this.currentRepo) this.onLoad(r.dataset.repo); }
    });
    this.$('#hubClose').addEventListener('click', () => d.close());
    d.querySelectorAll('[data-tab]').forEach((b) => b.addEventListener('click', () => this.setTab(b.dataset.tab)));
    d.addEventListener('click', (e) => { const g = e.target.closest('[data-tab-go]'); if (g) this.setTab(g.dataset.tabGo); });
    let t = 0;
    this.$('#hubQ').addEventListener('input', (e) => { this.state.q = e.target.value.trim(); clearTimeout(t); t = setTimeout(() => this.refresh(), 350); });
    this.$('#hubQ').addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && /^[\w.-]+\/[\w.-]+$/.test(this.state.q)) { e.preventDefault(); this.preflight(this.state.q); }  // a repo id: open it directly
    });
    this.$('#hubSort').addEventListener('change', (e) => { this.state.sort = e.target.value; this.refresh(); });
    this.$('#hubAll').addEventListener('change', (e) => { this.state.all = e.target.checked; this.refresh(); });
    this.$('#hubChips').innerHTML = ['', ...Object.keys(FAMILIES)].map((f) => `<button type="button" data-f="${f}" aria-pressed="${f === ''}">${f ? FAMILIES[f] : 'All supported'}</button>`).join('');
    this.$('#hubChips').addEventListener('click', (e) => {
      const b = e.target.closest('[data-f]'); if (!b) return;
      this.state.family = b.dataset.f;
      this.$('#hubChips').querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', x === b));
      this.refresh();
    });
    this.$('#hubList').addEventListener('click', (e) => { const r = e.target.closest('[data-id]'); if (r) this.preflight(r.dataset.id); });
    this.$('#hubBack').addEventListener('click', () => this.setTab('find'));
    this.$('#hubTokenSave').addEventListener('click', () => {
      const v = this.$('#hubToken').value.trim();
      if (v && !/^hf_[A-Za-z0-9]{20,}$/.test(v)) { this.$('#hubTokenMsg').textContent = 'That does not look like a Hugging Face token (they start with hf_).'; return; }
      setToken(v); this.$('#hubToken').value = '';
      this.syncToken(); this.loaded = false;
    });
    this.$('#hubTokenClear').addEventListener('click', () => { setToken(''); this.syncToken(); this.loaded = false; });
    this.$('#hubStored').addEventListener('click', async (e) => {
      const del = e.target.closest('[data-del]'), use = e.target.closest('[data-use]');
      if (use) { d.close(); this.onLoad(use.dataset.use); }
      const ex = e.target.closest('[data-export]');
      if (ex) { this.exportPick(ex.dataset.export, ex.dataset.name, ex); return; }
      if (del && confirm(`Delete ${del.dataset.name} (${fmtB(+del.dataset.bytes)}) from this browser? It can be downloaded again later.`)) { del.disabled = true; del.textContent = 'Deleting…'; await removeStored(del.dataset.del).catch(() => {}); this.renderStored(); }
    });
  }
  setTab(tab) {
    this.state.tab = tab;
    for (const t of ['find', 'check', 'stored', 'token']) this.$(`#hubTab-${t}`).hidden = t !== tab;
    this.dlg.querySelectorAll('[data-tab]').forEach((b) => b.setAttribute('aria-selected', b.dataset.tab === tab || (tab === 'check' && b.dataset.tab === 'find')));
    if (tab === 'stored') this.renderStored();
    if (tab === 'token') this.syncToken();
  }
  syncToken() {
    const has = !!getToken();
    this.$('#hubTokenState').textContent = has ? `A token is saved in this browser (hf_…${getToken().slice(-4)}).` : 'No token saved. Public models need none.';
    this.$('#hubTokenClear').hidden = !has;
    this.$('#hubTokenMsg').textContent = '';
  }

  // The list for the current search: from this browser's cache when there is one, checked
  // against Hugging Face in the background once it is over an hour old.
  async refresh() {
    const my = (this.seq = (this.seq || 0) + 1), key = cacheKey(this.state), hit = cacheGet(key);
    if (hit?.models) { this.renderList(hit.models, hit.t); this.loaded = true; }
    if (hit && Date.now() - hit.t < CACHE_FRESH) return;
    if (!hit) this.$('#hubNote').textContent = 'Searching Hugging Face…';
    try {
      const models = await search(this.state);
      if (my !== this.seq) return;
      cachePut(key, models);
      if (!hit || JSON.stringify(models) !== JSON.stringify(hit.models)) this.renderList(models, Date.now());
      else this.noteFor(this.rowCount, Date.now());  // unchanged: just say it was checked
      this.loaded = true;
    } catch (e) { if (my === this.seq && !hit) this.$('#hubNote').textContent = e.message; }
  }
  noteFor(n, t) {
    const age = Date.now() - t, ago = age < 90e3 ? 'just now' : age < 90 * 60e3 ? `${Math.round(age / 60e3)} min ago` : `${Math.round(age / 3600e3)} h ago`;
    this.$('#hubNote').textContent = n ? `${n} models${this.state.all ? ', including ones this tool cannot run yet' : ' this tool can run'} · checked with Hugging Face ${ago}` : 'No matching models.';
  }
  renderList(models, t) {
    const list = this.$('#hubList');
    this.lastList = [models, t];
    const rows = models.map((m) => ({ m, c: compat(m, this.verified, this.gpu) }))
      .filter(({ m, c }) => this.state.all || ((m.safetensors?.total || 0) > 5e6 && c.status !== 'no'));  // hide tiny test repos
    this.rowCount = rows.length; this.noteFor(rows.length, t);
    list.innerHTML = rows.map(({ m, c }) => {
      const [owner, name] = m.id.split('/');
      const p = m.safetensors?.total;
      return `<li><button type="button" class="hub-row" data-id="${esc(m.id)}">
        <span class="hub-name"><b>${esc(name)}</b><span>${esc(owner)}</span></span>
        <span class="hub-meta">${p ? `${fmtN(p)} params · ${fmtB(downloadBytes(m))}` : '–'}${m.cardData?.license ? ` · ${esc(m.cardData.license)}` : ''} · ${fmtD(m.downloads || 0)} downloads</span>
        <span class="hub-badge ${c.status}" title="${esc(c.reason)}">${c.label}</span>
        ${c.status === 'no' || c.status === 'big' ? `<span class="hub-why">${esc(c.reason)}</span>` : ''}
      </button></li>`;
    }).join('');
  }

  // Before downloading: everything we can check without fetching the weights.
  async preflight(repo) {
    this.setTab('check');
    const box = this.$('#hubCheck');
    box.innerHTML = `<h3>${esc(repo)}</h3><p class="note">Checking…</p>`;
    const headers = getToken() ? { Authorization: `Bearer ${getToken()}` } : {};
    try {
      const r = await fetch(`${HF}/api/models/${repo}?${EXPAND}&expand[]=siblings&expand[]=sha`, { headers });
      if (!r.ok) throw new Error(r.status === 404 || r.status === 401 ? `No public model called ${repo}.` : `Hugging Face API: HTTP ${r.status}`);
      const m = await r.json();
      try {  // the API's config is a summary; the sizes the memory check needs are in config.json
        const full = await fetch(`${HF}/${repo}/resolve/main/config.json`, { headers });
        if (full.ok) m.config = { ...(m.config || {}), ...(await full.json()) };
      } catch {}
      const c = compat(m, this.verified, this.gpu), cfg = m.config || {};
      const files = (m.siblings || []).map((s) => s.rfilename);
      const checks = [];
      const add = (ok, text) => checks.push(`<li class="${ok ? 'ok' : 'bad'}">${ok ? '✓' : '✕'} ${text}</li>`);
      add(!!FAMILIES[cfg.model_type], FAMILIES[cfg.model_type] ? `Model family: ${FAMILIES[cfg.model_type]}` : `Model type ${esc(cfg.model_type || 'unknown')}: not supported yet`);
      add(files.some((f) => f === 'model.safetensors' || f === 'model.safetensors.index.json'), 'Weights in a format it can read');
      add(!cfg.quantization_config, cfg.quantization_config ? 'Compressed (quantized) weights: not readable yet' : 'Full, uncompressed weights');
      add(files.includes('tokenizer.json') && !sentencePiece(cfg), !files.includes('tokenizer.json') ? 'No tokenizer it can read'
        : sentencePiece(cfg) ? 'A tokenizer it can\'t read yet (Llama 2 style)' : 'A tokenizer it can read');
      const p = m.safetensors?.total || 0, need = streamedBytes(cfg, p);
      if (p * 2 <= LIMIT_BYTES) add(true, `Fits in memory (about ${fmtB(p * 2)} of ${fmtB(LIMIT_BYTES)})`);
      else if (this.gpu && p * 2 <= GPU_BUDGET) add(true, `Runs on your GPU (${esc(this.gpu.name)}): about ${fmtB(p * 2)} of GPU memory. If your GPU has less, it loads with 8-bit weights (about ${fmtB(p * Q8_BYTES)}), and failing that streams from storage on the CPU`);
      else if (this.gpu && p * Q8_BYTES <= GPU_BUDGET) add(true, `Runs on your GPU (${esc(this.gpu.name)}) with 8-bit weights: about ${fmtB(p * Q8_BYTES)} of GPU memory. The 8-bit weights are within about 0.6% of the published ones, so predictions move slightly; the status line gives the exact figure. The download is still the full 16-bit model`);
      else add(need <= STREAM_LIMIT, need <= STREAM_LIMIT
        ? `Too big to hold in memory (${fmtB(p * 2)}), so it streams: its weights stay in storage and are read one layer at a time, about ${fmtB(need)} in memory. Exact, but each word takes longer`
        : `Too large even to stream: its word tables and one layer need about ${fmtB(need)}; a browser page gets about ${fmtB(STREAM_LIMIT)}`);
      let est = null; try { est = await navigator.storage.estimate(); } catch {}
      const free = est?.quota ? est.quota - (est.usage || 0) : Infinity, dl = downloadBytes(m);
      if (dl > 1e9) add(dl * 1.1 < free, dl * 1.1 < free ? `Room in this browser's storage (${fmtB(dl)} needed)` : `Not enough room in this browser's storage (${fmtB(dl)} needed, ${fmtB(free)} free)`);
      if (m.gated) add(!!getToken(), getToken() ? 'Gated: will use your saved token (you must have accepted the licence)' : 'Gated: needs a Hugging Face token');
      const can = checks.every((x) => x.startsWith('<li class="ok"'));
      const stored = (await listLocal().catch(() => [])).some((s) => s.id === localId(repo));
      box.innerHTML = `
        <div class="hub-check-head"><h3>${esc(repo)}</h3><span class="hub-badge ${c.status}">${c.label}</span></div>
        <dl class="hub-facts">
          <div><dt>Parameters</dt><dd>${p ? fmtN(p) : '–'}</dd></div>
          <div><dt>Download</dt><dd>${stored ? 'already stored' : fmtB(downloadBytes(m))}</dd></div>
          <div><dt>Speed</dt><dd>${speed(p, this.gpu && p * Q8_BYTES <= GPU_BUDGET)}${this.gpu && p * Q8_BYTES <= GPU_BUDGET ? ' on GPU' : ''}</dd></div>
          <div><dt>Licence</dt><dd>${esc(m.cardData?.license || 'see model card')}</dd></div>
        </dl>
        <ul class="hub-checks">${checks.join('')}</ul>
        <p class="note">The model comes straight from Hugging Face and is kept in this browser; nothing passes through our servers. Speed is for a recent desktop; phones are slower.</p>
        <div class="hub-actions">
          <a class="hub-link" href="${HF}/${esc(repo)}" target="_blank" rel="noopener">Model card ↗</a>
          ${m.gated && !getToken() ? '<button type="button" id="hubGoToken">Add a token</button>' : ''}
          <button type="button" class="primary" id="hubGo" ${can ? '' : 'disabled'}>${stored ? 'Open' : `Download ${p ? fmtB(downloadBytes(m)) : ''} and open`}</button>
        </div>`;
      this.$('#hubGo')?.addEventListener('click', () => { this.dlg.close(); this.onLoad(repo); });
      this.$('#hubGoToken')?.addEventListener('click', () => this.setTab('token'));
    } catch (e) { box.innerHTML = `<h3>${esc(repo)}</h3><p class="note">${esc(e.message)}</p>`; }
  }

  // Copying a model to or from a folder: progress, then the result, in the Storage tab.
  xfer(text, done, total) {
    const el = this.$('#hubXfer'); if (!el) return;
    el.hidden = false;
    el.innerHTML = `<span>${text}</span>${total ? `<span class="st-bar"><i style="width:${Math.min(100, (done / total) * 100).toFixed(1)}%;background:var(--cyan)"></i></span>` : ''}`;
  }
  busy(on) { this.dlg.querySelectorAll('#hubTab-stored button').forEach((b) => { if (on) { b.dataset.wasDisabled = b.disabled; b.disabled = true; } else b.disabled = b.dataset.wasDisabled === 'true'; }); }
  // The Export button doubles as the progress indicator: it reads "Exporting… 23%" while the copy
  // runs, and the panel above gives bytes, file and speed. Only a cancelled folder choice is silent.
  async exportPick(id, name, btn) {
    const label = (t) => { if (btn) btn.textContent = t; };
    this.xfer(`Choose a folder for ${esc(name)}. A folder called “${esc(id)}” will be made inside it.`);
    let dir;
    try { dir = await window.showDirectoryPicker({ id: 'lf-models', mode: 'readwrite', startIn: 'downloads' }); }
    catch (e) {
      if (e && e.name === 'AbortError') { this.$('#hubXfer').hidden = true; return; }  // the user closed the folder window
      try { dir = await window.showDirectoryPicker({ mode: 'readwrite' }); }  // some browsers refuse the extra options
      catch (e2) {
        if (e2 && e2.name === 'AbortError') { this.$('#hubXfer').hidden = true; return; }
        this.xfer(`This browser would not let the page write to a folder (${esc(e2 && e2.message || String(e2))}). Export works in Chrome and Edge; try the site there.`);
        return;
      }
    }
    this.busy(true); label('Exporting… 0%');
    this.xfer(`Preparing to export ${esc(name)} to “${esc(dir.name)}”…`, 0, 1);
    const t0 = performance.now();
    let last = 0;
    try {
      const folder = await exportModel(id, dir, (d, t, f) => {
        const now = performance.now(); if (now - last < 150 && d < t) return; last = now;  // redraw a few times a second
        const pct = t ? Math.floor((d / t) * 100) : 0, secs = (now - t0) / 1000, rate = secs > 1 ? d / secs : 0;
        label(`Exporting… ${pct}%`);
        this.xfer(`Exporting ${esc(name)} to “${esc(dir.name)}/${esc(id)}” · <b>${pct}%</b> · ${fmtB(d)} of ${fmtB(t)}${rate ? ` · ${Math.round(rate / 1e6)} MB/s · about ${Math.max(1, Math.round((t - d) / rate))} s left` : ''}${f ? `<br><span class="note">${esc(f)}</span>` : ''}`, d, t);
      });
      this.busy(false); label('Export');
      this.xfer(`✓ Exported ${esc(name)} to “${esc(dir.name)}/${esc(folder)}” in ${((performance.now() - t0) / 1000).toFixed(0)} s. To use it elsewhere, open the model menu there, choose Storage, then Import model from a folder, and pick that folder.`);
    } catch (e) { this.busy(false); label('Export'); this.xfer(`Could not export ${esc(name)}: ${esc(e.message)}`); }
  }
  // Choose a folder holding one exported model, or several, and copy them in.
  async importPick() {
    let pick;
    if (canExport()) {
      this.xfer('Choose the folder an export made, or a folder holding several.');
      try { pick = await window.showDirectoryPicker({ id: 'lf-models', mode: 'read' }); }
      catch (e) {
        if (e && e.name === 'AbortError') { this.$('#hubXfer').hidden = true; return; }
        try { pick = await window.showDirectoryPicker(); }
        catch (e2) { if (e2 && e2.name === 'AbortError') { this.$('#hubXfer').hidden = true; return; } this.xfer(`This browser would not open a folder (${esc(e2 && e2.message || String(e2))}).`); return; }
      }
    } else {
      pick = await new Promise((res) => {
        const inp = document.createElement('input'); inp.type = 'file'; inp.webkitdirectory = true;
        inp.addEventListener('change', () => res(inp.files)); inp.addEventListener('cancel', () => res(null)); inp.click();
      });
      if (!pick || !pick.length) return;
    }
    this.busy(true); this.xfer('Looking for exported models…');
    try {
      const found = await findExports(pick);
      if (!found.length) { this.busy(false); this.xfer('No exported model in that folder. Pick the folder an export made (it holds meta.json), or a folder of several.'); return; }
      const need = found.reduce((t, x) => t + x.bytes, 0);
      let est = null; try { est = await navigator.storage.estimate(); } catch {}
      if (est?.quota && need > est.quota - (est.usage || 0)) { this.busy(false); this.xfer(`Not enough room: ${fmtB(need)} needed, ${fmtB(est.quota - (est.usage || 0))} free for this site. Free up space first.`); return; }
      const stored = new Set((await listLocal().catch(() => [])).map((m) => m.repo));
      const list = found.map((x) => `${x.name} (${fmtB(x.bytes)})${stored.has(x.repo) ? ', replacing the copy here' : ''}`).join('\n');
      if (!confirm(`Import ${found.length > 1 ? `${found.length} models` : 'this model'} into this browser?\n\n${list}`)) { this.busy(false); this.xfer(''); this.$('#hubXfer').hidden = true; return; }
      const t0 = performance.now(); let base = 0;
      for (const x of found) {
        await importModel(x, (d, t, f) => this.xfer(`Importing ${esc(x.name)} · ${fmtB(base + d)} of ${fmtB(need)}${f ? ` · ${esc(f)}` : ''}`, base + d, need));
        base += x.bytes;
      }
      this.busy(false); await this.renderStored();
      this.xfer(`Imported ${found.map((x) => esc(x.name)).join(', ')} (${fmtB(need)}) in ${((performance.now() - t0) / 1000).toFixed(0)} s, checked file by file. Open ${found.length > 1 ? 'them' : 'it'} from the Choose tab, or with Open below.`);
    } catch (e) { this.busy(false); this.xfer(`Import stopped: ${esc(e.message)}. Nothing half-copied is treated as a model; delete it under Unfinished downloads.`); }
  }

  // The storage manager: what this site keeps in the browser, how much space it takes, and ways
  // to free it: one item at a time, or everything except the model open now.
  async renderStored() {
    this.renderPicks();
    const box = this.$('#hubStored'), usage = this.$('#hubUsage');
    box.innerHTML = '<li class="note">Reading storage…</li>';
    const items = await listStored().catch(() => []);
    let est = null;
    try { est = await navigator.storage.estimate(); } catch {}
    const open = this.currentRepo ? localId(this.currentRepo) : null;
    const total = items.reduce((t, m) => t + m.bytes, 0), freeable = items.filter((m) => m.id !== open).reduce((t, m) => t + m.bytes, 0);
    const kinds = [['model', 'Models', 'var(--purple)'], ['features', 'Feature dictionaries', 'var(--cyan)'], ['partial', 'Unfinished downloads', 'var(--amber)']];
    const bar = est?.quota ? kinds.map(([k, , col]) => {
      const b = items.filter((m) => m.kind === k).reduce((t, m) => t + m.bytes, 0);
      return b ? `<i style="width:${Math.max(0.6, (b / est.quota) * 100)}%;background:${col}" title="${fmtB(b)}"></i>` : '';
    }).join('') : '';
    usage.innerHTML = items.length
      ? `<span class="st-sum"><b>${fmtB(total)}</b> used by ${items.length} item${items.length > 1 ? 's' : ''}${est?.quota ? ` · this site may use up to ${fmtB(est.quota)}` : ''}</span>`
        + (bar ? `<span class="st-bar" aria-hidden="true">${bar}</span>` : '')
        + `<span class="st-actions"><button type="button" class="danger" id="hubFree" ${freeable ? '' : 'disabled'}>Free up space · ${fmtB(freeable)}</button><span class="note">${open ? 'Keeps the model open now; everything else can be downloaded again.' : 'Everything here can be downloaded again.'}</span></span>`
      : '<span class="st-sum">Nothing stored yet. Models you open from Hugging Face are kept here, so they load without downloading next time.</span>';
    usage.insertAdjacentHTML('beforeend', `<span class="st-actions"><button type="button" id="hubImport">Import model from a folder…</button><span class="note">${canExport() ? 'Export a model to a folder, then import it here in another browser, profile or copy of the site, instead of downloading it again.' : 'Bring in a model exported from Chrome or Edge. Exporting needs one of those browsers.'}</span></span><span class="st-xfer" id="hubXfer" hidden></span>`);
    const why = { model: 'loads without downloading', features: 'GPT-2 small features for one layer, streamed from Hugging Face', partial: 'a download that did not finish; safe to delete' };
    box.innerHTML = kinds.map(([k, title]) => {
      const list = items.filter((m) => m.kind === k).sort((a, b) => b.bytes - a.bytes);
      if (!list.length) return '';
      return `<li class="hub-sec st-head">${title} · ${fmtB(list.reduce((t, m) => t + m.bytes, 0))}</li>` + list.map((m) => {
        const inUse = m.id === open;
        return `<li class="hub-stored-row">
          <span class="hub-name"><b>${esc(m.name)}</b><span>${m.repo ? esc(m.repo) + ' · ' : ''}${fmtB(m.bytes)} · ${inUse ? 'open now' : why[k]}</span></span>
          <span class="hub-actions">${k === 'model' && !inUse ? `<button type="button" data-use="${esc(m.repo)}">Open</button>` : ''}${k === 'model' && canExport() ? `<button type="button" data-export="${esc(m.id)}" data-name="${esc(m.name)}" title="Copy this model to a folder on your disk, to import it elsewhere without downloading it again">Export</button>` : ''}<button type="button" data-del="${esc(m.id)}" data-name="${esc(m.name)}" data-bytes="${m.bytes}" ${inUse ? 'disabled title="In use: open another model first"' : ''}>Delete</button></span>
        </li>`;
      }).join('');
    }).join('');
    this.storedItems = items;
    this.$('#hubImport').addEventListener('click', () => this.importPick());
    this.$('#hubFree')?.addEventListener('click', async () => {
      const gone = items.filter((m) => m.id !== open);
      if (!confirm(`Delete ${gone.length} item${gone.length > 1 ? 's' : ''} (${fmtB(freeable)}) from this browser?${open ? ' The model open now is kept.' : ''} They can be downloaded again later.`)) return;
      usage.innerHTML = 'Deleting…';
      for (const m of gone) await removeStored(m.id).catch(() => {});
      this.renderStored();
    });
  }

}
