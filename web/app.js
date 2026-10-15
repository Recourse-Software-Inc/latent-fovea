// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// Latent Fovea viewer. Draws what the wasm kernel decides; streams what it asks for.
if (/[?&]stats\b/.test(location.search)) document.body.classList.add('stats');  // show the technical readouts
import { ForwardView } from './field.js';
import { DiffusionView } from './diffusion.js';
import { KdStore } from './kdquery.js';
import { ExpertView } from './expertview.js';
import { dataFetch, localBase, localId, exists, writeFile } from './store.js';
import { ModelBrowser, getToken } from './hubui.js';
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
// Which model: '#<id>' or '#<id>~embeddings' (some embedding pages pass only a plain hash),
// or ?data=<path>/ for a data set outside the registry. '#embeddings' alone still works.
const [hashModel, hashMode] = (() => {
  const h = location.hash.slice(1).split('~');
  return h[0] === 'embeddings' ? ['', 'embeddings'] : [h[0] || '', h[1] || ''];
})();
let DATA = params.get('data') ? params.get('data').replace(/\/?$/, '/') : null;
let registry = null, modelId = null, browser = null;
// '#hf:<owner>/<repo>' loads a model straight from Hugging Face: converted in the browser
// on first use (build-worker.js), then served from the browser's own storage.
const hfRepo = hashModel.startsWith('hf:') ? hashModel.slice(3) : null;
function buildLocal(repo, token) {
  return new Promise((resolve, reject) => {
    const w = new Worker('build-worker.js', { type: 'module' });
    w.onmessage = ({ data: m }) => {
      if (m.type === 'progress') { $('loadMsg').textContent = m.text; $('loadBar').hidden = false; $('loadBar').value = m.stage === 'download' ? m.p : m.stage === 'info' ? 0 : 1; }
      else if (m.type === 'done') { w.terminate(); resolve(m.base); }
      else if (m.type === 'error') { w.terminate(); reject(new Error(m.message)); }
    };
    w.onerror = (e) => { w.terminate(); reject(new Error(e.message || 'the model builder failed')); };
    w.postMessage({ type: 'build', repo, token });
  });
}
async function pickData() {
  if (DATA) return;
  if (hfRepo) {
    try { registry = await (await fetch('data/models.json', { cache: 'no-cache' })).json(); } catch { registry = null; }
    modelId = localId(hfRepo);
    if (!(await exists(`${modelId}/meta.json`))) {
      $('loadName').textContent = hfRepo; $('loadName').hidden = false;
      $('loadAbout').textContent = 'Downloading it from Hugging Face and preparing it in your browser. This happens once; next time it loads from your browser\'s storage.'; $('loadAbout').hidden = false;
      await buildLocal(hfRepo, getToken());
      navigator.storage?.persist?.().catch(() => {});  // ask the browser not to evict it
    }
    DATA = localBase(hfRepo);
    return;
  }
  try { registry = await (await fetch('data/models.json', { cache: 'no-cache' })).json(); } catch { registry = null; }
  const list = registry?.models || [];
  let saved = null;
  try { saved = localStorage.getItem('lf-model'); } catch {}
  const local = list.filter((m) => !m.url);  // entries with a url live on another page
  const entry = local.find((m) => m.id === hashModel) || local.find((m) => m.id === saved) || local.find((m) => m.id === registry.default) || local[0];
  modelId = entry ? entry.id : 'gpt2';
  DATA = 'data/' + (entry ? entry.path : 'gpt2/');
}
async function buildPicker() {
  const btn = $('modelPick'), list = (registry?.models || []).slice();
  const current = list.find((m) => m.id === modelId);
  $('modelPickName').textContent = current ? current.name : `${meta.model.name}${hfRepo ? ' · Hugging Face' : ''}`;
  btn.title = 'Choose a model: built in, downloaded, or from Hugging Face';
  const openRepo = (repo) => { location.hash = 'hf:' + repo + (state.mode === 'embed' ? '~embeddings' : ''); location.reload(); };
  const link = $('modelLink');
  const onPick = (id) => {
    const m = list.find((x) => x.id === id); if (!m) return;
    if (m.url) {  // hosted on its own page: offer a real link, keep this page as it is
      link.href = m.url; link.textContent = `Open ${m.name.replace(/\s*\(.*\)$/, '')} ↗`; link.hidden = false;
      return;
    }
    link.hidden = true;
    try { localStorage.setItem('lf-model', id); } catch {}
    location.hash = id + (state.mode === 'embed' ? '~embeddings' : '');
    location.reload();
  };
  browser = new ModelBrowser({ onLoad: openRepo, onPick, currentId: hfRepo ? null : modelId, currentRepo: hfRepo || null, builtins: list, anchor: btn });
  btn.disabled = false;
  btn.addEventListener('click', () => browser.toggle());
  btn.addEventListener('keydown', (e) => { if (e.key === 'ArrowDown') { e.preventDefault(); browser.open('find'); } });
}

// What kind of model it is, in a sentence, by family.
const KINDS = {
  'causal-lm': ['Autoregressive', 'It writes one token at a time, left to right. Each new token can only look back at the ones before it.'],
  'experts': ['Mixture of experts', "Each point is one of GLM-5.2's routed experts, placed by its row of the router: the direction in the hidden state that wakes it."],
  'kdstore': ['Attention layer, no decoder', 'A KnowDrive store: no weights, and nothing is generated. Each point is an atom, a passage stored as a key; a question is scored against every key, and a mask decides which keys can be attended to.'],
  'masked-diffusion': ['Masked diffusion', 'It starts with the whole answer masked and fills it in over several denoising steps, in any order. Every slot looks at the whole sequence, in both directions.'],
};
function showAbout(m) {  // m: a registry entry or meta.model
  if (!m) return;
  const [kind, how] = KINDS[m.family] || KINDS['causal-lm'];
  for (const id of ['qAbout', 'loadAbout']) {
    const el = $(id); if (!el) continue;
    el.innerHTML = '';
    const tag = document.createElement('span'); tag.className = 'kind'; tag.textContent = kind;
    const txt = document.createElement('span'); txt.textContent = ` ${how}${m.blurb ? ' ' + m.blurb : ''}`;
    el.append(tag, txt); el.hidden = false;
  }
  const n = $('loadName'); if (n) { n.textContent = m.name; n.hidden = false; }
}

// Category colours on #08080e (additive), in meta.categories order.
const CAT_COLORS = ['#9d8ff5', '#5b8def', '#35e0e6', '#f2b56b', '#9aa4b8', '#ef6fa6'];
const CAT_LABELS = { word: 'word', subword: 'word piece', number: 'number', punctuation: 'punctuation', whitespace: 'whitespace', other: 'other script / bytes' };
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);

const state = {
  fov: true, lens: true, rings: true, auto: true,
  gaze: { x: 0, y: 0 }, lastInput: -1e9, lensMix: 1, frame: 0,
  cam: { yaw: 0.6, pitch: 0.25, dist: 2.1, target: [0, 0, 0] },
  fly: null, pinned: -1, focus: -1, shownFor: -2, mode: 'embed', shift: 0,
};
let ex, mem, meta, tokens, cats, points, cells, knn, knncos, norms, simd = false, posBuf = null;
let N = 0, D = 0, NC = 0, rowBytes = 0, KD = false, kd = null, EX = false, xv = null;  // KD: a KnowDrive store (atoms, not tokens); EX: GLM-5.2's experts (xv: their view)
const dpr = Math.min(window.devicePixelRatio || 1, 2);
let W = 0, H = 0;

// ---------------- boot ----------------
// Hosts that only serve web file types can carry binaries under a suffix (meta.binary_suffix).
const bin = (name) => (/\.(json|txt)$/.test(name) ? name : name + ((meta && meta.binary_suffix) || ''));
async function fetchBuf(name) {
  const parts = meta && meta.parts && meta.parts[name];  // split by tools/stage_static.py: rejoin
  if (parts) {
    const bufs = await Promise.all(parts.map((p) => fetchBuf(p.file)));
    const out = new Uint8Array(bufs.reduce((s, b) => s + b.byteLength, 0));
    let o = 0; for (const b of bufs) { out.set(new Uint8Array(b), o); o += b.byteLength; }
    return out.buffer;
  }
  const r = await dataFetch(DATA + bin(name));
  if (!r.ok) throw new Error(`${DATA + name}: HTTP ${r.status}`);
  return r.arrayBuffer();
}
async function boot() {
  const msg = $('loadMsg');
  msg.textContent = 'Loading…';
  const simdBytes = await (await fetch('lf.simd.wasm?v=share1')).arrayBuffer();  // bump when its exports change
  simd = WebAssembly.validate(simdBytes);
  const bytes = simd ? simdBytes : await (await fetch('lf.wasm?v=share1')).arrayBuffer();
  ex = (await WebAssembly.instantiate(bytes, {})).instance.exports;

  msg.textContent = 'Loading the map…';
  try { await pickData(); } catch (e) {
    msg.textContent = `Could not load ${hfRepo || 'the model'}: ${e.message} `;
    const back = document.createElement('a'); back.href = './'; back.textContent = 'Back to the default model'; back.className = 'model-link'; msg.append(back);
    $('loadBar').hidden = true; return;
  }
  $('loadBar').hidden = true;
  showAbout(registry?.models?.find((m) => m.id === modelId));
  try {
    meta = await (await dataFetch(DATA + 'meta.json')).json();
  } catch {
    msg.innerHTML = `No data at <b>${DATA}</b>. Build it with <b>tools/build_data.py</b> or run <b>tools/get_data.sh</b>.`;
    return;
  }
  N = meta.n; D = meta.d; NC = meta.cells; rowBytes = meta.row_bytes;
  KD = meta.kind === 'kdstore'; document.body.classList.toggle('kd', KD);
  EX = meta.kind === 'experts'; document.body.classList.toggle('ex', EX);
  if (KD) { $('iId').previousSibling.textContent = 'Row '; $('findForm').querySelector('label').textContent = 'Find an atom'; $('find').placeholder = 'a word in its label, e.g. tariff'; }
  $('iDims').textContent = `Its ${D.toLocaleString()} coordinates`;
  loadIds();
  const lazyKnn = meta.knn === 'lazy';
  const [p, c, ce, kn, kc, no, tk] = await Promise.all([
    fetchBuf('points.f32'), fetchBuf('cat.u8'), fetchBuf('cells.u32'), lazyKnn ? null : fetchBuf('knn.u32'),
    lazyKnn ? null : fetchBuf('knncos.f16'), fetchBuf('norms.f32'), dataFetch(DATA + 'tokens.json').then((r) => r.json()),
  ]);
  points = new Float32Array(p); cats = new Uint8Array(c); cells = new Uint32Array(ce);
  if (!lazyKnn) { knn = new Uint32Array(kn); knncos = new Uint16Array(kc); } else startKnnWorker();
  norms = new Float32Array(no); tokens = tk;
  if (EX) {
    $('ex').hidden = false;
    xv = new ExpertView({ $, DATA, meta, N, dataFetch, escapeHtml, onChange: () => {
      state.shownFor = -2;
      // following the run: the radial view re-places every expert at its rank for each new token
      if (layoutId === 'radial' && radial && radial.i < 0 && xv.dist) placeRadial(-1, xv.dist, xv.anchor(layouts[radial.from] || points));
    } });
    xv.init().catch((e) => { console.warn('experts', e); });
    $('exRadial').onclick = () => toggleRadial(true);
  }
  if (KD) {
    kd = new KdStore({ $, DATA, meta, tokens, cats, N, dataFetch, escapeHtml, onChange: () => { state.shownFor = -2; }, flyTo: (i) => { state.pinned = i; state.shownFor = -2; flyTo(i); }, radialAt: () => toggleRadial(true) });
    kd.init().catch((e) => { console.warn('kd', e); });
  }

  if (ex.init(N, D, NC, 96 * 1024 * 1024) !== 0) { msg.textContent = 'Not enough memory for the vector cache.'; return; }
  mem = () => ex.memory.buffer;
  new Float32Array(mem(), ex.ptr_P(), N * 3).set(points);
  new Uint8Array(mem(), ex.ptr_cat(), N).set(cats);
  new Uint32Array(mem(), ex.ptr_cells(), NC + 1).set(cells);
  ex.prepare();
  ex.set_cache(+$('cache').value * 1e6);

  $('modelName').textContent = meta.model.name;
  showAbout(meta.model);
  $('qKicker').textContent = `${meta.model.name.replace(/\s*\(.*\)$/, '')}${meta.model.params ? ` · ${meta.model.params} parameters` : ''} · running in your browser`;
  if (!meta.forward) $('modeFwd').hidden = true;
  await buildPicker();
  showSub(meta.projection.method === 'umap' ? 'umap' : 'pca');
  buildLegend();
  initGL();
  resize();
  state.gaze = { x: W / 2, y: H / 2 };
  $('loading').hidden = true;
  setMode(hashMode === 'embeddings' || !meta.forward ? 'embed' : 'forward');
  requestAnimationFrame(tick);
  initLayouts().catch((e) => { console.warn('layouts', e); });
}
function showSub(layout) {
  const unit = KD ? 'atoms' : EX ? 'experts' : 'tokens';
  $('modelSub').textContent = `${N.toLocaleString()} ${unit}, ${D} dimensions each · ${layout === 'umap' ? 'clusters' : layout === 'axes' ? 'your axes' : layout === 'radial' ? (radial && radial.i < 0 ? (EX ? 'router ranks for the last token' : 'distances from the question') : `distances from one ${unit.slice(0, -1)}`) : 'PCA'} · ${meta.model.author} · ${meta.model.license}`;
}

// ---------------- layouts: clusters (UMAP) and PCA ----------------
// Every data set has one layout built in (points.f32: UMAP for prebuilt sets, PCA for models
// built in the browser). The other comes from points.<id>.f32 beside it, from a cluster map
// precomputed for popular models (data/layouts/, token order), or, failing both, from UMAP
// run in the background in this browser (layout-worker.js), saved for next time.
const layouts = {};
let layoutId = null, layoutPref = null, morph = null, clustering = null;
try { layoutPref = localStorage.getItem('lf-layout'); } catch {}
async function initLayouts() {
  loadFid();
  const native = meta.projection.method === 'umap' ? 'umap' : 'pca', other = native === 'umap' ? 'pca' : 'umap';
  layouts[native] = points.slice(); layoutId = native;
  try { layouts[other] = new Float32Array(await fetchBuf(`points.${other}.f32`)); } catch {}
  if (!layouts[other] && other === 'umap' && DATA.startsWith('opfs:')) {
    const r = await dataFetch(`data/layouts/${modelId}.umap.f32`).catch(() => null);
    if (r && r.ok) {  // token order -> this data set's row order
      const tok = new Float32Array(await r.arrayBuffer()), ids = new Uint32Array(await fetchBuf('ids.u32'));
      if (tok.length >= ids.length * 3) {
        const P = new Float32Array(N * 3);
        for (let i = 0; i < N; i++) P.set(tok.subarray(ids[i] * 3, ids[i] * 3 + 3), i * 3);
        layouts.umap = P;
        writeFile(`${modelId}/points.umap.f32`, P).catch(() => {});
      }
    }
    if (!layouts.umap) startClustering();
  }
  syncLayoutUi();
  const want = layoutPref || 'umap';  // clusters unless someone chose PCA
  if (layouts[want] && want !== layoutId) setLayout(want, true);
}
function startClustering() {
  clustering = { p: 0 };
  const w = new Worker('layout-worker.js', { type: 'module' });
  w.onmessage = ({ data: m }) => {
    if (m.type === 'progress') { clustering.p = m.p; $('layoutNote').textContent = `arranging clusters ${Math.round(m.p * 100)}%`; $('layoutNote').title = m.text; }
    else if (m.type === 'done') {
      w.terminate(); clustering = null; layouts.umap = m.points;
      syncLayoutUi();
      if ((layoutPref || 'umap') === 'umap') setLayout('umap');
    } else if (m.type === 'error') { w.terminate(); clustering = null; $('layoutNote').textContent = 'clusters unavailable'; $('layoutNote').title = m.message; }
  };
  w.postMessage({ base: DATA, dir: DATA.startsWith('opfs:') ? modelId : null });
}
function syncLayoutUi() {
  for (const b of document.querySelectorAll('#layoutPick [data-layout]')) {
    const id = b.dataset.layout;
    b.setAttribute('aria-checked', id === layoutId);
    b.disabled = !layouts[id] && id !== 'axes';  // Axes opens its form before it has a layout
  }
  if (!clustering) $('layoutNote').textContent = '';
  showSub(layoutId);
  showFid(); measureFidelity();
}

// ---------------- fidelity: how much of the real geometry each layout keeps ----------------
// Neighbour recall: for 150 tokens spread across the vocabulary, how many of each one's 15
// nearest neighbours in the full space are also among its 15 nearest in the 3-D layout. True
// neighbours come from the precomputed table, or the knn worker (exact unless its shortlist
// missed). PCA's share of the variation comes from the knn worker too. Cached per model.
let fid = {}, fidBusy = false;
const FID_S = 150, FID_K = 15;
const fidKey = () => `lf-fid2:${modelId}`;
const FID_REGION = 0.01;  // "the same region": within a token's nearest 1% of the cloud, about one visible blob
function loadFid() { try { fid = JSON.parse(localStorage.getItem(fidKey())) || {}; } catch { fid = {}; } }
function saveFid() { try { localStorage.setItem(fidKey(), JSON.stringify(fid)); } catch {} }
function showFid() {
  const el = $('layoutFid'), t = $('layoutFidText'), r = fid[`recall:${layoutId}`], g = fid[`region:${layoutId}`], rho = fid[`rank:${layoutId}`];
  if (!layoutId) { el.hidden = true; return; }
  el.hidden = false;
  if (layoutId === 'radial' && radial) { t.innerHTML = radial.i < 0 && xv ? `Router ranks for “${escapeHtml(xv.token?.text || '')}”: every expert's radius is its <b>exact</b> rank in its own layer's choice (score + balance bias); the eight chosen per layer are inside the first ring` : `Distances from “${escapeHtml(radial.i < 0 ? kd.queries[kd.q].text : disp(tokens[radial.i]))}”: every radius is the <b>exact</b> cosine distance from it · the direction around the disc is kept from the cloud and means little`; return; }
  const pct = (x) => `${x < 0.095 ? (x * 100).toFixed(1) : Math.round(x * 100)}%`;
  const name = layoutId === 'pca' ? 'PCA' : layoutId === 'axes' ? 'Axes' : 'Clusters';
  // Lead with what the eye uses: do true neighbours land in the same region, and does near/far agree.
  const parts = [];
  if (g != null) parts.push(`true neighbours land in the same region <b>${pct(g)}</b> of the time`);
  if (rho != null) parts.push(`near/far agreement <b>${rho.toFixed(2)}</b>`);
  if (layoutId === 'pca' && fid.pca != null) parts.push(`shows <b>${pct(fid.pca)}</b> of the variation`);
  if (layoutId === 'axes' && axes) parts.unshift(`x holds <b>${pct(axes.shares[0])}</b> of the variation, y <b>${pct(axes.shares[1])}</b>, depth <b>${pct(axes.shares[2])}</b>`);
  t.innerHTML = parts.length ? `${name}: ${parts.join(' · ')}` : 'Measuring how much this layout keeps…';
  if (local) t.innerHTML = `This region, re-arranged by its own layout: keeps <b>${pct(local.after)}</b> of its true 15 nearest neighbours (was <b>${pct(local.before)}</b> in the global cloud) · ${t.innerHTML}`;
  const strict = $('layoutFidStrict');
  if (strict) strict.textContent = r != null ? `Strictly, ${pct(r)} of each token's ${FID_K} nearest neighbours are also among its ${FID_K} nearest in the cloud: a 15-of-${N.toLocaleString()} exact-rank test, far stricter than what the eye reads.` : '';
}
async function measureFidelity() {
  if (fidBusy || !layoutId || !N) return;
  const todo = Object.keys(layouts).filter((id) => id !== 'radial' && layouts[id] && fid[`recall:${id}`] == null);
  if (fid.pca == null && layouts.pca) { if (!knnWorker) startKnnWorker(); knnWorker.postMessage({ type: 'pca' }); }
  const rankTodo = Object.keys(layouts).filter((id) => id !== 'radial' && layouts[id] && fid[`rank:${id}`] == null);
  if (rankTodo.length) {  // near/far agreement needs the full rows: the knn worker has them
    if (!knnWorker) startKnnWorker();
    const sample = Array.from({ length: FID_S }, (_, s) => Math.floor(((s + 0.5) * N) / FID_S));
    knnWorker.postMessage({ type: 'rank', sample, layouts: Object.fromEntries(rankTodo.map((id) => [id, layouts[id]])) });
  }
  if (!todo.length) return;
  fidBusy = true;
  try {
    const sample = Array.from({ length: FID_S }, (_, s) => Math.floor(((s + 0.5) * N) / FID_S));
    const truth = new Map();
    for (let tries = 0; tries < 600 && truth.size < sample.length; tries++) {  // ask the knn worker if there is no table
      for (const i of sample) { if (truth.has(i)) continue; const nb = nbrs(i); if (nb) truth.set(i, Array.from(nb.idx).filter((j) => j !== i).slice(0, FID_K)); }
      if (truth.size < sample.length) await new Promise((r) => setTimeout(r, 200));
    }
    for (const id of todo) {
      const P = layouts[id]; let hit = 0, tot = 0, inRegion = 0;
      const regionN = Math.max(FID_K, Math.round(N * FID_REGION));
      for (const i of sample) {
        const nb = truth.get(i); if (!nb) continue;
        const x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2], best = new Float64Array(FID_K).fill(Infinity), bi = new Int32Array(FID_K).fill(-1);
        // 3-D distance to each true neighbour, and how many points sit closer than it (its 3-D rank).
        const ddNb = nb.map((j) => (P[3 * j] - x) ** 2 + (P[3 * j + 1] - y) ** 2 + (P[3 * j + 2] - z) ** 2), closer = new Int32Array(nb.length);
        for (let j = 0; j < N; j++) {
          if (j === i) continue;
          const dd = (P[3 * j] - x) ** 2 + (P[3 * j + 1] - y) ** 2 + (P[3 * j + 2] - z) ** 2;
          if (dd < best[FID_K - 1]) { let p = FID_K - 1; while (p > 0 && best[p - 1] > dd) { best[p] = best[p - 1]; bi[p] = bi[p - 1]; p--; } best[p] = dd; bi[p] = j; }
          for (let q = 0; q < ddNb.length; q++) if (dd < ddNb[q]) closer[q]++;
        }
        const near = new Set(bi);
        nb.forEach((j, q) => { tot++; if (near.has(j)) hit++; if (closer[q] < regionN) inRegion++; });
        await new Promise((r) => setTimeout(r, 0));  // keep the page responsive
      }
      if (tot) { fid[`recall:${id}`] = hit / tot; fid[`region:${id}`] = inRegion / tot; saveFid(); showFid(); }
    }
  } finally { fidBusy = false; }
}
// Move every point to the chosen layout, easing over ~1.2 s (or at once).
function setLayout(id, instant = false) {
  if (!layouts[id] || id === layoutId) return;
  layoutId = id; local = null; syncLayoutUi();
  morphPoints(layouts[id], instant ? 0 : 1200);
}
// Ease every point from where it is to `to` over `dur` ms (0 = at once).
function morphPoints(to, dur) {
  const from = points.slice(), t0 = performance.now();
  const apply = () => {
    new Float32Array(mem(), ex.ptr_P(), N * 3).set(points);
    ex.prepare();  // tile centres follow the points
    gl.bindBuffer(gl.ARRAY_BUFFER, posBuf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, points);
  };
  cancelAnimationFrame(morph);
  const step = (now) => {
    const t = dur ? Math.min(1, (now - t0) / dur) : 1, e = t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2;
    for (let i = 0; i < points.length; i++) points[i] = from[i] + (to[i] - from[i]) * e;
    apply();
    if (t < 1) morph = requestAnimationFrame(step);
  };
  step(performance.now());
}

// ---------------- semantic zoom: re-arrange the region in view with its own layout ----------------
// The global cloud squeezes the whole vocabulary into three dimensions at once. Press Z (or
// the button) and the tokens under the parafovea are laid out again by their own UMAP, in
// place, so that region becomes faithful; the rest of the cloud stays as context. The worker
// also measures how much of the region's true 15-NN the old and new arrangement keep.
let local = null, localWorker = null, localBusy = false;
const LOCAL_MIN = 60, LOCAL_MAX = 8000;
function regionInView() {  // indices of the points within the parafovea on screen
  if (!MVP || state.mode !== 'embed') return [];
  const r1 = Math.max(+$('rFov').value * dpr * 1.4, +$('rPara').value * dpr), g = state.gaze, out = [];
  for (let i = 0; i < N; i++) { const s = toScreen(i); if (s && Math.hypot(s[0] - g.x, s[1] - g.y) <= r1) out.push(i); }
  return out;
}
function syncLocalUi() {
  const b = $('localFit'); if (!b) return;
  if (state.mode !== 'embed') { b.hidden = true; return; }
  b.hidden = false;
  if (local) { b.textContent = 'Restore the global layout'; b.disabled = false; return; }
  const n = localBusy ? 0 : regionInView().length;
  b.disabled = localBusy || n < LOCAL_MIN || n > LOCAL_MAX;
  b.textContent = localBusy ? 'Re-arranging…' : n < LOCAL_MIN ? 'Re-arrange this region (zoom in)' : n > LOCAL_MAX ? `Re-arrange this region (${n.toLocaleString()}: zoom in more)` : `Re-arrange these ${n.toLocaleString()} ${KD ? 'atoms' : 'tokens'}`;
}
function toggleLocal() {
  if (localBusy) return;
  if (local) {  // back to the global layout, where the region was
    const to = points.slice(), G = layouts[layoutId];
    for (const i of local.idx) { to[3 * i] = G[3 * i]; to[3 * i + 1] = G[3 * i + 1]; to[3 * i + 2] = G[3 * i + 2]; }
    local = null; morphPoints(to, 900); showFid(); syncLocalUi(); return;
  }
  const idx = regionInView();
  if (idx.length < LOCAL_MIN || idx.length > LOCAL_MAX) return;
  if (!localWorker) {
    localWorker = new Worker('layout-worker.js', { type: 'module' });
    localWorker.onmessage = ({ data: m }) => {
      if (m.type === 'localProgress') { $('layoutNote').textContent = `re-arranging ${Math.round(m.p * 100)}%`; return; }
      if (m.type === 'localError') { localBusy = false; $('layoutNote').textContent = 'could not re-arrange this region'; $('layoutNote').title = m.message; syncLocalUi(); return; }
      if (m.type !== 'local') return;
      localBusy = false; $('layoutNote').textContent = '';
      placeLocal(m.idx, m.points, m.before, m.after);
    };
  }
  localBusy = true; syncLocalUi();
  const cur = new Float32Array(idx.length * 3);
  idx.forEach((i, k) => { cur[3 * k] = points[3 * i]; cur[3 * k + 1] = points[3 * i + 1]; cur[3 * k + 2] = points[3 * i + 2]; });
  localWorker.postMessage({ type: 'local', base: DATA, idx: Uint32Array.from(idx), cur });
}
// Put the region's new arrangement where the region is: same centre, a little more spread,
// and turned to match the old arrangement as closely as a rotation can (polar decomposition).
function placeLocal(idx, Y, before, after) {
  const n = idx.length, c = [0, 0, 0];
  for (const i of idx) for (let k = 0; k < 3; k++) c[k] += points[3 * i + k] / n;
  let rr = 0, yy = 0; const M = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  idx.forEach((i, j) => {
    const a = [points[3 * i] - c[0], points[3 * i + 1] - c[1], points[3 * i + 2] - c[2]], b = [Y[3 * j], Y[3 * j + 1], Y[3 * j + 2]];
    rr += a[0] ** 2 + a[1] ** 2 + a[2] ** 2; yy += b[0] ** 2 + b[1] ** 2 + b[2] ** 2;
    for (let p = 0; p < 3; p++) for (let q = 0; q < 3; q++) M[3 * p + q] += a[p] * b[q];
  });
  let R = M.slice();  // R <- (R + R^-T) / 2 converges to the nearest rotation to M
  for (let it = 0; it < 30; it++) {
    const inv = inv3(R); if (!inv) break;
    const nx = R.map((v, k) => (v + inv[3 * (k % 3) + Math.floor(k / 3)]) / 2);
    let diff = 0; for (let k = 0; k < 9; k++) diff += Math.abs(nx[k] - R[k]);
    R = nx; if (diff < 1e-7) break;
  }
  if (!Number.isFinite(R[0])) R = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const s = 1.25 * Math.sqrt(rr / Math.max(yy, 1e-9));  // keep the region's spread, a little roomier
  const to = points.slice();
  idx.forEach((i, j) => {
    const b = [Y[3 * j], Y[3 * j + 1], Y[3 * j + 2]];
    for (let p = 0; p < 3; p++) to[3 * i + p] = c[p] + s * (R[3 * p] * b[0] + R[3 * p + 1] * b[1] + R[3 * p + 2] * b[2]);
  });
  local = { idx, before, after };
  morphPoints(to, 1200); showFid(); syncLocalUi();
  state.lastInput = performance.now();
}
function inv3(m) {
  const [a, b, c, d, e, f, g, h, i] = m, det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (Math.abs(det) < 1e-12) return null;
  const k = 1 / det;
  return [(e * i - f * h) * k, (c * h - b * i) * k, (b * f - c * e) * k, (f * g - d * i) * k, (a * i - c * g) * k, (c * d - a * f) * k, (d * h - e * g) * k, (b * g - a * h) * k, (a * e - b * d) * k];
}
function chooseLayout(id) {
  if (!layouts[id] && !(id === 'umap' && clustering)) return;
  layoutPref = id; try { localStorage.setItem('lf-layout', id); } catch {}
  if (layouts[id]) setLayout(id);
  else $('layoutNote').textContent = `clusters in ${Math.round((1 - clustering.p) * 100)}%…`;  // switches when ready
}
// ---------------- "What am I seeing here?" in a window over the viewer ----------------
const explain = $('explainDlg');
function openExplain() {
  const f = $('explainFrame');
  if (!f.src) f.src = 'real.html?embed';
  explain.showModal();
}
document.addEventListener('click', (e) => {
  const a = e.target.closest('[data-explain]');
  if (!a || e.metaKey || e.ctrlKey || e.shiftKey) return;  // modified clicks still open the page itself
  e.preventDefault(); openExplain();
});
$('explainClose').addEventListener('click', () => explain.close());
explain.addEventListener('click', (e) => { if (e.target === explain) explain.close(); });  // the backdrop
addEventListener('message', (e) => { if (e.origin === location.origin && e.data === 'lf-close-explain') explain.close(); });

$('layoutPick').addEventListener('click', (e) => {
  const b = e.target.closest('[data-layout]'); if (!b) return;
  if (b.dataset.layout === 'axes') { $('axesForm').hidden = false; if (!layouts.axes) { $('axX').focus(); return; } }
  else $('axesForm').hidden = true;
  if (!b.disabled) chooseLayout(b.dataset.layout);
});
// ---------------- radial view: exact distances from the pinned token ----------------
let radial = null;  // { i, scale, basis: [R, U], centre }
function syncRadialUi() {
  const b = $('radialBtn'), info = document.querySelector('.radial-info');
  const show = state.mode === 'embed' && (state.pinned >= 0 || layoutId === 'radial');
  b.hidden = !show; if (info) info.hidden = !show;
  if (!show) return;
  b.textContent = layoutId === 'radial' ? 'Back to the cloud' : `Distances from “${disp(tokens[state.pinned]).trim() || disp(tokens[state.pinned])}”`;
}
function toggleRadial(fromQuery = false) {
  if (fromQuery && xv && xv.dist) {  // every expert at its rank in its own layer's choice for the last token
    if (layoutId === 'radial' && radial && radial.i < 0) { setLayout(radial.from && layouts[radial.from] ? radial.from : 'umap'); return; }
    placeRadial(-1, xv.dist, xv.anchor(points)); return;
  }
  if (fromQuery && kd && kd.dist) {
    if (layoutId === 'radial' && radial && radial.i < 0) { setLayout(radial.from && layouts[radial.from] ? radial.from : 'umap'); return; }
    placeRadial(-1, kd.dist, kd.anchor(points)); return;
  }
  if (layoutId === 'radial') { const back = radial && radial.from && layouts[radial.from] ? radial.from : layouts.umap ? 'umap' : 'pca'; setLayout(back); return; }
  if (state.pinned < 0) return;
  if (!knnWorker) startKnnWorker();
  $('layoutNote').textContent = 'measuring distances…';
  knnWorker.postMessage({ type: 'radial', i: state.pinned });
}
function placeRadial(i, dist, at = null) {  // i < 0: from a point that is not an atom (a question at 'at')
  // A disc facing the camera, centred where the pinned token is; each token keeps the bearing
  // it has on screen now and moves to a radius equal to its true cosine distance.
  const c = state.cam, e = [Math.cos(c.pitch) * Math.sin(c.yaw), Math.sin(c.pitch), Math.cos(c.pitch) * Math.cos(c.yaw)];
  const R = [Math.cos(c.yaw), 0, -Math.sin(c.yaw)], U = [e[1] * R[2] - e[2] * R[1], e[2] * R[0] - e[0] * R[2], e[0] * R[1] - e[1] * R[0]];
  const centre = i >= 0 ? [points[3 * i], points[3 * i + 1], points[3 * i + 2]] : at, si = (i >= 0 ? toScreen(i) : project(...at)) || [W / 2, H / 2];
  const sorted = Float32Array.from(dist).sort(), scale = 1 / (sorted[Math.floor(0.98 * (N - 1))] || 1);  // 98% of tokens within radius 1, like the other layouts
  const P = new Float32Array(N * 3);
  let seed = 7; const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
  for (let j = 0; j < N; j++) {
    const s = toScreen(j), r = dist[j] * scale;
    const th = j === i ? 0 : s ? Math.atan2(-(s[1] - si[1]), s[0] - si[0]) : rnd() * Math.PI * 2;  // behind the camera: anywhere
    for (let k = 0; k < 3; k++) P[3 * j + k] = centre[k] + r * (Math.cos(th) * R[k] + Math.sin(th) * U[k]);
  }
  radial = { i, scale, basis: [R, U], centre, from: layoutId === 'radial' && radial ? radial.from : layoutId };
  layouts.radial = P;
  setLayout('radial');
  $('layoutNote').textContent = ''; syncRadialUi();
}
$('radialBtn').addEventListener('click', toggleRadial);

// ---------------- semantic axes: x and y are word pairs the user names ----------------
let axes = null;  // { pairs: [[a, b], [c, d]], words, shares }
function findToken(q) {  // the same lookup as the Find box
  q = q.trim(); if (!q) return -1;
  for (const t of [' ' + q, q, ' ' + q.toLowerCase(), q.toLowerCase()]) { const i = tokens.indexOf(t); if (i >= 0) return i; }
  return -1;
}
$('axesForm').addEventListener('submit', (e) => {
  e.preventDefault();
  const parse = (s) => s.split(/\s*[−\-–]\s*/).map((w) => w.trim()).filter(Boolean);
  const px = parse($('axX').value), py = parse($('axY').value), msg = $('axesMsg');
  if (px.length !== 2 || py.length !== 2) { msg.textContent = 'Each axis is a pair: a − b.'; return; }
  const idx = [...px, ...py].map(findToken), bad = [...px, ...py].filter((w, k) => idx[k] < 0);
  if (bad.length) { msg.textContent = `Not a single token: ${bad.map((w) => `“${w}”`).join(', ')}.`; return; }
  msg.textContent = 'Laying out…';
  if (!knnWorker) startKnnWorker();
  axes = { pairs: [[idx[0], idx[1]], [idx[2], idx[3]]], words: [px, py], shares: null };
  for (const k of ['recall:axes', 'region:axes', 'rank:axes']) delete fid[k];
  knnWorker.postMessage({ type: 'axes', pairs: axes.pairs });
});

function buildLegend() {
  $('legend').innerHTML = meta.categories.map((k, i) =>
    `<span data-c="${i}"${KD ? ` role="button" tabindex="0" title="Mask ${k} out (or back in)"` : ''}><i style="color:${CAT_COLORS[i]};background:${CAT_COLORS[i]}"></i>${KD || EX ? k : CAT_LABELS[k] || k} ${meta.category_counts[i].toLocaleString()}</span>`).join('');
  if (KD) $('legend').onclick = (e) => {
    const s = e.target.closest('span[data-c]'); if (!s || !kd) return;
    kd.toggleDomain(+s.dataset.c); s.classList.toggle('off', kd.off.has(+s.dataset.c));
  };
}

// ---------------- WebGL ----------------
let maskAttr = null;
let gl, progPts, progFov, vaoPts, vaoFov, attrBuf, fovBuf;
const VS_PTS = `#version 300 es
layout(location=0) in vec3 a_pos; layout(location=1) in float a_cat; layout(location=2) in vec2 a_attr;
uniform mat4 u_mvp; uniform float u_dpr; uniform float u_dim; uniform vec3 u_col[6];
uniform vec2 u_res; uniform vec2 u_gaze; uniform float u_r1; uniform float u_lensDim; uniform float u_density;
out vec3 v_col; out float v_a;
void main(){
  gl_Position = u_mvp * vec4(a_pos, 1.0);
  float w = max(gl_Position.w, 0.05);
  float fog = clamp(1.6 - w * 0.22, 0.25, 1.0);
  v_col = u_col[int(a_cat)];
  // Under the fovea and parafovea the cloud is dimmed (u_lensDim), so a dense core never
  // blows out to white over what the fovea shows; it eases back to full at the parafovea's edge.
  vec2 scr = vec2((gl_Position.x / w * 0.5 + 0.5) * u_res.x, (0.5 - gl_Position.y / w * 0.5) * u_res.y);
  float lens = mix(u_lensDim, 1.0, smoothstep(u_r1 * 0.8, u_r1 * 1.05, distance(scr, u_gaze)));
  // Crisp dots, blended "over" rather than added: dense regions build up to the colour of their
  // tokens instead of blooming to white. u_density evens out vocabularies of different sizes.
  v_a = min(1.0, a_attr.x * fog * u_dim * lens * u_density * 1.05);
  gl_PointSize = (1.0 + a_attr.y * 3.2) * u_dpr * clamp(2.4 / w, 0.5, 1.9);
  if (a_attr.x == 0.0) gl_PointSize = 0.0;
}`;
const FS_PTS = `#version 300 es
precision mediump float; in vec3 v_col; in float v_a; out vec4 o;
void main(){ float d = length(gl_PointCoord - 0.5) * 2.0; float a = (1.0 - smoothstep(0.7, 1.0, d)) * v_a; o = vec4(v_col * a, a); }`;
const VS_FOV = `#version 300 es
layout(location=0) in vec2 a_scr; layout(location=1) in float a_cat;
uniform vec2 u_res; uniform float u_dpr; uniform vec3 u_col[6];
out vec3 v_col;
void main(){
  gl_Position = vec4(a_scr.x / u_res.x * 2.0 - 1.0, 1.0 - a_scr.y / u_res.y * 2.0, 0.0, 1.0);
  v_col = u_col[int(a_cat)];
  gl_PointSize = 5.5 * u_dpr;
}`;
const FS_FOV = `#version 300 es
precision mediump float; in vec3 v_col; out vec4 o;
void main(){
  float d = length(gl_PointCoord - 0.5) * 2.0;
  float core = 1.0 - smoothstep(0.5, 0.62, d), ring = 1.0 - smoothstep(0.85, 1.0, d);  // a crisp dot with a fine dark rim
  vec3 c = mix(vec3(0.03, 0.03, 0.06), mix(v_col, vec3(1.0), 0.12), core);
  float a = ring * 0.95; o = vec4(c * a, a);
}`;

function shader(type, src) {
  const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
function program(vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, shader(gl.VERTEX_SHADER, vs)); gl.attachShader(p, shader(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}
function initGL() {
  gl = $('gl').getContext('webgl2', { antialias: false, premultipliedAlpha: true, alpha: false });
  if (!gl) throw new Error('This browser has no WebGL2.');
  progPts = program(VS_PTS, FS_PTS); progFov = program(VS_FOV, FS_FOV);
  const cols = new Float32Array(CAT_COLORS.flatMap(hex));
  for (const p of [progPts, progFov]) { gl.useProgram(p); gl.uniform3fv(gl.getUniformLocation(p, 'u_col'), cols); }

  vaoPts = gl.createVertexArray(); gl.bindVertexArray(vaoPts);
  posBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, posBuf); gl.bufferData(gl.ARRAY_BUFFER, points, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  const cb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, cb); gl.bufferData(gl.ARRAY_BUFFER, cats, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 1, gl.UNSIGNED_BYTE, false, 0, 0);
  attrBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, attrBuf); gl.bufferData(gl.ARRAY_BUFFER, N * 2, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 2, gl.UNSIGNED_BYTE, true, 0, 0);

  vaoFov = gl.createVertexArray(); gl.bindVertexArray(vaoFov);
  fovBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, fovBuf); gl.bufferData(gl.ARRAY_BUFFER, 2048 * 12, gl.DYNAMIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 12, 0);
  gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 12, 8);
  gl.bindVertexArray(null);
  gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);  // premultiplied "over": no blooming
  gl.disable(gl.DEPTH_TEST);
}

function resize() {
  W = Math.round(innerWidth * dpr); H = Math.round(innerHeight * dpr);
  for (const c of [$('gl'), $('ov')]) { c.width = W; c.height = H; }
}
addEventListener('resize', () => gl && resize());

// ---------------- camera ----------------
function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0];
}
function lookAt(e, t) {
  let z = [e[0] - t[0], e[1] - t[1], e[2] - t[2]]; const zl = Math.hypot(...z); z = z.map((v) => v / zl);
  let x = [z[2], 0, -z[0]]; const xl = Math.hypot(...x) || 1; x = x.map((v) => v / xl);
  const y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]];
  const d = (a) => -(a[0] * e[0] + a[1] * e[1] + a[2] * e[2]);
  return [x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, d(x), d(y), d(z), 1];
}
function mul(a, b) {
  const o = new Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k]; o[c * 4 + r] = s;
  }
  return o;
}
let MVP = null, projF = 1;
function updateCamera() {
  const c = state.cam, t = c.target;
  const eye = [t[0] + c.dist * Math.cos(c.pitch) * Math.sin(c.yaw), t[1] + c.dist * Math.sin(c.pitch), t[2] + c.dist * Math.cos(c.pitch) * Math.cos(c.yaw)];
  const P = perspective(0.9, W / H, 0.02, 60);
  P[9] = -state.shift;  // while the field panel is open, lift the scene into the space above it
  projF = P[5];
  MVP = mul(P, lookAt(eye, t));
}
function project(x, y, z) {
  const m = MVP, w = m[3] * x + m[7] * y + m[11] * z + m[15];
  if (w <= 0.05) return null;
  return [((m[0] * x + m[4] * y + m[8] * z + m[12]) / w) * W / 2 + W / 2, H / 2 - ((m[1] * x + m[5] * y + m[9] * z + m[13]) / w) * H / 2];
}
function toScreen(i) {
  const m = MVP, x = points[3 * i], y = points[3 * i + 1], z = points[3 * i + 2];
  const w = m[3] * x + m[7] * y + m[11] * z + m[15];
  if (w <= 0.05) return null;
  return [((m[0] * x + m[4] * y + m[8] * z + m[12]) / w) * W / 2 + W / 2, H / 2 - ((m[1] * x + m[5] * y + m[9] * z + m[13]) / w) * H / 2];
}
// Fly-through: glide after the thought path, a little behind it, looking the way it is heading.
function followFlight(goal, dt) {
  const c = state.cam, t = c.target, v = [goal[0] - t[0], goal[1] - t[1], goal[2] - t[2]];
  const k = Math.min(1, dt * 2.5);
  for (let i = 0; i < 3; i++) t[i] += v[i] * k;
  c.dist += (0.75 - c.dist) * Math.min(1, dt * 1.5);
  c.pitch += (0.18 - c.pitch) * Math.min(1, dt * 1.5);
  if (Math.hypot(v[0], v[2]) > 0.02) {  // the eye sits behind the direction of travel
    const want = Math.atan2(-v[0], -v[2]), turn = Math.atan2(Math.sin(want - c.yaw), Math.cos(want - c.yaw));
    c.yaw += turn * Math.min(1, dt * 1.2);
  }
}
function flyToPoint(p, dist) {
  state.fly = { from: [...state.cam.target], to: p, d0: state.cam.dist, d1: dist, t0: performance.now() };
}
function flyTo(i) {
  state.fly = { from: [...state.cam.target], to: [points[3 * i], points[3 * i + 1], points[3 * i + 2]], d0: state.cam.dist, d1: Math.min(state.cam.dist, 0.9), t0: performance.now() };
  state.gaze = { x: W / 2, y: H / 2 }; state.lastInput = performance.now();
}

// ---------------- streaming ----------------
const stream = { inflight: 0, bucket: 0, bytes: 0, t: 0, rate: 0, full: null, noRange: false, cacheFull: 0 };
// Read bytes [a, b] of the vector table. It may be one file or split into parts
// (meta.vector_parts, for hosts with a per-file size limit). Uses HTTP Range when
// the host honours it, else downloads the whole file or part once and slices it.
const whole = new Map();
async function readRange(file, a, b) {
  if (whole.has(file)) return (await whole.get(file)).slice(a, b + 1);
  const r = await dataFetch(DATA + bin(file), { headers: { Range: `bytes=${a}-${b}` } });
  if (r.status === 206) return r.arrayBuffer();
  if (r.status !== 200) throw new Error(`HTTP ${r.status}`);
  stream.noRange = true;
  const all = r.arrayBuffer(); whole.set(file, all);
  return (await all).slice(a, b + 1);
}
async function readVectors(a, b) {
  const parts = meta.vector_parts;
  if (!parts) return readRange('vectors.f16', a, b);
  const out = new Uint8Array(b - a + 1);
  let base = 0;
  for (const p of parts) {
    const end = base + p.bytes;
    if (a < end && b >= base) {
      const lo = Math.max(a, base), hi = Math.min(b, end - 1);
      out.set(new Uint8Array(await readRange(p.file, lo - base, hi - base)), lo - a);
    }
    base = end;
  }
  return out.buffer;
}
async function fetchCell(c) {
  const a = cells[c] * rowBytes, b = cells[c + 1] * rowBytes - 1;
  ex.mark_requested(c); stream.inflight++;
  try {
    const buf = await readVectors(a, b);
    const ptr = ex.in_buffer(buf.byteLength);
    new Uint8Array(mem(), ptr, buf.byteLength).set(new Uint8Array(buf));
    if (!ex.load_cell(c, state.frame)) stream.cacheFull = performance.now();
    stream.bytes += buf.byteLength;
  } catch (e) {
    ex.mark_failed(c); console.warn('tile', c, e);
  } finally { stream.inflight--; }
}
function schedule(dt) {
  const bw = +$('bw').value * 1e6;
  stream.bucket = bw ? Math.min(stream.bucket + bw * dt, bw * 0.25) : Infinity;
  const nreq = new Float32Array(mem(), ex.ptr_stats(), 16)[5] | 0;
  if (!nreq) return;
  const req = new Uint32Array(mem(), ex.ptr_req(), nreq).slice();
  for (const c of req) {
    if (stream.inflight >= 8) break;
    const bytes = (cells[c + 1] - cells[c]) * rowBytes;
    if (bytes > stream.bucket) break;
    stream.bucket -= bytes;
    fetchCell(c);
  }
}

// ---------------- autopilot ----------------
const ap = { next: 0, from: null, to: null, t0: 0 };
function autopilot(now) {
  if (now >= ap.next) {
    let tgt = null;
    for (let k = 0; k < 30 && !tgt; k++) {
      const s = toScreen((Math.random() * N) | 0);
      if (s && s[0] > W * 0.25 && s[0] < W * 0.75 && s[1] > H * 0.2 && s[1] < H * 0.8) tgt = s;
    }
    ap.from = { ...state.gaze }; ap.to = tgt ? { x: tgt[0], y: tgt[1] } : { ...state.gaze }; ap.t0 = now;
    ap.next = now + 1400 + Math.random() * 1400;
  }
  const t = Math.min(1, (now - ap.t0) / 80), e = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
  state.gaze.x = ap.from.x + (ap.to.x - ap.from.x) * e; state.gaze.y = ap.from.y + (ap.to.y - ap.from.y) * e;
}

// ---------------- frame ----------------
let lastT = performance.now(), statT = 0, fps = 60, kms = 0, lastShow = 0;
function tick(now) {
  requestAnimationFrame(tick);
  const dt = Math.min(0.1, (now - lastT) / 1000); lastT = now;
  if (state.fly) {
    const f = state.fly, t = Math.min(1, (now - f.t0) / 700), e = 1 - (1 - t) ** 3;
    state.cam.target = f.from.map((v, i) => v + (f.to[i] - v) * e); state.cam.dist = f.d0 + (f.d1 - f.d0) * e;
    if (t >= 1) state.fly = null;
  }
  const shiftTo = state.mode === 'forward' && fwd ? fwd.sceneShift() : 0;
  state.shift += (shiftTo - state.shift) * Math.min(1, dt * 6);
  // Forward mode: fly through along the thought path when it's on, otherwise a slow orbit (an idle
  // drift behind the query box, and a slower turn while a pass plays). Either yields to the user.
  const goal = state.mode === 'forward' && fwd ? fwd.flyGoal() : null;
  if (state.mode === 'forward' && !state.fly && now - state.lastInput > 1500) {
    if (goal) followFlight(goal, dt); else state.cam.yaw += dt * (running() ? 0.035 : 0.06);
  }
  updateCamera();
  if (state.auto && state.mode === 'embed' && now - state.lastInput > 4000 && state.pinned < 0) autopilot(now);
  const target = state.lens && state.fov ? 1 : 0;
  state.lensMix += (target - state.lensMix) * Math.min(1, dt * 6);
  new Float32Array(mem(), ex.ptr_mat(), 16).set(MVP);
  ex.set_view(projF * H / 2);
  const r0 = +$('rFov').value * dpr, r1 = Math.max(r0 * 1.4, +$('rPara').value * dpr);
  state.frame++;
  const t0 = performance.now();
  const fovOn = state.fov && state.mode === 'embed';
  ex.frame(W, H, state.gaze.x, state.gaze.y, r0, r1, (fovOn ? 1 : 0) | (fovOn && state.lensMix > 0.001 ? 2 : 0), state.lensMix, state.frame);
  kms += (performance.now() - t0 - kms) * 0.1;
  const st = new Float32Array(mem(), ex.ptr_stats(), 16);
  const nf = st[0] | 0;
  state.lensShare = st[12];  // share of the fovea rows' variation its local PCA shows (-1: none)

  // draw
  gl.viewport(0, 0, W, H); gl.clearColor(8 / 255, 8 / 255, 14 / 255, 1); gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(progPts);
  gl.uniformMatrix4fv(gl.getUniformLocation(progPts, 'u_mvp'), false, MVP);
  gl.uniform1f(gl.getUniformLocation(progPts, 'u_dpr'), dpr);
  const bright = +$('cloudBright').value / 100;
  gl.uniform1f(gl.getUniformLocation(progPts, 'u_dim'), (state.mode === 'forward' ? (running() ? 0.28 : 0.55) * fwd.cloudDim() : 1) * bright);  // the hidden-state space dims the words
  gl.uniform2f(gl.getUniformLocation(progPts, 'u_res'), W, H);
  gl.uniform2f(gl.getUniformLocation(progPts, 'u_gaze'), state.gaze.x, state.gaze.y);
  gl.uniform1f(gl.getUniformLocation(progPts, 'u_r1'), r1);
  gl.uniform1f(gl.getUniformLocation(progPts, 'u_lensDim'), fovOn ? 0.22 : 1.0);
  gl.uniform1f(gl.getUniformLocation(progPts, 'u_density'), Math.min(1, Math.max(0.35, Math.sqrt(60000 / N))));
  gl.bindVertexArray(vaoPts);
  let attr = new Uint8Array(mem(), ex.ptr_attr(), N * 2);
  if (kd && kd.lit) {  // the mask: keys that can't be attended to fade, in place
    if (!maskAttr) maskAttr = new Uint8Array(N * 2);
    maskAttr.set(attr); const lit = kd.lit;
    for (let i = 0; i < N; i++) if (!lit[i]) maskAttr[2 * i] >>= 3;
    attr = maskAttr;
  }
  gl.bindBuffer(gl.ARRAY_BUFFER, attrBuf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, attr);
  gl.drawArrays(gl.POINTS, 0, N);
  if (nf) {
    const fi = new Uint32Array(mem(), ex.ptr_fidx(), nf), fd = new Float32Array(mem(), ex.ptr_fdisp(), nf * 2);
    const v = new Float32Array(nf * 3);
    for (let f = 0; f < nf; f++) { v[3 * f] = fd[2 * f]; v[3 * f + 1] = fd[2 * f + 1]; v[3 * f + 2] = cats[fi[f]]; }
    gl.useProgram(progFov);
    gl.uniform2f(gl.getUniformLocation(progFov, 'u_res'), W, H);
    gl.uniform1f(gl.getUniformLocation(progFov, 'u_dpr'), dpr);
    gl.bindVertexArray(vaoFov);
    gl.bindBuffer(gl.ARRAY_BUFFER, fovBuf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, v);
    gl.drawArrays(gl.POINTS, 0, nf);
  }
  gl.bindVertexArray(null);

  if (fovOn) schedule(dt);
  state.focus = state.pinned >= 0 ? state.pinned : (st[4] | 0);
  if (st[4] < 0 && state.pinned < 0) state.focus = -1;
  if (state.mode === 'forward') { ov.clearRect(0, 0, W, H); fwd.drawOverlay(ov, W, H); fwd.draw(now); }
  else drawOverlay(r0, r1, nf);
  if (state.focus !== state.shownFor && now - lastShow > 120) { lastShow = now; showInspector(state.focus); }
  else if (state.focus >= 0 && !inspRowShown && ex.row(state.focus)) showInspector(state.focus);

  fps += (1 / Math.max(dt, 1e-3) - fps) * 0.05;
  stream.t += dt;
  if (now - statT > 300) { statT = now; updateHud(st); }
}

// ---------------- overlay ----------------
const ov = $('ov').getContext('2d');
const disp = (s) => s.replace(/^ /, '␣').replace(/\n/g, '↵').replace(/\t/g, '⇥')
  .replace(/[\x00-\x08\x0b-\x1f\x7f-\x9f\ufffd]/g, (c) => c === '\ufffd' ? '\ufffd' : `\\x${c.charCodeAt(0).toString(16).padStart(2, '0')}`);
const deg = (px) => (Math.atan(((px / dpr) * 25.4 / 96) / 600) * 180 / Math.PI).toFixed(1);
function drawOverlay(r0, r1, nf) {
  ov.clearRect(0, 0, W, H);
  const g = state.gaze;
  ov.font = `${10.5 * dpr}px "JetBrains Mono", monospace`;
  ov.lineWidth = 1 * dpr;
  if (state.fov && state.rings) {
    ov.strokeStyle = 'rgba(53,224,230,0.55)';
    ov.beginPath(); ov.arc(g.x, g.y, r0, 0, Math.PI * 2); ov.stroke();
    ov.setLineDash([3 * dpr, 6 * dpr]); ov.strokeStyle = 'rgba(143,130,238,0.45)';
    ov.beginPath(); ov.arc(g.x, g.y, r1, 0, Math.PI * 2); ov.stroke(); ov.setLineDash([]);
    const share = state.lens && state.lensShare > 0 ? ` · local PCA shows ${Math.round(state.lensShare * 100)}%` : '';
    ov.fillStyle = 'rgba(53,224,230,0.8)'; ov.fillText(`FOVEA ${deg(r0)}°${share}`, g.x + r0 * 0.74, g.y - r0 * 0.74);
    ov.fillStyle = 'rgba(143,130,238,0.75)'; ov.fillText(`PARAFOVEA ${deg(r1)}°`, g.x + r1 * 0.72, g.y - r1 * 0.72);
  }
  // Foveal positions by point index, so neighbours inside the lens are drawn where they are shown.
  const fpos = new Map();
  if (nf) {
    const fi = new Uint32Array(mem(), ex.ptr_fidx(), nf), fd = new Float32Array(mem(), ex.ptr_fdisp(), nf * 2);
    for (let f = 0; f < nf; f++) fpos.set(fi[f], [fd[2 * f], fd[2 * f + 1]]);
    // Labels: nearest to the gaze first, one per 110x20 px bin.
    const order = [...fpos.entries()].map(([i, p]) => [i, p, (p[0] - g.x) ** 2 + (p[1] - g.y) ** 2]).sort((a, b) => a[2] - b[2]);
    const used = new Set(); let n = 0;
    ov.textBaseline = 'middle';
    for (const [i, p] of order) {
      if (n >= (KD ? 16 : 36)) break;  // a store's labels are phrases, not tokens: fewer of them
      const key = KD ? `${Math.floor(p[0] / (190 * dpr))},${Math.floor(p[1] / (26 * dpr))}` : `${Math.floor(p[0] / (110 * dpr))},${Math.floor(p[1] / (20 * dpr))}`;
      if (used.has(key)) continue;
      used.add(key); n++;
      ov.fillStyle = i === state.focus ? '#ffffff' : 'rgba(232,230,245,0.78)';
      const lab = disp(tokens[i]);
      ov.fillText(KD && lab.length > 24 && i !== state.focus ? lab.slice(0, 23) + '…' : lab, p[0] + 7 * dpr, p[1]);
    }
  }
  if (layoutId === 'radial' && radial) {  // rings of equal cosine similarity around the pinned token
    const [R, U] = radial.basis, C = radial.centre;
    ov.lineWidth = 1 * dpr; ov.font = `${10 * dpr}px "JetBrains Mono", monospace`; ov.textBaseline = 'middle'; ov.textAlign = 'left';
    const rings = radial.i < 0 && xv ? [8, 32, 64, 128].map((k) => [1 - k / (xv.E - 1), `rank ${k}`]) : [0.9, 0.7, 0.5, 0.3, 0.1].map((c) => [c, `cos ${c}`]);
    for (const [cs, ringLabel] of rings) {
      const r = (1 - cs) * radial.scale; let first = true, lab = null;
      ov.strokeStyle = `rgba(53,224,230,${cs >= 0.5 ? 0.5 : 0.28})`; ov.beginPath();
      for (let k = 0; k <= 72; k++) {
        const th = (k / 72) * Math.PI * 2, p = project(C[0] + r * (Math.cos(th) * R[0] + Math.sin(th) * U[0]), C[1] + r * (Math.cos(th) * R[1] + Math.sin(th) * U[1]), C[2] + r * (Math.cos(th) * R[2] + Math.sin(th) * U[2]));
        if (!p) { first = true; continue; }
        if (first) ov.moveTo(p[0], p[1]); else ov.lineTo(p[0], p[1]); first = false;
        if (k === 9) lab = p;  // upper right
      }
      ov.stroke();
      if (lab) { ov.fillStyle = 'rgba(53,224,230,0.8)'; ov.fillText(ringLabel, lab[0] + 4 * dpr, lab[1]); }
    }
  }
  if (xv) xv.drawOverlay(ov, dpr, (i) => fpos.get(i) || toScreen(i));  // the last token's experts, by where they came from
  if (kd && kd.dist) {  // the question, and arcs to the page it attends to
    const c = layoutId === 'radial' && radial && radial.i < 0 ? radial.centre : kd.anchor(points);
    if (c) kd.drawOverlay(ov, dpr, project(...c), toScreen);
    if (layoutId === 'radial' && radial && radial.i < 0 && kd.cut != null) {  // the page's edge: the last atom it took
      const [R, U] = radial.basis, C = radial.centre, r = kd.cut * radial.scale; let first = true;
      ov.save(); ov.strokeStyle = 'rgba(53,224,230,.9)'; ov.setLineDash([5 * dpr, 4 * dpr]); ov.lineWidth = 1.5 * dpr; ov.beginPath();
      for (let k = 0; k <= 96; k++) {
        const th = (k / 96) * Math.PI * 2, p = project(C[0] + r * (Math.cos(th) * R[0] + Math.sin(th) * U[0]), C[1] + r * (Math.cos(th) * R[1] + Math.sin(th) * U[1]), C[2] + r * (Math.cos(th) * R[2] + Math.sin(th) * U[2]));
        if (!p) { first = true; continue; } if (first) ov.moveTo(p[0], p[1]); else ov.lineTo(p[0], p[1]); first = false;
        if (k === 60) { ov.fillStyle = 'rgba(53,224,230,.95)'; ov.fillText(`page edge · top ${kd.page.length}`, p[0] + 4 * dpr, p[1] + 10 * dpr); }
      }
      ov.stroke(); ov.restore();
    }
  }
  if (layoutId === 'axes' && axes && axes.shares) {  // name the axes at their ends
    ov.font = `${11 * dpr}px "JetBrains Mono", monospace`; ov.textBaseline = 'middle'; ov.textAlign = 'center';
    const ends = [[[1.15, 0, 0], axes.words[0][0]], [[-1.15, 0, 0], axes.words[0][1]], [[0, 1.15, 0], axes.words[1][0]], [[0, -1.15, 0], axes.words[1][1]]];
    ov.strokeStyle = 'rgba(232,230,245,0.25)'; ov.lineWidth = 1 * dpr; ov.setLineDash([4 * dpr, 6 * dpr]);
    for (const [[x, y, z], w] of ends) {
      const p = project(x, y, z), o = project(0, 0, 0); if (!p || !o) continue;
      ov.beginPath(); ov.moveTo(o[0], o[1]); ov.lineTo(p[0], p[1]); ov.stroke();
      ov.fillStyle = 'rgba(232,230,245,0.9)'; ov.fillText(`${w} →`, p[0], p[1]);
    }
    ov.setLineDash([]); ov.textAlign = 'left';
  }
  const f = state.focus;
  if (f >= 0) {
    const fp = fpos.get(f) || toScreen(f);
    if (fp) {
      ov.strokeStyle = 'rgba(53,224,230,0.55)';
      const nb = nbrs(f), kk = nb ? nb.idx.length : 0;
      for (let k = 0; k < kk; k++) {
        const j = nb.idx[k], p = fpos.get(j) || toScreen(j);
        if (!p) continue;
        ov.globalAlpha = 0.25 + 0.6 * (1 - k / kk);
        ov.beginPath(); ov.moveTo(fp[0], fp[1]); ov.lineTo(p[0], p[1]); ov.stroke();
        ov.beginPath(); ov.arc(p[0], p[1], 3 * dpr, 0, Math.PI * 2); ov.stroke();
      }
      ov.globalAlpha = 1;
      ov.strokeStyle = '#fff'; ov.lineWidth = 1.5 * dpr;
      ov.beginPath(); ov.arc(fp[0], fp[1], 7 * dpr, 0, Math.PI * 2); ov.stroke();
      if (state.pinned >= 0) { ov.beginPath(); ov.arc(fp[0], fp[1], 11 * dpr, 0, Math.PI * 2); ov.stroke(); }
    }
  }
}

// ---------------- neighbours ----------------
// Precomputed (knn.u32 + knncos.f16) for prebuilt data sets; computed on demand by
// knn-worker.js for models built in the browser. nbrs(i) -> { idx, cos } or null (asked for).
const lazyNbrs = new Map();
let knnWorker = null;
function startKnnWorker() {
  knnWorker = new Worker('knn-worker.js', { type: 'module' });
  knnWorker.onmessage = ({ data: m }) => {
    if (m.type === 'pca') { fid.pca = m.share; saveFid(); showFid(); return; }
    if (m.type === 'rank') { for (const [id, rho] of Object.entries(m.rho)) fid[`rank:${id}`] = rho; saveFid(); showFid(); return; }
    if (m.type === 'radial') { placeRadial(m.i, m.dist); return; }
    if (m.type === 'axes') { if (!axes) return; axes.shares = m.shares; layouts.axes = m.points; $('axesMsg').textContent = ''; syncLayoutUi(); setLayout('axes'); return; }
    lazyNbrs.set(m.i, { idx: m.idx, cos: m.cos });
    if (state.shownFor === m.i) state.shownFor = -2;  // redraw the inspector with them
  };
  knnWorker.postMessage({ type: 'load', base: DATA });
}
function nbrs(i) {
  if (knn) return { idx: knn.subarray(i * meta.k, (i + 1) * meta.k), cos: Float32Array.from(knncos.subarray(i * meta.k, (i + 1) * meta.k), half) };
  const got = lazyNbrs.get(i);
  if (got) return got;
  if (got === undefined && knnWorker) { lazyNbrs.set(i, null); knnWorker.postMessage({ type: 'knn', i, k: meta.k }); }
  return null;
}

// ---------------- inspector ----------------
let inspRowShown = false;
const half = (h) => { const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023; return e === 0 ? s * 2 ** -14 * (m / 1024) : e === 31 ? NaN : s * 2 ** (e - 15) * (1 + m / 1024); };
function showInspector(i) {
  state.shownFor = i;
  $('inspEmpty').hidden = i >= 0; $('inspBody').hidden = i < 0;
  if (i < 0) return;
  const s = tokens[i];
  $('iToken').innerHTML = `“${escapeHtml(s).replace(/^ /, '<span class="sp">␣</span>').replace(/\n/g, '<span class="sp">↵</span>')}”`;
  $('iCat').textContent = KD || EX ? meta.categories[cats[i]] : CAT_LABELS[meta.categories[cats[i]]] || meta.categories[cats[i]];
  $('iKd').hidden = !kd && !xv; if (kd && kd.store) $('iKd').innerHTML = kd.inspectHtml(i); else if (xv) $('iKd').innerHTML = xv.inspectHtml(i);
  $('iPin').textContent = state.pinned >= 0 ? 'Pinned' : 'Following gaze';
  $('iPin').className = 'pill ' + (state.pinned >= 0 ? 'cyan' : 'ghost');
  $('iId').textContent = idOf(i);
  $('iNorm').textContent = norms[i].toFixed(3);
  const nb = [], near = nbrs(i);
  if (!near) nb.push('<li class="note">Searching the full space…</li>');
  for (let k = 0; near && k < near.idx.length; k++) {
    const j = near.idx[k], cs = near.cos[k];
    nb.push(`<li><button type="button" data-i="${j}"><span>${escapeHtml(disp(tokens[j]))}<span class="bar" style="width:${Math.max(4, cs * 100).toFixed(0)}%"></span></span><span class="cos">${cs.toFixed(3)}</span></button></li>`);
  }
  $('iNbrs').innerHTML = nb.join('');
  drawBarcode(i);
}
let idsCache = null;
function idOf(i) { return idsCache ? idsCache[i] : '…'; }
const loadIds = () => fetchBuf('ids.u32').then((b) => { idsCache = new Uint32Array(b); if (state.shownFor >= 0) $('iId').textContent = idsCache[state.shownFor]; }).catch(() => {});
function escapeHtml(s) { return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

function drawBarcode(i) {
  const cv = $('barcode'), c = cv.getContext('2d');
  c.clearRect(0, 0, cv.width, cv.height);
  const ptr = ex.row(i);
  inspRowShown = !!ptr;
  if (!ptr) {
    $('iRow').textContent = state.fov ? 'Keep looking here and its coordinates load.' : 'Turn foveation on (F) to load its coordinates.';
    $('iCheck').textContent = '';
    return;
  }
  const v = new Float32Array(mem(), ptr, D).slice();
  let mx = 1e-9; for (const x of v) mx = Math.max(mx, Math.abs(x));
  const img = c.createImageData(D, 1);
  for (let k = 0; k < D; k++) {
    const t = v[k] / mx, a = Math.min(1, Math.abs(t) ** 0.6);
    const col = t >= 0 ? [53, 224, 230] : [143, 110, 255];
    img.data[4 * k] = 8 + col[0] * a; img.data[4 * k + 1] = 8 + col[1] * a; img.data[4 * k + 2] = 14 + col[2] * a; img.data[4 * k + 3] = 255;
  }
  for (let y = 0; y < cv.height; y++) c.putImageData(img, 0, y);
  let arg = 0; for (let k = 1; k < D; k++) if (Math.abs(v[k]) > Math.abs(v[arg])) arg = k;
  $('iRow').textContent = `Largest coordinate: number ${arg}, at ${v[arg].toFixed(3)}.`;
  // Throughput check: exact kNN over the rows resident right now, in wasm.
  const t0 = performance.now(), got = ex.knn_resident(i, meta.k), ms = performance.now() - t0;
  const res = new Uint32Array(mem(), ex.ptr_knn(), got), near = nbrs(i);
  const resident = (new Float32Array(mem(), ex.ptr_stats(), 16)[8] / (D * 4)) | 0;
  if (!near || !document.body.classList.contains('stats')) { $('iCheck').textContent = ''; return; }
  const truth = new Set(near.idx);
  let hit = 0; for (const j of res) if (truth.has(j)) hit++;
  $('iCheck').textContent = `WebAssembly${simd ? ' SIMD' : ''} searched the ${resident.toLocaleString()} rows in memory in ${ms.toFixed(1)} ms and found ${hit} of these ${near.idx.length}.`;
}
$('iNbrs').addEventListener('click', (e) => {
  const b = e.target.closest('button[data-i]'); if (!b) return;
  const j = +b.dataset.i; state.pinned = j; state.shownFor = -2; flyTo(j);
});

// ---------------- HUD ----------------
function updateHud(st) {
  const mb = (b) => (b / 1e6).toFixed(b >= 1e7 ? 0 : 1);
  $('sFps').innerHTML = `${fps.toFixed(0)} <small>fps</small>`;
  $('sFov').innerHTML = `${st[0] | 0} <small>rows${st[3] ? ` · ${st[3] | 0} pending` : ''}</small>`;
  const rate = stream.bytes / Math.max(stream.t, 1e-3); stream.bytes = 0; stream.t = 0;
  stream.rate += (rate - stream.rate) * 0.5;
  $('sStream').innerHTML = state.fov ? `${mb(stream.rate)} <small>MB/s${stream.noRange ? ' · no range support' : ''}</small>` : '<small>off</small>';
  const full = performance.now() - stream.cacheFull < 1500;
  $('sCache').innerHTML = `${mb(st[8])} <small>/ ${mb(st[9])} MB${full ? ' · full' : ''}</small>`;
  const cs = new Uint8Array(mem(), ex.ptr_cstate(), NC);
  let res = 0; for (let c = 0; c < NC; c++) if (cs[c] === 2) res++;
  $('sTiles').innerHTML = `${res} <small>resident · ${(st[7] | 0)}/${(st[6] | 0)} wanted</small>`;
  $('sKernel').innerHTML = `${kms.toFixed(2)} <small>ms${simd ? ' · SIMD' : ''}</small>`;
  $('modePill').textContent = state.mode === 'forward' ? 'Forward pass' : state.fov ? (state.lens ? 'Foveated · lens' : 'Foveated') : 'Full detail';
  $('modePill').className = 'pill ' + (state.mode === 'forward' || state.fov ? 'cyan' : 'warn');
  const r0 = +$('rFov').value, r1 = +$('rPara').value;
  $('rFovOut').textContent = `${r0}px · ${deg(r0 * dpr)}°`; $('rParaOut').textContent = `${r1}px · ${deg(r1 * dpr)}°`;
}

// ---------------- input ----------------
const canvas = $('gl');
const pointers = new Map();
let drag = null, pinch = null, moved = 0;
const pos = (e) => ({ x: e.clientX * dpr, y: e.clientY * dpr });
canvas.addEventListener('pointerdown', (e) => {
  canvas.setPointerCapture(e.pointerId); pointers.set(e.pointerId, pos(e));
  state.lastInput = performance.now(); moved = 0;
  if (pointers.size === 2) { const [a, b] = [...pointers.values()]; pinch = Math.hypot(a.x - b.x, a.y - b.y); drag = null; }
  else drag = { ...pos(e), pan: e.button === 2 || e.shiftKey };
  if (e.pointerType !== 'mouse') state.gaze = pos(e);
});
canvas.addEventListener('pointermove', (e) => {
  const p = pos(e); state.lastInput = performance.now();
  if (pointers.has(e.pointerId)) pointers.set(e.pointerId, p);
  if (pinch && pointers.size === 2) {
    const [a, b] = [...pointers.values()], d = Math.hypot(a.x - b.x, a.y - b.y);
    state.cam.dist = Math.max(0.15, Math.min(12, state.cam.dist * pinch / d)); pinch = d; return;
  }
  if (drag) {
    const dx = p.x - drag.x, dy = p.y - drag.y; moved += Math.abs(dx) + Math.abs(dy);
    if (drag.pan) {
      const s = state.cam.dist / H * 1.2, c = state.cam;
      const right = [Math.cos(c.yaw), 0, -Math.sin(c.yaw)];
      c.target = c.target.map((v, i) => v - right[i] * dx * s + (i === 1 ? dy * s : 0));
    } else {
      state.cam.yaw -= dx * 0.005; state.cam.pitch = Math.max(-1.45, Math.min(1.45, state.cam.pitch + dy * 0.005));
    }
    drag.x = p.x; drag.y = p.y;
  }
  state.gaze = p;
});
const up = (e) => {
  pointers.delete(e.pointerId); if (pointers.size < 2) pinch = null;
  if (drag && moved < 6 * dpr && e.type === 'pointerup' && state.mode === 'forward' && fwd && fwd.sceneClick && fwd.sceneClick()) { drag = null; return; }
  if (drag && moved < 6 * dpr && e.type === 'pointerup') {  // a click pins the focused token
    state.pinned = state.focus >= 0 && state.pinned !== state.focus ? state.focus : -1; state.shownFor = -2;
  }
  if (!pointers.size) drag = null;
};
canvas.addEventListener('pointerup', up); canvas.addEventListener('pointercancel', up);
canvas.addEventListener('contextmenu', (e) => e.preventDefault());
canvas.addEventListener('wheel', (e) => {
  e.preventDefault(); state.lastInput = performance.now();
  state.cam.dist = Math.max(0.15, Math.min(12, state.cam.dist * Math.exp(e.deltaY * 0.0012)));
}, { passive: false });

addEventListener('keydown', (e) => {
  if (e.target.closest && e.target.closest('input, select, textarea')) { if (e.key === 'Escape') e.target.blur(); return; }
  const k = e.key.toLowerCase();
  if (k === 'm') setMode(state.mode === 'embed' ? 'forward' : 'embed');
  else if (k === 'p' && state.mode === 'forward' && fwd) fwd.toggleCollapsed();
  else if (k === 't' && state.mode === 'forward' && fwd) fwd.toggleFly();
  else if (k === 'a' && state.mode === 'forward' && fwd) fwd.toggleAttr();
  else if (k === 'k' && state.mode === 'forward' && fwd) fwd.koAtGaze();
  else if (k === 's' && state.mode === 'forward' && fwd && fwd.toggleStates) fwd.toggleStates();
  else if ((k === '1' || k === '2' || k === '3') && state.mode === 'forward' && fwd && fwd.setView) fwd.setView(['field', 'river', 'flow'][+k - 1]);
  else if (k === 'f') state.fov = !state.fov;
  else if (k === 'c') chooseLayout(layoutId === 'umap' ? 'pca' : 'umap');
  else if (k === 'z' && state.mode === 'embed') toggleLocal();
  else if (k === 'd' && state.mode === 'embed') toggleRadial();
  else if (k === 'l') state.lens = !state.lens;
  else if (k === 'r') state.rings = !state.rings;
  else if (k === 'h') { state.cam = { yaw: 0.6, pitch: 0.25, dist: 2.1, target: [0, 0, 0] }; state.fly = null; }
  else if (k === '/') { e.preventDefault(); (state.mode === 'forward' ? $('q') : $('find')).focus(); }
  else if (k === 'escape') { state.pinned = -1; state.shownFor = -2; if (layoutId === 'radial') toggleRadial(); }
  else return;
  state.lastInput = performance.now();
});

$('findForm').addEventListener('submit', (e) => {
  e.preventDefault();
  const q = $('find').value; if (!q) return;
  const tries = [' ' + q, q, ' ' + q.toLowerCase(), q.toLowerCase()];
  let i = -1;
  for (const t of tries) { i = tokens.indexOf(t); if (i >= 0) break; }
  if (i < 0) { const ql = q.toLowerCase(); i = tokens.findIndex((t) => t.trim().toLowerCase().startsWith(ql)); }
  if (i < 0 && KD) { const ql = q.toLowerCase(); i = tokens.findIndex((t) => t.toLowerCase().includes(ql)); }
  if (i < 0) { $('findMsg').textContent = KD ? `No atom's label has “${q}”.` : `“${q}” is not a single GPT-2 token.`; return; }
  $('findMsg').textContent = `Found “${disp(tokens[i])}”.`;
  state.pinned = i; state.shownFor = -2; flyTo(i); $('find').blur();
});
$('optAuto').addEventListener('change', (e) => { state.auto = e.target.checked; });
$('cache').addEventListener('change', () => ex.set_cache(+$('cache').value * 1e6));
$('flush').addEventListener('click', () => { ex.flush_cache(); state.shownFor = -2; });

let fwd = null;
function setMode(mode) {
  if (mode === 'forward' && !meta.forward) mode = 'embed';
  state.mode = mode;
  document.body.classList.toggle('fwd-mode', mode === 'forward');
  $('modeEmbed').setAttribute('aria-selected', mode === 'embed'); $('modeFwd').setAttribute('aria-selected', mode === 'forward');
  const View = meta.model.family === 'masked-diffusion' ? DiffusionView : ForwardView;
  if (!fwd && View === DiffusionView) { $('fwdCount').value = 32; $('fwdSample').value = 'sample'; }
  if (!fwd) fwd = new View({ $, DATA, tokens, points, cats, dpr, project, disp, flyToPoint, pointer: () => (performance.now() - state.lastInput < 5000 ? state.gaze : null), arch: meta.model.arch || 'gpt2', repo: hfRepo, family: meta.model.family,
    modelName: meta.model.name.replace(/\s*\(.*\)$/, ''), modelMB: Math.round((meta.forward_bytes || 250e6) / 1e6) });
  if (mode === 'forward') { fwd.enter(); if (!fwd.running) setTimeout(() => $('q').focus(), 50); } else fwd.exit();
}
// Info boxes (the small i): their text is shown in one floating layer on the page, placed
// beside the icon and kept on screen, so a scrolling panel or drawer never cuts it off.
const tipLayer = document.createElement('div');
tipLayer.id = 'tipLayer'; tipLayer.hidden = true; tipLayer.setAttribute('role', 'tooltip');
document.body.append(tipLayer);
function showTip(icon) {
  const tip = icon.querySelector('.tip'); if (!tip) return;
  tipLayer.innerHTML = tip.innerHTML; tipLayer.hidden = false;
  const r = icon.getBoundingClientRect(), w = tipLayer.offsetWidth, h = tipLayer.offsetHeight, m = 8;
  let x = r.right + m;
  if (x + w > innerWidth - m) x = r.left - m - w;  // no room on the right: open to the left
  x = Math.max(m, Math.min(x, innerWidth - m - w));
  const y = Math.max(m, Math.min(r.top - 10, innerHeight - m - h));
  tipLayer.style.left = `${x}px`; tipLayer.style.top = `${y}px`;
}
const hideTip = () => { tipLayer.hidden = true; };
document.addEventListener('pointerover', (e) => { const i = e.target.closest && e.target.closest('.info'); if (i) showTip(i); });
document.addEventListener('pointerout', (e) => { const i = e.target.closest && e.target.closest('.info'); if (i && !i.contains(e.relatedTarget)) hideTip(); });
document.addEventListener('focusin', (e) => { const i = e.target.closest && e.target.closest('.info'); if (i) showTip(i); });
document.addEventListener('focusout', (e) => { if (e.target.closest && e.target.closest('.info')) hideTip(); });
document.addEventListener('scroll', hideTip, true);

// Dot brightness: applies in both views, remembered.
try { const b = localStorage.getItem('lf-bright'); if (b) $('cloudBright').value = b; } catch {}
const showBright = () => { $('cloudBrightOut').textContent = `${$('cloudBright').value}%`; };
$('cloudBright').addEventListener('input', () => { showBright(); try { localStorage.setItem('lf-bright', $('cloudBright').value); } catch {} });
$('localFit').addEventListener('click', toggleLocal);
setInterval(() => { syncLocalUi(); syncRadialUi(); }, 500);  // the count under the parafovea, and the pinned token, change as you look around
showBright();

$('modeEmbed').addEventListener('click', () => setMode('embed'));
$('modeFwd').addEventListener('click', () => setMode('forward'));
const running = () => state.mode === 'forward' && fwd && fwd.running;
$('query').addEventListener('submit', (e) => { e.preventDefault(); $('q').blur(); fwd.submit($('q').value); });
$('q').addEventListener('input', () => fwd && fwd.prefetch());
$('qExamples').addEventListener('click', (e) => {
  const b = e.target.closest('button'); if (!b) return;
  $('q').value = b.textContent; fwd.submit(b.textContent);
});

boot().catch((err) => { $('loadMsg').textContent = `Could not start: ${err.message}`; console.error(err); });
