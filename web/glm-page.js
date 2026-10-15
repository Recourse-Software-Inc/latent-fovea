// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// The GLM-5.2 page (glm.html): opens the model folder, runs the worker, draws the expert grid.
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const w = new Worker('glm/glm-worker.js', { type: 'module' });
let drafts = [], draftStats = { right: 0, n: 0 };
let info = null, running = false, out = [], pending = '', session = { tokens: 0, hits: 0, misses: 0, read: 0 };
const heat = new Map();  // layer*E+e -> times routed
const resident = new Set();
let lastExperts = null, lastTokens = [];

// ---- the folder: remembered between visits (IndexedDB), permission asked again when needed
const db = () => new Promise((res, rej) => { const r = indexedDB.open('lf-glm', 1); r.onupgradeneeded = () => r.result.createObjectStore('h'); r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error); });
const saveHandle = async (h) => { try { const d = await db(); d.transaction('h', 'readwrite').objectStore('h').put(h, 'dir'); } catch {} };
const loadHandle = async () => { try { const d = await db(); return await new Promise((res) => { const q = d.transaction('h').objectStore('h').get('dir'); q.onsuccess = () => res(q.result || null); q.onerror = () => res(null); }); } catch { return null; } };

function openWith(msg) {
  $('pick').disabled = true; $('bar').hidden = false; $('msg').textContent = 'Reading the shards\' headers…';
  w.postMessage({ type: 'open', expertBudget: +$('pool').value * 1e9, ramBudget: +(params.get('ram') ?? $('ramSel').value) * 1e9, prefetchAhead: +(params.get('ahead') ?? $('aheadSel').value), ...msg });
}
$('pick').onclick = async () => {
  try {
    let h = await loadHandle();
    if (h && (await h.requestPermission({ mode: 'read' })) !== 'granted') h = null;
    if (!h) { h = await showDirectoryPicker({ id: 'glm52', mode: 'read' }); await saveHandle(h); }
    openWith({ dir: h });
  } catch (e) { if (e.name !== 'AbortError') $('msg').textContent = e.message; }
};
(async () => {
  // Apple GPUs share one memory with the CPU: a RAM tier there takes the same gigabytes the GPU cache
  // would use (and pushes the machine into swap). Off by default; give the GPU cache more instead.
  try {
    const ad = await navigator.gpu?.requestAdapter();
    if (ad && /apple/i.test(ad.info?.vendor || '')) {
      $('ramSel').value = '0';
      $('memNote').hidden = false;
      $('memNote').textContent = 'This GPU shares its memory with the rest of the computer, so a separate RAM tier would only take space from the GPU cache: it is off. Leave room for your other apps: the page needs about 11 GB plus the cache you choose.';
    }
  } catch {}
  if (params.get('data')) {  // tests: a container served over HTTP
    // ?shards=N names out-00000…: a whole container served by the dev server (tests on the real model)
    const shards = +params.get('shards') || 0;
    const names = shards ? ['config.json', 'tokenizer.json', '.coli_usage', ...Array.from({ length: shards }, (_, i) => `out-${String(i).padStart(5, '0')}.safetensors`)]
      : (params.get('files') || 'config.json,out-00000.safetensors,tokenizer.json').split(',');
    return openWith({ url: new URL(`data/${params.get('data')}/`, location.href).href, names });  // absolute: the worker lives in glm/
  }
  const h = await loadHandle();
  if (h) { $('pick').textContent = `Open “${h.name}” again`; if ((await h.queryPermission({ mode: 'read' })) === 'granted') openWith({ dir: h }); }
  if (!window.showDirectoryPicker) { $('pick').disabled = true; $('msg').textContent = 'This needs a browser that can read a folder you pick (Chrome or Edge).'; }
})();

// ---- worker messages
w.onmessage = ({ data: m }) => {
  if (m.type === 'progress') {
    $('bar').value = m.total ? m.done / m.total : 0;
    $('msg').textContent = m.stage === 'prewarm' ? `Warming the cache: ${m.done} of ${m.total} experts` : `Loading the resident part onto the GPU: ${m.done} of ${m.total} weights`;
  } else if (m.type === 'ready') {
    info = m; $('setup').hidden = true; $('query').hidden = false; $('side').hidden = false; $('expCard').hidden = false;
    $('warm').hidden = !m.hasUsage;
    $('kicker').textContent = `GLM-5.2 · ${m.L} layers · ${m.E} experts per layer, ${m.K} per token · running in your browser`;
    $('sub').textContent = `${(m.gpuBytes / 1e9).toFixed(1)} GB on the GPU (the resident part, plus room for ${m.nSlots.toLocaleString()} experts of ${(m.slotBytes / 1e6).toFixed(1)} MB), and room for ${(m.ramPer * (m.L - m.dense)).toLocaleString()} more in RAM. Context up to ${m.maxT.toLocaleString()} tokens.`;
    drawExperts(); setTimeout(() => $('q').focus(), 50);
  } else if (m.type === 'prewarmed') {
    $('warm').disabled = false; $('warm').textContent = `Cache warmed: ${m.resident} experts on the GPU`; $('msg').textContent = '';
  } else if (m.type === 'drafts') { drafts = m.words; renderAnswer(); }
  else if (m.type === 'prefill') $('answer').innerHTML = `<span class="p">Reading your question (${m.n} tokens at once): layer ${m.layer + 1} of ${m.L}…</span>`;
  else if (m.type === 'token') onToken(m);
  else if (m.type === 'done') { running = false; $('go').textContent = 'Run'; $('tps').textContent = `${m.tokens} tokens in ${(m.ms / 1000).toFixed(0)} s` + (draftStats.n ? ` · previews right first time: ${draftStats.right}/${draftStats.n}` : ''); renderAnswer(); }
  else if (m.type === 'error') { running = false; $('go').textContent = 'Run'; $('pick').disabled = false; $('msg').textContent = m.message; $('answer').textContent += `\n[${m.message}]`; }
};
$('warm').onclick = () => { $('warm').disabled = true; $('warm').textContent = 'Warming…'; w.postMessage({ type: 'prewarm' }); };

// ---- asking
const template = (text) => `[gMASK]<sop><|user|>${text}<|assistant|><think></think>`;
$('query').onsubmit = (e) => {
  e.preventDefault();
  if (running) { w.postMessage({ type: 'stop' }); return; }
  const text = $('q').value.trim(); if (!text || !info) return;
  running = true; $('go').textContent = 'Stop'; $('answerCard').hidden = false; $('answer').innerHTML = '<span class="p">Reading your question…</span>';
  out = []; lastTokens = [];
  const depths = []; for (let d = 0; d <= info.L; d += 13) depths.push(d); if (depths[depths.length - 1] !== info.L) depths.push(info.L);
  w.postMessage({ type: 'generate', prompt: template(text), max: +(params.get('max') || 128), temperature: 0, lensDepths: depths, draftAhead: +(params.get('draft') ?? $('draftSel').value) });
  drafts = []; draftStats = { right: 0, n: 0 };
};
$('ex').onclick = (e) => { const b = e.target.closest('button'); if (b) { $('q').value = b.textContent; $('query').requestSubmit(); } };

const chan = new BroadcastChannel('lf-glm');  // an open expert cloud (index.html, GLM-5.2 experts) follows along
function onToken(m) {
  if (m.next != null) chan.postMessage({ type: 'token', text: m.text, experts: m.experts, scores: m.scores });  // the token whose routing this is
  session.hits += m.hits + m.ramHits + m.preHits; session.misses += m.misses; session.read += m.readBytes; session.ram = m.ramBytes;
  lastTokens.push(m.text);
  lastExperts = m.experts;
  m.experts.forEach((ch, l) => ch.forEach(([e, , tier]) => { const k = l * info.E + e; heat.set(k, (heat.get(k) || 0) + 1); resident.add(k); }));
  if (m.next != null) {
    session.tokens++;
    if (!m.prompt) out.push(m.text);
    if (drafts.length && m.nextText != null && !m.prompt) { draftStats.n++; if (drafts[0] === m.nextText) draftStats.right++; }  // the preview's first word vs the exact one
    drafts = []; lastNext = running ? m.nextText || '' : '';
    renderAnswer();
    if (m.lens) drawLens(m.lens, m.depths);
    if (m.att) drawAtt(m.att);
  } else $('answer').innerHTML = `<span class="p">Reading your question: token ${m.pos + 1}…</span>`;
  $('sMs').textContent = (m.ms / 1000).toFixed(1);
  $('sHit').textContent = `${m.hits} / ${m.preHits} / ${m.ramHits} / ${m.misses}`;
  session.pred = (session.pred || 0) + m.predicted; session.right = (session.right || 0) + m.predictedRight;
  session.pre = (session.pre || 0) + m.prefetched; session.preUsed = (session.preUsed || 0) + m.preHits;
  $('tAhead').textContent = session.pred ? `${Math.round(100 * session.right / session.pred)}% · ${session.preUsed}/${session.pre}` : '–';
  $('sRead').textContent = (m.readBytes / 1e9).toFixed(2);
  $('sRate').textContent = m.readMs > 0 ? Math.round(m.readBytes / 1e6 / (m.readMs / 1000)) : '–';
  $('tTok').textContent = session.tokens;
  $('tHit').textContent = session.hits + session.misses ? `${Math.round(100 * session.hits / (session.hits + session.misses))}%` : '–';
  $('tRead').textContent = (session.read / 1e9).toFixed(1);
  $('tPool').textContent = ((session.ram || 0) / 1e9).toFixed(1);
  $('expInfo').textContent = `${m.hits + m.ramHits + m.misses} experts · ${m.ramHits} from RAM · ${m.misses} read from the SSD`;
  drawExperts();
}
let lastNext = '';
function renderAnswer() {  // exact words sharp; the next exact word underlined; the preview faint
  $('answer').innerHTML = esc(out.join('')) + (lastNext ? `<span class="now">${esc(lastNext)}</span>` : '')
    + (drafts.length && running ? `<span class="draft" title="Preview: drafted from the experts already on the GPU; replaced by the exact words as they arrive">${esc(drafts.join(''))}</span>` : '');
  if (draftStats.n) $('tps').textContent = `previews right first time: ${draftStats.right}/${draftStats.n}`;
}
const esc = (s) => s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

// ---- views
function drawExperts() {
  const cv = $('experts'), c = cv.getContext('2d'), E = info.E, rows = info.L - info.dense;
  const cw = cv.width / E, ch = cv.height / rows;
  c.fillStyle = '#06060b'; c.fillRect(0, 0, cv.width, cv.height);
  let mx = 1; for (const v of heat.values()) mx = Math.max(mx, v);
  for (const [k, v] of heat) {
    const l = Math.floor(k / E) - info.dense, e = k % E;
    const a = 0.15 + 0.6 * Math.sqrt(v / mx);
    c.fillStyle = `rgba(108,92,231,${a})`; c.fillRect(e * cw, l * ch, Math.max(1, cw - 0.5), Math.max(1, ch - 0.5));
  }
  if (lastExperts) lastExperts.forEach((chs, l) => chs.forEach(([e, wt, tier]) => {
    c.fillStyle = tier === 'ssd' ? '#f2b56b' : tier === 'ram' ? '#ef6fa6' : tier === 'pre' ? '#7be08a' : '#35e0e6';
    c.fillRect(e * cw - 0.5, (l - info.dense) * ch - 0.5, cw + 1, ch + 1);
  }));
}
function drawLens(lens, depths) {
  $('lens').innerHTML = lens.map((r, k) => {
    const t = r.top[0];
    return `<div><span class="d">${depths[k] === 0 ? 'emb' : 'L' + depths[k]}</span><span class="w">${esc(JSON.stringify(t.text ?? '').slice(1, -1))}</span><span class="p">${(100 * t.p).toFixed(1)}%</span></div>`;
  }).join('');
}
function drawAtt(att) {
  const cv = $('att'), c = cv.getContext('2d'), L = att.length, T = att[0].length;
  c.fillStyle = '#06060b'; c.fillRect(0, 0, cv.width, cv.height);
  const cw = cv.width / T, chh = cv.height / L;
  for (let l = 0; l < L; l++) {
    let mx = 1e-9; for (const v of att[l]) mx = Math.max(mx, v);
    for (let t = 0; t < T; t++) { const a = att[l][t] / mx; if (a < 0.02) continue; c.fillStyle = `rgba(53,224,230,${a})`; c.fillRect(t * cw, l * chh, Math.ceil(cw), Math.ceil(chh)); }
  }
}
// expert tooltip
$('experts').onmousemove = (e) => {
  if (!info) return;
  const r = e.target.getBoundingClientRect(), E = info.E, rows = info.L - info.dense;
  const ex = Math.floor((e.clientX - r.left) / r.width * E), l = Math.floor((e.clientY - r.top) / r.height * rows) + info.dense;
  const n = heat.get(l * E + ex) || 0, cur = lastExperts?.[l]?.find(([x]) => x === ex);
  $('tip').hidden = false; $('tip').style.left = e.clientX + 12 + 'px'; $('tip').style.top = e.clientY + 12 + 'px';
  $('tip').textContent = `layer ${l} · expert ${ex} · routed ${n}× this session${cur ? ` · now: weight ${cur[1].toFixed(3)}, ${cur[2] === 'ssd' ? 'read from the SSD' : 'from the GPU cache'}` : ''}`;
};
$('experts').onmouseleave = () => { $('tip').hidden = true; };
