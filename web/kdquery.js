// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// A KnowDrive store as an attention layer (data sets with meta.kind === 'kdstore'): the atoms are
// the keys, a query is scored against every one of them, and a mask decides which can be attended
// to. This module adds what a store has and a language model doesn't: the atom's passage, file and
// tags in the inspector; example queries with KnowDB's own answers (vector, full-text and hybrid
// heads); every atom's exact distance from the query (radial view); the mask, live, over the
// static keys; the recall shape; and the facet read-out (the tags the matched set carries).
//
// What is exact and what is replayed: the query vectors are recovered from KnowDB's vector scores
// (tools/kd_queries.py prints the fit), so the vector head's ranking, with or without a mask, is
// recomputed here over every atom. Full-text (BM25) needs the server's index: those lists are
// KnowDB's, replayed, and under a mask they are filtered, not refilled.
const HEADS = { vector: ['Vector', '#35e0e6'], fts: ['Full-text', '#f2b56b'], hybrid: ['Hybrid', '#b9a8ff'] };
const PAGE = 20;

export class KdStore {
  constructor(o) {
    Object.assign(this, o);  // { $, DATA, meta, tokens, cats, N, dataFetch, escapeHtml, onChange, flyTo, radialAt, layoutId, points }
    this.store = null; this.rowfile = null; this.rowseq = null; this.snips = new Map();
    this.queries = []; this.q = -1; this.dist = null; this.head = 'vector';
    this.off = new Set(); this.lit = null; this.page = [];
  }
  async init() {
    const get = async (f) => { const r = await this.dataFetch(this.DATA + 'kd/' + f); if (!r.ok) throw new Error(`${f}: HTTP ${r.status}`); return r; };
    const [st, rf, rs] = await Promise.all([get('store.json').then((r) => r.json()), get('rowfile.u32').then((r) => r.arrayBuffer()), get('rowseq.u32').then((r) => r.arrayBuffer())]);
    this.store = st; this.rowfile = new Uint32Array(rf); this.rowseq = new Uint32Array(rs);
    // Tags nearly every file carries (jurisdiction, format, licence) say nothing about a page.
    const df = new Uint32Array(st.tags.length);
    for (const f of st.files) for (const t of f.tags) df[t]++;
    this.common = new Set([...df.keys()].filter((t) => df[t] > 0.4 * st.files.length));
    try { this.queries = (await get('queries.json').then((r) => r.json())).queries; } catch { this.queries = []; }
    this.buildPanel();
  }
  // ---- inspector ----
  snippet(i) {
    const k = Math.floor(i / (this.meta.snippet_shard || 4096));
    if (!this.snips.has(k)) {
      this.snips.set(k, null);
      this.dataFetch(this.DATA + `kd/snip.${String(k).padStart(3, '0')}.json`).then((r) => r.json()).then((a) => { this.snips.set(k, a); this.onChange(); }).catch(() => {});
    }
    const a = this.snips.get(k);
    return a ? a[i - k * (this.meta.snippet_shard || 4096)] : null;
  }
  file(i) { return this.store.files[this.rowfile[i]]; }
  inspectHtml(i) {
    const f = this.file(i), raw = this.snippet(i), e = this.escapeHtml;
    // The passage as text: markup (HTML pages are stored as written) shown as its words.
    const s = raw == null ? null : raw.replace(/<[^>]*>?/g, ' ').replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&').replace(/\s+/g, ' ').trim();
    const tags = f.tags.map((t) => this.store.tags[t]).filter((t) => !/^(path|name|stem):/.test(t));
    let q = '';
    if (this.dist) {
      const d = this.dist[i], rank = this.rankOf(i);
      q = `<div class="kd-q">Distance from the query <b>${d.toFixed(3)}</b> (cosine ${(1 - d).toFixed(3)}) · ${this.lit && !this.lit[i] ? 'masked out' : `rank <b>${rank.toLocaleString()}</b> of ${this.litCount().toLocaleString()} by the vector head`}</div>`;
    }
    return `${q}<div class="kd-snip">${s == null ? '<span class="note">Loading the passage…</span>' : e(s) + (raw.length >= 240 ? '…' : '')}</div>
      <div class="kd-file" title="${e(f.path)}">${e(f.path)} <span class="note">· atom ${this.rowseq[i]}</span></div>
      <div class="kd-tags">${tags.slice(0, 12).map((t) => `<span>${e(t)}</span>`).join('')}</div>`;
  }
  // ---- mask ----
  litCount() { if (!this.lit) return this.N; let c = 0; for (let i = 0; i < this.N; i++) c += this.lit[i]; return c; }
  toggleDomain(k) {
    if (this.off.has(k)) this.off.delete(k); else this.off.add(k);
    if (!this.off.size) this.lit = null;
    else { this.lit = new Uint8Array(this.N); for (let i = 0; i < this.N; i++) this.lit[i] = this.off.has(this.cats[i]) ? 0 : 1; }
    this.rank(); this.render(); this.onChange();
  }
  // ---- queries ----
  async pick(qi) {
    this.q = qi; this.dist = null; this.page = [];
    if (qi < 0) { this.render(); this.onChange(); return; }
    this.$('kdqNote').textContent = 'Loading its distances to every atom…';
    const Q = this.queries[qi];
    const buf = await (await this.dataFetch(this.DATA + 'kd/' + Q.dist)).arrayBuffer();
    if (this.q !== qi) return;
    let d;
    if (Q.dist.endsWith('.f32')) d = new Float32Array(buf);  // full precision: near-ties keep KnowDB's order
    else { const h = new Uint16Array(buf); d = new Float32Array(h.length); for (let i = 0; i < h.length; i++) d[i] = half(h[i]); }
    this.dist = d;
    this.rank(); this.render(); this.onChange();
  }
  rank() {  // the page for the chosen head, under the mask
    if (!this.dist) { this.page = []; return; }
    const lit = this.lit, Q = this.queries[this.q];
    if (this.head === 'vector') {  // exact: every lit atom by its distance
      const top = [];
      for (let i = 0; i < this.N; i++) {
        if (lit && !lit[i]) continue;
        const d = this.dist[i];
        if (top.length < PAGE) { top.push([i, d]); top.sort((a, b) => a[1] - b[1]); } else if (d < top[PAGE - 1][1]) { top[PAGE - 1] = [i, d]; top.sort((a, b) => a[1] - b[1]); }
      }
      this.page = top.map(([i, d]) => ({ i, score: 1 - d, label: `cos ${(1 - d).toFixed(3)}` }));
    } else {  // replayed from KnowDB, filtered by the mask
      const L = Q.lists[this.head].hits.filter(([i]) => i >= 0 && (!lit || lit[i])).slice(0, PAGE);
      this.page = L.map(([i, s]) => ({ i, score: s, label: `${this.head === 'hybrid' ? 'RRF' : 'BM25'} ${s.toFixed(3)}` }));
    }
    this.cut = this.head === 'vector' && this.page.length ? 1 - this.page[this.page.length - 1].score : null;
  }
  rankOf(i) {
    const d = this.dist[i], lit = this.lit; let r = 1;
    for (let j = 0; j < this.N; j++) if (this.dist[j] < d && (!lit || lit[j])) r++;
    return r;
  }
  // Where the query sits in a cloud layout: among the atoms it is nearest to (a placement, not a
  // projection: a layout has no place for a vector it was not built from).
  anchor(P) {
    if (!this.page.length) return null;
    const c = [0, 0, 0], k = Math.min(5, this.page.length);
    for (let t = 0; t < k; t++) for (let a = 0; a < 3; a++) c[a] += P[3 * this.page[t].i + a] / k;
    return c;
  }
  // ---- panel ----
  buildPanel() {
    const $ = this.$, el = $('kdq');
    el.hidden = false;
    $('kdqPick').innerHTML = `<option value="-1">Choose a question…</option>` + this.queries.map((q, i) => `<option value="${i}">${this.escapeHtml(q.text)}</option>`).join('');
    if (!this.queries.length) $('kdqNote').textContent = 'No example queries in this data set (tools/kd_queries.py adds them).';
    $('kdqPick').addEventListener('change', () => this.pick(+$('kdqPick').value));
    $('kdqHeads').addEventListener('click', (e) => {
      const b = e.target.closest('button[data-head]'); if (!b) return;
      this.head = b.dataset.head; this.rank(); this.render(); this.onChange();
    });
    $('kdqHits').addEventListener('click', (e) => { const b = e.target.closest('button[data-i]'); if (b) this.flyTo(+b.dataset.i); });
    $('kdqRadial').addEventListener('click', () => this.radialAt());
    this.render();
  }
  render() {
    const $ = this.$, e = this.escapeHtml, Q = this.queries[this.q];
    for (const b of $('kdqHeads').querySelectorAll('button')) b.setAttribute('aria-checked', b.dataset.head === this.head);
    $('kdqBody').hidden = !this.dist;
    $('kdqRadial').hidden = !this.dist;
    if (!this.dist) { if (this.queries.length) $('kdqNote').textContent = 'Pick a question: the store scores it against every atom.'; return; }
    const fit = Q.fit;
    $('kdqNote').innerHTML = this.head === 'vector'
      ? `Exact: every ${this.lit ? 'unmasked ' : ''}atom scored here, as KnowDB scores it. The query's vector was recovered from KnowDB's own scores (largest misfit ${fit.max_error.toExponential(1)}).`
      : `KnowDB's ${HEADS[this.head][0].toLowerCase()} list, replayed${this.lit ? ' and filtered by the mask (the server would refill the page)' : ''}: ${this.head === 'fts' ? 'BM25 needs the full-text index, which stays on the server' : 'the vector and full-text lists fused by Reciprocal Rank Fusion'}.`;
    // recall shape: the page's scores, best first
    const cv = $('kdqShape'), c = cv.getContext('2d'), w = cv.width, h = cv.height;
    c.clearRect(0, 0, w, h);
    const s = this.page.map((p) => p.score), mx = Math.max(...s, 1e-9), mn = this.head === 'vector' ? 0 : 0;
    const bw = w / PAGE;
    s.forEach((v, k) => { const y = ((v - mn) / (mx - mn)) * (h - 4); c.fillStyle = HEADS[this.head][1]; c.globalAlpha = k < 5 ? 1 : 0.5; c.fillRect(k * bw + 1, h - y, bw - 2, y); });
    c.globalAlpha = 1;
    const tot = s.reduce((a, b) => a + b, 0), top5 = s.slice(0, 5).reduce((a, b) => a + b, 0);
    $('kdqShapeNote').textContent = s.length ? `Top 5 hold ${Math.round((100 * top5) / tot)}% of the page's score${this.head === 'vector' ? ` · cosine ${s[0].toFixed(3)} down to ${s[s.length - 1].toFixed(3)}` : ''}` : 'Nothing left under this mask.';
    $('kdqHits').innerHTML = this.page.slice(0, 8).map((p, k) => `<li><button type="button" data-i="${p.i}"><span>${k + 1}. ${e(this.tokens[p.i])}</span><span class="cos">${p.label}</span></button></li>`).join('');
    // facet read-out: tags carried by the page's files (what the matched set is about)
    const cnt = new Map();
    for (const p of this.page) for (const t of this.file(p.i).tags) { const n = this.store.tags[t]; if (!this.common.has(t) && !/^(path|name|stem|dir|mime):/.test(n)) cnt.set(n, (cnt.get(n) || 0) + 1); }
    const fac = [...cnt].sort((a, b) => b[1] - a[1]).slice(0, 6);
    $('kdqFacets').innerHTML = fac.map(([t, n]) => `<div><span>${e(t)}</span><i style="width:${(100 * n) / Math.max(1, this.page.length)}%"></i><b>${n}</b></div>`).join('');
  }
  // ---- overlay: arcs from the query to its page, coloured by head ----
  drawOverlay(ov, dpr, at, toScreen) {
    if (!this.dist || !at) return;
    const col = HEADS[this.head][1];
    ov.save(); ov.lineCap = 'round';
    for (let k = this.page.length - 1; k >= 0; k--) {
      const p = toScreen(this.page[k].i); if (!p) continue;
      ov.strokeStyle = col; ov.globalAlpha = 0.25 + 0.65 * (1 - k / PAGE); ov.lineWidth = (0.8 + 2.2 * (1 - k / PAGE)) * dpr;
      const mx = (at[0] + p[0]) / 2, my = (at[1] + p[1]) / 2 - 0.12 * Math.hypot(p[0] - at[0], p[1] - at[1]);
      ov.beginPath(); ov.moveTo(at[0], at[1]); ov.quadraticCurveTo(mx, my, p[0], p[1]); ov.stroke();
      ov.beginPath(); ov.arc(p[0], p[1], 3 * dpr, 0, Math.PI * 2); ov.stroke();
      if (k < 5) { ov.globalAlpha = 0.95; ov.fillStyle = col; ov.font = `${10 * dpr}px "JetBrains Mono", monospace`; ov.fillText(String(k + 1), p[0] + 5 * dpr, p[1] - 6 * dpr); }
    }
    ov.globalAlpha = 1; ov.fillStyle = '#fff'; ov.strokeStyle = col; ov.lineWidth = 2 * dpr;
    const r = 7 * dpr; ov.beginPath(); ov.moveTo(at[0], at[1] - r); ov.lineTo(at[0] + r, at[1]); ov.lineTo(at[0], at[1] + r); ov.lineTo(at[0] - r, at[1]); ov.closePath(); ov.fill(); ov.stroke();
    ov.font = `600 ${12 * dpr}px "Plus Jakarta Sans", system-ui, sans-serif`; ov.textBaseline = 'middle';
    const t = `“${this.queries[this.q].text}”`, tw = ov.measureText(t).width;
    ov.fillStyle = 'rgba(8,8,14,.8)'; ov.fillRect(at[0] + 12 * dpr, at[1] - 10 * dpr, tw + 10 * dpr, 20 * dpr);
    ov.fillStyle = '#fff'; ov.fillText(t, at[0] + 17 * dpr, at[1]);
    ov.restore();
  }
}
function half(h) { const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023; return e === 0 ? s * 2 ** -14 * (m / 1024) : e === 31 ? NaN : s * 2 ** (e - 15) * (1 + m / 1024); }
