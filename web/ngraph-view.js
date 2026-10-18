// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// The neuron graph for one word (web/ngraph.js), drawn over the page: layers left to right, each
// of the model's own neurons sized by how much of the word it carries (directly and through the
// neurons it feeds), edges by how much flows along them. Click a neuron to read it (the graph
// docks to a bar, one click away); shift-click to add it to a group, then knock the group out and
// Rerun, beside two controls: as many random neurons, and the most active ones, of the same layers.

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

export class NeuronGraph {
  constructor(view) {
    this.v = view; this.sel = new Set(); this.min = false;
    const el = document.createElement('section');
    el.id = 'ngraph'; el.hidden = true; el.setAttribute('aria-label', 'Neuron graph');
    document.body.appendChild(el);
    this.el = el;
    el.addEventListener('click', (e) => {
      if (e.target.closest('.x')) { this.close(); return; }
      if (e.target.closest('[data-act="expand"]')) { this.min = false; this.render(this.v); return; }
      if (e.target.closest('[data-act="dock"]')) { this.min = true; this.render(this.v); return; }
      const n = e.target.closest('[data-node]');
      if (n) { this.pick(n.dataset.node, e.shiftKey || e.metaKey || e.ctrlKey); return; }
      const b = e.target.closest('[data-act]'); if (!b) return;
      const act = b.dataset.act;
      if (act === 'restore') { this.restore(); return; }
      const g = this.v.graph && this.v.graph.data; if (!g) return;
      if (act === 'top') { this.sel = new Set([...g.nodes].sort((p, q) => q.total - p.total).slice(0, 4).map((x) => x.key)); this.render(this.v); }
      if (act === 'clear') { this.sel.clear(); this.render(this.v); }
      if (act === 'ko') this.knockOut([...this.sel], 'the group');
      if (act === 'random' || act === 'active') this.knockOut(this.control(act), act === 'random' ? `${this.sel.size} random neurons of the same layers` : `the ${this.sel.size} most active neurons of the same layers`);
    });
    el.addEventListener('keydown', (e) => {
      const n = e.target.closest('[data-node]');
      if (n && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); this.pick(n.dataset.node, e.shiftKey); }
    });
    addEventListener('keydown', (e) => { if (e.key === 'Escape' && !this.el.hidden && !this.min && !/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) { this.min = true; this.render(this.v); } });
  }
  pick(key, group) {
    if (group) { if (this.sel.has(key)) this.sel.delete(key); else this.sel.add(key); this.render(this.v); return; }
    const [l, i] = key.split(':').map(Number);
    this.min = true; this.render(this.v);  // dock, so the neuron's drawer is in view
    this.v.gotoNeuron(l, i);
  }
  // As many neurons as the group, from the same layers: chosen at random, or the most active there.
  control(kind) {
    // "most active" on the model the graph was traced on: the unchanged run once a test has rerun it
    const v = this.v, h = this.tracedRun(), F = v.F, out = [];
    if (kind !== 'random' && !(h && h.mlp)) return out;
    for (const k of this.sel) {
      const l = +k.split(':')[0];
      if (kind === 'random') { let i; do i = Math.floor(Math.random() * F); while (this.sel.has(`${l}:${i}`) || out.includes(`${l}:${i}`)); out.push(`${l}:${i}`); }
      else {
        const a = h.mlp.subarray(l * F, (l + 1) * F), order = Array.from({ length: F }, (_, i) => i).sort((p, q) => Math.abs(a[q]) - Math.abs(a[p]));
        out.push(`${l}:${order.find((i) => !this.sel.has(`${l}:${i}`) && !out.includes(`${l}:${i}`))}`);
      }
    }
    return out;
  }
  // One test at a time: the reader's own neuron changes are set aside (and put back by Restore);
  // head and MLP knockouts stay, and the note says so.
  // the token's capture on the model the graph was traced on, kept when its first test reran the text
  tracedRun() { const v = this.v; return this.saved ? this.traced : v.graph && v.hist[v.graph.pos]; }
  knockOut(keys, what) {
    if (!keys.length) return;
    const v = this.v;
    if (!this.saved) { this.saved = new Map(v.ko.neurons); this.traced = v.hist[v.graph.pos]; }
    v.ko.neurons.clear();
    for (const k of keys) v.ko.neurons.set(k, { s: 0, add: 0, pre: 0, label: 'knocked out' });
    this.test = new Map([...v.ko.neurons].map(([k, x]) => [k, { ...x }]));  // what this test set, to tell it from the reader's later edits
    v.ivVer++; v.rerun();
    const kept = v.ko.heads.size + v.ko.mlps.size, set = this.saved.size;
    this.note = `Knocked out ${what} (${keys.map((k) => `L${k.replace(':', ' #')}`).join(', ')}) and reran.${set ? ` Your ${set} other neuron change${set > 1 ? 's are' : ' is'} set aside until Restore.` : ''}${kept ? ` ${kept} head or MLP knockout${kept > 1 ? 's' : ''} stay on.` : ''} The unchanged run is kept as A: compare the word's probability in the readout, and check what else moved.`;
    this.render(v);
  }
  restore() {
    const v = this.v;
    if (!this.saved) return;  // nothing of the graph's to undo: the reader's own changes stay
    const kept = this.merge();
    v.ivVer++; v.rerun();
    if (kept) { this.note = `Restored your own changes, and kept ${kept === 1 ? 'the one' : `the ${kept}`} you made during the test.`; this.render(v); return; }
    this.note = v.koCount() ? 'Restored your own changes.' : 'Restored the unchanged model.';
    this.render(v);
  }
  show() { this.min = false; this.render(this.v); }
  close() { this.el.hidden = true; this.v.graph = null; this.putBack(); this.note = ''; this.v.syncUrl(); }
  // The graph is going away: its test's knockouts give way to the reader's own changes, staged
  // for the next Rerun (the change count says so), never thrown away.
  putBack() {
    if (!this.saved) return;
    const v = this.v; this.merge();
    v.ivVer++; v.nrnKey = ''; v.syncIv();
  }
  // The reader's changes from before the test, plus any neuron they changed during it (anything
  // not exactly as the test set it); the test's own knockouts go. Returns how many were kept.
  merge() {
    const v = this.v, test = this.test || new Map(), during = [];
    for (const [k, x] of v.ko.neurons) { const t = test.get(k); if (!t || t.s !== x.s || t.add !== x.add || (t.pre || 0) !== (x.pre || 0)) during.push([k, x]); }
    v.ko.neurons.clear();
    for (const [k, x] of this.saved) v.ko.neurons.set(k, x);
    for (const [k, x] of during) v.ko.neurons.set(k, x);
    this.saved = null; this.test = null; this.traced = null;
    return during.length;
  }
  render(v) {
    const G = v.graph; if (!G) { this.el.hidden = true; return; }
    const tok = (r) => esc(v.disp(v.tokens[r] || ''));
    const title = `The neurons behind “${tok(G.row)}”</b><span class="note">at “${esc(v.disp(G.at))}” (${v.tokName ? v.tokName(G.pos) : `token ${G.pos + 1}`}) · traced on ${G.ko ? `the model with ${G.ko} change${G.ko > 1 ? 's' : ''}` : 'the unchanged model'}</span>`;
    this.el.classList.toggle('min', this.min);
    if (this.min) {
      this.el.innerHTML = `<header><b>${title}<button type="button" data-act="expand">Show graph</button><button type="button" class="x" aria-label="Close">✕</button></header>`;
      this.el.hidden = false; return;
    }
    let html = `<header><b>${title}<button type="button" data-act="dock" title="Dock the graph (Esc)">Dock</button><button type="button" class="x" aria-label="Close">✕</button></header>`;
    const g = G.data;
    if (!g) { this.el.innerHTML = html + `<p class="note">${esc(G.progress || 'Tracing…')}</p>`; this.el.hidden = false; return; }
    if (g.error) { this.el.innerHTML = html + `<p class="note">${esc(g.error)}</p>`; this.el.hidden = false; return; }
    const L = v.L, W = Math.max(640, Math.min(innerWidth - 80, 1280)), Hh = Math.max(360, Math.min(innerHeight - 300, 620));
    const padL = 110, padR = 150, colW = (W - padL - padR) / Math.max(1, L - 1);
    const xOf = (l) => padL + l * colW;
    // neurons stacked per layer by their total; heads (and the embedding) as small marks at the bottom
    const byL = new Map();
    for (const n of g.nodes) { if (!byL.has(n.l)) byL.set(n.l, []); byL.get(n.l).push(n); }
    const tmax = Math.max(1e-6, ...g.nodes.map((n) => Math.abs(n.total)));
    const pos = new Map();
    for (const [l, list] of byL) {
      list.sort((p, q) => q.total - p.total);
      list.forEach((n, k) => pos.set(n.key, { x: xOf(l), y: 40 + (k + 0.5) * ((Hh - 120) / list.length) }));
    }
    const heads = [...g.heads].sort((p, q) => Math.abs(q.total) - Math.abs(p.total)).slice(0, 14);
    const hset = new Set(heads.map((h) => h.key));
    const hByL = new Map();
    for (const h of heads) { const l = h.kind === 'emb' ? -1 : h.l; hByL.set(l, (hByL.get(l) || 0) + 1); pos.set(h.key, { x: h.kind === 'emb' ? 18 : xOf(h.l) - colW * 0.3, y: Hh - 64 + ((hByL.get(l) - 1) % 3) * 18 }); }
    const emax = Math.max(1e-6, ...g.edges.map((e) => Math.abs(e.w)));
    let svg = `<svg viewBox="0 0 ${W} ${Hh}" width="${W}" height="${Hh}" role="group" aria-label="Neuron graph">`;
    for (let l = 0; l < L; l++) svg += `<text x="${xOf(l)}" y="16" class="lab" text-anchor="middle">L${l}</text>`;
    svg += `<text x="8" y="${Hh - 76}" class="lab">attention heads (what they moved here from this and earlier tokens) and the embedding</text>`;
    for (const e of [...g.edges].sort((p, q) => Math.abs(p.w) - Math.abs(q.w))) {
      if ((e.from.startsWith('h') || e.from === 'emb') && !hset.has(e.from)) continue;
      const a = pos.get(e.from), b = pos.get(e.to); if (!a || !b) continue;
      const t = Math.abs(e.w) / emax; if (t < 0.03) continue;
      const mx = (a.x + b.x) / 2;
      svg += `<path d="M${a.x},${a.y} C${mx},${a.y} ${mx},${b.y} ${b.x},${b.y}" class="${e.w >= 0 ? 'ep' : 'en'}" style="stroke-width:${(0.6 + t * 6).toFixed(2)};opacity:${(0.25 + t * 0.7).toFixed(2)}"><title>${e.w >= 0 ? '+' : ''}${e.w.toFixed(3)} logits of the word's score along this route</title></path>`;
    }
    for (const h of heads) { const p = pos.get(h.key); svg += `<rect x="${p.x - 4}" y="${p.y - 4}" width="8" height="8" transform="rotate(45 ${p.x} ${p.y})" class="${h.total >= 0 ? 'hp' : 'hn'}"><title>${esc(h.label)}: ${h.total >= 0 ? '+' : ''}${h.total.toFixed(3)} logits through the neurons shown</title></rect><text x="${p.x + 8}" y="${p.y + 4}" class="lab small">${esc(h.label)}</text>`; }
    for (const n of g.nodes) {
      const p = pos.get(n.key), r = 4 + 14 * Math.sqrt(Math.abs(n.total) / tmax), ko = v.ko.neurons.has(n.key), s = this.sel.has(n.key);
      const lbl = v.label(n.l, n.i);  // the reader's own name for it, if any: the map fills in as they read
      svg += `<g data-node="${n.key}" class="node${s ? ' sel' : ''}${ko ? ' ko' : ''}${lbl ? ' named' : ''}" tabindex="0" role="button" aria-label="L${n.l} #${n.i}${lbl ? ` ${esc(lbl)}` : ''}"><circle cx="${p.x}" cy="${p.y}" r="${r.toFixed(1)}" class="${n.total >= 0 ? 'np' : 'nn'}"></circle><title>L${n.l} #${n.i}${lbl ? ` “${esc(lbl)}”` : ''} · total ${n.total >= 0 ? '+' : ''}${n.total.toFixed(3)} logits (direct ${n.direct.toFixed(3)}) · activation ${n.act.toFixed(2)}</title><text x="${p.x + r + 3}" y="${p.y + 4}" class="lab">#${n.i}${lbl ? ` ${esc(lbl.length > 18 ? lbl.slice(0, 17) + '…' : lbl)}` : ''}${ko ? ' ✕' : ''}</text></g>`;
    }
    svg += '</svg>';
    const direct = g.nodes.reduce((s, n) => s + n.direct, 0);
    html += `<p class="note">${g.nodes.length} neurons. Their direct pushes add up to ${direct.toFixed(2)} of the ${g.mlpsTotal.toFixed(2)} logits all MLPs give “${tok(G.row)}”, out of a score of ${g.score.toFixed(2)} above the average word. Circle size: each neuron's total, directly and through the neurons it feeds. <span class="c">Cyan</span> pushes toward the word, <span class="v">violet</span> against. Click a neuron to read it (Enter from the keyboard); shift-click to group. Names you give neurons show here.</p>`;
    const n = this.sel.size, any = !!this.saved;  // only a graph test's own knockout can be undone here
    html += `<div class="fwd-row" style="gap:6px;margin:4px 0 8px;flex-wrap:wrap"><button type="button" data-act="top">Group the top 4</button><button type="button" data-act="clear" ${n ? '' : 'disabled'}>Clear group</button><button type="button" data-act="ko" ${n ? '' : 'disabled'}>Knock out the group (${n}) and rerun</button><button type="button" data-act="random" ${n ? '' : 'disabled'}>Control: ${n || 'as many'} random, same layers</button><button type="button" data-act="active" ${n && this.tracedRun() ? '' : 'disabled'}>Control: ${n || 'as many'} most active, same layers</button><button type="button" data-act="restore" ${any ? '' : 'disabled'} title="Undo the graph's last knockout and put back the neuron changes you had before it">Restore</button></div>`;
    if (this.note) html += `<p class="note">${esc(this.note)}</p>`;
    html += `<div class="gwrap">${svg}</div>`;
    html += '<p class="note limit">Each edge is an exact share of a neuron\'s input passed through the slope of its activation function at this token: a first-order account, which can miss a neuron pushed far into its flat side. Attention heads end the walk; the graph doesn\'t follow them back to other tokens. A knockout is exact, but it acts at every token, and it tests whether the group is needed, not whether it is specific to this word: run both controls, and check what else the model now says.</p>';
    this.el.innerHTML = html; this.el.hidden = false;
  }
}
