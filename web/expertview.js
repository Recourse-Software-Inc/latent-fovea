// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// GLM-5.2's experts as a cloud (data sets with meta.kind === 'experts', tools/build_experts.py): each
// point is one routed expert, placed by its row of the router, the direction in the hidden state
// that wakes it. While glm.html runs, each token's routing arrives here (BroadcastChannel 'lf-glm'):
// the experts it chose light up by where they came from (on the GPU, read ahead, RAM, the drive),
// and "Distances from the token" puts every expert at its rank in its own layer's choice for that
// token (sigmoid score + balance bias, as the router decides): the eight chosen per layer innermost.
const TIER = { gpu: '#35e0e6', pre: '#7be08a', ram: '#ef6fa6', ssd: '#f2b56b' };

export class ExpertView {
  constructor(o) {
    Object.assign(this, o);  // { $, DATA, meta, N, dataFetch, escapeHtml, onChange }
    this.rows = null; this.rowOf = new Map(); this.token = null; this.dist = null; this.rank = null;
  }
  async init() {
    const d = await (await this.dataFetch(this.DATA + 'experts.json')).json();
    this.rows = d.rows; this.E = d.E;
    d.rows.forEach((r, i) => this.rowOf.set(r.layer * d.E + r.expert, i));
    this.chan = new BroadcastChannel('lf-glm');
    this.chan.onmessage = ({ data: m }) => { if (m.type === 'token') this.onToken(m); };
    this.$('exNote').textContent = 'Open GLM-5.2 (glm.html) in another tab and ask it something: each token\'s experts light up here.';
  }
  // m: { text, experts: [l][[e, weight, tier]], scores: [l][E] (sigmoid) | null }
  onToken(m) {
    const chosen = [];
    m.experts.forEach((ch, l) => (ch || []).forEach(([e, w, tier]) => { const i = this.rowOf.get(l * this.E + e); if (i !== undefined) chosen.push({ i, l, e, w, tier }); }));
    this.token = { text: m.text, chosen };
    if (m.scores) {  // rank of every expert in its own layer's choice, as the router ranks them
      const dist = new Float32Array(this.N).fill(1), rank = new Uint16Array(this.N).fill(this.E);
      m.scores.forEach((sig, l) => {
        if (!sig) return;
        const order = [...sig.keys()].map((e) => { const i = this.rowOf.get(l * this.E + e); return [sig[e] + (i !== undefined ? this.rows[i].bias : 0), e, i]; })
          .sort((a, b) => b[0] - a[0] || a[1] - b[1]);
        order.forEach(([, , i], r) => { if (i !== undefined) { rank[i] = r; dist[i] = r / (this.E - 1); } });
      });
      this.dist = dist; this.rank = rank;
    }
    const tiers = chosen.reduce((t, c) => ((t[c.tier] = (t[c.tier] || 0) + 1), t), {});
    this.$('exNote').innerHTML = `Token <b>${this.escapeHtml(JSON.stringify(m.text ?? '').slice(1, -1))}</b>: ${chosen.length} experts chosen · ` +
      Object.entries(tiers).map(([k, v]) => `<span style="color:${TIER[k]}">${v} ${k === 'gpu' ? 'on the GPU' : k === 'pre' ? 'read ahead' : k === 'ram' ? 'from RAM' : 'from the drive'}</span>`).join(' · ');
    this.$('exRadial').hidden = !this.dist;
    this.onChange();
  }
  inspectHtml(i) {
    const r = this.rows?.[i]; if (!r) return '';
    const e = this.escapeHtml;
    let now = '';
    const c = this.token?.chosen.find((x) => x.i === i);
    if (c) now = `<div class="kd-q">Chosen for <b>${e(this.token.text || '')}</b> · weight <b>${c.w.toFixed(3)}</b> · ${c.tier === 'gpu' ? 'already on the GPU' : c.tier === 'pre' ? 'read ahead in time' : c.tier === 'ram' ? 'from RAM' : 'read from the drive'}</div>`;
    else if (this.rank) now = `<div class="kd-q">Rank <b>${this.rank[i] + 1}</b> of ${this.E} in layer ${r.layer} for the last token</div>`;
    return `${now}<div class="kd-file">Layer <b>${r.layer}</b> · expert <b>${r.expert}</b> · balance bias ${r.bias.toFixed(2)}${r.routed ? ` · routed ${r.routed}× in colibrì's history` : ''}</div>
      <div class="kd-tags">${r.words.map((w) => `<span>${e(w)}</span>`).join('')}</div>
      <div class="note">Words whose embeddings align with this expert's router row: a rough guide only (the router reads hidden states, not raw words).</div>`;
  }
  anchor(P) {
    const ch = this.token?.chosen; if (!ch?.length) return null;
    const c = [0, 0, 0]; for (const { i } of ch) for (let a = 0; a < 3; a++) c[a] += P[3 * i + a] / ch.length;
    return c;
  }
  drawOverlay(ov, dpr, toScreen) {
    const ch = this.token?.chosen; if (!ch?.length) return;
    ov.save(); ov.lineWidth = 1.6 * dpr;
    for (const { i, tier, w } of ch) {
      const p = toScreen(i); if (!p) continue;
      ov.strokeStyle = TIER[tier] || '#fff'; ov.globalAlpha = 0.95;
      ov.beginPath(); ov.arc(p[0], p[1], (3 + 9 * Math.min(1, w)) * dpr, 0, Math.PI * 2); ov.stroke();
    }
    ov.restore();
  }
}
