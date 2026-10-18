// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// A shared link's interventions survive the trip through the page address: every tuple
// (layer, neuron, scale, add, pre) comes back as it went in, including fractional and negative values.
// Run: node tools/check_links.mjs
import { ForwardView, SHARED_KEYS } from '../web/field.js';

const cases = [
  [[5, 880, 0, 0, 0]],                                   // a knockout
  [[5, 880, 0, 4.123, 0]],                               // held at its maximum
  [[3, 2427, 0, -0.17, 0], [9, 1806, 0, -0.0123, 0]],    // troughs
  [[11, 3071, 1, 0, -1.5], [0, 0, 0.5, 2.25, 0.3333]],   // a steered input, a scaled one
];
let bad = 0;
for (const c of cases) {
  const enc = ForwardView.encodeIv(c), q = new URLSearchParams({ iv: enc });
  const back = ForwardView.decodeIv(new URLSearchParams(q.toString()).get('iv'));
  const ok = JSON.stringify(back) === JSON.stringify(c.map((x) => x.map((y) => +(+y).toFixed(4))));
  if (!ok) bad++;
  console.log(`${ok ? 'ok ' : 'BAD'} ${JSON.stringify(c)} -> iv=${enc} -> ${JSON.stringify(back)}`);
}
// malformed input is dropped, not half-applied
for (const s of ['5:880:0:x:0', '5.5:880', '1:2:3:4:5:6', '']) {
  const back = ForwardView.decodeIv(s);
  if (back.length) bad++;
  console.log(`${back.length ? 'BAD' : 'ok '} iv=${JSON.stringify(s)} -> ${JSON.stringify(back)}`);
}
if (!SHARED_KEYS.includes('iv') || !SHARED_KEYS.includes('gk')) { bad++; console.log('BAD shared keys'); }
if (bad) { console.error(`${bad} failed`); process.exit(1); }
console.log('all links round-trip');
