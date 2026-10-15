// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// Build a Hugging Face model in a real browser (headless Chromium via Playwright), run a
// prompt through the same worker the site uses, and write what it computed to a JSON file
// for tools/ref_hf.py to compare against transformers.
//
//   python3 tools/serve.py --root site &      # a staged site (tools/stage_static.py site)
//   node tools/check_hf.mjs Qwen/Qwen2.5-0.5B out.json [--site http://127.0.0.1:8080/] [--prompt "..."]
//
// Env: CHROMIUM (browser binary; default Playwright's), PLAYWRIGHT (module path), LF_PROFILE (profile dir).
import fs from 'fs';
import os from 'os';
import path from 'path';

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args.splice(i, 2)[1] : d; };
const site = opt('--site', 'http://127.0.0.1:8080/'), prompt = opt('--prompt', 'The capital of France is');
const [repo, outFile = 'hf-check.json'] = args;
if (!repo) { console.error('usage: node tools/check_hf.mjs <owner/repo> [out.json]'); process.exit(2); }
const { chromium } = await import(process.env.PLAYWRIGHT || 'playwright');
const profile = process.env.LF_PROFILE || fs.mkdtempSync(path.join(os.tmpdir(), 'lf-'));
const ctx = await chromium.launchPersistentContext(profile, {
  executablePath: process.env.CHROMIUM || undefined, args: ['--ignore-certificate-errors'],
});
const p = await ctx.newPage();
const t0 = Date.now();
await p.goto(`${site}#hf:${repo}~embeddings`);
for (;;) {
  await p.waitForTimeout(1000);
  const msg = await p.textContent('#loadMsg');
  if (await p.evaluate(() => document.getElementById('loading').hidden)) break;
  if (/Could not/.test(msg)) { console.error(msg); await ctx.close(); process.exit(1); }
}
const buildS = (Date.now() - t0) / 1000;
const out = await p.evaluate(async ({ repo, prompt }) => {
  const { HFTokenizer } = await import('./bpe.js');
  const { dataFetch, localBase } = await import('./store.js');
  const base = localBase(repo);
  const tok = new HFTokenizer(await (await dataFetch(base + 'model/tokenizer.json')).json());
  const ids = tok.encode(prompt);
  const w = new Worker('forward-worker.js', { type: 'module' });
  const next = () => new Promise((ok) => { w.onmessage = (e) => { if (e.data.type !== 'progress') ok(e.data); }; });
  let r = next(); w.postMessage({ type: 'load', data: base });
  const ready = await r;
  if (ready.type === 'error') return { error: ready.message };
  let cap;
  for (let pos = 0; pos < ids.length; pos++) {
    r = next(); w.postMessage({ type: 'step', id: pos, row: ready.rowOf[ids[pos]], pos, lensMode: pos === ids.length - 1 ? 2 : 0 });
    cap = await r;
    if (cap.type === 'error') return { error: cap.message };
  }
  const idsT = new Uint32Array(await (await dataFetch(base + 'ids.u32')).arrayBuffer());
  const fin = cap.lens[ready.L];
  return { ids, L: ready.L, H: ready.H, ms: cap.ms, res: Array.from(cap.res), att00: Array.from(cap.att.slice(0, ids.length)),
    top: fin.rows.map((row, j) => [idsT[row], fin.probs[j]]) };
}, { repo, prompt });
await ctx.close();
if (out.error) { console.error(out.error); process.exit(1); }
Object.assign(out, { repo, prompt, build_s: buildS });
fs.writeFileSync(outFile, JSON.stringify(out));
console.log(`${repo}: built/loaded in ${buildS.toFixed(0)} s, ${out.ms.toFixed(0)} ms per token, ${out.ids.length} tokens`);
