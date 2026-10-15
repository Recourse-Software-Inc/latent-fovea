// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// Run the WebAssembly GPT-2 on a prompt from local data and print what it predicts.
//   node tools/check_forward.mjs "The capital of France is" [data/gpt2]
import fs from 'fs';
import path from 'path';
import { Forward } from '../web/forward.js';
import { GPT2Tokenizer } from '../web/bpe.js';

const prompt = process.argv[2] || 'The capital of France is';
const dir = path.resolve('web', process.argv[3] || 'data/gpt2');
const read = (n) => { const b = fs.readFileSync(path.join(dir, n)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const manifest = JSON.parse(fs.readFileSync(path.join(dir, 'model/manifest.json')));
const tok = new GPT2Tokenizer(JSON.parse(fs.readFileSync(path.join(dir, 'model/vocab.json'))), fs.readFileSync(path.join(dir, 'model/merges.txt'), 'utf8'));
const tokens = JSON.parse(fs.readFileSync(path.join(dir, 'tokens.json')));
const ids = new Uint32Array(read('ids.u32'));
const fwd = await Forward.create(fs.readFileSync('web/gpt2.simd.wasm'), fs.readFileSync('web/gpt2.wasm'));
let t = performance.now();
await fwd.load(read, manifest, ids);
console.error(`load ${(performance.now() - t).toFixed(0)} ms, memory ${(fwd.ex.memory.buffer.byteLength / 1e6).toFixed(0)} MB`);
const tokIds = tok.encode(prompt), rows = tokIds.map((i) => fwd.rowOf[i]);
t = performance.now();
rows.forEach((r, p) => fwd.step(r, p, p === rows.length - 1 ? 2 : 0));
const ms = performance.now() - t;
const out = fwd.lens(fwd.L);
console.log(JSON.stringify({
  prompt, token_ids: tokIds, rows, ms_total: +ms.toFixed(1), ms_per_token: +(ms / rows.length).toFixed(1),
  top: out.rows.map((r, j) => [tokens[r], +out.probs[j].toFixed(4)]),
  lens_top1: Array.from({ length: fwd.L + 1 }, (_, d) => tokens[fwd.lens(d).rows[0]]),
  att_l0h0: Array.from(fwd.attRow(0, 0, rows.length - 1)).map((v) => +v.toFixed(4)),
  att_l5h1: Array.from(fwd.attRow(5, 1, rows.length - 1)).map((v) => +v.toFixed(4)),
}));
