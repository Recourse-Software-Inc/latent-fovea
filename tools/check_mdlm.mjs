// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// Run one MDLM forward pass in WebAssembly: prompt + N masked slots.
//   node tools/check_mdlm.mjs "The capital of France is" [N=16] [data/mdlm]
import fs from 'fs';
import path from 'path';
import { MDLM } from '../web/mdlm.js';
import { GPT2Tokenizer } from '../web/bpe.js';

const prompt = process.argv[2] || 'The capital of France is';
const N = +(process.argv[3] || 16);
const dir = path.resolve('web', process.argv[4] || 'data/mdlm');
const read = (n) => { const b = fs.readFileSync(path.join(dir, n)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const manifest = JSON.parse(fs.readFileSync(path.join(dir, 'model/manifest.json')));
const tok = new GPT2Tokenizer(JSON.parse(fs.readFileSync(path.join(dir, 'model/vocab.json'))), fs.readFileSync(path.join(dir, 'model/merges.txt'), 'utf8'));
const tokens = JSON.parse(fs.readFileSync(path.join(dir, 'tokens.json')));
const ids = new Uint32Array(read('ids.u32'));
const m = await MDLM.create(fs.readFileSync('web/mdlm.simd.wasm'), fs.readFileSync('web/mdlm.wasm'));
let t = performance.now();
await m.load(read, manifest, ids);
console.error(`load ${(performance.now() - t).toFixed(0)} ms, memory ${(m.ex.memory.buffer.byteLength / 1e6).toFixed(0)} MB`);
const rows = [...tok.encode(prompt).map((i) => m.rowOf[i]), ...Array(N).fill(m.maskRow)];
m.forward(rows);  // warm-up: V8 tiers wasm up from its baseline compiler
t = performance.now();
m.forward(rows);
const ms = performance.now() - t;
const beliefs = {};
rows.forEach((r, i) => { if (r === m.maskRow && Object.keys(beliefs).length < 4) { const b = m.belief(i); beliefs[i] = b.rows.slice(0, 5).map((r2, j) => [tokens[r2], +b.probs[j].toFixed(5)]); } });
const r5 = (a) => Array.from(a).map((v) => +v.toFixed(5));
console.log(JSON.stringify({ rows, ms_forward: +ms.toFixed(0), n: rows.length, beliefs, att_l0h0_q0: r5(m.attRow(0, 0, 0)), att_l7h3_qlast: r5(m.attRow(7, 3, rows.length - 1)) }));
