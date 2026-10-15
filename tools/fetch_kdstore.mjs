// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// Fetch every atom of a KnowDrive store, with its text vector, onto local disk, for
// tools/build_kdstore.py to turn into a Latent Fovea data set. Run with a short-lived read
// token in a file (re-read before every request, so it can be replaced while this runs):
//
//   node tools/fetch_kdstore.mjs --store vs_… --base https://public.knowdrive.ai \
//        --token-file ~/.cache/latent-fovea/token --out ~/.cache/latent-fovea/kdstore-canada
//
// The token needs functions:/api/v1/vector_stores:read and vector_stores:<store>:read.
// Resumable: files already fetched are skipped. When the token expires (401), it waits for the
// token file to change and carries on. Nothing is sent but the token and the store's own ids.
import fs from 'fs';
import path from 'path';

const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
const home = process.env.HOME;
const STORE = arg('--store'), BASE = arg('--base', 'https://public.knowdrive.ai');
const TOKEN_FILE = arg('--token-file', `${home}/.cache/latent-fovea/token`);
const OUT = arg('--out', `${home}/.cache/latent-fovea/kdstore`);
const CONC = +arg('--concurrency', 8), SNIP = 240;
if (!STORE) { console.error('usage: node tools/fetch_kdstore.mjs --store <vector_store_id> [--base URL] [--token-file F] [--out DIR]'); process.exit(2); }
fs.mkdirSync(path.join(OUT, 'files'), { recursive: true });

const token = () => (process.env.KNOWDRIVE_TOKEN || fs.readFileSync(TOKEN_FILE, 'utf8')).trim();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let waitingSince = 0;
async function get(url) {
  for (let attempt = 0; ; attempt++) {
    let tok = '';
    try { tok = token(); } catch { tok = ''; }
    const r = await fetch(url, { headers: { Authorization: `Bearer ${tok}` } }).catch((e) => ({ ok: false, status: 0, e }));
    if (r.ok) { waitingSince = 0; return r.json(); }
    if (r.status === 401 || r.status === 403 || !tok) {  // expired: wait for a fresh token in the file
      if (!waitingSince) { waitingSince = Date.now(); console.error(`\n[token] ${r.status || 'no token'}: waiting for a new token in ${TOKEN_FILE}`); }
      const before = tok;
      while (true) { await sleep(2000); let now = ''; try { now = token(); } catch {} if (now && now !== before) break; }
      continue;
    }
    if (attempt >= 6) throw new Error(`${r.status} ${url}`);
    await sleep(1000 * 2 ** attempt);  // 429 / 5xx / network: back off
  }
}

try { if (!token()) throw 0; } catch {
  console.error(`No KnowDrive read token: put one in ${TOKEN_FILE} or $KNOWDRIVE_TOKEN (a KnowDrive account is needed).`);
  process.exit(1);
}

// 1. Every file in the store (id, path, tags, size), cached.
const filesPath = path.join(OUT, 'files.json');
let files;
if (fs.existsSync(filesPath)) files = JSON.parse(fs.readFileSync(filesPath, 'utf8'));
else {
  files = [];
  let after = '';
  do {
    const page = await get(`${BASE}/api/v1/vector_stores/${STORE}/files?limit=10000${after ? `&after=${encodeURIComponent(after)}` : ''}`);
    for (const f of page.data || page.files || []) files.push({ id: f.id || f.file_id, path: f.filename || f.path || f.uri, tags: f.tags || [], bytes: f.bytes || 0, atoms: f.atom_count || 0, mime: f.mime_type, created_at: f.created_at });
    after = page.has_more ? (page.next_after || page.last_id || (page.data || page.files).at(-1).id) : '';
    process.stderr.write(`\rlisting files: ${files.length}`);
  } while (after);
  fs.writeFileSync(filesPath, JSON.stringify(files));
  console.error(`\nlisted ${files.length} files`);
}

// 2. Each file's atoms: metadata (JSON) and text vectors (float32, little-endian), one pair per file.
const FIELDS = 'id,sequence_index,text,tags,uri,text_vector';
let done = 0, atoms = 0, t0 = Date.now();
const todo = files.filter((f) => !fs.existsSync(path.join(OUT, 'files', `${f.id}.json`)));
done = files.length - todo.length;
async function fetchFile(f) {
  const meta = [], vecs = [];
  let after = '';
  do {
    const page = await get(`${BASE}/api/v1/vector_stores/${STORE}/files/${f.id}/atoms?fields=${FIELDS}&limit=100&text=raw${after ? `&after=${encodeURIComponent(after)}` : ''}`);
    for (const a of page.data || page.atoms || []) {
      // The vector comes as base64 of 1024 little-endian float32s (or, on some builds, a JSON array).
      const v = typeof a.text_vector === 'string' ? new Float32Array(Uint8Array.from(Buffer.from(a.text_vector, 'base64')).buffer) : a.text_vector;
      if (!v || v.length !== 1024) continue;  // an atom without a text vector (an image, say) is not in the text space
      meta.push({ id: a.id, seq: a.sequence_index, text: (a.text || '').slice(0, SNIP), tags: a.tags || [], uri: a.uri || f.path });
      vecs.push(Float32Array.from(v));
    }
    after = page.has_more ? (page.next_after || '') : '';
  } while (after);
  const buf = new Float32Array(vecs.length * 1024);
  vecs.forEach((v, i) => buf.set(v, i * 1024));
  fs.writeFileSync(path.join(OUT, 'files', `${f.id}.f32`), Buffer.from(buf.buffer));
  fs.writeFileSync(path.join(OUT, 'files', `${f.id}.json`), JSON.stringify(meta));  // written last: marks the file done
  atoms += meta.length; done++;
  const s = (Date.now() - t0) / 1000;
  process.stderr.write(`\rfiles ${done}/${files.length} · atoms this run ${atoms} · ${(atoms / s).toFixed(0)} atoms/s   `);
}
let next = 0;
await Promise.all(Array.from({ length: CONC }, async () => { while (next < todo.length) await fetchFile(todo[next++]); }));
console.error(`\ndone: ${files.length} files`);
