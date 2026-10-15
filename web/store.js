// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Recourse Software Inc.
// Models fetched from Hugging Face are converted in the browser and kept in the
// origin-private file system (OPFS), one directory per model, in the same file layout
// as a prebuilt data set. `dataFetch` is a drop-in for fetch(): URLs that start with
// `opfs:` are served from there (Range included); anything else goes to the network.

export const LOCAL = 'opfs:';
const ROOT = 'models';

export const localId = (repo) => 'hf--' + repo.replace(/\//g, '--');
export const localBase = (repo) => `${LOCAL}${localId(repo)}/`;

async function root() {
  if (!navigator.storage?.getDirectory) throw new Error('This browser can\'t store models, so they can\'t be downloaded here. Try a recent Chrome, Edge, Firefox or Safari, outside private browsing.');
  return (await navigator.storage.getDirectory()).getDirectoryHandle(ROOT, { create: true });
}
// Walk "a/b/c.json" from the root; create directories (and the file) when asked.
async function handle(path, create = false) {
  const parts = path.split('/').filter(Boolean);
  let dir = await root();
  for (const p of parts.slice(0, -1)) dir = await dir.getDirectoryHandle(p, { create });
  return dir.getFileHandle(parts[parts.length - 1], { create });
}

export async function dataFetch(url, init = {}) {
  if (typeof url !== 'string' || !url.startsWith(LOCAL)) {
    // Static hosts may answer a missing file with the home page: treat HTML as missing.
    const r = await fetch(url, init);
    if (r.ok && !/\.html?$/.test(url) && (r.headers.get('Content-Type') || '').includes('text/html')) return new Response(null, { status: 404 });
    return r;
  }
  const path = decodeURIComponent(url.slice(LOCAL.length).split(/[?#]/)[0]);
  let file;
  try { file = await (await handle(path)).getFile(); } catch { return new Response(null, { status: 404 }); }
  const range = /bytes=(\d+)-(\d*)/.exec((init.headers && (init.headers.Range || init.headers.range)) || '');
  if (!range) return new Response(file, { status: 200, headers: { 'Content-Length': String(file.size) } });
  const a = +range[1], b = range[2] ? Math.min(+range[2], file.size - 1) : file.size - 1;
  return new Response(file.slice(a, b + 1), { status: 206, headers: { 'Content-Range': `bytes ${a}-${b}/${file.size}` } });
}

// Write a whole file (ArrayBuffer, typed array, string or Blob).
export async function writeFile(path, data) {
  const h = await handle(path, true);
  if (h.createWritable) {
    const w = await h.createWritable();
    await w.write(typeof data === 'string' ? new Blob([data]) : data);
    await w.close();
    return;
  }
  // Safari: writable streams are missing, sync access handles exist (workers only).
  if (typeof h.createSyncAccessHandle !== 'function') return writeInWorker(path, data);  // Safari, main thread
  const s = await h.createSyncAccessHandle();
  const bytes = typeof data === 'string' ? new TextEncoder().encode(data)
    : data instanceof Blob ? new Uint8Array(await data.arrayBuffer())
    : ArrayBuffer.isView(data) ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(data);
  s.truncate(0); s.write(bytes, { at: 0 }); s.flush(); s.close();
}

// Hand a write to write-worker.js, where Safari allows sync access handles.
function writeInWorker(path, data) {
  return new Promise((resolve, reject) => {
    const w = new Worker(new URL('./write-worker.js', import.meta.url), { type: 'module' });
    w.onmessage = ({ data: m }) => { w.terminate(); if (m.ok) resolve(); else reject(new Error(m.message)); };
    w.onerror = (e) => { w.terminate(); reject(new Error(e.message || 'could not save the file')); };
    w.postMessage({ path, data });
  });
}

// A synchronous reader for one stored file (workers only): streaming a model's weights reads
// each layer straight into the kernel's memory with it, no copies. Close it when done.
export async function syncReader(path) {
  const h = await handle(path);
  if (!h.createSyncAccessHandle) throw new Error('This browser can\'t stream model weights from its storage.');
  return h.createSyncAccessHandle();
}

export async function readBuf(path) {
  return (await (await handle(path)).getFile()).arrayBuffer();
}
export async function removeFile(path) {
  const parts = path.split('/').filter(Boolean);
  let dir = await root();
  for (const p of parts.slice(0, -1)) dir = await dir.getDirectoryHandle(p);
  await dir.removeEntry(parts[parts.length - 1], { recursive: true });
}
export async function fileSize(path) {
  try { return (await (await handle(path)).getFile()).size; } catch { return -1; }
}

export async function exists(path) {
  try { await handle(path); return true; } catch { return false; }
}

// Models stored here: [{ id, repo, name, bytes }]. meta.json is written last, so a
// directory without it is an interrupted download and is left out (and can be removed).
export async function listLocal() {
  const out = [];
  let dir;
  try { dir = await root(); } catch { return out; }
  for await (const [name, h] of dir.entries()) {
    if (h.kind !== 'directory') continue;
    try {
      const meta = JSON.parse(await (await (await h.getFileHandle('meta.json')).getFile()).text());
      out.push({ id: name, repo: meta.model.repo, name: meta.model.name, bytes: await dirBytes(h) });
    } catch { /* incomplete */ }
  }
  return out;
}
async function dirBytes(dir) {
  let n = 0;
  for await (const [, h] of dir.entries()) n += h.kind === 'file' ? (await h.getFile()).size : await dirBytes(h);
  return n;
}
// Everything this site keeps in the browser's storage, for the storage manager: downloaded models,
// downloads that never finished, and SAE feature dictionaries (one entry per layer).
// [{ id (path to delete), kind: 'model' | 'partial' | 'features', name, repo?, bytes }]
export async function listStored() {
  const out = [];
  let dir;
  try { dir = await root(); } catch { return out; }
  for await (const [name, h] of dir.entries()) {
    if (h.kind !== 'directory') { out.push({ id: name, kind: 'partial', name, bytes: (await h.getFile()).size }); continue; }
    if (name.startsWith('sae--')) {
      for await (const [hook, hh] of h.entries()) {
        // blocks.N.hook_resid_pre is the stream entering layer N (the field's depth N); resid_post after the last layer.
        const m = /blocks\.(\d+)\.hook_resid_(pre|post)/.exec(hook), where = m ? (m[2] === 'pre' ? (+m[1] === 0 ? 'the embedding' : `after layer ${+m[1] - 1}`) : `after layer ${m[1]}`) : hook;
        if (hh.kind === 'directory') out.push({ id: `${name}/${hook}`, kind: 'features', name: `${name.slice(5).replace('gpt2-small', 'GPT-2 small')} features · ${where}`, bytes: await dirBytes(hh) });
      }
      continue;
    }
    let meta = null;
    try { meta = JSON.parse(await (await (await h.getFileHandle('meta.json')).getFile()).text()); } catch {}
    out.push(meta ? { id: name, kind: 'model', name: meta.model.name, repo: meta.model.repo, bytes: await dirBytes(h) }
      : { id: name, kind: 'partial', name: name.replace(/^hf--/, '').replace(/--/g, '/'), bytes: await dirBytes(h) });
  }
  return out;
}
// Remove one stored item by its path ('hf--owner--name', or 'sae--set/hook').
export async function removeStored(path) {
  const parts = path.split('/').filter(Boolean);
  let dir = await root();
  for (const p of parts.slice(0, -1)) dir = await dir.getDirectoryHandle(p);
  await dir.removeEntry(parts[parts.length - 1], { recursive: true });
  if (parts.length > 1) {  // an emptied parent folder goes too
    let empty = true;
    for await (const _ of dir.entries()) { empty = false; break; }
    if (empty) { let up = await root(); for (const p of parts.slice(0, -2)) up = await up.getDirectoryHandle(p); await up.removeEntry(parts[parts.length - 2]).catch(() => {}); }
  }
}
// ---------- export and import ----------
// A downloaded model can be copied to a folder on disk and back, so another copy of the site (or
// another browser profile, or another machine) can use it without downloading it again. The
// export is the model's folder as stored here, plus a manifest of every file and its size, which
// the import checks. meta.json is copied last, so a half-finished import is never taken for a model.
export const EXPORT_MANIFEST = 'latent-fovea-export.json';
export const canExport = () => typeof window !== 'undefined' && typeof window.showDirectoryPicker === 'function';

async function* walkFiles(dir, prefix = '') {  // [relative path, FileSystemFileHandle]
  for await (const [name, h] of dir.entries()) {
    if (h.kind === 'directory') yield* walkFiles(h, `${prefix}${name}/`);
    else yield [`${prefix}${name}`, h];
  }
}
async function diskFile(dir, path, create) {  // a file handle at path under dir, making folders
  const parts = path.split('/');
  for (const p of parts.slice(0, -1)) dir = await dir.getDirectoryHandle(p, { create });
  return dir.getFileHandle(parts[parts.length - 1], { create });
}
// Count bytes as a stream passes through.
const counter = (on) => new TransformStream({ transform(chunk, c) { on(chunk.byteLength); c.enqueue(chunk); } });

// Copy stored model `id` into a new folder inside `target` (a directory handle the user picked).
// onProgress(bytesDone, bytesTotal, fileName). Returns the folder's name.
export async function exportModel(id, target, onProgress = () => {}) {
  const src = await (await root()).getDirectoryHandle(id);
  const files = [];
  for await (const [path, h] of walkFiles(src)) files.push({ path, h, size: (await h.getFile()).size });
  files.sort((a, b) => (a.path === 'meta.json') - (b.path === 'meta.json'));  // meta.json last
  const total = files.reduce((t, f) => t + f.size, 0), out = await target.getDirectoryHandle(id, { create: true });
  let done = 0;
  for (const f of files) {
    onProgress(done, total, f.path);
    const w = await (await diskFile(out, f.path, true)).createWritable();
    await (await f.h.getFile()).stream().pipeThrough(counter((n) => { done += n; onProgress(done, total, f.path); })).pipeTo(w);
  }
  const meta = JSON.parse(await (await (await src.getFileHandle('meta.json')).getFile()).text());
  const man = { format: 'latent-fovea-model/1', id, repo: meta.model.repo, name: meta.model.name, bytes: total, exported: new Date().toISOString(), files: files.map((f) => ({ path: f.path, bytes: f.size })) };
  const mw = await (await out.getFileHandle(EXPORT_MANIFEST, { create: true })).createWritable();
  await mw.write(JSON.stringify(man, null, 1)); await mw.close();
  onProgress(total, total, '');
  return id;
}

// Find exported models in what the user picked: a directory handle (Chrome, Edge) or the File
// list of an <input webkitdirectory> (every browser). The pick may be one exported model or a
// folder of several. Returns [{ repo, name, bytes, files: [{ path, bytes, get() -> File }] }].
export async function findExports(pick) {
  const all = [];  // { path (from the pick), bytes, get }
  if (pick && typeof pick.entries === 'function') {
    for await (const [path, h] of walkFiles(pick)) all.push({ path, get: () => h.getFile() });
    for (const f of all) f.bytes = (await f.get()).size;
  } else for (const file of pick || []) {
    const rel = file.webkitRelativePath || file.name;
    all.push({ path: rel.split('/').slice(1).join('/') || rel, bytes: file.size, get: async () => file });
  }
  const roots = all.filter((f) => f.path === 'meta.json' || f.path.endsWith('/meta.json')).map((f) => f.path.slice(0, -'meta.json'.length));
  const out = [];
  for (const r of roots) {
    if (roots.some((o) => o !== r && r.startsWith(o))) continue;  // a meta.json nested inside another model
    const files = all.filter((f) => f.path.startsWith(r)).map((f) => ({ ...f, path: f.path.slice(r.length) }));
    let meta, man = null;
    try { meta = JSON.parse(await (await files.find((f) => f.path === 'meta.json').get()).text()); } catch { continue; }
    const mf = files.find((f) => f.path === EXPORT_MANIFEST);
    if (mf) try { man = JSON.parse(await (await mf.get()).text()); } catch {}
    if (!meta.model || !meta.model.repo) continue;
    const model = files.filter((f) => f.path !== EXPORT_MANIFEST);
    out.push({ repo: meta.model.repo, name: meta.model.name, bytes: model.reduce((t, f) => t + f.bytes, 0), files: model, manifest: man });
  }
  return out;
}

// Copy one found export into this site's storage, checking it against its manifest.
export async function importModel(ex, onProgress = () => {}) {
  if (ex.manifest) {  // every file the export listed, at its size
    const have = new Map(ex.files.map((f) => [f.path, f.bytes]));
    const bad = ex.manifest.files.find((f) => have.get(f.path) !== f.bytes);
    if (bad) throw new Error(`${bad.path} is ${have.has(bad.path) ? 'the wrong size' : 'missing'} in the exported folder`);
  }
  const id = localId(ex.repo), files = ex.files.slice().sort((a, b) => (a.path === 'meta.json') - (b.path === 'meta.json'));
  await (await root()).removeEntry(id, { recursive: true }).catch(() => {});  // replace any earlier copy
  let done = 0;
  for (const f of files) {
    onProgress(done, ex.bytes, f.path);
    await writeFile(`${id}/${f.path}`, await f.get());
    done += f.bytes;
    if ((await fileSize(`${id}/${f.path}`)) !== f.bytes) throw new Error(`${f.path} did not copy completely`);
  }
  onProgress(ex.bytes, ex.bytes, '');
  return id;
}

export async function removeLocal(id) {
  await (await root()).removeEntry(id, { recursive: true });
}
