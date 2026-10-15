#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Recourse Software Inc.
"""Turn a KnowDrive store fetched by tools/fetch_kdstore.mjs into a Latent Fovea data set: the
store's atoms as a cloud, in the same layout as a model's vocabulary (see tools/build_data.py), so
the Embeddings view, the fovea, the inspector, layouts and neighbours all work on it unchanged.

    ~/.cache/latent-fovea/venv/bin/python tools/build_kdstore.py \\
        --cache ~/.cache/latent-fovea/kdstore-canada --out web/data/canada --name "Canada 2025/26 Public Data"

Atoms are the store's keys (KnowDrive reads retrieval as attention: score a query against frozen
keys, mask, emit). Nearest neighbours are exact (cosine, blockwise over every row). Besides the
model-shaped files it writes kd/: the files (path, tags), the tag table, and each row's file and
sequence position, for the inspector and for masking in the browser.
"""
import argparse, collections, glob, json, os, time
import numpy as np

GRID, K = 16, 16
ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--name", default="KnowDrive store"); ap.add_argument("--store", default="")
ap.add_argument("--host", default="public.knowdrive.ai")
ap.add_argument("--license", default="Canadian public records")
ap.add_argument("--label-chars", type=int, default=40)
ap.add_argument("--max-part-mb", type=float, default=24)
ap.add_argument("--limit-files", type=int, default=0, help="for a quick test: only the first N fetched files")
a = ap.parse_args()
t0 = time.time()
log = lambda *x: print(f"[{time.time() - t0:7.1f}s]", *x, flush=True)

# ---- load the cache ----
files = json.load(open(os.path.join(a.cache, "files.json")))
tag_ids, tags = {}, []
def tid(t):
    if t not in tag_ids: tag_ids[t] = len(tags); tags.append(t)
    return tag_ids[t]
rows_meta, vecs, file_rows = [], [], []
for fi, f in enumerate(files):
    mp = os.path.join(a.cache, "files", f"{f['id']}.json")
    if not os.path.exists(mp) or (a.limit_files and len(file_rows) >= a.limit_files): continue
    meta = json.load(open(mp))
    v = np.fromfile(os.path.join(a.cache, "files", f"{f['id']}.f32"), dtype="<f4").reshape(-1, 1024)
    assert len(v) == len(meta), f["id"]
    for m in meta: rows_meta.append((fi, m["seq"], m["text"]))
    vecs.append(v)
    file_rows.append(fi)
X = np.concatenate(vecs).astype(np.float32); del vecs
n, d = X.shape
log(f"{n} atoms from {len(file_rows)} of {len(files)} files, d={d}")
norms = np.linalg.norm(X, axis=1).astype(np.float32)
Xn = X / np.maximum(norms[:, None], 1e-12)

# ---- categories: the domain: tag ----
def domain(f):
    for t in f["tags"]:
        if t.startswith("domain:"): return t[7:]
    return "other"
dc = collections.Counter(domain(files[fi]) for fi, _, _ in rows_meta)
cats = [c for c, _ in dc.most_common() if c != "other"][:5] + ["other"]
cat_of = {c: i for i, c in enumerate(cats)}
cat = np.array([cat_of.get(domain(files[fi]), len(cats) - 1) for fi, _, _ in rows_meta], np.uint8)

# ---- exact cosine nearest neighbours, blockwise ----
knn = np.zeros((n, K), np.uint32); knncos = np.zeros((n, K), np.float16)
B = 1024
for s in range(0, n, B):
    S = Xn[s:s + B] @ Xn.T
    S[np.arange(len(S)), np.arange(s, s + len(S))] = -2  # not itself
    top = np.argpartition(-S, K, axis=1)[:, :K]
    sc = np.take_along_axis(S, top, axis=1)
    o = np.argsort(-sc, axis=1)
    knn[s:s + B] = np.take_along_axis(top, o, axis=1); knncos[s:s + B] = np.take_along_axis(sc, o, axis=1)
    if (s // B) % 40 == 0: log(f"knn {s}/{n}")
log("knn done")

# ---- layouts: PCA, and UMAP (cosine) on 50 principal components ----
mu = Xn.mean(0)
U, Sv, Vt = np.linalg.svd(Xn[np.random.default_rng(7).choice(n, min(n, 50000), replace=False)] - mu, full_matrices=False)
P50 = (Xn - mu) @ Vt[:50].T
pca = P50[:, :3].astype(np.float32)
log("pca done")
import umap
P = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.15, metric="cosine", random_state=7, low_memory=True, verbose=True).fit_transform(P50).astype(np.float32)
log("umap done")

def norm_points(P):
    c = (P.min(0) + P.max(0)) / 2; r = np.percentile(np.linalg.norm(P - c, axis=1), 99)
    return ((P - c) / r).astype(np.float32)
P = norm_points(P); pca = norm_points(pca)

# ---- tiles: rows sorted into a 16^3 grid in Morton order ----
def morton3(x, y, z):
    def spread(v):
        v = v.astype(np.uint32) & 0x3FF
        v = (v | (v << 16)) & 0x030000FF; v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3; v = (v | (v << 2)) & 0x09249249
        return v
    return spread(x) | (spread(y) << 1) | (spread(z) << 2)
lo, hi = P.min(0), P.max(0)
g = np.clip(((P - lo) / (hi - lo + 1e-9) * GRID).astype(int), 0, GRID - 1)
cell = morton3(g[:, 0], g[:, 1], g[:, 2])
order = np.argsort(cell, kind="stable")
inv = np.empty(n, np.uint32); inv[order] = np.arange(n, dtype=np.uint32)
cells = np.zeros(GRID ** 3 + 1, np.uint32)
np.add.at(cells, cell[order].astype(np.int64) + 1, 1)
cells = np.cumsum(cells).astype(np.uint32)

# ---- write ----
os.makedirs(os.path.join(a.out, "kd"), exist_ok=True)
w = lambda name, arr: arr.tofile(os.path.join(a.out, name))
Xo = X[order]
w("norms.f32", norms[order]); w("points.f32", P[order]); w("points.pca.f32", pca[order])
w("cat.u8", cat[order]); w("cells.u32", cells); w("ids.u32", order.astype(np.uint32))
w("knn.u32", inv[knn[order]].astype(np.uint32)); w("knncos.f16", knncos[order])
rb = d * 2
rows_per = int(a.max_part_mb * 1e6) // rb
parts = []
for k, s in enumerate(range(0, n, rows_per)):
    name = f"vectors.{k:02d}.f16"
    Xo[s:s + rows_per].astype(np.float16).tofile(os.path.join(a.out, name))
    parts.append({"file": name, "bytes": int(min(rows_per, n - s) * rb)})
# Labels: an atom is a passage, not a word, so its label is a short name: the 3 words most
# particular to it (frequent in it, rare across the store: TF-IDF), in the order they appear.
# The passage itself is in kd/snip.*.json, for the inspector.
import re
STOP = set("""a an and are as at be been being but by can could did do does for from had has have he her his i if in
into is it its may more most must no not of on or our over shall she should so such than that the their them then there
these they this those to under up was we were what when where which while who will with would you your also any all each
other per page pages date dates table total name names number item items type use used new one two three
le la les de des du et en un une pour par sur au aux est span div class var display style border""".split())
WORD = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'\-]{2,}|[0-9]{4}")
MARKUP = re.compile(r"<[^>]*>?|\{[^}]*\}?|[\w-]+\s*:\s*[^;\s]+;|https?://\S+|\S*[=/]\S*")
def wordlike(w):  # drop encoded data (random case, no vowels) and numbers that are not years
    if w.isdigit(): return 1900 <= int(w) <= 2099
    core = w[1:]
    return w.lower() not in STOP and re.search(r"[aeiouyàâéèêëîïôûü]", w.lower()) and (core.islower() or w.isupper() or "-" in w)
toks = [[w for w in WORD.findall(MARKUP.sub(" ", t)) if wordlike(w)] for _, _, t in rows_meta]
df = collections.Counter()
for ws in toks: df.update(set(w.lower() for w in ws))
def label(ws):
    tf = collections.Counter(w.lower() for w in ws)
    if not tf: return "…"
    score = {w: c * np.log(n / df[w]) for w, c in tf.items() if df[w] > 1}  # a word seen once in the store is noise
    best = set(sorted(score, key=lambda w: -score[w])[:3])
    seen, out = set(), []
    for w in ws:  # first appearance, original case
        if w.lower() in best and w.lower() not in seen: seen.add(w.lower()); out.append(w)
    return " · ".join(out)[:a.label_chars] or "…"
json.dump([label(toks[i]) for i in order], open(os.path.join(a.out, "tokens.json"), "w"), ensure_ascii=False)
# The store table: files and tags once, and for each row (in tile order) its file and position.
used = sorted(set(fi for fi, _, _ in rows_meta))
fidx = {fi: j for j, fi in enumerate(used)}
json.dump({"files": [{"id": files[fi]["id"], "path": files[fi]["path"], "tags": [tid(t) for t in files[fi]["tags"]]} for fi in used], "tags": tags},
          open(os.path.join(a.out, "kd", "store.json"), "w"), ensure_ascii=False, separators=(",", ":"))
np.array([fidx[rows_meta[i][0]] for i in order], np.uint32).tofile(os.path.join(a.out, "kd", "rowfile.u32"))
np.array([rows_meta[i][1] for i in order], np.uint32).tofile(os.path.join(a.out, "kd", "rowseq.u32"))
# Snippets (up to 240 characters each) in shards of 4,096 rows, fetched when an atom is inspected.
SNIP = 4096
for k, s in enumerate(range(0, n, SNIP)):
    json.dump([" ".join(rows_meta[i][2].split()) for i in order[s:s + SNIP]], open(os.path.join(a.out, "kd", f"snip.{k:03d}.json"), "w"), ensure_ascii=False)
meta = {
    "schema": "latent-fovea/1", "id": os.path.basename(os.path.normpath(a.out)), "kind": "kdstore",
    "model": {"name": a.name, "author": "KnowDrive", "license": a.license, "family": "kdstore", "params": f"{n:,} atoms",
              "blurb": f"A KnowDrive store on {a.host}: {n:,} atoms (spans of {len(used):,} public-record files), each a 1,024-number Jina-CLIP-v2 vector. KnowDrive reads retrieval as attention, so these are the keys a query is scored against.",
              "store": a.store, "host": a.host, "files": len(used), "embedding": "jina-clip-v2"},
    "n": int(n), "d": int(d), "k": K, "grid": GRID, "cells": GRID ** 3, "vector_dtype": "f16", "row_bytes": rb,
    "projection": {"method": "umap", "n_neighbors": 30, "min_dist": 0.15, "metric": "cosine", "pre": "pca50", "seed": 7},
    "bounds": [P.min(0).tolist(), P.max(0).tolist()], "categories": cats,
    "category_counts": [int((cat == i).sum()) for i in range(len(cats))],
    "vector_parts": parts, "snippet_shard": SNIP, "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
tj = os.path.getsize(os.path.join(a.out, "tokens.json"))
assert tj < 24e6, f"tokens.json is {tj / 1e6:.1f} MB; lower --label-chars (static hosts take 25 MiB per file)"
log(f"wrote {a.out}: {n} rows, {len(parts)} vector parts, {sum(p['bytes'] for p in parts) / 1e6:.0f} MB")
