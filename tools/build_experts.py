#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Recourse Software Inc.
"""GLM-5.2's experts as a Latent Fovea data set: one point per routed expert, its vector the
expert's row of the router (mlp.gate.weight, F32 [256, 6144] per MoE layer). The router scores an
expert as sigmoid(row · h) for the token's normalised hidden state h, so the row is the direction
that wakes the expert: experts whose rows point the same way answer to the same kind of state.

    python3 tools/build_experts.py --model /Volumes/<drive>/glm52_i4 --out web/data/glm52-experts

Writes the same layout as tools/build_data.py (tiles, exact kNN, UMAP and PCA, labels, categories)
with meta.kind = "experts", plus experts.json: per row its layer, index, correction bias, routing
count from colibri's .coli_usage when present, and the vocabulary words its row aligns with best.
Those words are a rough guide only: the router reads hidden states, not raw embeddings.
"""
import argparse, json, os, time
import numpy as np
from glm_ref import read_container, raw

GRID, K = 16, 16
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--max-part-mb", type=float, default=24); ap.add_argument("--words", type=int, default=3)
a = ap.parse_args()
t0 = time.time(); log = lambda *x: print(f"[{time.time() - t0:6.1f}s]", *x, flush=True)
cfg = json.load(open(os.path.join(a.model, "config.json")))
idx = read_container(a.model)
L, dense, E, C, V = cfg["num_hidden_layers"], cfg["first_k_dense_replace"], cfg["n_routed_experts"], cfg["hidden_size"], cfg["vocab_size"]
layers = list(range(dense, L))

# ---- router rows
rows, meta_rows = [], []
for l in layers:
    w = raw(idx, f"model.layers.{l}.mlp.gate.weight").astype(np.float32).reshape(E, C)
    b = raw(idx, f"model.layers.{l}.mlp.gate.e_score_correction_bias").astype(np.float32).ravel()
    rows.append(w)
    for e in range(E): meta_rows.append({"layer": l, "expert": e, "bias": round(float(b[e]), 5)})
X = np.concatenate(rows); n = len(X)
norms = np.linalg.norm(X, axis=1).astype(np.float32)
Xn = X / np.maximum(norms[:, None], 1e-12)
log(f"{n} experts from {len(layers)} layers, d={C}")

# ---- routing history (colibri's .coli_usage: "layer expert count", -1/-2 header lines)
usage = {}
up = os.path.join(a.model, ".coli_usage")
if os.path.exists(up):
    for line in open(up):
        p = line.split()
        if len(p) == 3 and int(p[0]) >= 0: usage[(int(p[0]), int(p[1]))] = int(p[2])

# ---- words each row aligns with (rough guide): router row · token embedding, top few
tok = json.load(open(os.path.join(a.model, "tokenizer.json")))
vocab = {i: t for t, i in tok["model"]["vocab"].items()}
for t in tok.get("added_tokens", []): vocab[t["id"]] = t["content"]
byte_dec = {}
bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
cs = bs[:]; k = 0
for b_ in range(256):
    if b_ not in bs: bs.append(b_); cs.append(256 + k); k += 1
for b_, c_ in zip(bs, cs): byte_dec[chr(c_)] = b_
def word(i):
    s = vocab.get(i, "")
    try: return bytes(byte_dec[ch] for ch in s).decode("utf-8")
    except Exception: return s
q = raw(idx, "model.embed_tokens.weight").reshape(V, C).view(np.int8)
s = raw(idx, "model.embed_tokens.weight.qs").astype(np.float32)
emb = q.astype(np.float32) * s[:, None]
emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
good = np.array([bool(w.strip()) and w.strip().isprintable() and len(w.strip()) > 1 and i < 151000 for i, w in ((i, word(i)) for i in range(V))])
words = []
for s0 in range(0, n, 2048):
    sc = Xn[s0:s0 + 2048] @ emb.T
    sc[:, ~good] = -9
    top = np.argpartition(-sc, a.words, axis=1)[:, :a.words]
    for r, t in enumerate(top):
        t = t[np.argsort(-sc[r, t])]
        words.append([word(int(i)).strip() for i in t])
del emb
log("words done")

# ---- exact cosine kNN
knn = np.zeros((n, K), np.uint32); knncos = np.zeros((n, K), np.float16)
for s0 in range(0, n, 2048):
    S = Xn[s0:s0 + 2048] @ Xn.T
    S[np.arange(len(S)), np.arange(s0, s0 + len(S))] = -2
    top = np.argpartition(-S, K, axis=1)[:, :K]
    sc = np.take_along_axis(S, top, axis=1); o = np.argsort(-sc, axis=1)
    knn[s0:s0 + 2048] = np.take_along_axis(top, o, axis=1); knncos[s0:s0 + 2048] = np.take_along_axis(sc, o, axis=1)
log("knn done")

# ---- layouts
mu = Xn.mean(0)
_, _, Vt = np.linalg.svd(Xn - mu, full_matrices=False)
P50 = (Xn - mu) @ Vt[:50].T
pca = P50[:, :3].astype(np.float32)
import umap
P = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.15, metric="cosine", random_state=7).fit_transform(P50).astype(np.float32)
def norm_points(P):
    c = (P.min(0) + P.max(0)) / 2; r = np.percentile(np.linalg.norm(P - c, axis=1), 99)
    return ((P - c) / r).astype(np.float32)
P = norm_points(P); pca = norm_points(pca)
log("layouts done")

# ---- categories: bands of layers
bands = 6; per = -(-len(layers) // bands)
cat_names = [f"layers {layers[i]}–{layers[min(i + per, len(layers)) - 1]}" for i in range(0, len(layers), per)]
cat = np.array([layers.index(m["layer"]) // per for m in meta_rows], np.uint8)

# ---- tiles (16^3 Morton) and write
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
cells = np.zeros(GRID ** 3 + 1, np.uint32); np.add.at(cells, cell[order].astype(np.int64) + 1, 1); cells = np.cumsum(cells).astype(np.uint32)
os.makedirs(a.out, exist_ok=True)
w = lambda name, arr: arr.tofile(os.path.join(a.out, name))
w("norms.f32", norms[order]); w("points.f32", P[order]); w("points.pca.f32", pca[order])
w("cat.u8", cat[order]); w("cells.u32", cells); w("ids.u32", order.astype(np.uint32))
w("knn.u32", inv[knn[order]].astype(np.uint32)); w("knncos.f16", knncos[order])
rb = C * 2; rows_per = int(a.max_part_mb * 1e6) // rb; parts = []
Xo = X[order]
for k, s0 in enumerate(range(0, n, rows_per)):
    name = f"vectors.{k:02d}.f16"; Xo[s0:s0 + rows_per].astype(np.float16).tofile(os.path.join(a.out, name))
    parts.append({"file": name, "bytes": int(min(rows_per, n - s0) * rb)})
label = lambda m, wd: f"L{m['layer']}·e{m['expert']} " + "·".join(wd[:2])
json.dump([label(meta_rows[i], words[i]) for i in order], open(os.path.join(a.out, "tokens.json"), "w"), ensure_ascii=False)
json.dump({"model": "GLM-5.2", "E": E, "layers": layers,
           "rows": [{**meta_rows[i], "words": words[i], "routed": usage.get((meta_rows[i]["layer"], meta_rows[i]["expert"]), 0)} for i in order]},
          open(os.path.join(a.out, "experts.json"), "w"), ensure_ascii=False, separators=(",", ":"))
meta = {
    "schema": "latent-fovea/1", "id": os.path.basename(os.path.normpath(a.out)), "kind": "experts",
    "model": {"name": "GLM-5.2 experts", "author": "Z.ai", "license": "MIT", "family": "experts", "params": f"{n:,} experts",
              "blurb": f"The {n:,} routed experts of GLM-5.2 ({len(layers)} layers × {E}), each placed by its row of the router: the direction in the hidden state that wakes it."},
    "n": int(n), "d": int(C), "k": K, "grid": GRID, "cells": GRID ** 3, "vector_dtype": "f16", "row_bytes": rb,
    "projection": {"method": "umap", "n_neighbors": 30, "min_dist": 0.15, "metric": "cosine", "pre": "pca50", "seed": 7},
    "bounds": [P.min(0).tolist(), P.max(0).tolist()], "categories": cat_names,
    "category_counts": [int((cat == i).sum()) for i in range(len(cat_names))],
    "vector_parts": parts, "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
log(f"wrote {a.out}: {n} rows, {sum(p['bytes'] for p in parts) / 1e6:.0f} MB of vectors in {len(parts)} parts")
