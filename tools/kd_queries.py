#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Recourse Software Inc.
"""Example queries for a KnowDrive data set built by tools/build_kdstore.py: KnowDB's own answers,
and each query's vector, recovered exactly from those answers.

    python3 tools/kd_queries.py --data web/data/canada \\
        --cache ~/.cache/latent-fovea/kdstore-canada --store vs_… --token-file ~/.cache/latent-fovea/token

KnowDrive's search does not return the query's embedding, and a browser cannot run Jina-CLIP-v2.
But the vector head's score for an atom is a function of one dot product, q·k, and every k is in
the data set; so a few hundred scored atoms pin down all 1,024 numbers of q (least squares, which
must then fit every score to rounding: the check is printed and stored). With q, the browser can
score every atom exactly, as the store does.

Writes kd/queries.json (per query: text, the three result lists as rows and scores, the fit) and
kd/q.NN.f32 (the recovered query vector) and kd/d.NN.f32 (its cosine distance to every atom).
"""
import argparse, json, os, sys, time, urllib.request
import numpy as np

QUERIES = [
    "lobbying on pharmaceutical drug pricing",
    "tariffs on steel and aluminum imports from the United States",
    "federal contracts for IT consulting services",
    "third-party election advertising and campaign contributions",
    "who lobbied about the carbon tax",
    "committee testimony on housing affordability",
    "softwood lumber duties",
    "clean drinking water on First Nations reserves",
    "regulating artificial intelligence",
    "grants to charities and non-profits",
]
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--cache", required=True)
ap.add_argument("--store", required=True); ap.add_argument("--base", default="https://public.knowdrive.ai")
ap.add_argument("--token-file", default=os.path.expanduser("~/.cache/latent-fovea/token"))
ap.add_argument("--fit-results", type=int, default=1500, help="vector-head results used to recover q")
ap.add_argument("--page", type=int, default=50, help="results kept per list for the app")
a = ap.parse_args()

def search(q, mode, offset, n=100):
    body = json.dumps({"query": q, "mode": mode, "max_num_results": n, "offset": offset}).encode()
    import hashlib
    cp = os.path.join(a.cache, "searches", hashlib.sha1(body).hexdigest() + ".json")  # pages kept, so a rerun needn't search again
    if os.path.exists(cp): return json.load(open(cp))
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    page = _search(body)
    json.dump(page, open(cp, "w"))
    return page
def _search(body):
    for attempt in range(6):
        req = urllib.request.Request(f"{a.base}/api/v1/vector_stores/{a.store}/search", data=body, method="POST",
                                     headers={"Authorization": "Bearer " + open(a.token_file).read().strip(), "Content-Type": "application/json"})
        try:
            return json.load(urllib.request.urlopen(req, timeout=60))
        except Exception as e:
            if attempt == 5: raise
            time.sleep(2 ** attempt)

meta = json.load(open(os.path.join(a.data, "meta.json")))
n, d = meta["n"], meta["d"]
store = json.load(open(os.path.join(a.data, "kd", "store.json")))
rowfile = np.fromfile(os.path.join(a.data, "kd", "rowfile.u32"), dtype=np.uint32)
rowseq = np.fromfile(os.path.join(a.data, "kd", "rowseq.u32"), dtype=np.uint32)
# The keys at full precision, as KnowDB holds them (the fetch cache), in the data set's row order.
V = np.zeros((n, d), np.float32)
where = {(int(f), int(q)): r for r, (f, q) in enumerate(zip(rowfile, rowseq))}
for j, f in enumerate(store["files"]):
    seqs = [m["seq"] for m in json.load(open(os.path.join(a.cache, "files", f["id"] + ".json")))]
    vf = np.fromfile(os.path.join(a.cache, "files", f["id"] + ".f32"), dtype="<f4").reshape(-1, d)
    for i, q in enumerate(seqs): V[where[(j, q)]] = vf[i]
V /= np.linalg.norm(V, axis=1, keepdims=True)
# Search hits carry the file and the atom's place in it (source_offset), not its sequence number;
# their text is the atom's with context around it, so text is no key. Fetch each atom's offset once
# (sequence and offset only, no vectors; cached beside the fetch) and match hits on (file, offset).
from concurrent.futures import ThreadPoolExecutor
off_path = os.path.join(a.cache, "offsets.json")
offsets = json.load(open(off_path)) if os.path.exists(off_path) else {}
def get(url):
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"Authorization": "Bearer " + open(a.token_file).read().strip()})
            return json.load(urllib.request.urlopen(req, timeout=60))
        except Exception:
            if attempt == 5: raise
            time.sleep(2 ** attempt)
def fetch_offsets(fid):
    seq_off, after = {}, ""
    while True:
        page = get(f"{a.base}/api/v1/vector_stores/{a.store}/files/{fid}/atoms?fields=sequence_index,metadata&limit=100" + (f"&after={after}" if after else ""))
        for at in page["data"]:
            md = at.get("metadata") or {}
            if isinstance(md, str): md = json.loads(md)
            seq_off[str(at["sequence_index"])] = md.get("source_offset")
        after = page.get("next_after") if page.get("has_more") else ""
        if not after: return fid, seq_off
todo = [f["id"] for f in store["files"] if f["id"] not in offsets]
if todo:
    with ThreadPoolExecutor(8) as ex:
        for k, (fid, so) in enumerate(ex.map(fetch_offsets, todo)):
            offsets[fid] = so
            if k % 500 == 0: print(f"offsets {k}/{len(todo)}", flush=True)
    json.dump(offsets, open(off_path, "w"))
fid_of = {f["id"]: j for j, f in enumerate(store["files"])}
by_off = {}  # (file, offset) -> rows; a long passage split into several atoms shares one offset
for r in range(n):
    f = store["files"][int(rowfile[r])]["id"]
    by_off.setdefault((f, offsets.get(f, {}).get(str(int(rowseq[r])))), []).append(r)
norm = lambda t: " ".join((t or "").split())
texts = {}
def opening(r):  # the atom's own text, from the fetch cache
    f = store["files"][int(rowfile[r])]["id"]
    if f not in texts: texts[f] = {m["seq"]: norm(m["text"])[:80] for m in json.load(open(os.path.join(a.cache, "files", f + ".json")))}
    return texts[f].get(int(rowseq[r]), "")
def row_of(hit):
    c = by_off.get((hit.get("file_id"), (hit.get("attributes") or {}).get("source_offset")), [])
    if len(c) > 1:  # siblings: the hit's text starts with its own atom's, so the earliest opening wins
        t = norm(hit["content"][0]["text"]) if hit.get("content") else ""
        pos = sorted((t.find(opening(r)), r) for r in c if opening(r) and opening(r) in t)
        c = [pos[0][1]] if pos else []
    return c[0] if len(c) == 1 else -1

out = []
for qi, text in enumerate(QUERIES):
    lists = {}
    for mode in ("vector", "hybrid", "fts"):
        hits, off, want = [], 0, a.fit_results if mode == "vector" else a.page
        while len(hits) < want:
            page = search(text, mode, off, min(100, want - len(hits)))
            hits += page["data"]; off += len(page["data"])
            if not page.get("has_more") or not page["data"]: break
        lists[mode] = {"score_kind": page.get("score_kind"), "hits": [(row_of(h), float(h["score"])) for h in hits]}
    # Recover q from the vector head: score = q·k or 1 - q·k (cosine similarity or distance); try both.
    pairs = [(r, s) for r, s in lists["vector"]["hits"] if r >= 0]
    K = V[[r for r, _ in pairs]]; s = np.array([s for _, s in pairs], np.float64)
    best = None
    for kind, y in (("similarity", s), ("distance", 1 - s)):
        q, *_ = np.linalg.lstsq(K.astype(np.float64), y, rcond=None)
        res = np.abs(K @ q - y).max()
        if best is None or res < best[2]: best = (kind, q, res)
    kind, q, res = best
    qn = q / np.linalg.norm(q)
    cos = V @ qn.astype(np.float32)
    # Does the browser's ranking (cosine over every atom) reproduce KnowDB's vector list?
    mine = np.argsort(-cos)[:a.page]
    theirs = [r for r, _ in lists["vector"]["hits"][:a.page]]
    agree = sum(1 for x, y in zip(mine, theirs) if x == y)
    print(f"{qi:2d} {text[:48]:48s} fit on {len(pairs)} atoms ({kind}): |q|={np.linalg.norm(q):.4f} max error {res:.2e}; top-{a.page} order matches {agree}/{a.page}", flush=True)
    np.asarray(qn, np.float32).tofile(os.path.join(a.data, "kd", f"q.{qi:02d}.f32"))
    (1 - cos).astype(np.float32).tofile(os.path.join(a.data, "kd", f"d.{qi:02d}.f32"))  # every atom's cosine distance (32-bit: near-ties keep their order)
    out.append({"text": text, "vector": f"q.{qi:02d}.f32", "dist": f"d.{qi:02d}.f32", "score_kind": kind,
                "fit": {"atoms": len(pairs), "norm": float(np.linalg.norm(q)), "max_error": float(res), "order_agree": agree},
                "lists": {m: {"score_kind": L["score_kind"], "hits": L["hits"][:a.page]} for m, L in lists.items()}})
json.dump({"store": a.store, "host": a.base, "queries": out}, open(os.path.join(a.data, "kd", "queries.json"), "w"), indent=1)
print("wrote", os.path.join(a.data, "kd", "queries.json"))
