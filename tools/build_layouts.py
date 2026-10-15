#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Recourse Software Inc.
"""Precompute cloud layouts so the viewer can switch between clusters and PCA at once.

  --hf REPO ...     cluster maps (UMAP, as build_data.py) for models loaded from Hugging Face,
                    written to OUT/layouts/hf--owner--name.umap.f32 in *token-id order*
                    (the browser maps them onto its own tile order). Only the embedding
                    tensor is downloaded, by HTTP Range.
  --verified FILE   also every model listed in web/verified.json
  --pca DIR ...     PCA layouts for prebuilt data sets: DIR/points.pca.f32 in DIR's row order

  python3 tools/build_layouts.py --out web/data --verified web/verified.json --hf Qwen/Qwen2.5-0.5B-Instruct
  python3 tools/build_layouts.py --pca web/data/gpt2 web/data/distilgpt2 web/data/mdlm
"""
import argparse, json, os, sys, time, urllib.request
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_data import HF, SafeTensors, http_get, project, log  # noqa: E402

EMBED = {"gpt2": "wte.weight", "llama": "model.embed_tokens.weight", "qwen2": "model.embed_tokens.weight",
         "qwen3": "model.embed_tokens.weight", "gpt_neox": "gpt_neox.embed_in.weight"}


def scale(P):  # as build_data: median-centred, 98% of points inside radius 1
    P = P - np.median(P, 0)
    return (P / np.percentile(np.linalg.norm(P, axis=1), 98)).astype(np.float32)


def embedding(repo):
    cfg = json.loads(http_get(f"{HF}/{repo}/resolve/main/config.json"))
    suffix = EMBED[cfg["model_type"]]
    try:
        idx = json.loads(http_get(f"{HF}/{repo}/resolve/main/model.safetensors.index.json"))
        name = next(k for k in idx["weight_map"] if k == suffix or k.endswith("." + suffix))
        url = f"{HF}/{repo}/resolve/main/{idx['weight_map'][name]}"
    except Exception:  # noqa: BLE001  single-file model
        url = f"{HF}/{repo}/resolve/main/model.safetensors"
    _, X, _ = SafeTensors(url).read(suffix)
    return np.asarray(X, np.float32), cfg["model_type"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="web/data")
    ap.add_argument("--hf", nargs="*", default=[])
    ap.add_argument("--verified")
    ap.add_argument("--pca", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    for d in a.pca:
        meta = json.load(open(os.path.join(d, "meta.json")))
        X = np.fromfile(os.path.join(d, "vectors.f16"), np.float16).reshape(meta["n"], meta["d"]).astype(np.float32)
        P, _ = project(X, "pca", a.seed)
        scale(P).tofile(os.path.join(d, "points.pca.f32"))
        log(f"{d}: points.pca.f32")

    repos = list(a.hf)
    if a.verified:
        repos += [r for r in json.load(open(a.verified))["models"] if r not in repos]
    lay = os.path.join(a.out, "layouts")
    os.makedirs(lay, exist_ok=True)
    index_path = os.path.join(lay, "index.json")
    index = json.load(open(index_path)) if os.path.exists(index_path) else {}
    for repo in repos:
        fid = "hf--" + repo.replace("/", "--")
        out = os.path.join(lay, fid + ".umap.f32")
        if os.path.exists(out) and not a.force:
            log(f"{repo}: have it"); continue
        t0 = time.time()
        X, arch = embedding(repo)
        log(f"{repo}: {X.shape} ({arch}); UMAP…")
        P, info = project(X, "umap", a.seed)
        scale(P).tofile(out)
        index[repo] = {"file": fid + ".umap.f32", "n": int(X.shape[0]), "arch": arch, "projection": info,
                       "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0)}
        json.dump(index, open(index_path, "w"), indent=2)
        log(f"{repo}: done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
