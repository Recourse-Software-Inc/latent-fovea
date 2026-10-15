#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Recourse Software Inc.
"""Build the latent-fovea data set from an open-weights model.

v0 target: GPT-2 small's token-embedding matrix (wte, 50,257 x 768).

Only the bytes of the one tensor we need are downloaded: the safetensors
header is read first, then a single HTTP Range request fetches wte.

Outputs (all little-endian, points in *tile order*):
  meta.json      model, licence, shapes, grid, file index
  tokens.json    decoded token strings, tile order
  ids.u32        tile order -> original token id
  cat.u8         token category (see CATEGORIES)
  points.f32     n x 3 global 3-D projection (UMAP, cosine), unit-ish radius
  norms.f32      L2 norm of each embedding row
  cells.u32      G^3 + 1 prefix offsets; cell c holds rows [cells[c], cells[c+1])
  knn.u32        n x K exact cosine neighbours (tile-order indices)
  knncos.f16     n x K their cosine similarities
  vectors.f16    n x d embedding rows, grouped by cell (Morton order): a cell is
                 one contiguous byte range, i.e. one streamable tile.
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import time
import http.client
import urllib.error
import urllib.request

import numpy as np

HF = "https://huggingface.co"
MODELS = {
    "gpt2": {
        "repo": "openai-community/gpt2",
        "name": "GPT-2 small (124M)",
        "author": "OpenAI",
        "license": "MIT",
        "tensor_suffix": "wte.weight",
        "tensor_label": "token embedding (wte)",
        "params": "124M", "family": "causal-lm",
        "blurb": "OpenAI's 2019 model, the classic subject of interpretability research.",
    },
    "distilgpt2": {
        "repo": "distilbert/distilgpt2",
        "name": "DistilGPT2 (82M)",
        "author": "Hugging Face",
        "license": "Apache-2.0",
        "tensor_suffix": "wte.weight",
        "tensor_label": "token embedding (wte)",
        "params": "82M", "family": "causal-lm",
        "blurb": "GPT-2 distilled to half the depth: 6 layers instead of 12, same vocabulary.",
    },
    "mdlm": {
        "repo": "kuleshov-group/mdlm-owt",
        "tokenizer_repo": "openai-community/gpt2",
        "name": "MDLM (OpenWebText, 169M)",
        "author": "Kuleshov Group, Cornell",
        "license": "Apache-2.0",
        "tensor_suffix": "vocab_embed.embedding",
        "tensor_label": "input token embedding (vocab_embed)",
        "params": "169M", "family": "masked-diffusion",
        "blurb": "From Sahoo et al., 2024, trained on OpenWebText. It uses GPT-2's vocabulary, so the same query can be compared with GPT-2.",
        "extra_tokens": {50257: "[MASK]"},
    },
}
CATEGORIES = ["word", "subword", "number", "punctuation", "whitespace", "other"]
GRID = 16  # cells per axis
K = 16


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def http_get(url, rng=None, tries=5):
    for attempt in range(tries):  # large downloads sometimes drop mid-transfer: retry
        req = urllib.request.Request(url, headers={"User-Agent": "latent-fovea/0.1"})
        if rng:
            req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
                if rng and r.status != 206:  # server ignored Range: slice locally
                    data = data[rng[0] : rng[1] + 1]
                return data
        except (http.client.IncompleteRead, ConnectionError, TimeoutError, urllib.error.URLError) as e:
            if attempt == tries - 1 or (isinstance(e, urllib.error.HTTPError) and e.code < 500 and e.code != 429):
                raise  # a missing file will not appear on retry
                raise
            log(f"retrying {url.rsplit('/', 1)[-1]} after {type(e).__name__}")
            time.sleep(2 ** attempt)


class SafeTensors:
    """Reads tensors from a local or remote (HTTP Range) .safetensors file."""

    def __init__(self, src):
        self.src = src
        self.remote = src.startswith("http")
        if self.remote:
            hlen = struct.unpack("<Q", http_get(src, (0, 7)))[0]
            self.header = json.loads(http_get(src, (8, 8 + hlen - 1)))
        else:
            self.f = open(src, "rb")
            hlen = struct.unpack("<Q", self.f.read(8))[0]
            self.header = json.loads(self.f.read(hlen))
        self.base = 8 + hlen
        self.names = [k for k in self.header if k != "__metadata__"]

    def find(self, suffix):
        return next(k for k in self.names if k == suffix or k.endswith("." + suffix))

    def read(self, suffix):
        name = self.find(suffix)
        info = self.header[name]
        assert info["dtype"] in ("F32", "F16", "BF16"), info["dtype"]
        b0, b1 = info["data_offsets"]
        start = self.base + b0
        if self.remote:
            raw = http_get(self.src, (start, start + (b1 - b0) - 1))
        else:
            self.f.seek(start)
            raw = self.f.read(b1 - b0)
        if info["dtype"] == "BF16":  # bfloat16 is the top half of a float32
            arr = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(info["shape"])
        else:
            dt = np.float32 if info["dtype"] == "F32" else np.float16
            arr = np.frombuffer(raw, dtype=dt).reshape(info["shape"]).astype(np.float32)
        return name, arr, hashlib.sha256(raw).hexdigest()


class TorchBin:
    """Same interface as SafeTensors for checkpoints only published as pytorch_model.bin."""

    def __init__(self, url):
        import io, torch
        sd = torch.load(io.BytesIO(http_get(url)), map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        self.sd = {k: v for k, v in sd.items() if hasattr(v, "shape")}
        self.names = list(self.sd)

    def find(self, suffix):
        return next(k for k in self.names if k == suffix or k.endswith("." + suffix))

    def read(self, suffix):
        name = self.find(suffix)
        arr = self.sd[name].float().numpy()
        return name, arr, hashlib.sha256(arr.tobytes()).hexdigest()


def open_weights(repo, local=None):
    if local:
        return SafeTensors(local), local
    url = f"{HF}/{repo}/resolve/main/model.safetensors"
    try:
        return SafeTensors(url), url
    except Exception as e:  # noqa: BLE001
        log(f"no model.safetensors ({e}); falling back to pytorch_model.bin")
        url = f"{HF}/{repo}/resolve/main/pytorch_model.bin"
        return TorchBin(url), url


def read_tensor(src, suffix):
    st = SafeTensors(src)
    name, arr, digest = st.read(suffix)
    log(f"tensor {name} {list(arr.shape)} = {arr.nbytes / 1e6:.1f} MB")
    return name, arr, digest


# Order of tensors inside each layerNN.f16 file. The kernel reads the same order.
LAYER_LAYOUT = [
    ("ln_1.weight", [768]), ("ln_1.bias", [768]),
    ("attn.c_attn.weight", [768, 2304]), ("attn.c_attn.bias", [2304]),
    ("attn.c_proj.weight", [768, 768]), ("attn.c_proj.bias", [768]),
    ("ln_2.weight", [768]), ("ln_2.bias", [768]),
    ("mlp.c_fc.weight", [768, 3072]), ("mlp.c_fc.bias", [3072]),
    ("mlp.c_proj.weight", [3072, 768]), ("mlp.c_proj.bias", [768]),
]


def export_weights(src, repo, out):
    """Write every non-embedding GPT-2 weight as fp16 files under out/model/.
    wte is not repeated: the kernel reuses vectors.f16 (tile order) + ids.u32."""
    st = SafeTensors(src)
    cfg = json.loads(http_get(f"{HF}/{repo}/resolve/main/config.json"))
    n_layer, n_head = cfg.get("n_layer", 12), cfg.get("n_head", 12)
    assert cfg.get("n_embd", 768) == 768 and n_head == 12, "the kernel is built for 768-wide, 12-head GPT-2 blocks"
    mdir = os.path.join(out, "model")
    os.makedirs(mdir, exist_ok=True)
    files = []
    for l in range(n_layer):
        parts = []
        for name, shape in LAYER_LAYOUT:
            _, arr, _ = st.read(f"h.{l}.{name}")
            assert list(arr.shape) == shape, (l, name, arr.shape)
            parts.append(arr.astype(np.float16).ravel())
        fn = f"layer{l:02d}.f16"
        np.concatenate(parts).tofile(os.path.join(mdir, fn))
        files.append(fn)
        log(f"  wrote {fn}")
    wpe = st.read("wpe.weight")[1]
    lnf_w, lnf_b = st.read("ln_f.weight")[1], st.read("ln_f.bias")[1]
    np.concatenate([wpe.ravel(), lnf_w, lnf_b]).astype(np.float16).tofile(os.path.join(mdir, "misc.f16"))
    for fn in ("vocab.json", "merges.txt"):
        with open(os.path.join(mdir, fn), "wb") as f:
            f.write(http_get(f"{HF}/{repo}/resolve/main/{fn}"))
    manifest = {
        "n_layer": n_layer, "n_head": n_head, "d_model": 768, "d_mlp": 3072, "n_ctx": int(wpe.shape[0]),
        "dtype": "f16", "layer_layout": LAYER_LAYOUT, "layer_files": files,
        "misc": [["wpe.weight", list(wpe.shape)], ["ln_f.weight", [768]], ["ln_f.bias", [768]]],
        "wte": "../vectors.f16 in tile order; ../ids.u32 maps tile row -> token id",
        "tokenizer": ["vocab.json", "merges.txt"],
    }
    json.dump(manifest, open(os.path.join(mdir, "manifest.json"), "w"), indent=2)


def bytes_to_unicode():
    # GPT-2's reversible byte <-> printable-unicode map.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def decode_vocab(vocab, n):
    inv = {v: k for k, v in bytes_to_unicode().items()}
    out = [""] * n
    for tok, i in vocab.items():
        if i < n:
            out[i] = bytes(inv[c] for c in tok).decode("utf-8", errors="replace")
    return out


def category(s):
    if s == "" or s.isspace():
        return 4
    core = s[1:] if s.startswith(" ") else s
    if core.isdigit() or (core.replace(",", "").replace(".", "").isdigit() and core[:1].isdigit()):
        return 2
    if core.isascii() and core.isalpha():
        return 0 if s.startswith(" ") else 1
    if all((not c.isalnum()) and c.isascii() for c in core):
        return 3
    return 5


def morton3(x, y, z):
    def spread(v):
        v = v.astype(np.uint32)
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        v = (v | (v << 2)) & 0x09249249
        return v
    return spread(x) | (spread(y) << 1) | (spread(z) << 2)


def project(X, method, seed):
    Xc = X - X.mean(0)
    # PCA-50 first: denoises and makes UMAP fast.
    U, S, Vt = np.linalg.svd(Xc[:: max(1, len(X) // 20000)], full_matrices=False)
    P50 = Xc @ Vt[:50].T
    if method == "pca":
        return P50[:, :3], {"method": "pca"}
    import umap
    reducer = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.15, metric="cosine", random_state=seed, low_memory=True)
    return reducer.fit_transform(P50).astype(np.float32), {
        "method": "umap", "n_neighbors": 30, "min_dist": 0.15, "metric": "cosine", "pre": "pca50", "seed": seed,
    }


def exact_knn(Xs, k, block=2048):
    Xn = Xs / np.maximum(np.linalg.norm(Xs, axis=1, keepdims=True), 1e-12)
    n = len(Xn)
    idx = np.empty((n, k), np.uint32)
    sim = np.empty((n, k), np.float32)
    for s in range(0, n, block):
        S = Xn[s : s + block] @ Xn.T
        for r in range(S.shape[0]):
            S[r, s + r] = -2.0  # exclude self
        top = np.argpartition(-S, k, axis=1)[:, :k]
        ts = np.take_along_axis(S, top, 1)
        order = np.argsort(-ts, 1)
        idx[s : s + block] = np.take_along_axis(top, order, 1)
        sim[s : s + block] = np.take_along_axis(ts, order, 1)
        if (s // block) % 5 == 0:
            log(f"  knn {s + S.shape[0]}/{n}")
    return idx, sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2", choices=MODELS)
    ap.add_argument("--out", default="web/data/gpt2")
    ap.add_argument("--safetensors", help="local file instead of downloading")
    ap.add_argument("--vocab", help="local vocab.json instead of downloading")
    ap.add_argument("--projection", default="umap", choices=["umap", "pca"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--weights", action="store_true", help="also export the full model for the forward pass")
    ap.add_argument("--registry", metavar="DIR", help="only (re)write DIR/models.json from the models built there")
    a = ap.parse_args()
    if a.registry:
        print(write_registry(a.registry))
        return
    m = MODELS[a.model]
    os.makedirs(a.out, exist_ok=True)

    st, src = open_weights(m["repo"], a.safetensors)
    log("reading", src)
    tname, X, digest = st.read(m["tensor_suffix"])
    log(f"tensor {tname} {list(X.shape)}")
    n, d = X.shape
    vrepo = m.get("tokenizer_repo", m["repo"])
    vocab = json.load(open(a.vocab)) if a.vocab else json.loads(http_get(f"{HF}/{vrepo}/resolve/main/vocab.json"))
    tokens = decode_vocab(vocab, n)
    for i, t in m.get("extra_tokens", {}).items():
        if i < n:
            tokens[i] = t
    cats = np.array([category(s) for s in tokens], np.uint8)

    log(f"projecting {n} x {d} with {a.projection}")
    P, proj_info = project(X, a.projection, a.seed)
    P = P - np.median(P, 0)
    P /= np.percentile(np.linalg.norm(P, axis=1), 98)

    lo, hi = P.min(0), P.max(0)
    g = np.clip(((P - lo) / (hi - lo + 1e-9) * GRID).astype(int), 0, GRID - 1)
    cell = morton3(g[:, 0], g[:, 1], g[:, 2])
    order = np.lexsort((np.arange(n), cell))
    cells = np.zeros(GRID**3 + 1, np.uint32)
    np.add.at(cells, cell[order].astype(np.int64) + 1, 1)
    cells = np.cumsum(cells).astype(np.uint32)

    Xs, Ps = X[order], P[order].astype(np.float32)
    log("exact cosine kNN")
    kidx, ksim = exact_knn(Xs, K)

    def w(name, arr):
        arr.tofile(os.path.join(a.out, name))
    w("ids.u32", order.astype(np.uint32))
    w("cat.u8", cats[order])
    w("points.f32", Ps)
    w("norms.f32", np.linalg.norm(Xs, axis=1).astype(np.float32))
    w("cells.u32", cells)
    w("knn.u32", kidx)
    w("knncos.f16", ksim.astype(np.float16))
    w("vectors.f16", Xs.astype(np.float16))
    json.dump([tokens[i] for i in order], open(os.path.join(a.out, "tokens.json"), "w"), ensure_ascii=True)  # escapes U+FFFD from partial-byte tokens

    if a.weights:
        log("exporting full model weights")
        if m["family"] == "masked-diffusion":
            export_mdlm(st, m["repo"], a.out, order)
        else:
            export_weights(src, m["repo"], a.out)

    meta = {
        "schema": "latent-fovea/1",
        "id": a.model,
        "model": {**{k: m[k] for k in ("name", "author", "license", "params", "family", "blurb")}, "repo": m["repo"],
                  "source": f"{HF}/{m['repo']}", "tensor": tname, "tensor_label": m["tensor_label"],
                  "tensor_sha256": digest},
        "n": int(n), "d": int(d), "k": K, "grid": GRID, "cells": GRID**3,
        "vector_dtype": "f16", "row_bytes": int(d * 2),
        "projection": proj_info, "bounds": [lo.tolist(), hi.tolist()],
        "categories": CATEGORIES,
        "category_counts": np.bincount(cats, minlength=len(CATEGORIES)).tolist(),
        "files": sorted(f for f in os.listdir(a.out) if f != "model") + ["meta.json"],
        "forward": "model/manifest.json" if a.weights else None,
        "forward_bytes": (sum(os.path.getsize(os.path.join(a.out, "model", f)) for f in os.listdir(os.path.join(a.out, "model")))
                          + os.path.getsize(os.path.join(a.out, "vectors.f16"))) if a.weights else None,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    log("done", {f: os.path.getsize(os.path.join(a.out, f)) for f in sorted(os.listdir(a.out))})


def export_mdlm(st, repo, out, order):
    """MDLM (DiT backbone, masked diffusion). time_conditioning is off for mdlm-owt, so the
    conditioning vector c = silu(sigma_map(0)) is a constant and every adaLN modulation is
    precomputed here. Linear weights are transposed to [in][out]; the unembedding is stored in
    tile order so logits index the viewer's point order directly."""
    cfg = json.loads(http_get(f"{HF}/{repo}/resolve/main/config.json"))
    log("mdlm config", {k: cfg.get(k) for k in ("hidden_dim", "cond_dim", "n_blocks", "n_heads", "model_length", "time_conditioning", "vocab_size")})
    if cfg.get("time_conditioning"):
        raise SystemExit("this checkpoint uses time conditioning; the kernel expects constant modulations")
    C, NH, NB = cfg.get("hidden_dim", 768), cfg.get("n_heads", 12), cfg.get("n_blocks", 12)
    assert C == 768 and NH == 12, "the kernel is built for 768-wide, 12-head blocks"
    rd = lambda sfx: st.read(sfx)[1]
    # c = silu(mlp(timestep_embedding(0))); timestep_embedding(0) = [cos 0]*128 + [sin 0]*128
    tfreq = np.concatenate([np.ones(128, np.float32), np.zeros(128, np.float32)])
    h = rd("sigma_map.mlp.0.weight") @ tfreq + rd("sigma_map.mlp.0.bias")
    h = h / (1 + np.exp(-h))
    c = rd("sigma_map.mlp.2.weight") @ h + rd("sigma_map.mlp.2.bias")
    c = (c / (1 + np.exp(-c))).astype(np.float32)
    mdir = os.path.join(out, "model")
    os.makedirs(mdir, exist_ok=True)
    mods, files = [], []
    for l in range(NB):
        b = f"blocks.{l}."
        mods.append(rd(b + "adaLN_modulation.weight") @ c + rd(b + "adaLN_modulation.bias"))  # 6*C
        parts = [rd(b + "norm1.weight"), rd(b + "attn_qkv.weight").T, rd(b + "attn_out.weight").T, rd(b + "norm2.weight"),
                 rd(b + "mlp.0.weight").T, rd(b + "mlp.0.bias"), rd(b + "mlp.2.weight").T, rd(b + "mlp.2.bias")]
        fn = f"block{l:02d}.f16"
        np.concatenate([x.astype(np.float32).ravel() for x in parts]).astype(np.float16).tofile(os.path.join(mdir, fn))
        files.append(fn)
        log(f"  wrote {fn}")
    mods.append(rd("output_layer.adaLN_modulation.weight") @ c + rd("output_layer.adaLN_modulation.bias"))  # 2*C
    np.concatenate([m.astype(np.float32).ravel() for m in mods]).astype(np.float32).tofile(os.path.join(mdir, "mods.f32"))
    W, bias = rd("output_layer.linear.weight"), rd("output_layer.linear.bias")  # [V, C], [V]
    np.concatenate([rd("output_layer.norm_final.weight"), bias[order]]).astype(np.float32).tofile(os.path.join(mdir, "final.f32"))
    W[order].astype(np.float16).tofile(os.path.join(mdir, "unembed.f16"))
    tok = MODELS["mdlm"]["tokenizer_repo"]
    for fn in ("vocab.json", "merges.txt"):
        with open(os.path.join(mdir, fn), "wb") as f:
            f.write(http_get(f"{HF}/{tok}/resolve/main/{fn}"))
    mask_row = int(np.where(order == 50257)[0][0])
    json.dump({
        "arch": "mdlm", "n_layer": NB, "n_head": NH, "d_model": C, "d_mlp": 4 * C, "n_ctx": cfg.get("model_length", 1024),
        "vocab": int(W.shape[0]), "mask_id": 50257, "mask_row": mask_row, "time_conditioning": False,
        "block_layout": [["norm1.weight", [C]], ["attn_qkv.weight^T", [C, 3 * C]], ["attn_out.weight^T", [C, C]], ["norm2.weight", [C]],
                         ["mlp.0.weight^T", [C, 4 * C]], ["mlp.0.bias", [4 * C]], ["mlp.2.weight^T", [4 * C, C]], ["mlp.2.bias", [C]]],
        "layer_files": files, "mods": "mods.f32: per block 6*C (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp), then final 2*C (shift, scale)",
        "final": "final.f32: norm_final.weight [C] then output bias [V] in tile order", "unembed": "unembed.f16: [V][C] in tile order",
        "embed": "../vectors.f16 (vocab_embed in tile order)", "tokenizer": ["vocab.json", "merges.txt"],
        "sampler": {"kind": "ddpm_cache", "eps": 1e-5, "steps": 64},
    }, open(os.path.join(mdir, "manifest.json"), "w"), indent=2)


def write_registry(root):
    """models.json at the data root: every built model, for the viewer's picker."""
    models = []
    for d in sorted(os.listdir(root)):
        mp = os.path.join(root, d, "meta.json")
        if not os.path.isfile(mp):
            continue
        meta = json.load(open(mp))
        mm = meta["model"]
        models.append({"id": d, "path": d + "/", "name": mm["name"], "author": mm["author"], "license": mm["license"],
                       "params": mm.get("params"), "family": mm.get("family", "causal-lm"), "blurb": mm.get("blurb", ""),
                       "forward": bool(meta.get("forward"))})
        if meta.get("kind") == "kdstore":  # a KnowDrive store: its size, and what must load before it opens
            files = [os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(root, d)) for f in fs]
            rel = lambda f: os.path.relpath(f, os.path.join(root, d))
            lazy = lambda r: r.startswith(("vectors", "kd/snip.", "kd/d.", "kd/q.", "ids."))  # streamed or fetched on demand
            models[-1].update({"kind": "kdstore", "atoms": meta["n"], "files": meta["model"].get("files"), "host": meta["model"].get("host"),
                               "bytes": sum(os.path.getsize(f) for f in files),
                               "open_bytes": sum(os.path.getsize(f) for f in files if not lazy(rel(f)))})
    order = list(MODELS)
    models.sort(key=lambda m: order.index(m["id"]) if m["id"] in order else len(order))
    json.dump({"schema": "latent-fovea-models/1", "default": models[0]["id"] if models else None, "models": models},
              open(os.path.join(root, "models.json"), "w"), indent=2)
    return models


if __name__ == "__main__":
    sys.exit(main())
