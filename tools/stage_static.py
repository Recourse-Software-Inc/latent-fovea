#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Recourse Software Inc.
"""Stage the viewer + built models for a static host with a per-file size limit.

  - vectors.f16 is split into row-aligned parts (meta.vector_parts, streamed by Range)
  - any other file over the limit is split into parts (meta.parts), reassembled on load
  - token JSON is written ASCII-escaped

Cloudflare Pages (25 MiB per file):
    python3 tools/stage_static.py site --max-mb 24
A host with a total size cap: one model's data here, others linked to where they are hosted:
    python3 tools/stage_static.py OUT --only gpt2 --link distilgpt2=URL
"""
import argparse, json, os, shutil

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--max-mb", type=float)
ap.add_argument("--only", help="stage only this model's data (for hosts with a total size cap)")
ap.add_argument("--link", action="append", default=[], metavar="ID=URL", help="list model ID in the picker as hosted at URL")
a = ap.parse_args()
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")
SUF = ""  # a suffix for binaries, for hosts that serve only web file types (meta.binary_suffix)
MAX = int((a.max_mb or 24) * 1e6)
if os.path.exists(a.out):
    shutil.rmtree(a.out)
os.makedirs(a.out)
for f in os.listdir(root):
    if f in ("glm-test.html", "glm-test.js"): continue  # a local self-check (needs fixtures under data/)
    if f.endswith((".js", ".wasm", ".css", ".jpg")) or f == "verified.json" or f.endswith(".html"):
        shutil.copy(os.path.join(root, f), a.out)
for sub in ("glm", "fonts"):  # subfolders: the GLM-5.2 engine and worker, the fonts
    if os.path.isdir(os.path.join(root, sub)):
        shutil.copytree(os.path.join(root, sub), os.path.join(a.out, sub))

data = os.path.join(root, "data")
os.makedirs(os.path.join(a.out, "data"))
if not os.path.isfile(os.path.join(data, "models.json")):
    raise SystemExit(f"no {data}/models.json: fetch data with tools/get_data.sh or build it with tools/build_data.py")
reg = json.load(open(os.path.join(data, "models.json")))
links = dict(x.split("=", 1) for x in a.link)
for m in reg["models"]:
    if m["id"] in links:
        m["url"] = links[m["id"]]
if a.only:
    reg["models"] = [m for m in reg["models"] if m["id"] == a.only or "url" in m]
    reg["default"] = a.only
json.dump(reg, open(os.path.join(a.out, "data", "models.json"), "w"), indent=2)
for m in reg["models"]:
    if "url" in m:
        continue
    src, dst = os.path.join(data, m["id"]), os.path.join(a.out, "data", m["id"])
    os.makedirs(os.path.join(dst, "model"), exist_ok=True)
    meta = json.load(open(os.path.join(src, "meta.json")))
    rb, rows_per = meta["row_bytes"], MAX // meta["row_bytes"]
    if meta.get("vector_parts") and all(p["bytes"] <= MAX for p in meta["vector_parts"]):
        for p in meta["vector_parts"]:  # already split by its builder (data too large for one file)
            shutil.copy(os.path.join(src, p["file"]), os.path.join(dst, p["file"] + SUF))
    else:
        if meta.get("vector_parts"):
            vec = b"".join(open(os.path.join(src, p["file"]), "rb").read() for p in meta["vector_parts"])
        else:
            vec = open(os.path.join(src, "vectors.f16"), "rb").read()
        parts = []
        for k, off in enumerate(range(0, len(vec), rows_per * rb)):
            chunk = vec[off : off + rows_per * rb]
            name = f"vectors.{k:02d}.f16"
            open(os.path.join(dst, name + SUF), "wb").write(chunk)
            parts.append({"file": name, "bytes": len(chunk)})
        meta["vector_parts"] = parts
    if SUF:
        meta["binary_suffix"] = SUF
    split = {}

    def put(src_path, rel):  # copy one data file, splitting it if it is over the limit
        name = rel + ("" if rel.endswith((".json", ".txt")) else SUF)
        size = os.path.getsize(src_path)
        if size <= MAX or rel.endswith((".json", ".txt")):
            shutil.copy(src_path, os.path.join(dst, name))
            return
        data = open(src_path, "rb").read()
        split[rel] = []
        for k, off in enumerate(range(0, size, MAX)):
            part = f"{rel}.part{k:02d}"
            open(os.path.join(dst, part + SUF), "wb").write(data[off : off + MAX])
            split[rel].append({"file": part, "bytes": len(data[off : off + MAX])})
    json.dump(json.load(open(os.path.join(src, "tokens.json"))), open(os.path.join(dst, "tokens.json"), "w"), ensure_ascii=True)
    for f in os.listdir(src):
        p = os.path.join(src, f)
        if f in ("meta.json", "tokens.json", "vectors.f16") or f.startswith("vectors.") or os.path.isdir(p):
            continue
        put(p, f)
    for sub in ("model", "kd"):  # the forward model's weights; a KnowDrive store's table and queries
        if os.path.isdir(os.path.join(src, sub)):
            os.makedirs(os.path.join(dst, sub), exist_ok=True)
            for f in os.listdir(os.path.join(src, sub)):
                put(os.path.join(src, sub, f), sub + "/" + f)
    if split:
        meta["parts"] = split
    json.dump(meta, open(os.path.join(dst, "meta.json"), "w"), indent=2)
lay = os.path.join(data, "layouts")  # precomputed cluster maps for models loaded from Hugging Face
if os.path.isdir(lay) and not a.only:
    shutil.copytree(lay, os.path.join(a.out, "data", "layouts"))
# Cloudflare Pages / Netlify-style headers: the dev server's security headers on every page, and
# caching for the model data
from serve import SECURITY_HEADERS
open(os.path.join(a.out, "_headers"), "w").write(
    "/*\n" + "".join(f"  {k}: {v}\n" for k, v in SECURITY_HEADERS.items()) +
    "\n/data/*\n  Cache-Control: public, max-age=86400\n\n/*.wasm\n  Cache-Control: public, max-age=3600\n")
big = [(os.path.getsize(os.path.join(d, f)), os.path.join(d, f)) for d, _, fs in os.walk(a.out) for f in fs]
print(f"staged {len(big)} files, {sum(s for s, _ in big) / 1e6:.0f} MB; largest {max(big)[0] / 1e6:.1f} MB")
