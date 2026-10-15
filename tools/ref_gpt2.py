#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Recourse Software Inc.
"""numpy reference forward pass over the exported data, for checking the kernel.
    python3 tools/ref_gpt2.py <rows as JSON> [web/data/gpt2]"""
import json, sys, os
import numpy as np

rows = json.loads(sys.argv[1])
d = sys.argv[2] if len(sys.argv) > 2 else "web/data/gpt2"
man = json.load(open(os.path.join(d, "model/manifest.json")))
C, L, NH, F, T = 768, man["n_layer"], man["n_head"], man["d_mlp"], man["n_ctx"]
wte = np.fromfile(os.path.join(d, "vectors.f16"), np.float16).reshape(-1, C).astype(np.float32)
misc = np.fromfile(os.path.join(d, "model/misc.f16"), np.float16).astype(np.float32)
wpe, lnfw, lnfb = misc[: T * C].reshape(T, C), misc[T * C : T * C + C], misc[T * C + C :]
tokens = json.load(open(os.path.join(d, "tokens.json")))

def ln(x, w, b):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * w + b

def gelu(x): return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

x = wte[rows] + wpe[: len(rows)]
n = len(rows); att = {}
for l in range(L):
    flat = np.fromfile(os.path.join(d, "model", man["layer_files"][l]), np.float16).astype(np.float32)
    P, o = {}, 0
    for name, shape in man["layer_layout"]:
        k = int(np.prod(shape)); P[name] = flat[o : o + k].reshape(shape); o += k
    h = ln(x, P["ln_1.weight"], P["ln_1.bias"]) @ P["attn.c_attn.weight"] + P["attn.c_attn.bias"]
    q, k, v = np.split(h, 3, -1)
    q, k, v = [a.reshape(n, NH, 64).transpose(1, 0, 2) for a in (q, k, v)]
    s = q @ k.transpose(0, 2, 1) / 8.0 + np.triu(np.full((n, n), -1e10), 1)
    a = np.exp(s - s.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
    att[l] = a
    y = (a @ v).transpose(1, 0, 2).reshape(n, C)
    x = x + y @ P["attn.c_proj.weight"] + P["attn.c_proj.bias"]
    m = gelu(ln(x, P["ln_2.weight"], P["ln_2.bias"]) @ P["mlp.c_fc.weight"] + P["mlp.c_fc.bias"])
    x = x + m @ P["mlp.c_proj.weight"] + P["mlp.c_proj.bias"]
logits = ln(x[-1], lnfw, lnfb) @ wte.T
p = np.exp(logits - logits.max()); p /= p.sum()
top = np.argsort(-p)[:10]
print(json.dumps({"top": [[tokens[i], round(float(p[i]), 4)] for i in top],
                  "att_l0h0": [round(float(v), 4) for v in att[0][0, -1]],
                  "att_l5h1": [round(float(v), 4) for v in att[5][1, -1]]}))
