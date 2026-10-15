#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Recourse Software Inc.
"""numpy reference for one MDLM forward pass over the exported data (checks kernel/mdlm.c).
    python3 tools/ref_mdlm.py '<rows JSON>' [web/data/mdlm]"""
import json, os, sys
import numpy as np

rows = np.array(json.loads(sys.argv[1]))
d = sys.argv[2] if len(sys.argv) > 2 else "web/data/mdlm"
man = json.load(open(os.path.join(d, "model/manifest.json")))
C, L, NH, F = man["d_model"], man["n_layer"], man["n_head"], man["d_mlp"]
HS = C // NH
emb = np.fromfile(os.path.join(d, "vectors.f16"), np.float16).reshape(-1, C).astype(np.float32)
V = emb.shape[0]
mods = np.fromfile(os.path.join(d, "model/mods.f32"), np.float32)
final = np.fromfile(os.path.join(d, "model/final.f32"), np.float32)
unemb = np.fromfile(os.path.join(d, "model/unembed.f16"), np.float16).reshape(V, C).astype(np.float32)
tokens = json.load(open(os.path.join(d, "tokens.json")))
mask = man["mask_row"]

def ln(x, w):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * w

def gelu(x): return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

n = len(rows)
x = emb[rows]
inv = 1.0 / (10000 ** (np.arange(0, HS, 2) / HS))
ang = np.arange(n)[:, None] * inv[None]
cos, sin = np.cos(ang), np.sin(ang)  # [n, HS/2]
def rope(t):  # t [n, NH, HS]
    a, b = t[..., : HS // 2], t[..., HS // 2 :]
    return np.concatenate([a * cos[:, None] - b * sin[:, None], a * sin[:, None] + b * cos[:, None]], -1)
att_keep = {}
for l in range(L):
    flat = np.fromfile(os.path.join(d, "model", man["layer_files"][l]), np.float16).astype(np.float32)
    P, o = {}, 0
    for name, shape in man["block_layout"]:
        k = int(np.prod(shape)); P[name] = flat[o : o + k].reshape(shape); o += k
    sa, ca, ga, sm, cm, gm = mods[l * 6 * C : (l + 1) * 6 * C].reshape(6, C)
    h = ln(x, P["norm1.weight"]) * (1 + ca) + sa
    qkv = h @ P["attn_qkv.weight^T"]
    q, k, v = [qkv[:, i * C : (i + 1) * C].reshape(n, NH, HS) for i in range(3)]
    q, k = rope(q), rope(k)
    s = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(HS)
    a = np.exp(s - s.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
    att_keep[l] = a
    y = np.einsum("hqk,khd->qhd", a, v).reshape(n, C)
    x = x + ga * (y @ P["attn_out.weight^T"])
    h = gelu((ln(x, P["norm2.weight"]) * (1 + cm) + sm) @ P["mlp.0.weight^T"] + P["mlp.0.bias"])
    x = x + gm * (h @ P["mlp.2.weight^T"] + P["mlp.2.bias"])
sf, cf = mods[L * 6 * C :].reshape(2, C)
z = ln(x, final[:C]) * (1 + cf) + sf
logits = z @ unemb.T + final[C:]
logits[:, mask] = -np.inf
p = np.exp(logits - logits.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
out = {"beliefs": {}, "att_l0h0_q0": [round(float(t), 5) for t in att_keep[0][0, 0]], "att_l7h3_qlast": [round(float(t), 5) for t in att_keep[7][3, -1]]}
for i in np.where(rows == mask)[0][:4]:
    top = np.argsort(-p[i])[:5]
    out["beliefs"][int(i)] = [[tokens[j], round(float(p[i, j]), 5)] for j in top]
print(json.dumps(out))
