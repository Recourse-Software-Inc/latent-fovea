#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Recourse Software Inc.
"""The NumPy reference of tools/glm_ref.py, for the real 744B container: layer by layer over a short
prompt, loading one layer's weights at a time and only the experts the router picks, so it fits in
memory. Writes the residual after every layer at every position (and the chosen experts and final
argmax), to compare with web/glm/glm-gpu.js on the same tokens.

    python tools/glm_ref_stream.py --model /Volumes/<drive>/glm52_i4 --ids 576,729,1265 --out <dir>
"""
import argparse, json, math, os, time
import numpy as np
from glm_ref import read_container, raw, weight

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--ids", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--layers", type=int, default=0, help="stop after this many layers (0: all)")
a = ap.parse_args()
ids = [int(x) for x in a.ids.split(",")]
c = json.load(open(os.path.join(a.model, "config.json")))
idx = read_container(a.model)
C, H, QL, KL = c["hidden_size"], c["num_attention_heads"], c["q_lora_rank"], c["kv_lora_rank"]
NOPE, R, VD, V = c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"], c["vocab_size"]
L = a.layers or c["num_hidden_layers"]; dense = c["first_k_dense_replace"]; K = c["num_experts_per_tok"]
FD, FE, eps = c["intermediate_size"], c["moe_intermediate_size"], c["rms_norm_eps"]
theta = (c.get("rope_parameters") or {}).get("rope_theta", c.get("rope_theta"))
W = lambda n, O, I: weight(idx, n, O, I)
F = lambda n: raw(idx, n).astype(np.float32).ravel()
rms = lambda x, w: x / np.sqrt(np.mean(x * x) + eps) * w
silu = lambda z: z / (1 + np.exp(-z))
def rope(v, pos):
    j = np.arange(R // 2); ang = pos * theta ** (-2.0 * j / R)
    a_, b_ = v[..., 0::2], v[..., 1::2]
    return np.concatenate([a_ * np.cos(ang) - b_ * np.sin(ang), a_ * np.sin(ang) + b_ * np.cos(ang)], axis=-1)

T = len(ids)
X = np.zeros((T, C), np.float32)
# embedding rows only (the full table is 951 MB as int8; 3.8 GB dequantized)
q = raw(idx, "model.embed_tokens.weight").reshape(V, C); s = raw(idx, "model.embed_tokens.weight.qs").astype(np.float32)
for t, tok in enumerate(ids): X[t] = q[tok].view(np.int8).astype(np.float32) * s[tok]
res = [X.copy()]; experts = []
t0 = time.time()
for l in range(L):
    p = f"model.layers.{l}."; at = p + "self_attn."
    ln1, ln2 = F(p + "input_layernorm.weight"), F(p + "post_attention_layernorm.weight")
    qa, qaln = W(at + "q_a_proj.weight", QL, C), F(at + "q_a_layernorm.weight")
    qb = W(at + "q_b_proj.weight", H * (NOPE + R), QL)
    kva, kvaln = W(at + "kv_a_proj_with_mqa.weight", KL + R, C), F(at + "kv_a_layernorm.weight")
    kvb = W(at + "kv_b_proj.weight", H * (NOPE + VD), KL).reshape(H, NOPE + VD, KL)
    o = W(at + "o_proj.weight", C, H * VD)
    Lc, Rc = [], []
    for t in range(T):  # attention, causal over the latent cache
        h = rms(X[t], ln1)
        qq = (qb @ rms(qa @ h, qaln)).reshape(H, NOPE + R)
        qn, qp = qq[:, :NOPE], rope(qq[:, NOPE:], t)
        ckv = kva @ h
        Lc.append(rms(ckv[:KL], kvaln)); Rc.append(rope(ckv[KL:], t))
        Lm, Rm = np.array(Lc), np.array(Rc)
        ctx = np.empty((H, VD), np.float32)
        for hh in range(H):
            qabs = qn[hh] @ kvb[hh, :NOPE]
            sc = (Lm @ qabs + Rm @ qp[hh]) / math.sqrt(NOPE + R)
            pr = np.exp(sc - sc.max()); pr /= pr.sum()
            ctx[hh] = kvb[hh, NOPE:] @ (pr @ Lm)
        X[t] = X[t] + o @ ctx.ravel()
    m = p + "mlp."
    if l < dense:
        g_, u_, d_ = W(m + "gate_proj.weight", FD, C), W(m + "up_proj.weight", FD, C), W(m + "down_proj.weight", C, FD)
        for t in range(T):
            h = rms(X[t], ln2); X[t] = X[t] + d_ @ (silu(g_ @ h) * (u_ @ h))
        experts.append([[] for _ in range(T)])
    else:
        router, bias = W(m + "gate.weight", c["n_routed_experts"], C), F(m + "gate.e_score_correction_bias")
        sg, su, sd = W(m + "shared_experts.gate_proj.weight", FE, C), W(m + "shared_experts.up_proj.weight", FE, C), W(m + "shared_experts.down_proj.weight", C, FE)
        hs = [rms(X[t], ln2) for t in range(T)]
        picks = []
        for t in range(T):
            sig = 1 / (1 + np.exp(-(router @ hs[t])))
            top = np.argsort(-(sig + bias), kind="stable")[:K]
            w = sig[top]; w = w / w.sum() * c["routed_scaling_factor"]
            picks.append(list(zip(top.tolist(), w.tolist())))
        cache = {}
        for e in sorted({e for pk in picks for e, _ in pk}):
            cache[e] = (W(m + f"experts.{e}.gate_proj.weight", FE, C), W(m + f"experts.{e}.up_proj.weight", FE, C), W(m + f"experts.{e}.down_proj.weight", C, FE))
        for t in range(T):
            h = hs[t]; y = sd @ (silu(sg @ h) * (su @ h))
            for e, we in picks[t]:
                g_, u_, d_ = cache[e]; y = y + we * (d_ @ (silu(g_ @ h) * (u_ @ h)))
            X[t] = X[t] + y
        experts.append([[e for e, _ in pk] for pk in picks])
    res.append(X.copy())
    print(f"layer {l} done at {time.time() - t0:.0f} s; |x| {np.abs(X).max():.2f}", flush=True)
out = {"ids": ids, "layers": L, "experts": experts}
if L == c["num_hidden_layers"]:
    head = raw(idx, "lm_head.weight").reshape(V, C).view(np.int8); hs_ = raw(idx, "lm_head.weight.qs").astype(np.float32)
    nrm = F("model.norm.weight")
    head = head.astype(np.float32)
    logits = [(head @ rms(X[t], nrm)) * hs_ for t in range(T)]
    out["argmax"] = [int(np.argmax(z)) for z in logits]
    out["top5"] = [np.argsort(-z)[:5].tolist() for z in logits]
os.makedirs(a.out, exist_ok=True)
json.dump(out, open(os.path.join(a.out, "stream_ref.json"), "w"))
np.array(res, np.float32).transpose(1, 0, 2).tofile(os.path.join(a.out, "stream_res.f32"))  # [T][L+1][C]
print("wrote", a.out, out.get("argmax"))
