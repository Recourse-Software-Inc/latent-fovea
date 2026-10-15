#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Recourse Software Inc.
"""A random GLM-5.2-shaped model in colibri's container format, at the REAL dimensions (hidden
6144, 64 heads, MLA 2048/512, rope 64, experts 2048 wide, vocab 154,880) but only a few layers and
experts, so web/glm/glm-gpu.js can be checked at full size (row chunking, 154,880-row dispatches,
binding limits) without the real model (429 GB). tools/glm_ref.py then computes its reference.

    python tools/glm_fixture.py --config <GLM-5.2 config.json> --out <dir> [--layers 2 --dense 1 --experts 16]

Quantized as colibri's converter does (c/tools/convert_fp8_to_int4.py): matrices int4 grouped
(group 64, q = clip(rint(w/s), -8, 7), s = max(absmax/7, 1e-8), low nibble = even column, stored
q+8); embed_tokens and lm_head int8 per row (s = absmax/127); norms, router, bias F32.
"""
import argparse, json, os, struct
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--layers", type=int, default=2); ap.add_argument("--dense", type=int, default=1)
ap.add_argument("--experts", type=int, default=16); ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
c = json.load(open(a.config))
c.update(num_hidden_layers=a.layers, first_k_dense_replace=a.dense, n_routed_experts=a.experts,
         mlp_layer_types=["dense"] * a.dense + ["sparse"] * (a.layers - a.dense),
         indexer_types=["full"] * a.layers, num_nextn_predict_layers=0)
c.pop("quantization_config", None)
rng = np.random.default_rng(a.seed)
C, H, QL, KL, NOPE, R, VD = c["hidden_size"], c["num_attention_heads"], c["q_lora_rank"], c["kv_lora_rank"], c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
FD, FE, E, V = c["intermediate_size"], c["moe_intermediate_size"], a.experts, c["vocab_size"]
tensors = []  # (name, dtype, bytes)

def int4(name, O, I, std):
    w = rng.standard_normal((O, I), dtype=np.float32) * std
    ng = -(-I // 64); pad = ng * 64 - I
    g = np.pad(w, ((0, 0), (0, pad))).reshape(O, ng, 64)
    s = np.maximum(np.abs(g).max(-1) / 7, 1e-8).astype(np.float32)
    q = np.clip(np.rint(g / s[..., None]), -8, 7).astype(np.int8).reshape(O, -1)[:, :I] + 8
    if I % 2: q = np.pad(q, ((0, 0), (0, 1)), constant_values=8)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    tensors.append((name, "U8", packed.tobytes())); tensors.append((name + ".qs", "F32", s.tobytes()))

def int8(name, O, I, std):
    w = rng.standard_normal((O, I), dtype=np.float32) * std
    s = np.maximum(np.abs(w).max(1) / 127, 1e-8).astype(np.float32)
    q = np.clip(np.rint(w / s[:, None]), -127, 127).astype(np.int8)
    tensors.append((name, "U8", q.view(np.uint8).tobytes())); tensors.append((name + ".qs", "F32", s.tobytes()))

def f32(name, v): tensors.append((name, "F32", np.asarray(v, np.float32).tobytes()))

std = 0.02
int8("model.embed_tokens.weight", V, C, 1.0)
for l in range(a.layers):
    p = f"model.layers.{l}."; at = p + "self_attn."
    f32(p + "input_layernorm.weight", 1 + 0.1 * rng.standard_normal(C)); f32(p + "post_attention_layernorm.weight", 1 + 0.1 * rng.standard_normal(C))
    int4(at + "q_a_proj.weight", QL, C, std); f32(at + "q_a_layernorm.weight", 1 + 0.1 * rng.standard_normal(QL))
    int4(at + "q_b_proj.weight", H * (NOPE + R), QL, std)
    int4(at + "kv_a_proj_with_mqa.weight", KL + R, C, std); f32(at + "kv_a_layernorm.weight", 1 + 0.1 * rng.standard_normal(KL))
    int4(at + "kv_b_proj.weight", H * (NOPE + VD), KL, std); int4(at + "o_proj.weight", C, H * VD, std)
    m = p + "mlp."
    if l < a.dense:
        int4(m + "gate_proj.weight", FD, C, std); int4(m + "up_proj.weight", FD, C, std); int4(m + "down_proj.weight", C, FD, std)
    else:
        f32(m + "gate.weight", rng.standard_normal((E, C)) * std); f32(m + "gate.e_score_correction_bias", 0.01 * rng.standard_normal(E))
        int4(m + "shared_experts.gate_proj.weight", FE, C, std); int4(m + "shared_experts.up_proj.weight", FE, C, std); int4(m + "shared_experts.down_proj.weight", C, FE, std)
        for e in range(E):
            int4(m + f"experts.{e}.gate_proj.weight", FE, C, std); int4(m + f"experts.{e}.up_proj.weight", FE, C, std); int4(m + f"experts.{e}.down_proj.weight", C, FE, std)
f32("model.norm.weight", 1 + 0.1 * rng.standard_normal(C))
int8("lm_head.weight", V, C, std)

os.makedirs(a.out, exist_ok=True)
hdr, off = {}, 0
for name, dt, b in tensors:
    n = len(b) // (4 if dt == "F32" else 1)
    hdr[name] = {"dtype": dt, "shape": [n], "data_offsets": [off, off + len(b)]}; off += len(b)
h = json.dumps(hdr).encode(); h += b" " * (-len(h) % 8)
with open(os.path.join(a.out, "out-00000.safetensors"), "wb") as f:
    f.write(struct.pack("<Q", len(h))); f.write(h)
    for _, _, b in tensors: f.write(b)
json.dump(c, open(os.path.join(a.out, "config.json"), "w"), indent=1)
json.dump({"prompt_ids": [154822, 154824, 3, 1000, 25000, 77, 9001, 154000],
           "full_ids": [154822, 154824, 3, 1000, 25000, 77, 9001, 154000, 12, 345]}, open(os.path.join(a.out, "prompt.json"), "w"))
print(f"wrote {a.out}: {off / 1e9:.2f} GB, {len(tensors)} tensors")
