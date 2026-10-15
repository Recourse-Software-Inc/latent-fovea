#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Recourse Software Inc.
"""Reference for Latent Fovea's GLM-5.2 engine (web/glm/): reads a colibri container (the int4
format JustVugg/colibri streams from disk), runs it two ways, and checks they agree.

  1. transformers' own GlmMoeDsa model, loaded with the container's DEQUANTIZED weights: the
     ground truth for this exact container (int4 changes the answers, so the fp32 oracle can't be).
  2. A plain NumPy decode loop, one token at a time, written the way the WebGPU kernels compute:
     absorbed MLA attention over a 576-float latent cache (kv_lora + rope), sigmoid router with the
     correction bias, top-k, normalised and scaled, SwiGLU experts plus the shared expert.

It writes ref.json (prompt, teacher-forced argmax, greedy continuation, the experts chosen per layer
per position) and ref_hidden.f32 (the residual after every layer, per position) for the browser test.

    python3 tools/glm_ref.py --model <container dir> --out <dir> [--prompt-from <ref json>]

Container format (colibri c/tools/convert_fp8_to_int4.py): safetensors shards, no index. A
quantized weight is `name` (U8, flat) + `name.qs` (F32): int4 grouped (group 64, symmetric,
low nibble = even column, stored q+8, w = (q-8)*s[o*ng + i/64]) or int8 per row (w = q*s[o]);
the format follows from the byte counts. Norms, router and its bias are plain F32.
"""
import argparse, glob, json, math, os, struct
import numpy as np


def read_container(d):
    """name -> (path, dtype, shape, offset, nbytes) over every *.safetensors in d."""
    idx = {}
    for p in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        for k, v in h.items():
            if k == "__metadata__": continue
            a, b = v["data_offsets"]
            idx[k] = (p, v["dtype"], v["shape"], 8 + n + a, b - a)
    return idx


def raw(idx, name):
    p, dt, shape, off, nb = idx[name]
    with open(p, "rb") as f:
        f.seek(off); b = f.read(nb)
    return np.frombuffer(b, {"U8": np.uint8, "F32": np.float32, "I8": np.int8}[dt]).reshape(shape)


def weight(idx, name, O, I):
    """Dequantized [O, I] float32 for a container tensor (any of the formats above)."""
    if idx[name][1] == "F32" and name + ".qs" not in idx:
        return raw(idx, name).reshape(O, I).astype(np.float32)
    q, s = raw(idx, name), raw(idx, name + ".qs").astype(np.float32)
    if q.size == O * I and s.size == O:  # int8, one scale per row
        return q.view(np.int8).reshape(O, I).astype(np.float32) * s[:, None]
    ng = -(-I // 64)
    if q.size == O * (-(-I // 2)) and s.size == O * ng:  # int4, group 64
        q = q.reshape(O, -1)
        w = np.empty((O, q.shape[1] * 2), np.float32)
        w[:, 0::2] = (q & 15).astype(np.float32) - 8
        w[:, 1::2] = (q >> 4).astype(np.float32) - 8
        w = w[:, :I]
        return w * np.repeat(s.reshape(O, ng), 64, axis=1)[:, :I]
    raise ValueError(f"{name}: {q.size} bytes, {s.size} scales do not match [{O},{I}]")


class Glm:
    def __init__(self, d):
        self.cfg = c = json.load(open(os.path.join(d, "config.json")))
        self.idx = idx = read_container(d)
        self.C, self.H = c["hidden_size"], c["num_attention_heads"]
        self.QL, self.KL = c["q_lora_rank"], c["kv_lora_rank"]
        self.NOPE, self.ROPE, self.VD = c["qk_nope_head_dim"], c["qk_rope_head_dim"], c["v_head_dim"]
        self.L, self.V = c["num_hidden_layers"], c["vocab_size"]
        self.FD, self.FE = c["intermediate_size"], c["moe_intermediate_size"]
        self.E, self.K = c["n_routed_experts"], c["num_experts_per_tok"]
        self.dense = c.get("first_k_dense_replace", 0)
        self.eps = c["rms_norm_eps"]
        rp = c.get("rope_parameters") or {}
        self.theta = rp.get("rope_theta", c.get("rope_theta", 10000.0))
        C, H = self.C, self.H
        W = lambda n, O, I: weight(idx, n, O, I)
        F = lambda n: raw(idx, n).astype(np.float32).ravel()
        self.emb = W("model.embed_tokens.weight", self.V, C)
        self.head = W("lm_head.weight", self.V, C)
        self.norm = F("model.norm.weight")
        self.layers = []
        for l in range(self.L):
            p = f"model.layers.{l}."
            a = p + "self_attn."
            ly = dict(
                ln1=F(p + "input_layernorm.weight"), ln2=F(p + "post_attention_layernorm.weight"),
                qa=W(a + "q_a_proj.weight", self.QL, C), qaln=F(a + "q_a_layernorm.weight"),
                qb=W(a + "q_b_proj.weight", H * (self.NOPE + self.ROPE), self.QL),
                kva=W(a + "kv_a_proj_with_mqa.weight", self.KL + self.ROPE, C), kvaln=F(a + "kv_a_layernorm.weight"),
                kvb=W(a + "kv_b_proj.weight", H * (self.NOPE + self.VD), self.KL),
                o=W(a + "o_proj.weight", C, H * self.VD))
            m = p + "mlp."
            if l < self.dense:
                ly.update(g=W(m + "gate_proj.weight", self.FD, C), u=W(m + "up_proj.weight", self.FD, C), d=W(m + "down_proj.weight", C, self.FD))
            else:
                ly.update(router=W(m + "gate.weight", self.E, C), bias=F(m + "gate.e_score_correction_bias"),
                          sg=W(m + "shared_experts.gate_proj.weight", self.FE * c.get("n_shared_experts", 1), C),
                          su=W(m + "shared_experts.up_proj.weight", self.FE * c.get("n_shared_experts", 1), C),
                          sd=W(m + "shared_experts.down_proj.weight", C, self.FE * c.get("n_shared_experts", 1)),
                          ex=[(W(m + f"experts.{e}.gate_proj.weight", self.FE, C), W(m + f"experts.{e}.up_proj.weight", self.FE, C),
                               W(m + f"experts.{e}.down_proj.weight", C, self.FE)) for e in range(self.E)])
            self.layers.append(ly)

    def rms(self, x, w): return x / np.sqrt(np.mean(x * x) + self.eps) * w

    def rope(self, v, pos):
        """Interleaved-pair input (2j, 2j+1) -> split-half output [j | j + R/2] (colibri rope_interleave)."""
        R = v.shape[-1]; j = np.arange(R // 2)
        ang = pos * self.theta ** (-2.0 * j / R)
        a, b = v[..., 0::2], v[..., 1::2]
        return np.concatenate([a * np.cos(ang) - b * np.sin(ang), a * np.sin(ang) + b * np.cos(ang)], axis=-1)

    def forward_token(self, tok, pos, cache):
        """One decode step; cache[l] = (latent[t][KL], rope[t][ROPE]). Returns logits, residuals, chosen experts."""
        x = self.emb[tok].astype(np.float32)  # float32 throughout: BLAS accumulates well enough, and f64 would copy every matrix
        res, chosen = [x.copy()], []
        H, NOPE, ROPE, VD, KL = self.H, self.NOPE, self.ROPE, self.VD, self.KL
        for l, ly in enumerate(self.layers):
            h = self.rms(x, ly["ln1"])
            qr = self.rms(ly["qa"] @ h, ly["qaln"])
            q = (ly["qb"] @ qr).reshape(H, NOPE + ROPE)
            qn, qp = q[:, :NOPE], self.rope(q[:, NOPE:], pos)
            ckv = ly["kva"] @ h
            lat, kr = self.rms(ckv[:KL], ly["kvaln"]), self.rope(ckv[KL:], pos)
            cache[l][0].append(lat); cache[l][1].append(kr)
            Lc, Rc = np.array(cache[l][0]), np.array(cache[l][1])  # [T, KL], [T, ROPE]
            kvb = ly["kvb"].reshape(H, NOPE + VD, KL)
            ctx = np.empty((H, VD), np.float32)
            for hh in range(H):
                qabs = qn[hh] @ kvb[hh, :NOPE]                 # absorb W_uk: [KL]
                s = (Lc @ qabs + Rc @ qp[hh]) / math.sqrt(NOPE + ROPE)
                p = np.exp(s - s.max()); p /= p.sum()
                ctx[hh] = kvb[hh, NOPE:] @ (p @ Lc)            # absorb W_uv after the weighted latent
            x = x + ly["o"] @ ctx.ravel()
            h = self.rms(x, ly["ln2"])
            silu = lambda z: z / (1 + np.exp(-z))
            if "g" in ly:
                y = ly["d"] @ (silu(ly["g"] @ h) * (ly["u"] @ h)); chosen.append([])
            else:
                sig = 1 / (1 + np.exp(-(ly["router"] @ h)))
                top = np.argsort(-(sig + ly["bias"]), kind="stable")[: self.K]
                w = sig[top]; w = w / w.sum() * self.cfg["routed_scaling_factor"]
                y = ly["sd"] @ (silu(ly["sg"] @ h) * (ly["su"] @ h))
                for e, we in zip(top, w):
                    g, u, d = ly["ex"][e]
                    y = y + we * (d @ (silu(g @ h) * (u @ h)))
                chosen.append([int(e) for e in top])
            x = x + y
            res.append(x.copy())
        logits = self.head @ self.rms(x, self.norm)
        return logits, res, chosen


def hf_reference(g, d, full_ids):
    """transformers on the container's dequantized weights: teacher-forced argmax over full_ids."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    m = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(d)).float().eval()
    sd = m.state_dict()
    for k in list(sd):
        if ".indexer." in k: continue  # absent from the container; unused while context <= index_topk
        l = int(k.split(".")[2]) if k.startswith("model.layers.") else None
        if k.endswith("experts.gate_up_proj"):
            sd[k] = torch.tensor(np.stack([np.concatenate([e[0], e[1]]) for e in g.layers[l]["ex"]]))
        elif k.endswith("experts.down_proj"):
            sd[k] = torch.tensor(np.stack([e[2] for e in g.layers[l]["ex"]]))
        else:
            O, I = (sd[k].shape[0], sd[k].shape[1]) if sd[k].dim() == 2 else (sd[k].shape[0], 1)
            sd[k] = torch.tensor(weight(g.idx, k, O, I).reshape(sd[k].shape))
    m.load_state_dict(sd)
    with torch.no_grad():
        out = m(torch.tensor([full_ids]), output_hidden_states=True)
    return out.logits[0].argmax(-1).tolist(), [h[0].numpy() for h in out.hidden_states]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--prompt-from", help="JSON with prompt_ids and full_ids (e.g. colibrì's oracle); default: the model folder's prompt.json, else 32 fixed random tokens")
    ap.add_argument("--out", required=True); ap.add_argument("--gen", type=int, default=16)
    a = ap.parse_args()
    g = Glm(a.model)
    pf = a.prompt_from or (os.path.join(a.model, "prompt.json") if os.path.isfile(os.path.join(a.model, "prompt.json")) else None)
    if pf:  # given, or the one tools/glm_fixture.py writes next to its model
        ref = json.load(open(pf))
    else:  # a fixture's weights are random, so any tokens test it equally well
        V = json.load(open(os.path.join(a.model, "config.json")))["vocab_size"]
        full = np.random.default_rng(7).integers(0, V, 32).tolist()
        ref = {"prompt_ids": full[:8], "full_ids": full}
    prompt, full = ref["prompt_ids"], ref["full_ids"]
    tf_hf, hid_hf = hf_reference(g, a.model, full)
    # NumPy decode over the same tokens (teacher forced), then greedy from the prompt.
    cache = [([], []) for _ in range(g.L)]
    tf_np, hidden, experts, worst = [], [], [], 0.0
    for pos, tok in enumerate(full):
        logits, res, ch = g.forward_token(tok, pos, cache)
        tf_np.append(int(np.argmax(logits))); experts.append(ch)
        hidden.append(np.array(res, np.float32))
        # hidden_states[i] from transformers is the residual after layer i-1 (the last one normed)
        for i in range(g.L):
            worst = max(worst, float(np.abs(hid_hf[i][pos] - res[i]).max()))
    cache = [([], []) for _ in range(g.L)]
    for pos, tok in enumerate(prompt[:-1]): g.forward_token(tok, pos, cache)
    greedy, tok = [], prompt[-1]
    for pos in range(len(prompt) - 1, len(prompt) - 1 + a.gen):
        logits, _, _ = g.forward_token(tok, pos, cache)
        tok = int(np.argmax(logits)); greedy.append(tok)
    agree = sum(x == y for x, y in zip(tf_np, tf_hf))
    print(f"teacher-forced argmax: numpy vs transformers {agree}/{len(full)}; residuals max |diff| {worst:.2e}")
    os.makedirs(a.out, exist_ok=True)
    json.dump({"model": os.path.abspath(a.model), "prompt_ids": prompt, "full_ids": full, "tf_pred": tf_np,
               "tf_pred_transformers": tf_hf, "greedy": greedy, "experts": experts,
               "hidden_shape": [len(full), g.L + 1, g.C]}, open(os.path.join(a.out, "ref.json"), "w"))
    np.array(hidden, np.float32).tofile(os.path.join(a.out, "ref_hidden.f32"))
    print("wrote", a.out)
