#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Recourse Software Inc.
"""Every MLP neuron's strongest examples, both poles, and the shape of its activations, for the
neuron inspector. Read a neuron the way its readers do: what drives it hardest, what pushes it
hardest the other way, and whether its meaning holds far down its range or falls off a cliff.

    python3 tools/build_neuron_examples.py --model gpt2 --out web/data/gpt2/neurons [--tokens 3000000]

Corpus: FineWeb-Edu (HuggingFaceFW/fineweb-edu, sample/10BT, ODC-By 1.0), read row group by row
group over HTTP, so only what is used is downloaded. Each document becomes one sequence that starts
at the document's own beginning, after <|endoftext|> (as GPT-2 saw documents in training), and
runs up to --ctx tokens: long-context neurons see their whole context, and no example's context
crosses into another document. Position 0, the <|endoftext|> the model parks attention on, and
the padding after short documents are left out of every ranking and count.

Two passes over the same tokens:
  1. per neuron, the --pos strongest activations (after GELU) and the --neg most negative inputs
     to GELU (its negative pole: GELU itself bottoms out at -0.17, but the input says how hard the
     neuron is pushed away), no two within --window tokens of each other;
  2. per neuron, a histogram of its activation as a fraction of its own maximum, in halving bins
     (1, 1/2, 1/4, ... 1/4096 of the maximum), the tokens at half its maximum or more, and the
     tokens where it is at or below zero.

Writes into --out:
  meta.json             layers, neurons per layer, sizes, corpus, tokens read, bins
  corpus.u16            every sequence, --ctx tokens each (PAD = 65535 after a document ends), so
                        the viewer can show an example's whole passage
  L<l>.ids.u16          per neuron: (pos + neg + negout) windows of (before + 1 + after) token ids
                        (65535 where the window runs past the document's start or end)
  L<l>.acts.f16         the same shape: the neuron's activation at every token of each window
                        (after GELU for the positive and negative-output examples, before it for the negative inputs)
  L<l>.at.u32           per neuron: each example's index into corpus.u16 (0xFFFFFFFF: none)
  L<l>.stats.f32        per neuron: max, min of the GELU input, count >= max/2, count <= 0, bins
  L<l>.bands.u32        per neuron: --band-samples random positions in each of bands 0..--bands (0: above
                        half its max, 1: 1/4 to 1/2, ...; band_first in meta.json), no two within --window
                        tokens in one document, into corpus.u16; L<l>.bandv.f16 their values
A neuron's records are contiguous, so the viewer reads one neuron with one Range request per file.
"""
import argparse, json, math, os, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gpt2", choices=["gpt2", "distilgpt2"])
ap.add_argument("--out", required=True)
ap.add_argument("--tokens", type=int, default=3_000_000, help="document tokens to read")
ap.add_argument("--ctx", type=int, default=1024, help="longest sequence: a document's first ctx - 1 tokens")
ap.add_argument("--pos", type=int, default=12, help="strongest positive examples per neuron")
ap.add_argument("--neg", type=int, default=6, help="most negative inputs per neuron (firmly off)")
ap.add_argument("--negout", type=int, default=0, help="most negative outputs per neuron (for GELU these are ties at -0.17: off by default)")
ap.add_argument("--band-samples", type=int, default=8, help="random examples kept from each of bands 1..--bands (1/4 to 1/2 of max, ...): the middle of its range")
ap.add_argument("--band-pool", type=int, default=4, help="draw this many times --band-samples at random per band, then keep the first that are --window apart")
ap.add_argument("--bands", type=int, default=5, help="how many bands below the top to sample")
ap.add_argument("--no-top-band", action="store_true", help="don't sample the top band (above half its max) too: by default it is, since for a neuron that fires hard often the strongest few are a sliver of it")
ap.add_argument("--before", type=int, default=24, help="tokens of context kept before the example")
ap.add_argument("--after", type=int, default=4, help="tokens after it")
ap.add_argument("--window", type=int, default=0, help="minimum distance between two examples of one neuron (default: the window's length, so no two share context)")
ap.add_argument("--batch", type=int, default=4)
a = ap.parse_args()
a.window = a.window or a.before + 1 + a.after

REPO = {"gpt2": "openai-community/gpt2", "distilgpt2": "distilbert/distilgpt2"}[a.model]
CORPUS = "HuggingFaceFW/fineweb-edu"
SHARD = "sample/10BT/000_00000.parquet"
PAD = 65535
dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
t0 = time.time(); log = lambda *x: print(f"[{time.time() - t0:6.1f}s]", *x, flush=True)

from transformers import AutoTokenizer, GPT2LMHeadModel
tok = AutoTokenizer.from_pretrained(REPO)
model = GPT2LMHeadModel.from_pretrained(REPO, torch_dtype=torch.float32).to(dev).eval()
L, F = model.config.n_layer, model.config.n_inner or 4 * model.config.n_embd
EOT = tok.eos_token_id

# ---- corpus: one sequence per document, from its start
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
pf = pq.ParquetFile(HfFileSystem().open(f"datasets/{CORPUS}/{SHARD}", "rb", block_size=8 << 20))
seqs, lens, real, used = [], [], 0, []
NRG = pf.metadata.num_row_groups
for rg in [(k * 97) % NRG for k in range(NRG)]:  # row groups spread across the shard, not just its start
    if real >= a.tokens: break
    used.append(rg)
    for text in pf.read_row_group(rg, columns=["text"]).column(0).to_pylist()[::4]:  # every 4th document: more sources per token
        ids = [EOT] + tok(text)["input_ids"][: a.ctx - 1]
        seqs.append(ids + [PAD] * (a.ctx - len(ids))); lens.append(len(ids)); real += len(ids) - 1
        if real >= a.tokens: break
corpus = np.array(seqs, np.int64)            # [n_seq, ctx], PAD after each document
n_seq = len(seqs)
lens = np.array(lens)
valid = (np.arange(a.ctx)[None, :] >= 1) & (np.arange(a.ctx)[None, :] < lens[:, None])  # no sink, no padding
log(f"{real:,} document tokens from {n_seq:,} documents (row groups {used}), up to {a.ctx} tokens each")

# ---- capture each MLP's GELU input (c_fc output)
pre = {}
def hook(l):
    def f(_m, _i, out): pre[l] = out
    return f
for l, block in enumerate(model.transformer.h):
    block.mlp.c_fc.register_forward_hook(hook(l))
act = lambda x: torch.nn.functional.gelu(x, approximate="tanh")  # GPT-2's gelu_new

def batches():
    for s in range(0, n_seq, a.batch):
        x = corpus[s : s + a.batch]
        T = int(lens[s : s + a.batch].max())  # padding after a document never reaches its tokens (causal)
        xin = torch.from_numpy(np.where(x[:, :T] == PAD, EOT, x[:, :T])).to(dev)
        yield s, xin, torch.from_numpy(valid[s : s + a.batch, :T]).to(dev), T

# ---- pass 1: candidates for each pole. Keep a generous pool per neuron (CAND) and space them
# out at the end, so repeated tokens of one passage don't take every slot.
CAND = 4 * (a.pos + a.neg + a.negout)
NEG_INF = -float("inf")
top_v = torch.full((L, CAND, F), NEG_INF, device=dev); top_i = torch.zeros((L, CAND, F), dtype=torch.int64, device=dev)
bot_v = torch.full((L, CAND, F), NEG_INF, device=dev); bot_i = torch.zeros((L, CAND, F), dtype=torch.int64, device=dev)
out_v = torch.full((L, CAND, F), NEG_INF, device=dev); out_i = torch.zeros((L, CAND, F), dtype=torch.int64, device=dev)
with torch.no_grad():
    for s, x, ok, T in batches():
        model(x)
        b = x.shape[0]
        flat = (torch.arange(s, s + b, device=dev)[:, None] * a.ctx + torch.arange(T, device=dev)[None, :]).reshape(-1)
        okf = ok.reshape(-1, 1)
        for l in range(L):
            h = pre[l].reshape(-1, F)
            k = min(CAND, h.shape[0])
            g = act(h)
            for v_all, keep_v, keep_i in ((g, top_v, top_i), (-h, bot_v, bot_i), (-g, out_v, out_i)):
                v, i = torch.topk(torch.where(okf, v_all, torch.full_like(v_all, NEG_INF)), k, dim=0)
                mv = torch.cat([keep_v[l], v]); mi = torch.cat([keep_i[l], flat[i]])
                v2, j = torch.topk(mv, CAND, dim=0)
                keep_v[l] = v2; keep_i[l] = torch.gather(mi, 0, j)
        if (s // a.batch) % 100 == 0: log(f"pass 1: {s + b}/{n_seq} documents")
top_v, top_i, bot_v, bot_i, out_v, out_i = (t.cpu().numpy() for t in (top_v, top_i, bot_v, bot_i, out_v, out_i))
neuron_max = np.maximum(top_v[:, 0, :], 0)   # strongest GELU output per neuron
neuron_min = -bot_v[:, 0, :]                  # most negative GELU input per neuron

def spaced(vals, idx, want):
    """Best `want` positions, no two in one document within --window tokens."""
    chosen = []
    if want <= 0: return chosen
    for v, p in zip(vals, idx):
        if not math.isfinite(v): break
        if all(p // a.ctx != q // a.ctx or abs(int(p) - int(q)) >= a.window for q in chosen):
            chosen.append(int(p))
            if len(chosen) == want: break
    return chosen

# ---- pass 2: histograms against each neuron's own maximum
NB = 13  # bins: (1/2,1], (1/4,1/2], ... the last catches everything above 0 and at most 1/4096
mx = torch.from_numpy(np.maximum(neuron_max, 1e-6)).to(dev)
hist = torch.zeros((L, NB, F), dtype=torch.int64, device=dev)
# a few random tokens from each band below the top: a random key per (token, neuron) in the band,
# keeping the largest keys, is a uniform sample without replacement; a pool of several times as
# many is drawn, so that the ones kept can be spaced apart like the examples
RS = a.band_samples
B0 = 1 if a.no_top_band else 0   # the first band sampled: 0 is the top, above half its max
NBS = a.bands + 1 - B0           # bands B0 .. a.bands
RP = RS * max(1, a.band_pool) if RS else 0
bk_v = torch.full((L, NBS, RP, F), NEG_INF, device=dev); bk_i = torch.zeros((L, NBS, RP, F), dtype=torch.int64, device=dev)
bk_a = torch.zeros((L, NBS, RP, F), device=dev)
gen = torch.Generator(device=dev); gen.manual_seed(7)
half = torch.zeros((L, F), dtype=torch.int64, device=dev)
nonpos = torch.zeros((L, F), dtype=torch.int64, device=dev)
with torch.no_grad():
    for s, x, ok, T in batches():
        model(x)
        okf = ok.reshape(-1, 1)
        b_ = x.shape[0]
        flat = (torch.arange(s, s + b_, device=dev)[:, None] * a.ctx + torch.arange(T, device=dev)[None, :]).reshape(-1)
        for l in range(L):
            g_ = act(pre[l].reshape(-1, F))
            frac = g_ / mx[l]
            pos = (frac > 0) & okf
            b = torch.clamp(torch.floor(-torch.log2(torch.clamp(frac, min=1e-30))), 0, NB - 1).to(torch.int64)
            hist[l].scatter_add_(0, torch.where(pos, b, torch.zeros_like(b)), pos.to(torch.int64))
            half[l] += ((frac >= 0.5) & okf).sum(0)
            nonpos[l] += ((frac <= 0) & okf).sum(0)
            if RS:
                band = torch.floor(-torch.log2(torch.clamp(frac, min=1e-30)))  # 0: (1/2, 1], 1: (1/4, 1/2], ...
                for k in range(NBS):
                    key = torch.where((band == k + B0) & pos, torch.rand(frac.shape, device=dev, generator=gen), torch.full_like(frac, NEG_INF))
                    kk = min(RP, key.shape[0])
                    v, i = torch.topk(key, kk, dim=0)
                    mv = torch.cat([bk_v[l, k], v]); mi = torch.cat([bk_i[l, k], flat[i]]); ma = torch.cat([bk_a[l, k], torch.gather(g_, 0, i)])
                    v2, j = torch.topk(mv, RP, dim=0)
                    bk_v[l, k] = v2; bk_i[l, k] = torch.gather(mi, 0, j); bk_a[l, k] = torch.gather(ma, 0, j)
hist, half, nonpos = hist.cpu().numpy(), half.cpu().numpy(), nonpos.cpu().numpy()
bk_v, bk_i, bk_a = bk_v.cpu().numpy(), bk_i.cpu().numpy(), bk_a.cpu().numpy()
log("pass 2 done")

# ---- windows: token ids and the neuron's activation over each. Every wanted window, for every
# layer, grouped by document, so each document runs once.
W = a.before + 1 + a.after
K = a.pos + a.neg + a.negout
os.makedirs(a.out, exist_ok=True)
ids_out = np.full((L, F, K, W), PAD, np.uint16)
acts_out = np.zeros((L, F, K, W), np.float16)
at_out = np.full((L, F, K), 0xFFFFFFFF, np.uint32)
by_seq = {}
for l in range(L):
    for n in range(F):
        P = spaced(top_v[l, :, n], top_i[l, :, n], a.pos)
        N = spaced(bot_v[l, :, n], bot_i[l, :, n], a.neg)
        for slot, p in enumerate(P): by_seq.setdefault(p // a.ctx, []).append((l, n, slot, p, True))
        for slot, p in enumerate(N): by_seq.setdefault(p // a.ctx, []).append((l, n, a.pos + slot, p, False))
        O = spaced(out_v[l, :, n], out_i[l, :, n], a.negout) if out_v[l, 0, n] > 1e-4 else []  # only if it ever goes negative
        for slot, p in enumerate(O): by_seq.setdefault(p // a.ctx, []).append((l, n, a.pos + a.neg + slot, p, True))
seqs_w = sorted(by_seq)
log(f"{sum(len(v) for v in by_seq.values()):,} windows from {len(seqs_w):,} documents")
with torch.no_grad():
    for s0 in range(0, len(seqs_w), a.batch):
        qs = seqs_w[s0 : s0 + a.batch]
        T = int(lens[qs].max())
        x = corpus[qs][:, :T]
        model(torch.from_numpy(np.where(x == PAD, EOT, x)).to(dev))
        hs = [pre[l].cpu().numpy() for l in range(L)]  # [b, T, F] per layer
        for b, q in enumerate(qs):
            n_q = lens[q]
            for l, n, slot, p, positive in by_seq[q]:
                t = p - q * a.ctx
                lo, hi = t - a.before, t + a.after + 1
                src_lo, src_hi = max(lo, 0), min(hi, n_q)  # inside the document only
                col = hs[l][b, src_lo:src_hi, n]
                v = act(torch.from_numpy(col)).numpy() if positive else col
                ids_out[l, n, slot, src_lo - lo : src_hi - lo] = corpus[q, src_lo:src_hi]
                acts_out[l, n, slot, src_lo - lo : src_hi - lo] = v
                at_out[l, n, slot] = p
        if (s0 // a.batch) % 200 == 0: log(f"windows: {s0 + len(qs)}/{len(seqs_w)} documents")
for l in range(L):
    stats = np.zeros((F, 4 + NB), np.float32)
    stats[:, 0] = neuron_max[l]; stats[:, 1] = neuron_min[l]; stats[:, 2] = half[l]; stats[:, 3] = nonpos[l]
    stats[:, 4:] = hist[l].T
    ids_out[l].tofile(os.path.join(a.out, f"L{l}.ids.u16")); acts_out[l].tofile(os.path.join(a.out, f"L{l}.acts.f16"))
    at_out[l].tofile(os.path.join(a.out, f"L{l}.at.u32")); stats.tofile(os.path.join(a.out, f"L{l}.stats.f32"))
    if RS:  # per neuron: bands x samples positions into corpus.u16 (0xFFFFFFFF: none) and their activations
        at_b = np.full((F, NBS, RS), 0xFFFFFFFF, np.uint32); v_b = np.zeros((F, NBS, RS), np.float16)
        for n in range(F):
            for k in range(NBS):
                pool = {int(p): v for p, v in zip(bk_i[l, k, :, n], bk_a[l, k, :, n])}
                keep = spaced(bk_v[l, k, :, n], bk_i[l, k, :, n], RS)  # pool in random order: a spaced random sample
                for j, p in enumerate(keep): at_b[n, k, j] = p; v_b[n, k, j] = pool[p]
        at_b.tofile(os.path.join(a.out, f"L{l}.bands.u32")); v_b.tofile(os.path.join(a.out, f"L{l}.bandv.f16"))
corpus.astype(np.uint16).tofile(os.path.join(a.out, "corpus.u16"))
log("layers written")

json.dump({
    "schema": "latent-fovea-neurons/2", "model": REPO, "layers": L, "neurons": F,
    "pos": a.pos, "neg": a.neg, "negout": a.negout, "band_samples": a.band_samples, "bands": NBS, "band_first": B0, "before": a.before, "after": a.after, "window": W, "spacing": a.window,
    "tokens": int(real), "documents": n_seq, "ctx": a.ctx, "eot": EOT, "pad": PAD,
    "corpus": {"name": "FineWeb-Edu", "repo": CORPUS, "file": SHARD, "row_groups": used, "license": "ODC-By-1.0",
               "url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
               "sequences": "each document from its start, after <|endoftext|>, up to ctx tokens; position 0 and padding excluded"},
    "positive": "GELU output", "negative": "GELU input: most negative (its other side; readers get between -0.17 and 0 from it)",
    "band_spacing": "band samples are drawn at random, then kept only if no two in one document are within `spacing` tokens",
    "bins": "count of tokens with activation in (max/2^(k+1), max/2^k], k = 0..11; the last bin is everything above 0 and at most max/4096",
    "stats": ["max", "min_input", "count_at_least_half_max", "count_at_or_below_zero"] + [f"bin{k}" for k in range(NB)],
    "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
# point the model's data set at its book, so the viewer only asks for it when there is one
mp = os.path.join(os.path.dirname(os.path.normpath(a.out)), "meta.json")
if os.path.isfile(mp):
    mm = json.load(open(mp)); mm["neuron_book"] = os.path.basename(os.path.normpath(a.out)) + "/meta.json"
    json.dump(mm, open(mp, "w"), indent=2)
log(f"wrote {a.out}")
