#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Recourse Software Inc.
"""Compare what the browser computed (tools/check_hf.mjs) with Hugging Face transformers.

    python3 tools/ref_hf.py out.json        # exit 0 = agrees, 1 = does not

Checks, for the last prompt position: the token ids (our tokenizer vs the model's), the
residual stream's norm after every block, layer 0 head 0's attention, and the next-token
distribution's top 10. Weights are fp16 in the browser, so small differences are expected;
models shipped as fp32 differ a little more than bf16/fp16 ones.
"""
import json, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

got = json.load(open(sys.argv[1]))
repo = got["repo"]
tok = AutoTokenizer.from_pretrained(repo)
want_ids = tok(got["prompt"])["input_ids"]
m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32, attn_implementation="eager")
with torch.no_grad():
    o = m(torch.tensor([got["ids"]]), output_hidden_states=True, output_attentions=True)
hs = o.hidden_states  # the last one has the final norm applied, so compare blocks 0..L-1
L = len(hs) - 1
res = max(abs(got["res"][d] - hs[d][0, -1].norm().item()) / hs[d][0, -1].norm().item() for d in range(L))
att = max(abs(a - b) for a, b in zip(got["att00"], o.attentions[0][0, 0, -1].tolist()))
tp, ti = torch.softmax(o.logits[0, -1], -1).topk(10)
ours = got["top"]
top1 = ours[0][0] == ti[0].item()
overlap = len({i for i, _ in ours} & set(ti.tolist()))
ref = dict(zip(ti.tolist(), tp.tolist()))
pdiff = max(abs(p - ref[i]) for i, p in ours if i in ref)

checks = {
    "token ids match the model's tokenizer": got["ids"] == want_ids,
    f"residual norms within 1e-3 (worst {res:.1e})": res < 1e-3,
    f"attention within 1e-3 (worst {att:.1e})": att < 1e-3,
    "same top-1 next token": top1,
    f"top-10 overlap >= 9 ({overlap})": overlap >= 9,
    f"top-10 probabilities within 5e-3 (worst {pdiff:.1e})": pdiff < 5e-3,
}
print(f"{repo} ({L} layers, {got['ms']:.0f} ms/token in the browser)")
for k, ok in checks.items():
    print(f"  {'PASS' if ok else 'FAIL'}  {k}")
sys.exit(0 if all(checks.values()) else 1)
