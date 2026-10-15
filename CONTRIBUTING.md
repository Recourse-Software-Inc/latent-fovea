# Contributing to Latent Fovea

Latent Fovea is maintained by KnowDrive.ai (Recourse Software Inc.). Issues and pull requests are welcome.

## Running it

```bash
tools/get_data.sh          # prebuilt data sets, or build your own (README, "Build the data yourself")
python3 tools/serve.py     # http://127.0.0.1:8080/
```

The viewer is plain ES modules with no build step and no dependencies. The WebAssembly kernels are committed; rebuild them with `kernel/build.sh` after changing `kernel/*.c` (README, "Rebuild the kernel").

## Checking a change

Each engine has a check against an independent reference. Run the ones your change touches:

- **GPT-2**: `node tools/check_forward.mjs "The capital of France is"`, compared with `python3 tools/ref_gpt2.py`.
- **MDLM**: `node tools/check_mdlm.mjs "The capital of France is"`, compared with `python3 tools/ref_mdlm.py`.
- **UMAP**: `node tools/check_umap.mjs web/data/gpt2`.
- **Hugging Face models**: `tools/check_hf.mjs` (headless Chromium) and `tools/ref_hf.py` (transformers).
- **GLM-5.2**: build a small model with `tools/glm_fixture.py`, write its reference with `tools/glm_ref.py` into `web/data/glm-tiny/`, then open `http://127.0.0.1:8080/glm-test.html`. Every line must report an exact match.

## Pull requests

- Keep the existing style: small modules, no frameworks, comments that explain why.
- New source files start with the licence header:
  ```
  // SPDX-License-Identifier: Apache-2.0
  // Copyright 2026 Recourse Software Inc.
  ```
- Don't commit model weights, data sets or tokens. `web/data/` is ignored.
- By submitting a contribution you agree that it is licensed under the Apache License 2.0 (section 5 of the licence).
