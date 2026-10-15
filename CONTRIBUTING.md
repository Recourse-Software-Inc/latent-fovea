# Contributing to Latent Fovea

Latent Fovea is maintained by KnowDrive.ai (Recourse Software Inc.). Issues and pull requests are welcome. For security problems, follow [SECURITY.md](SECURITY.md) instead of opening an issue.

## Running it

```bash
tools/get_data.sh          # prebuilt data sets (~970 MB), or build your own (README, "Build the data yourself")
python3 tools/serve.py     # http://127.0.0.1:8080/
```

The viewer is plain ES modules with no build step and no runtime dependencies. The WebAssembly kernels are committed; rebuild them with `kernel/build.sh` after changing `kernel/*.c` (README, "Rebuild the kernel"). Python 3.11+ (`pip install -r requirements.txt`, plus `requirements-dev.txt` for the reference checks) and Node 22+ cover every tool.

`tools/serve.py` sends the same Content-Security-Policy as the deployed site. If a change needs a new external host or an inline script, it will fail there first: update `SECURITY_HEADERS` in `tools/serve.py` deliberately, or avoid the need.

## Checking a change

CI runs the parse checks, the licence-header check and a dev-server smoke test on every push and pull request. Each engine also has a check against an independent reference; run the ones your change touches. They need the prebuilt data, or data built with `--weights`.

- **GPT-2**: the WebAssembly kernel against NumPy on the same prompt:
  ```bash
  node tools/check_forward.mjs "The capital of France is" > /tmp/gpt2.json
  python3 tools/ref_gpt2.py "$(python3 -c 'import json; print(json.dumps(json.load(open("/tmp/gpt2.json"))["rows"]))')"
  ```
- **MDLM**: the same with `tools/check_mdlm.mjs` and `tools/ref_mdlm.py` (data in `web/data/mdlm`).
- **UMAP**: `node tools/check_umap.mjs web/data/gpt2` scores the in-browser UMAP against the prebuilt layout.
- **Hugging Face models**: `tools/check_hf.mjs` runs a model in headless Chromium (`npm i -D playwright && npx playwright install chromium`) against a staged site (`python3 tools/stage_static.py site && python3 tools/serve.py --root site`); `tools/ref_hf.py` compares with transformers.
- **GLM-5.2**: a random model at the real dimensions, its reference, then the browser check:
  ```bash
  hf download zai-org/GLM-5.2 config.json --local-dir /tmp/glm52
  python3 tools/glm_fixture.py --config /tmp/glm52/config.json --out web/data/glm-tiny
  python3 tools/glm_ref.py --model web/data/glm-tiny --out web/data/glm-tiny
  ```
  Then open `http://127.0.0.1:8080/glm-test.html`. Every line must report an exact match.

## Pull requests

- Keep the existing style: small modules, no frameworks, comments that explain why.
- New source files start with the licence header:
  ```
  // SPDX-License-Identifier: Apache-2.0
  // Copyright 2026 Recourse Software Inc.
  ```
- Don't commit model weights, data sets or tokens. `web/data/` is ignored.
- By submitting a contribution you agree that it is licensed under the Apache License 2.0 (section 5 of the licence). Sign off your commits (`git commit -s`) to certify the [Developer Certificate of Origin](https://developercertificate.org).
