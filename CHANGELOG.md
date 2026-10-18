# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Reading a neuron the way the network reads it (handbook Chapter 11, after Nick Cammarata's *Neural Networks Are Already Clean*): for every GPT-2 and DistilGPT2 neuron, its strongest examples, where its input is most negative (its other side), and its activation shape with random samples from its top band and the middle of its range, over three million FineWeb-Edu tokens read document by document, with each example's whole passage on demand; the input before the activation function, and SwiGLU's gate, value and sign regimes; what feeds a neuron at a token (an exact split of its input) and what reads it; steering kept inside the network's own range; a blind test box; names and notes per neuron, a back/forward trail, export, and a shareable address.
- Neuron graphs: the model's own neurons behind a word, traced back layer by layer, with group knockouts and two controls (random and most active neurons of the same layers).
- Kernels record each neuron's input (and SwiGLU's value), take an additive steer (`iv_neuron2`), and split a neuron's input (`neuron_links`, `neuron_readers`).

### Changed
- GPT-2 and DistilGPT2 runs start after `<|endoftext|>` (a ⟨doc⟩ chip), as the model saw documents in training, so values match the neuron book and Test.
- Handbook §11.2 treats SAE features as a second step after the neurons, not a cleaner replacement for them.

## [0.1.1] - 2026-10-16

### Added
- CI (parse checks, licence headers, dev-server smoke test), Dependabot, issue and pull-request templates.
- `requirements.txt`, `requirements-dev.txt` and `package.json`.
- Prebuilt data on the `data` branch, each model folder with its licence and a statement of changes.
- `CITATION.cff`, for citing Latent Fovea.

### Changed
- The site sends a Content-Security-Policy and other security headers, and `tools/serve.py` sends the same ones. The page scripts of `glm.html` and `real.html` moved into files.
- `tools/build_data.py` updates `models.json` after every build. `tools/glm_ref.py` no longer needs a prompt file.
- Workflows pin their actions to commits, pass inputs through the environment, and work without a `data` branch.
- Fonts are served from the site (`web/fonts/`, OFL-1.1) instead of Google Fonts, so pages make no requests to Google; the Content-Security-Policy no longer allows Google's font hosts.
- Code of conduct reports go to conduct@recoursellm.com.

### Removed
- The Canada 2025/26 Public Data store is no longer listed or published. The tools that build a store data set remain.

### Fixed
- `?data=` accepts only a relative path on the same site, and data-set category names are escaped. A crafted link could otherwise run script in the page.
- With no data, the viewer explains how to get some, and a new model in the URL hash reloads the page.

## [0.1.0] - 2026-10-15

First public release.

### Added
- Foveated viewer for the token embeddings of GPT-2, DistilGPT2 and MDLM: tiles streamed by HTTP Range, exact kNN, UMAP and PCA layouts, local re-projection in WebAssembly.
- GPT-2's forward pass, slowed down: the attention field, the logit lens, the lens river, attention flow and rollout, hidden-state space and SAE features. Direct logit attribution, interventions (knock out heads, MLPs and neurons) and run comparison.
- MDLM masked diffusion, shown filling in its answer.
- Any compatible model from Hugging Face, checked, downloaded and built in the browser. A general decoder on the CPU and on WebGPU, with 16-bit and 8-bit weights, and model export and import between browsers.
- A KnowDrive store as a data set: retrieval shown as attention.
- GLM-5.2 (744B MoE) in the browser on WebGPU, with experts streamed from an SSD. Includes an optional RAM tier, router read-ahead, the expert cloud and drafted previews.

[Unreleased]: https://github.com/Recourse-Software-Inc/latent-fovea/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/Recourse-Software-Inc/latent-fovea/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Recourse-Software-Inc/latent-fovea/releases/tag/v0.1.0
