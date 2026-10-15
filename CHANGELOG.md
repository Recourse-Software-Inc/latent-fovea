# Changelog

## 0.1.0 (2026-10-15)

First public release.

- Foveated viewer for the token embeddings of GPT-2, DistilGPT2 and MDLM: tiles streamed by HTTP Range, exact kNN, UMAP and PCA layouts, local re-projection in WebAssembly.
- GPT-2's forward pass, slowed down: the attention field, the logit lens, hidden-state space, and SAE features.
- MDLM masked diffusion, shown filling in its answer.
- Any compatible model from Hugging Face, checked, downloaded and built in the browser.
- A KnowDrive store as a data set: retrieval shown as attention.
- GLM-5.2 (744B MoE) in the browser on WebGPU, with experts streamed from an SSD. Includes an optional RAM tier, router read-ahead, the expert cloud and drafted previews.
