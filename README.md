# Latent Fovea data

Prebuilt data sets for [Latent Fovea](https://github.com/Recourse-Software-Inc/latent-fovea), by KnowDrive.ai (Recourse Software Inc.). Fetch them into a clone with `tools/get_data.sh`.

Each folder holds a model's token embeddings (or, for `glm52-experts`, GLM-5.2's router rows) as tiles, with an exact nearest-neighbour graph, UMAP and PCA layouts and labels. `gpt2`, `distilgpt2` and `mdlm` also hold the model's weights in 16-bit form for the forward pass. `meta.json` records the source and, where one was read, the exact tensor and its SHA-256.

The data comes from the models, so it is not covered by Latent Fovea's Apache licence: each folder carries its model's licence (`LICENSE`) and what was changed (`NOTICE`). `layouts/README.md` lists the layouts' models and licences.

| Folder | Model | Licence |
|---|---|---|
| gpt2 | GPT-2 small, OpenAI | Modified MIT |
| distilgpt2 | DistilGPT2, Hugging Face | Apache-2.0 |
| mdlm | MDLM, Kuleshov Group, Cornell | Apache-2.0 |
| glm52-experts | GLM-5.2 router, Zhipu AI (Z.ai) | MIT |

Rebuild any of them with `tools/build_data.py` (or `tools/build_experts.py`), or with the repository's **build-data** workflow.
