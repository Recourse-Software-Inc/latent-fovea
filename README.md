# Latent Fovea

Fly through the **real latent space of an open-weights model** in the browser. Full detail only where you look.

Built by [KnowDrive.ai](https://knowdrive.ai) (Recourse Software Inc.). Open source under the [Apache License 2.0](LICENSE); see [NOTICE](NOTICE) for attribution. Live demo: [latent.knowdrive.ai](https://latent.knowdrive.ai).

**Requirements:** a desktop browser with WebGL2 and WebAssembly SIMD (current Chrome, Edge, Firefox or Safari). WebGPU (Chrome or Edge 113+, Safari 26) runs Hugging Face models faster; GLM-5.2 needs WebGPU plus a folder picker (Chrome or Edge) and its ~419 GB container on a local SSD. Building data needs Python 3.11+; the checks need Node 22+.

**Status:** an experimental research tool, 0.1.1. Not built for phones. Known limits: SentencePiece tokenizers (Llama 2, TinyLlama) aren't supported, GLM-5.2's context stops at 2,048 tokens, and UMAP layouts for Hugging Face models are computed in the browser unless prebuilt.

**Contents:** [Models](#models) · [What you see](#what-you-see) · [Any model from Hugging Face](#any-model-from-hugging-face) · [Forward pass](#forward-pass-watching-gpt-2-think-slowed-down) · [Diffusion](#diffusion-watching-mdlm-fill-in-the-answer) · [GPU](#running-on-the-gpu) · [Streaming](#how-the-streaming-works) · [Run it](#run-it) · [Build the data](#build-the-data-yourself) · [Files](#files) · [Credits](#credits)

VR headsets render the few degrees under your gaze at full resolution and the rest coarsely. Some also stream at full bitrate only where you look. Latent Fovea applies both ideas to a model's internals. The vectors under your gaze are streamed in, re-projected and searched in WebAssembly. Everything else is drawn from a small precomputed atlas.

**Is it real?** [`web/real.html`](web/real.html) (linked from the header and under every answer) explains what on screen is the model's raw data, what is derived from it (UMAP, logit lens, local PCA) and what is illustration (the thought path, averaged arcs, colours).

## Models

Pick the model from the **Model** menu at the top left. The choice is kept in the URL hash (`#gpt2`, `#distilgpt2`, add `~embeddings` for the explorer), so links go straight to a model.

| Model | Parameters | Licence | What it adds |
|---|---|---|---|
| **GPT-2 small**, OpenAI | 124M | MIT | The classic interpretability subject. 12 layers. |
| **DistilGPT2**, Hugging Face | 82M | Apache-2.0 | GPT-2 distilled to 6 layers with the same vocabulary. Compare how shallower "thinking" looks on the same query. |
| **MDLM** (OpenWebText), Kuleshov Group, Cornell | 169M | Apache-2.0 | A **masked diffusion** language model. It doesn't write left to right: the whole answer starts as `[MASK]` and words lock in over denoising steps, in any order. |

Every language model shows its token-embedding matrix (for GPT-2, `wte`: 50,257 tokens × 768 dimensions, the model's first latent space) and runs live in the forward-pass view. `tools/build_data.py` lists the models (`MODELS`), and `web/data/models.json` is the registry the picker reads. The **build-data** workflow takes a comma-separated list of models to build.

### GLM-5.2 (744B), streamed from an SSD

**GLM-5.2** under "Streamed from your drive" opens `glm.html`: Z.ai's 744B mixture-of-experts model (MIT), run in the browser with [colibrì](https://github.com/JustVugg/colibri)'s technique and container (Vincenzo Fornaro, Apache-2.0). About 40B parameters take part in each token, and only the routed experts (8 of 256 per layer, 21.2 MB each) change from token to token. So the attention, shared experts, dense layers and output layer (~10 GB, int4/int8 as stored) stay on the GPU; the 19,200 routed experts stay on a drive and are read in place, from a folder you pick, as the router asks for them, into per-layer GPU pools with least-recently-used eviction (optionally pre-warmed from colibrì's routing history), with an optional RAM tier below them for computers with a separate graphics card (off on Apple Silicon, where GPU and RAM are the same memory), and optional read-ahead: the next layer's router run early on the current state guesses its experts (measured 79% right) and starts reading them; on a busy USB SSD that costs more than it saves (31 s vs 22.5 s per token), so it is off by default. The engine (`web/glm/glm-gpu.js`, `web/glm/container.js`, worker `web/glm/glm-worker.js`) decodes the container's int4 (group 64) and int8 weights in WebGPU kernels, runs multi-head latent attention over a 576-float latent cache, and routes on the CPU each layer (sigmoid + correction bias, top-8, normalised, ×2.5). The page shows the experts chosen per layer (cached or read from the SSD), the logit lens, attention, and drive statistics. Context stops at 2,048 tokens (the sparse-attention indexer isn't in the container; dense attention is exact up to `index_topk`). Speed is set by the drive: a token that misses every expert reads 12.7 GB.

**Previews** show the next few words within a second while the exact token waits on the drive: each missing expert is replaced by the resident expert of its layer with the closest router row (no drive reads), the preview shown faintly and replaced by the exact words as they arrive. Measured: first word right 69–71%, exact word in the top three 92%, ~0.5 s per word against ~25 s per exact token; leaving missing experts out does worse (54%).

**GLM-5.2 experts** is a cloud of the 19,200 experts themselves (`tools/build_experts.py`): each expert's row of its layer's router (the direction in the hidden state that wakes it) as a vector, laid out like any vocabulary. With GLM-5.2 running in another tab, each token's routing lights up there (BroadcastChannel), and "Distances from the token" places every expert at its exact rank in its layer's choice (`web/expertview.js`).

Get the container (~419 GB without the 10 GB MTP head) onto an SSD:

```bash
hf download mastouri/GLM-5.2-colibri-int4-g64-with-int8-mtp --local-dir /Volumes/<drive>/glm52_i4 --exclude "out-mtp-*"
```

Checked against transformers on the same dequantized weights (`tools/glm_ref.py`), on colibrì's tiny oracle converted by its own converter and on a random model at full GLM-5.2 dimensions (`tools/glm_fixture.py`): identical teacher-forced predictions, chosen experts and greedy text, residuals within 2e-5. `web/glm-test.html?set=<fixture>` runs the check in the browser (local only).

### A KnowDrive store: attention without a decoder

A retrieval store scores a question against a fixed set of keys, masks some out, and reads from the best: what one attention head does, with no weights and nothing generated. A KnowDrive store built with the tools below (it needs a KnowDrive account) opens in the model menu like any model: each atom (a passage of a file) is one vector, 1,024 numbers for a store embedded with Jina-CLIP-v2. The atoms are the cloud, coloured by domain and labelled by their three most distinctive words; the inspector shows the passage, its file and tags; neighbours are exact. **Ask the store** picks an example question: arcs to the page each head returns (Vector, Full-text, Hybrid), the recall shape, the tags the page carries, and **Distances from the question**, every atom at its exact cosine distance with a ring at the page's edge. Click a legend domain to mask it: masked atoms fade in place and the vector page is re-ranked over what is left. The vector head is recomputed in the browser from a query vector recovered from KnowDB's own scores (on a 391,922-atom store it reproduced all 1,500 of them to 2×10⁻⁷, and the browser's top 50 matched KnowDB's in order for all ten example questions); full-text and hybrid are KnowDB's lists, replayed. Handbook, Chapter 18. (KnowDB is KnowDrive's retrieval engine.) No store data set is published with Latent Fovea.

## What you see

| Region | What is drawn | What it costs |
|---|---|---|
| **Periphery** | Every token as a faint point at its place in a 3-D UMAP (cosine) of the full matrix | 12 bytes per token, loaded once (600 KB) |
| **Parafovea** | Brighter points, thinned by a stable per-token sample that falls off with distance from the gaze | Nothing extra |
| **Fovea** (≈3° of visual angle) | Token labels. The **actual 768-d rows**, streamed as tiles. A **local PCA** of just those rows re-lays them out inside the fovea (toggle with `L`). **Exact cosine neighbours** in the full space, drawn as lines. The row itself as a 768-column barcode | Only the tiles under the gaze, at a bandwidth you set, in a cache you size |

Press `F` to turn foveation off and see the baseline.

Neighbour lines often reach far across the 3-D view. That's the point: any 3-D projection of a 768-d space distorts it, and the fovea is where you see the real geometry.

## Any model from Hugging Face

Pick a model from the Hugging Face list in the model dropdown, or open `#hf:<owner>/<repo>` (for example `#hf:Qwen/Qwen2.5-0.5B`). The browser builds everything itself, with no data build on our side:

1. It reads `config.json` and the safetensors headers, then streams the weights by HTTP Range in a few large requests, converting them to fp16.
2. It lays out the token embeddings with a 3-D PCA (UMAP is still offline-only) and sorts them into the same streamable tiles.
3. It stores the result in the browser's private file system (OPFS). The next visit loads it from there.

**Clusters or PCA.** The **Clusters / PCA** switch under the model picker (key `C`) morphs the cloud between the two layouts. Clusters is UMAP: tokens sit with their nearest neighbours, and it is the default. PCA shows the three directions of greatest spread. Models from Hugging Face open clustered straight away when there is a precomputed cluster map for them (`tools/build_layouts.py`, the **build-layouts** workflow, `data/layouts/`). Otherwise they open in PCA while UMAP runs in the background in the browser (`kernel/umap.c`: PCA to 50 dimensions, exact cosine neighbours, the fuzzy graph, 200 epochs). It takes about a minute for GPT-2's vocabulary, then the cloud morphs and the result is saved. On GPT-2 it keeps as many true neighbours as umap-learn does offline (`tools/check_umap.mjs`).

**Stats.** Add `?stats` to the address to show the technical readouts: frame rate, streaming, cache and timings.

Nearest neighbours are computed exactly when you look at a token (`knn-worker.js`) instead of being precomputed.

**Finding models.** The model button opens one dropdown (`web/hubui.js`): at the top, the models ready to use (the built-in ones, then any downloaded to this browser, the open one marked); below, a scrolling search of Hugging Face by family, sorted by downloads, trends or likes, with every result badged. Nothing is fetched until the dropdown opens, and search results are cached in the browser (trimmed to what the list shows, the 12 most recent searches): for an hour the cached list is used as is; after that it is shown at once and checked in the background, where the browser revalidates each request by ETag (an unchanged list is a 304 with no body) and the list is redrawn only if it changed. Picking a result checks it before anything downloads; the **Storage** tab (or **Manage storage** at the top of the menu) shows how much browser storage is used and by what (models, SAE feature dictionaries, unfinished downloads) with a usage bar, deletes any item, and **Free up space** deletes everything except the model open now (`listStored` / `removeStored` in `web/store.js`); **Export** (Chrome and Edge, via `showDirectoryPicker`) copies a stored model to a folder on disk with a manifest of every file and its size, and **Import model from a folder…** (any browser; `<input webkitdirectory>` where there is no directory picker) copies one or several exported models in, checking each file against the manifest and writing `meta.json` last so an interrupted import is never taken for a model (`exportModel` / `findExports` / `importModel` in `web/store.js`). Browser storage is per site and per profile, so this is how a model downloaded once (say on a local copy) reaches latent.knowdrive.ai, another profile or another machine without downloading it again. The **Token** tab holds a token for gated models. The badges:

| Badge | Meaning |
|---|---|
| **Verified** | checked end to end against `transformers` (listed in `web/verified.json`) |
| **Should load** | a supported architecture, safetensors, not quantized, byte-level tokenizer, and it fits |
| **Needs HF token** | gated: accept the licence on Hugging Face, then add a read token under **Token** |
| **Streams** | too big to hold in memory, so its weights are read from storage one layer at a time (see below): exact, but slower |
| **Too big** / **Unsupported** | shown with the reason, if you tick "Also show models it can't run yet" |

Choosing a model runs a check before anything downloads: architecture, weight format, tokenizer, memory, estimated speed and licence. Downloads are pinned to one revision and saved tensor by tensor, so an interrupted download resumes. **Downloaded** lists what is stored, with sizes, and lets you delete models. A token is kept only in this browser and sent only to huggingface.co.

These models run in a general decoder kernel, `kernel/dec.c`. It keeps weights in fp16 and widens them on the fly, which lets models up to about 1.5B parameters fit in WebAssembly's 4 GB.

**Bigger models stream.** Past that, the weights stay in the browser's storage and each layer is read into one shared buffer as the forward pass reaches it (`step_begin` / `step_layer` / `step_end` in the kernel, a synchronous OPFS reader in the worker). Memory then holds only the word tables, one layer, the context cache and what the view records, so models up to about 8B parameters run, for example Qwen2.5-3B, Qwen2.5-7B and Qwen3-8B. The weights are the same fp16 numbers, so streaming is exact; it only costs time, since every token reads the whole model back from disk. Streaming Qwen2.5-0.5B (forced with `?stream`) gives the same output as holding it in memory. The model browser marks such models **Streams**, and its pre-download check reads the model's `config.json` to work out the memory it will need and checks the browser has room to store it.

| Family (`model_type`) | Examples | What the kernel handles |
|---|---|---|
| `gpt2` | GPT-2 small/medium/large, DistilGPT2 | LayerNorm, learned positions, GELU |
| `llama` | SmolLM2, Llama 3.2 (gated) | RMSNorm, rotary (incl. Llama 3 scaling), SwiGLU, grouped-query attention |
| `qwen2`, `qwen3` | Qwen2.5-0.5B/1.5B, Qwen3-0.6B | the same, plus qkv bias (Qwen2) or per-head q/k norm (Qwen3) |
| `gpt_neox` | Pythia | partial rotary, parallel attention + MLP |

Tokenizers are read from `tokenizer.json`: byte-level BPE, including Llama 3 and Qwen's split rules. SentencePiece-style tokenizers (Llama 2, TinyLlama) are not supported yet. For large vocabularies, the logit lens runs at an even spread of depths as the model generates, so that it costs about as much as the forward pass; every other depth is read out the moment you look at it (see **Foveated readouts**).

**Checked against `transformers`.** `tools/check_hf.mjs` builds a model in headless Chromium and runs a prompt through the site's own worker. `tools/ref_hf.py` then compares the result with PyTorch on the same weights: token ids, the residual norm after every block, attention, and the top-10 next tokens.

| Model | Worst residual diff | Worst top-10 prob diff | ms/token (Apple M-series, CPU) |
|---|---|---|---|
| GPT-2 small | 1.3e-4 | 1e-4 | 79 |
| GPT-2 medium | 1.1e-4 | 1.3e-3 | 243 |
| SmolLM2-135M | 1.7e-7 | 3e-6 | 87 |
| Pythia-160M | 1.2e-5 | <1e-4 | 84 |
| Qwen2.5-0.5B | 9.8e-7 | 1e-4 | 345 |
| Qwen3-0.6B | 3.8e-7 | 1e-4 | 359 |
| Qwen2.5-1.5B | 6.4e-7 | 1.2e-5 | 1265 |

The GPT-2 models ship fp32 weights, so storing them as fp16 costs a little more precision than for the bf16 models.

## Forward pass: watching GPT-2 think, slowed down

The page opens on a single box: **type your query**, press Enter, and watch it run. The answer streams in under the box. While each token is forming, the text shows the model's current guess at the layer being replayed. GPT-2 runs entirely in the browser; its weights start downloading on your first keystroke. The embeddings explorer is one tab away (`M`, or open the page with `#embeddings`). It is a from-scratch C forward pass compiled to WebAssembly, running in a Web Worker, and it matches a numpy reference to 4 decimal places. After it computes each token, it replays that computation **layer by layer** at a speed you choose: live, 160 ms, 0.5 s or 1.5 s per layer. You can also pause and **step one layer at a time**.

The panel shows the whole network for the selected token, from the embedding at the bottom to layer 11 at the top:

- **Heads (12 × 12).** Each cell is one attention head. Its fill is how strongly that head writes into the residual stream. Its comb shows how much attention the head pays to each of the most recent tokens.
- **MLP.** All 3,072 neurons of every layer's MLP, 36,864 in total, coloured by activation.
- **Logit lens.** At every depth, the token the model would predict if it stopped there. For "The capital of France is the capital" you can watch it go from ` capital` (echoing the input) to ` city` in the middle layers to ` of` at the top.
- **Foveated detail.** The head under your cursor opens a loupe showing exactly which earlier tokens it attends to. Click to pin it. Many heads send most of their attention to the first token, an "attention sink".

In the 3-D view above, the prompt and the generated text are drawn as a path through the embedding space. The token being processed shows a **thought path**: at each depth, the probability-weighted position of the logit-lens guesses. **Attention arcs** run from the current depth back to the tokens that layer (or the pinned head) attends to.

**Fly-through.** Press `T`, or **Fly-through** under the playback buttons, and the camera travels along the thought path instead of orbiting. For each new word it sits just behind the path and follows it layer by layer, from the token being processed toward the word the model settles on, then carries on from there. The top guesses at the layer on screen are labelled with their probabilities as you pass them. Drag or zoom at any time; the flight resumes 1.5 s later. The setting is remembered. It is for word-by-word models only: MDLM fills every slot at once, so it has no path to follow.

**Attribution.** Press `A`, or **Attribution** in the playback panel, to colour the field by what pushed the model toward a word. A word's score (its logit, compared with the average word's, which is what sets its odds) is a sum of what the embedding, every head and every MLP wrote into the residual stream, so with the final normalisation's scale held at this token it splits exactly into one share per part: direct logit attribution. Heads turn cyan where they pushed toward the word and amber where they pushed against it, each MLP row gets a frame of the same colour, and a **Push** column adds up each layer. The word explained is the one written next; click a row of the logit lens to explain that layer's guess instead. It counts direct effects only, so a head that matters by feeding later layers shows up in those layers. The kernels check out against independent computations to about 1e-5 (per-layer residual changes, the embedding, the biases, and the logit gap against the lens's own probabilities).

**Interventions.** Attribution says which parts pushed toward a word; interventions test it. Shift-click a head (or press `K` over it) to knock it out, or knock out a layer's MLP or a single neuron in the neuron inspector; knocked-out parts get an amber ✕. **Rerun** (under Playback) sends the same text through the changed model: the answer is shown above the unchanged model's answer, with the first difference underlined, and the readout gives the probability the unchanged model's next word has now. Both kernels apply the change inside the forward pass (`iv_head`, `iv_mlp`, `iv_neuron`; scale 0 knocks out), so every later layer and every later token sees it, and it counts every route a part's effect can take. Example, GPT-2 on "The country of Canada is a country of immigrants. We are a country of": the four heads attribution ranks highest for " immigrants" take it from 44.9% to 9.1% when knocked out, while four other early heads leave it at 56.4%; add layer 10's MLP and the model stops repeating itself ("a country of people who have been through the"). A new query always runs the unchanged model first, so there is something to compare with; **Clear** undoes everything.

**Comparing runs.** **Pin as A** keeps the run on screen; then run something else (change one word, switch Greedy to Sample, knock parts out) and the two are compared. The answer shows A's beneath it with the first difference underlined, and chips whose token differs from A's are outlined in amber. In the field, the **Δ A** column gives, layer by layer, how far this run's residual stream has moved from A's at the token on screen (cosine distance, 0 = the same direction), and marks the depth where they part ways: the first to reach a quarter of that token's largest difference. Heads whose attention pattern differs from A's (total variation over the keys) get an amber outline, and the readout gives how likely A's next word is here. Positions are matched by index, which is exact when both texts have the same number of tokens. Example, GPT-2 with "The capital of France is" pinned and "The capital of Germany is" run: at " Germany" the runs part ways at the embedding, as they must, but at " is" the internal state barely moves (cosine distance about 0.001) until layer 9, where it jumps tenfold: the country only reaches the last token's state late. A Rerun with interventions pins the unchanged run for you.

**Features (SAE).** A single neuron responds to many unrelated things; a sparse autoencoder rewrites a layer's residual stream as a sum of a few directions from a learned dictionary, and each of those tends to mean one thing. For GPT-2 small, click a layer's label in the field (EMB, L0, L1…) to open its features at the token on screen. The first time, that layer's dictionary (Joseph Bloom's residual-stream SAEs, `jbloom/GPT2-Small-SAEs-Reformatted`, 24,576 features, MIT) streams from Hugging Face by HTTP Range and is kept in the browser at 16 bits; nothing is fetched for layers you don't open (`web/sae.js`, run in the worker on the stored residual stream). The drawer says how faithfully the active features rebuild this token's state (cosine and the size of what they miss), lists the most active ones with Neuronpedia's description of each (fetched from neuronpedia.org; only the layer and feature number are sent), and for a picked feature shows where it fires across the text, the words its direction writes, and a link to its Neuronpedia page. These SAEs were trained on TransformerLens's GPT-2, which centres every vector written into the residual stream, so the stream is centred before encoding; without that, 1,349 features fired and the rebuild missed 357% of the state, with it 56 fire at a cosine of 0.968 (26% missed), in line with the SAEs' published quality.

**Semantic zoom.** No single 3-D layout can be faithful to the whole vocabulary, but a layout can be faithful to the region you are looking at. Zoom in until the parafovea holds a few hundred to a few thousand tokens and press `Z` (or the button by the layout switch): the layout worker keeps `vectors.f16` loaded, runs the in-browser UMAP on just those rows, and the result is placed where the region is (same centre, 25% more spread, rotated to match the old arrangement by polar decomposition) and morphed in; the rest of the cloud stays as context. The worker also reports how much of the region's true 15-NN (from the UMAP kernel's own neighbour search) the old and new arrangement keep, and the fidelity line shows both. `Z` again, or switching layouts, restores the global cloud. It is the fovea idea applied to the layout itself.

**Axes you choose.** The third layout, **Axes**, takes two word pairs (default "king − queen" and "Paris − France"): x = e(a) − e(b), y = e(c) − e(d) orthogonalised to x, z = the top principal direction of what remains, all computed on the full rows in `knn-worker.js` (`axes` job) and scaled like the other layouts with one factor so the axes stay comparable. Positions along x and y are true projections, so the view is interpretable by construction; the fidelity line reports how much of the variation each axis holds, and the axes are named at their ends in the cloud. Only single-token words work.

**Distances from one token.** Pin a token (click it) and press `D`, or the button by the layout switch: the knn worker measures the exact cosine distance from it to every token (`radial` job, fp16 rows, no shortlist), and each token slides along the bearing it already has on screen to that radius, on a disc facing the camera centred where the pinned token is, scaled so 98% of tokens sit within radius 1. Rings mark cosine 0.9, 0.7, 0.5, 0.3 and 0.1. The radius is exact; the angle is kept from the cloud for continuity and is not meaningful. `D` again, Esc or another layout brings the cloud back.

**Fidelity on screen.** Every derived view says how much it keeps, for the model you are looking at, measured live and cached: under the Clusters / PCA switch, measured at the scale a human reads the cloud: how often a token's 15 true nearest neighbours land within its nearest 1% of the cloud ("the same region"), and near/far agreement: for 40 tokens, partners at Fibonacci true ranks (1st, 2nd, 3rd, 5th… nearest, to half the vocabulary) and the Spearman correlation between those ranks and their 3-D distances, averaged (`rank` in `knn-worker.js`; random partners would only ask about far-vs-far ordering, which a cluster layout does not try to keep); the strict 15-of-15 recall stays in the ⓘ as the caveat, and for PCA the share of the variation its three directions hold (`pca` in `knn-worker.js`, on a spread sample of 4,000 rows); beside the fovea, the share of the local variation its 2-D local PCA shows (`stats[12]` from `kernel/lf.c`); in the features drawer, how faithfully the SAE rebuilds the state; and in the forward pass's status line, how closely the model was checked against its reference implementation (or that it wasn't, one by one). Strictly, GPT-2's clusters keep about 6% of 15-NN and Qwen2.5-0.5B's about 2%; at region scale the numbers are far higher, which is the point of the reframing.

**Foveated readouts.** Detail is computed where you look, not everywhere up front. The worker keeps every token's residual stream at every depth, so any readout can be made later, exactly: point at a layer row (or replay it) and its logit lens is worked out for the token on screen if it wasn't already; rows not yet read out show three faint dots. Type any word into **Explain** ("why not Paris?") and its attribution is worked out at the token on screen, by replaying that one position; the kernel's key-value cache makes the replay bit-exact. Results join the replay history, so the thought path and the fly-through fill in as you look around.

**The hidden states, seen directly.** The cloud is the vocabulary's space; the model thinks in another one, the residual stream at every layer. Press `S`, or **Hidden states** under Playback, and the word cloud dims under every captured token's state at every depth, each token a line from its embedding to its last layer. The worker (`web/hidden.js`, the `hidden` job) normalises each vector as the model's normalisation would (centre, divide by RMS), fits a 3-D PCA on this run's states, leaving out the first token (an attention sink with outsized values that would stretch the axes; it is still projected and drawn) (subspace iteration on an even spread of at most 1,500 of them; about 80 ms for 26 GPT-2 tokens, 250 ms for 250) and projects them all, refitting as the text grows (at most every 1.5 s while it plays) with each axis kept pointing the same way; the readout gives the variance share (43% for "The capital of France is Paris. The capital of Germany is"). Each layer's step on a line is split into what attention changed (cyan) and what the MLP changed (violet), as thick as the change: both kernels export the MLP's output (`ptr_mout`), and the rest of the layer's change is attention's, exact for sequential and parallel layers alike (knocking an MLP out leaves exactly that change, to 2e-7). The slice at the replayed depth is lit; point at a state for its token, depth and logit-lens guess, click it to put that token on screen, and the fly-through follows the token's real state. In GPT-2 the countries and capitals of that text gather apart from the function words by layer 6.

**Lens river.** The panel's **River** tab (or `2`) draws the logit lens for the token on screen at every depth at once: stacked bands, bottom to top, as wide as each word's probability, so a guess can be watched appearing, growing and being overtaken; the empty space above is the rest of the vocabulary, and a white line shows how unsure it is as e^entropy ("as unsure as choosing among N words"). The readout says when it settles, e.g. " the" in that text "settles on ' United' from layer 11 on, in its top ten from layer 6". Big vocabularies fill in the depths they skipped when the river is open and playback is paused.

**Attention flow.** The **Flow** tab (or `3`) lays out tokens left to right and layers bottom to top, with a link wherever attention moves information from an earlier token's state into a later one's: each head's weight times the length of the value vector it reads (`vnorm` in both kernels, after Kobayashi et al., 2020), averaged over heads, links under 3% and beyond each state's top four dropped, drawn once into an offscreen canvas per run length. Weighting matters: in that text GPT-2's last layer gives 66% of its attention to the first token but, weighted by what it offers, 21%. **Rollout** (Abnar and Zuidema, 2020) chains the flow through the layers, residual counted as half, and draws the cone feeding the token on screen; the first token only attends to itself, so rollout piles up there (89% for the second " is") and its share is reported apart, the rest shown among the other tokens (" capital" 33%, " of" 17%, " France" 12%). Point at a state for what it took in at that layer. All three views run at the display's full frame rate with 214 tokens.

**Neuron inspector.** Click any MLP row to open that layer's neurons at the token on screen. It lists the neurons firing hardest, and, for the word being explained, the neurons pushing toward it and against it (activation × how much the neuron's output weights point at the word, through the final normalisation). Pick a neuron to see its activation across the whole text so far and the words its output weights push up and down. In GPT-2, for example, layer 9's neuron 1806 writes toward " an", " a" and " another", and fires on "We are" just before an article. The ⓘ boxes explain each part.

Click any token chip to scrub back to it. Every capture is kept.

**Just watch the cloud.** Press `P`, or the chevron at the panel's top right, to hide the field panel. The run keeps playing, the answer keeps streaming at the top, and the scene re-centres. The **Field** tab at the bottom (or `P` again) brings the panel back. Drag the grip on its top edge to resize it (arrow keys work when the grip has focus). The height and whether it is hidden are remembered.

Cost: about 250 MB of fp16 weights, loaded once when you open the view. The kernel decodes them to f32, so it uses about 575 MB of memory. Each token takes about 110 ms in Node on an Apple M-series laptop, including 13 logit-lens readouts over the full 50,257-token vocabulary. For GPT-2, generation is greedy by default, and "Sample" draws from the top 10 at temperature 0.8. For MDLM, "Sample" (the default) draws from the full distribution, as the paper does. The context window is 256 tokens.

## Diffusion: watching MDLM fill in the answer

With **MDLM** selected, the same query box does something different. Your prompt is followed by N `[MASK]` slots (the **Tokens** setting). At each denoising step, the model reads the whole sequence at once, since its attention runs in both directions. Each still-masked slot then unmasks with probability *dt/t* and takes a token from the model's belief for that slot. This is the paper's `ddpm_cache` sampler. The model only re-runs when something changed, and a final pass fills any slots still masked.

- **The text** shows every slot. Locked words are bright, the word that just locked is highlighted, and masked slots show their current best guess, faded by confidence. Early on the guesses often repeat, because each slot guesses on its own before the words start conditioning on each other.
- **The field** replays each model pass layer by layer for one slot: by default the slot that unmasks at this step. Click a chip to pick another. The heads show that slot's attention over the whole sequence, including words to its right. The logit-lens column shows what the slot would become at each depth.
- **In 3D**, masked slots hang at the probability-weighted centroid of their guesses and snap onto a token when they lock in. Arcs show whom the slot attends to at the current layer.

MDLM runs without time conditioning, so its adaLN modulations are constants. `tools/build_data.py` computes them from the real weights. The browser kernel (`kernel/mdlm.c`) is a bidirectional DiT with rotary attention. It runs all positions together, tile by tile, so the 340 MB of weights stream once per pass. It matches `tools/ref_mdlm.py` (numpy) on the real weights to within about 1e-5. One pass over about 40 positions takes about 1 s in WebAssembly SIMD, and a 32-slot answer usually needs about 20 passes. They're computed ahead in the worker while the replay runs.

## Running on the GPU

Models from Hugging Face (the general kernel's families: GPT-2, Llama and SmolLM, Qwen2/2.5, Qwen3, Pythia) run on the graphics card when the browser has WebGPU (Chrome and Edge 113+, Safari 26, Firefox on Windows), and on the CPU otherwise. `web/gpu-decoder.js` is `kernel/dec.c` rewritten as WGSL compute shaders, one per step: embedding, LayerNorm/RMSNorm, the fp16 matrix-vector products (weights stay fp16 and are read with `unpack2x16float`, so no `shader-f16` is needed), Qwen3's per-head norms, rotary positions from a table worked out in double precision, grouped-query attention over the key-value cache, each head's output (for head strengths and attribution), GELU / SwiGLU with knocked-out neurons, and the MLP output. It has the same interface as `decoder.js`, so the worker and every view are unchanged; its methods are async, and the worker awaits them and handles messages one at a time while the GPU is in use.

- **Bigger models.** GPU memory is outside WebAssembly's 4 GB, so a model up to about 7 GB of fp16 weights loads whole instead of streaming one layer at a time from storage. The model picker searches up to 8B parameters when the browser has a GPU, and badges models too big for the CPU path **Runs on your GPU**. Browsers don't say how much memory a GPU has, so that is a planning figure; a model that doesn't fit (the out-of-memory error scope fires at load) runs on the CPU, and the status line says why.
- **A token is one compute pass** (about 15 dispatches per layer), then one copy back of what the views read: attention, head strengths, every MLP activation, the residual stream at every depth, the MLP outputs and the value-vector sizes. Readouts are batched: the logit lens for every depth in one pass (`mvk` reads each unembedding row once for all depths; `stats` finds each depth's maximum, normaliser, entropy and top candidates on the GPU, so about 750 KB comes back instead of the vocabulary times the depth), and attribution toward every candidate word in one pass. The CPU still computes the "average word" row's sum in double precision from blocks of 256 rows the GPU sums, because GPT-2's huge residual dimensions magnify any rounding there.
- **Checked against the CPU kernel**, recording by recording, on the same prompt: Qwen2.5-0.5B residual stream within 1.3e-5 (relative), attention within 2.4e-5, the same top 10 at every depth, probabilities within 5e-5, attribution within 4e-5, knockouts the same (p(" Berlin") 0.74484 vs 0.74479); GPT-2 (Hugging Face build) residuals within 1.2e-6, attribution within 1e-5; Qwen3-0.6B (with its per-head norms) within 2.8e-6, the same top 10 at all 29 depths; Pythia-160M within 1.3e-4 (its large activations magnify rounding gradually over the layers, from 5e-7 after the first). Every pipeline and binding is validated at load, so a mistake makes the model run on the CPU rather than compute nothing.
- **Speed, Apple M-series GPU, every readout included** (the lens at all depths and attribution toward every candidate): Qwen2.5-0.5B about 11 ms a token for the pass (CPU 93 ms); Qwen2.5-3B about 170 ms a token resident on the GPU, against 1.6 s a token streamed on the CPU with the lens at a sample of depths only.
- **8-bit weights.** When a model's 16-bit weights don't fit, it loads them as 8-bit integers instead: the weight matrices (attention, MLP, embedding and unembedding; norms and biases stay 16-bit) are quantized on the GPU as they upload, one f32 scale per 64 weights (absolute maximum / 127), about 1.06 bytes per weight, so the download and the stored files are unchanged. Every kernel that reads weights has an 8-bit form (`K8` for the hot matrix-vector products, a `wq()` accessor for the rest), and the quantizer adds up its own error as it goes, so the status line can say how far the 8-bit weights are from the published ones (0.64% RMS for Qwen2.5-0.5B, 0.62% for Qwen2.5-3B). Against the 16-bit GPU path: Qwen2.5-0.5B keeps the same top guess at 23–25 of 25 depths and final probabilities within 0.009, in 587 MB instead of 1,016 MB; Qwen2.5-3B fits in 3.4 GB instead of 6.2 GB, runs 2.6× faster (19.8 vs 52.5 ms a step, since a token's pass is limited by reading the weights), gives p(" Berlin") 0.883 against 0.887 and the same 20-token continuation. Qwen2.5-7B (on an M3 Max, which also holds it at 16 bits): 8.2 GB instead of 15.3 GB, 52 ms a step instead of 110, the same top guess at all 29 depths, final probabilities within 0.003, the same 20-token continuation, weights 0.63% off. Greedy text can part ways where two words are nearly tied, as it would under any small change to the model. The picker's planning figure is 9 GB of weights, so 16-bit reaches about 4.5B parameters and 8-bit about 8.5B (Qwen2.5-7B, DeepSeek-R1-Distill-Qwen-7B: "GPU, 8-bit").
- **Run on** under Playback: GPU when available (16-bit, then 8-bit if 16 doesn't fit, then the CPU), GPU with 8-bit weights, or CPU only (remembered; the model loads again). `?backend=cpu` or `?backend=gpu8` forces one for testing. The built-in GPT-2, DistilGPT2 and MDLM keep their own CPU kernels.

## How the streaming works

`tools/build_data.py` sorts tokens into a 16³ grid over the projection, in Morton order. It writes the rows as fp16 in that order, so **each grid cell is one contiguous byte range** of `vectors.f16`: a tile.

Each frame, the WebAssembly kernel does five things:

1. Projects all 50k points and assigns each one a tier from its distance to the gaze.
2. Ranks the grid cells under the fovea, nearest the gaze first, then nearest the eye.
3. JS turns those cells into HTTP `Range` requests within a bandwidth budget.
4. Arriving tiles are decoded fp16 → f32 into a fixed-size slot pool. A clock sweep evicts cells nobody has looked at for a few frames.
5. It runs the local PCA and picks the focus.

JS only uploads the per-point attribute buffer to WebGL2 and draws the overlay.

The GPT-2 data set is 77 MB of vectors. The cache holds decoded f32 rows, so the default 24 MB holds about 7,800 rows: a sixth of the vocabulary.

## Run it

```bash
git clone https://github.com/Recourse-Software-Inc/latent-fovea && cd latent-fovea
tools/get_data.sh          # the prebuilt data sets (~970 MB) from the `data` branch
python3 tools/serve.py     # http://127.0.0.1:8080/  (static server with Range support)
```

The prebuilt data holds GPT-2, DistilGPT2 and MDLM (with weights for the forward pass), the GLM-5.2 expert cloud and layouts for Hugging Face models; each folder carries its model's licence. With no data at all, the viewer can still build any supported model from Hugging Face in the browser: open `http://127.0.0.1:8080/#hf:openai-community/gpt2`. The models it fetches are public and need no Hugging Face token.

Any static host that honours `Range` works, including GitHub Pages. Without Range support the viewer falls back to one full download and says so in the HUD. Point it at another data set with `?data=path/`.

### Deploy to Cloudflare Pages

The **deploy-pages** workflow publishes the viewer and every built model to Cloudflare Pages. It runs on every push to `main`, after every data build, and on demand. `tools/stage_static.py site` splits files larger than Pages' 25 MiB limit into parts, which the viewer reassembles, and adds cache headers. With every model built, the site is about 330 files and 2 GB.

It needs two repository secrets (**Settings → Secrets and variables → Actions**):

- `CLOUDFLARE_API_TOKEN`: a Cloudflare API token with the **Cloudflare Pages: Edit** permission.
- `CLOUDFLARE_ACCOUNT_ID`: your Cloudflare account id.

Without them the job skips with a warning. With them, the first run creates the Pages project, named by the repository variable `PAGES_PROJECT` (default `latent-fovea`); the site is served at `https://<project>.pages.dev` unless that name is taken. Put Cloudflare Access in front of it if it should stay private.

**Other hosts.** `tools/stage_static.py OUT --max-mb N` stages the same site for any static host: it splits files above `N` MB into parts and writes a `_headers` file with cache rules (used by Cloudflare and Netlify; other hosts ignore it). A custom domain is a CNAME to the host's address, set at your DNS provider.

## Build the data yourself

```bash
pip install -r requirements.txt
python3 tools/build_data.py --model gpt2 --out web/data/gpt2 --weights    # ~3 min; downloads 548 MB, writes ~260 MB
```

Each build also rewrites `web/data/models.json`, the registry the model picker reads (`--registry web/data` rewrites only that). `--weights` exports the full model for the forward pass: 12 fp16 layer files of 14 MB each, `wpe`/`ln_f`, and the tokenizer; without it, only the embeddings view works. The forward pass reuses the tile-ordered `vectors.f16` as its embedding and unembedding matrix. Without `--weights`, only the `wte` tensor is downloaded (154 MB). The script reads the safetensors header, then makes one Range request for that tensor's 154 MB. The **build-data** GitHub Action runs the same script and publishes the result to the `data` branch.

A KnowDrive store is built from the store itself. This needs a KnowDrive account and a read token for the store, in `~/.cache/latent-fovea/token` or `$KNOWDRIVE_TOKEN` (never committed):

```bash
node tools/fetch_kdstore.mjs --store vs_… --out ~/.cache/latent-fovea/kdstore-mystore     # every atom with its vector; resumable
python3 tools/build_kdstore.py --cache ~/.cache/latent-fovea/kdstore-mystore --out web/data/mystore --name "My store"
python3 tools/kd_queries.py --data web/data/mystore --cache ~/.cache/latent-fovea/kdstore-mystore --store vs_…   # example questions
```

### Rebuild the kernel

```bash
kernel/build.sh    # clang --target=wasm32, SIMD + baseline builds, no libc, no imports
```

The `.wasm` builds (five kernels, each with a SIMD and a baseline build) are committed, so running the viewer needs no toolchain. It needs a clang with the WebAssembly target and `wasm-ld`; Apple's clang has neither. On macOS, Homebrew's `llvm@21` and `lld@21` work: `PATH="$(brew --prefix llvm@21)/bin:$(brew --prefix lld@21)/bin:$PATH" kernel/build.sh`.

## Files

**Kernels** (`kernel/`, C compiled to WebAssembly by `kernel/build.sh`, no libc):
- `lf.c`: projection and tiers, tile scheduling, slot cache, fp16 decode, local PCA, kNN.
- `gpt2.c`: GPT-2 small with a KV cache, recording attention, head write strength, MLP activations, residuals and the logit lens at every depth.
- `mdlm.c`: the MDLM forward pass.
- `dec.c`: a general decoder-only transformer for the families loaded from Hugging Face.
- `umap.c`: UMAP in the browser.

**Viewer** (`web/`):
- `index.html`, `style.css`, `app.js`: the viewer (WebGL2, camera, streaming, overlay, inspector).
- `field.js` (the forward-pass view: field, lens river, attention flow), `hidden.js` (hidden states in 3-D), `diffusion.js` (MDLM), `sae.js` (sparse-autoencoder features).
- `forward.js` + `forward-worker.js` (built-in models), `mdlm.js`, `decoder.js` and `gpu-decoder.js` (the general decoder on the CPU and on WebGPU), `bpe.js` (the GPT-2 tokenizer).
- `hub.js`, `hubui.js`, `build-worker.js`, `store.js`, `write-worker.js`: the Hugging Face picker, building a model in the browser and keeping it in the browser's storage.
- `knn-worker.js`, `layout-worker.js`, `umap.js`: neighbours and layouts for models built in the browser.
- `kdquery.js`: a KnowDrive store's panel, questions and mask. `expertview.js`: the GLM-5.2 expert cloud.
- `glm.html`, `glm-page.js`, `glm/` (container reader, WebGPU engine, worker): GLM-5.2 streamed from an SSD. `glm-test.html`: its check against a reference (local only).
- `real.html` + `real.js`: the handbook ("Is it real?").

**Tools** (`tools/`):
- `serve.py`: dev server with Range support and the site's security headers. `get_data.sh`: fetches the prebuilt data. `stage_static.py`: stages the site for a static host.
- `build_data.py` (built-in models, with `meta.json` recording source and licence), `build_layouts.py` (precomputed layouts), `build_experts.py` (the GLM-5.2 expert cloud).
- `fetch_kdstore.mjs`, `build_kdstore.py`, `kd_queries.py`: a KnowDrive store as a data set (needs a KnowDrive read token).
- Checks against independent references: `check_forward.mjs` + `ref_gpt2.py`, `check_mdlm.mjs` + `ref_mdlm.py`, `check_hf.mjs` + `ref_hf.py`, `check_umap.mjs`, and `glm_ref.py`, `glm_ref_stream.py`, `glm_fixture.py` for GLM-5.2.

## Roadmap

Done in 0.1.0: GPT-2's token embeddings foveated and streamed; GPT-2 thinking, slowed down (forward pass, attention field, logit lens); DistilGPT2 and MDLM; Hugging Face models on the CPU and WebGPU; a KnowDrive store as attention; GLM-5.2 streamed from an SSD.

Next:
1. **An encoder classifier** (e.g. GLiClass / ModernBERT) that returns a calibrated choice instead of text.
2. **A tiny image-diffusion model**, where the fovea decodes image regions.
3. **SentencePiece tokenizers**, for Llama 2-family models.

## Credits

Latent Fovea is designed and built by KnowDrive.ai, Recourse Software Inc. Copyright 2026 Recourse Software Inc., licensed under the Apache License 2.0. Third-party work it builds on (full notices in [NOTICE](NOTICE)):

- **colibrì** by Vincenzo Fornaro ([JustVugg/colibri](https://github.com/JustVugg/colibri), Apache-2.0): the technique of keeping a mixture-of-experts model's dense part on the GPU and streaming routed experts from disk, and the int4 container format that `web/glm/` and `tools/glm_*` read. The code here is an independent WebGPU implementation.
- **GLM-5.2** by Z.ai, released under the MIT licence.
- **GPT-2 byte-pair encoding**: `web/bpe.js` contains portions translated from OpenAI's GPT-2 encoder (Modified MIT licence, reproduced in NOTICE).
- **UMAP** (McInnes, Healy and Melville; [umap-learn](https://github.com/lmcinnes/umap), BSD-3-Clause) builds the layouts in `tools/`; `kernel/umap.c` is an independent implementation for the browser.
- **GPT-2 sparse autoencoders** by Joseph Bloom (MIT), with feature descriptions from [Neuronpedia](https://www.neuronpedia.org).
- **Jina-CLIP-v2** by Jina AI (CC BY-NC 4.0) embeds the KnowDrive stores the store tools read; a store data set holds those vectors, not the model.
- **Fonts**: Plus Jakarta Sans and JetBrains Mono (SIL Open Font License 1.1), served from the site itself (`web/fonts/`), so pages make no requests to Google.

GPT-2 weights © OpenAI, released under the MIT licence ([model card](https://huggingface.co/openai-community/gpt2)). DistilGPT2 by Hugging Face, released under Apache-2.0 ([model card](https://huggingface.co/distilbert/distilgpt2)). MDLM by Sahoo et al., Kuleshov Group, Cornell, released under Apache-2.0 ([code](https://github.com/kuleshov-group/mdlm), [weights](https://huggingface.co/kuleshov-group/mdlm-owt)). The data set records the exact tensor and its SHA-256 in `meta.json`; each folder of the prebuilt data carries its model's licence.

**Trademarks.** KnowDrive, KnowDrive.ai, KnowDB and the KnowDrive logo (`web/kd-logo.jpg`) are trademarks of Recourse Software Inc. The Apache License does not grant permission to use them (section 6): forks should replace the logo and must not suggest endorsement.
