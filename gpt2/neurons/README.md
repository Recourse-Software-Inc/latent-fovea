# Neuron book

Every MLP neuron's strongest examples, most negative inputs, activation shape and random samples
from each band of its range, over about three million tokens (4,829 documents) of FineWeb-Edu.
Built by `tools/build_neuron_examples.py`; `meta.json` describes every file and the build settings.

`corpus.u16` holds the sampled documents as GPT-2 token ids, so the viewer can show each example's
whole passage.

FineWeb-Edu: Hugging Face, https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu, released
under the Open Data Commons Attribution License (ODC-By) 1.0. This folder holds a sample of it,
tokenized; no other change was made to the text.
