#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Recourse Software Inc.
# Build the kernel: a SIMD build and a baseline fallback, both import-free.
set -euo pipefail
cd "$(dirname "$0")/.."
FLAGS=(--target=wasm32 -nostdlib -O3 -ffast-math -fno-builtin -mbulk-memory -fvisibility=hidden
       -Wl,--no-entry -Wl,--export-dynamic -Wl,--strip-all -Wl,-z,stack-size=2097152)
clang "${FLAGS[@]}" -msimd128 -o web/lf.simd.wasm kernel/lf.c
clang "${FLAGS[@]}"           -o web/lf.wasm      kernel/lf.c
ls -l web/*.wasm
clang "${FLAGS[@]}" -msimd128 -o web/gpt2.simd.wasm kernel/gpt2.c
clang "${FLAGS[@]}"           -o web/gpt2.wasm      kernel/gpt2.c
ls -l web/gpt2*.wasm
clang "${FLAGS[@]}" -msimd128 -o web/mdlm.simd.wasm kernel/mdlm.c
clang "${FLAGS[@]}"           -o web/mdlm.wasm      kernel/mdlm.c
ls -l web/mdlm*.wasm
clang "${FLAGS[@]}" -msimd128 -o web/dec.simd.wasm kernel/dec.c
clang "${FLAGS[@]}"           -o web/dec.wasm      kernel/dec.c
ls -l web/dec*.wasm
clang "${FLAGS[@]}" -msimd128 -o web/umap.simd.wasm kernel/umap.c
clang "${FLAGS[@]}"           -o web/umap.wasm      kernel/umap.c
ls -l web/umap*.wasm
