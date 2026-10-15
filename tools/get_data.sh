#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Recourse Software Inc.
# Fetch the prebuilt data set from the repo's `data` branch into web/data/.
set -euo pipefail
cd "$(dirname "$0")/.."
git fetch --depth 1 origin data
mkdir -p web/data
git --work-tree=web/data checkout FETCH_HEAD -- .
git reset -q  # keep the index clean; web/data is gitignored
ls web/data
