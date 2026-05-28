#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"

find \
  "work_dir/test_fixed_020_640_ep${EPOCHS}_bs${BATCH_SIZE}" \
  "work_dir/test_layer_adaptive_640_ep${EPOCHS}_bs${BATCH_SIZE}" \
  -name "*.json" -print
