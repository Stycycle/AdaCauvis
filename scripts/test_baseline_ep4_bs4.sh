#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CHECKPOINT="${CHECKPOINT:-weights/cauvis_dinohead.pth}"
WORK_DIR="${WORK_DIR:-work_dir/test_baseline_640_ep${EPOCHS}_bs${BATCH_SIZE}}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
XFORMERS_DISABLED="${XFORMERS_DISABLED:-1}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python tools/test.py \
  configs/cauvis/baseline_dinov2_dinohead_bs1x4_sdgod.py \
  "${CHECKPOINT}" \
  --work-dir "${WORK_DIR}"