#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
WORK_DIR="${WORK_DIR:-work_dir/train_baseline_640_ep${EPOCHS}_bs${BATCH_SIZE}}"

PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
XFORMERS_DISABLED="${XFORMERS_DISABLED:-1}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python tools/train.py \
  configs/cauvis/baseline_dinov2_dinohead_bs1x4_sdgod.py \
  --amp \
  --work-dir "${WORK_DIR}" \
  --cfg-options \
  train_cfg.max_epochs="${EPOCHS}" \
  train_cfg.val_interval="${EPOCHS}" \
  default_hooks.checkpoint.interval=1 \
  train_dataloader.batch_size="${BATCH_SIZE}"
