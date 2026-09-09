#!/usr/bin/env bash
set -euo pipefail
python -m save_feats --dataset_name "${1:-cifar100}" --batch_size "${BATCH_SIZE:-128}" --num_workers "${NUM_WORKERS:-4}" --num_primitives "${NUM_PRIMITIVES:-16}" --num_heads "${NUM_HEADS:-8}" --use_ssb_splits True --warmup_model_dir "${WARMUP_MODEL_DIR:?Set WARMUP_MODEL_DIR}"
