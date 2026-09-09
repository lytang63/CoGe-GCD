#!/usr/bin/env bash
set -euo pipefail
python -m methods.clustering.extract_features --dataset "${1:-cifar100}" --use_best_model True --use_composition "${USE_COMPOSITION:-True}" --warmup_model_dir "${WARMUP_MODEL_DIR:?Set WARMUP_MODEL_DIR}"
