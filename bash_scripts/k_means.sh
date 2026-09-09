#!/usr/bin/env bash
set -euo pipefail
python -m methods.clustering.k_means --dataset_name "${1:-cifar100}" --semi_sup True --use_ssb_splits True --use_composition "${USE_COMPOSITION:-True}" --use_best_model True
