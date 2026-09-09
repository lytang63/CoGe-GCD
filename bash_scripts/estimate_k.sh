#!/usr/bin/env bash
set -euo pipefail
python -m methods.estimate_k.estimate_k --max_classes "${MAX_CLASSES:-1000}" --dataset_name "${1:-cifar100}" --search_mode other --use_composition "${USE_COMPOSITION:-True}"
