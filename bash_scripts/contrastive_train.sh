#!/usr/bin/env bash
set -euo pipefail
python -m methods.contrastive_training.contrastive_training \
  --dataset_name "${1:-cifar100}" --batch_size "${BATCH_SIZE:-128}" \
  --epochs "${EPOCHS:-200}" --base_model vit_dino --num_workers "${NUM_WORKERS:-4}" \
  --use_ssb_splits True --sup_con_weight 0.35 --weight_decay 5e-5 \
  --transform imagenet --lr 0.1 --eval_funcs v1 v2 \
  --unsupervised_smoothing 0.1 --exp_name "${EXP_NAME:-coge_gcd}" "${@:2}"
