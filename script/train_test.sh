#!/usr/bin/env bash
set -euo pipefail

output_dir=${1:-out/test_training}

# Minimal smoke training on 1 GPU.
deepspeed --include localhost:0 --master_port 29511 fine_tuning.py \
  --batch-size 2 \
  --gradient-accumulation-steps 1 \
  --epochs 1 \
  --opt adamw \
  --lr 3e-4 \
  --output_dir "$output_dir" \
  --dataset CSL_Daily \
  --task SLT
