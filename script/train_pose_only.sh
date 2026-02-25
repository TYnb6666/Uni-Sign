#!/usr/bin/env bash
set -euo pipefail

output_dir=${1:-out/pose_only_training}
input_setting=${2:-lrbf}  # lr | lrb | lrbf

# Using fine_tuning.py + data_loader_multigraph.py
# "From scratch" here means: no Uni-Sign stage checkpoint (--finetune omitted).
# Note: mT5 backbone still loads from ./pretrained_weight/mt5-base.

deepspeed --include localhost:0,1,2,3 --master_port 29511 fine_tuning.py \
  --batch-size 8 \
  --gradient-accumulation-steps 1 \
  --epochs 50 \
  --opt adamw \
  --lr 3e-4 \
  --output_dir "$output_dir" \
  --dataset CSL_Daily \
  --task SLT \
  --input_setting "$input_setting"

# If you have a compatible checkpoint, append:
#   --finetune path/to/checkpoint.pth

# Examples:
# bash ./script/train_pose_only.sh out/ablation_lr lr
# bash ./script/train_pose_only.sh out/ablation_lrb lrb
# bash ./script/train_pose_only.sh out/ablation_lrbf lrbf
