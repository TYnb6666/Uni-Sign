output_dir=out/pose_only_training

# Using fine_tuning.py as it uses the updated data_loader_multigraph.py
# We do NOT leverage Stage 1/2 pre-training here as their scripts use the old datasets.py
# This serves as a "Train from Scratch" or "Fine-tune" script depending on if you provide --finetune

deepspeed --include localhost:0,1,2,3 --master_port 29511 fine_tuning.py \
  --batch-size 8 \
  --gradient-accumulation-steps 1 \
  --epochs 50 \
  --opt AdamW \
  --lr 3e-4 \
  --output_dir $output_dir \
  --dataset CSL_Daily \
  --task SLT \
  # --finetune out/stage1/best.pth # Uncomment if you have a compatible Stage 1 checkpoint
  # --rgb_support # Disabled for pose-only
