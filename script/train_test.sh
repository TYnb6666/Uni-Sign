
output_dir=out/test_training

# Using 'CSL-Daily' as dataset name to trigger 'Chinese' behavior in loader
deepspeed --include localhost:0 --master_port 29511 fine_tuning.py \
  --batch-size 2 \
  --gradient-accumulation-steps 1 \
  --epochs 1 \
  --opt AdamW \
  --lr 3e-4 \
  --output_dir $output_dir \
  --dataset CSL-Daily \
  --task SLT \
  # --rgb_support Removed as we deleted the branch
