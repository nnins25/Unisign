#!/usr/bin/env bash

# AIHub_KSL pose-only CSLR smoke test.
# 100 step만 학습해 DataLoader, forward/backward, checkpoint 저장을 빠르게 확인한다.
# RGB 영상은 사용하지 않으므로 --rgb_support를 넣지 않는다.
# DeepSpeed는 사용하지 않으므로 --enable_deepspeed를 넣지 않는다.
#
# Linux/Git Bash 실행:
# bash script/smoke_aihub_ksl_cslr_pose_only.sh
#
# Windows PowerShell 직접 실행:
# python fine_tuning.py --dataset AIHub_KSL --task CSLR --batch-size 1 --gradient-accumulation-steps 8 --epochs 1 --lr 1e-4 --max_length 256 --opt AdamW --max_train_steps 100 --skip_eval --output_dir out/aihub_ksl_smoke100

python fine_tuning.py \
  --dataset AIHub_KSL \
  --task CSLR \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --epochs 1 \
  --lr 1e-4 \
  --max_length 256 \
  --opt AdamW \
  --max_train_steps 100 \
  --skip_eval \
  --output_dir out/aihub_ksl_smoke100
