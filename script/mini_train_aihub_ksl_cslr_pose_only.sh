#!/usr/bin/env bash

# AIHub_KSL pose-only CSLR mini train.
# 1000 step만 학습해 짧은 시간 안에 loss 경향을 확인한다.
# RGB 영상은 사용하지 않으므로 --rgb_support를 넣지 않는다.
# DeepSpeed는 사용하지 않으므로 --enable_deepspeed를 넣지 않는다.
# pretrained mT5 본체는 로드하지 않고, tokenizer만 사용한 작은 랜덤 mT5를 처음부터 학습한다.
#
# Linux/Git Bash 실행:
# bash script/mini_train_aihub_ksl_cslr_pose_only.sh
#
# Windows PowerShell 직접 실행:
# python fine_tuning.py --dataset AIHub_KSL --task CSLR --batch-size 1 --gradient-accumulation-steps 8 --epochs 1 --lr 1e-4 --max_length 256 --opt AdamW --mt5_init tiny_random --mt5_d_model 256 --mt5_d_ff 512 --mt5_num_layers 2 --mt5_num_decoder_layers 2 --mt5_num_heads 4 --max_train_steps 1000 --skip_eval --output_dir out/aihub_ksl_mini1000

python fine_tuning.py \
  --dataset AIHub_KSL \
  --task CSLR \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --epochs 1 \
  --lr 1e-4 \
  --max_length 256 \
  --opt AdamW \
  --mt5_init tiny_random \
  --mt5_d_model 256 \
  --mt5_d_ff 512 \
  --mt5_num_layers 2 \
  --mt5_num_decoder_layers 2 \
  --mt5_num_heads 4 \
  --max_train_steps 1000 \
  --skip_eval \
  --output_dir out/aihub_ksl_mini1000
