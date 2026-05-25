#!/usr/bin/env bash

# AIHub_KSL pose-only CSLR 학습 스크립트.
# RGB 영상은 사용하지 않으므로 --rgb_support를 넣지 않는다.
# DeepSpeed는 사용하지 않으므로 --enable_deepspeed를 넣지 않는다.
# pretrained mT5 본체는 로드하지 않고, tokenizer만 사용한 작은 랜덤 mT5를 처음부터 학습한다.
# batch-size 1에서도 약 11GB VRAM을 사용하므로 batch-size 4는 OOM 가능성이 크다.
# 따라서 기본값은 batch-size 1 + gradient accumulation 8로 둔다.
#
# 권장 실행 순서:
# 1) bash script/smoke_aihub_ksl_cslr_pose_only.sh
# 2) bash script/mini_train_aihub_ksl_cslr_pose_only.sh
# 3) bash script/train_aihub_ksl_cslr_pose_only.sh
# 4) bash script/eval_aihub_ksl_cslr_pose_only.sh
#
# Windows single GPU에서는 Git Bash 또는 WSL이 아니라면 아래 python 명령을
# PowerShell에서 직접 실행해도 된다.
#
# Linux/Git Bash single GPU 실행 예시:
# bash script/train_aihub_ksl_cslr_pose_only.sh
#
# PowerShell 직접 실행 예시:
# python fine_tuning.py --dataset AIHub_KSL --task CSLR --batch-size 1 --gradient-accumulation-steps 8 --epochs 30 --lr 1e-4 --max_length 256 --opt AdamW --mt5_init tiny_random --mt5_d_model 256 --mt5_d_ff 512 --mt5_num_layers 2 --mt5_num_decoder_layers 2 --mt5_num_heads 4 --output_dir out/aihub_ksl_cslr_pose_only

python fine_tuning.py \
  --dataset AIHub_KSL \
  --task CSLR \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --epochs 30 \
  --lr 1e-4 \
  --max_length 256 \
  --opt AdamW \
  --mt5_init tiny_random \
  --mt5_d_model 256 \
  --mt5_d_ff 512 \
  --mt5_num_layers 2 \
  --mt5_num_decoder_layers 2 \
  --mt5_num_heads 4 \
  --output_dir out/aihub_ksl_cslr_pose_only
