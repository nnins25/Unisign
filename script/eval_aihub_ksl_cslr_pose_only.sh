#!/usr/bin/env bash

# AIHub_KSL pose-only CSLR 평가 스크립트.
# fine_tuning.py는 epoch마다 checkpoint_{epoch}.pth를 저장한다.
# 아래 기본값은 full train의 첫 epoch checkpoint인 checkpoint_0.pth를 평가한다.
# TODO: 장기 학습에서는 dev WER 기준 best checkpoint 저장/선택 로직을 점검하고
#       필요하면 best_checkpoint.pth를 --finetune에 지정한다.
# RGB 영상은 사용하지 않으므로 --rgb_support를 넣지 않는다.
# DeepSpeed는 사용하지 않으므로 --enable_deepspeed를 넣지 않는다.
#
# Linux/Git Bash 실행:
# bash script/eval_aihub_ksl_cslr_pose_only.sh
#
# Windows PowerShell 직접 실행:
# python fine_tuning.py --dataset AIHub_KSL --task CSLR --batch-size 1 --eval --max_length 256 --output_dir out/aihub_ksl_cslr_pose_only --finetune out/aihub_ksl_cslr_pose_only/checkpoint_0.pth
#
# smoke/mini 결과를 평가하려면 --finetune 경로를 예를 들어 다음처럼 바꾼다.
# --finetune out/aihub_ksl_smoke100/checkpoint_0.pth
# --finetune out/aihub_ksl_mini1000/checkpoint_0.pth

python fine_tuning.py \
  --dataset AIHub_KSL \
  --task CSLR \
  --batch-size 1 \
  --eval \
  --max_length 256 \
  --output_dir out/aihub_ksl_cslr_pose_only \
  --finetune out/aihub_ksl_cslr_pose_only/checkpoint_0.pth
