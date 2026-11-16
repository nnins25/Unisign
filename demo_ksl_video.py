"""
AIHub_KSL 학습한 모델로 영상 한 개를 입력받아 한국어 morpheme 시퀀스를 출력한다.

사용 예:
    python demo_ksl_video.py --video path/to/video.mp4 ^
        --checkpoint out/aihub_ksl_cslr_pose_only_stgcn_v2/best_checkpoint.pth

내부적으로는 demo/online_inference.py 와 같은 흐름이지만
1) 결과 출력을 깔끔하게 표시하고
2) rtmlib 경로 자동 추가
3) KSL 학습 시 사용한 mT5 / hidden_dim 기본값을 미리 채워둔다.
"""
import argparse
import os
import sys
import time
from pathlib import Path

# rtmlib (Wholebody) import 경로
RTMLIB_LOCAL = Path(__file__).parent / "demo" / "rtmlib-main"
if RTMLIB_LOCAL.exists() and str(RTMLIB_LOCAL) not in sys.path:
    sys.path.insert(0, str(RTMLIB_LOCAL))

import cv2
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from rtmlib import Wholebody
from models import Uni_Sign
import utils
from datasets import S2T_Dataset_online


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, type=str,
                   help="Path to an MP4 video file.")
    p.add_argument("--checkpoint", default="out/aihub_ksl_cslr_pose_only_stgcn_v2/best_checkpoint.pth",
                   type=str, help="Trained checkpoint (.pth) path.")
    p.add_argument("--max_length", default=256, type=int)
    p.add_argument("--max_new_tokens", default=100, type=int)
    p.add_argument("--num_beams", default=4, type=int)
    p.add_argument("--rtmpose_mode", default="lightweight",
                   choices=["lightweight", "balanced", "performance"])
    # 학습 시 사용한 모델 구조 (그대로 둬야 checkpoint 와 호환)
    p.add_argument("--hidden_dim", default=256, type=int)
    p.add_argument("--dataset", default="AIHub_KSL", type=str)
    p.add_argument("--task", default="CSLR", type=str)
    p.add_argument("--rgb_support", action="store_true")
    p.add_argument("--mt5_init", default="tiny_random", type=str)
    p.add_argument("--mt5_d_model", default=256, type=int)
    p.add_argument("--mt5_d_kv", default=64, type=int)
    p.add_argument("--mt5_d_ff", default=512, type=int)
    p.add_argument("--mt5_num_layers", default=2, type=int)
    p.add_argument("--mt5_num_decoder_layers", default=2, type=int)
    p.add_argument("--mt5_num_heads", default=4, type=int)
    p.add_argument("--mt5_dropout", default=0.1, type=float)
    p.add_argument("--label_smoothing", default=0.0, type=float)
    p.add_argument("--seed", default=0, type=int)
    return p.parse_args()


def process_frame(frame, wholebody):
    frame = np.uint8(frame)
    keypoints, scores = wholebody(frame)
    h, w, _ = frame.shape
    return keypoints, scores, [w, h]


def extract_pose(video_path: str, mode: str):
    print(f"[1/3] Pose extraction from: {video_path}")
    wholebody = Wholebody(
        to_openpose=False,
        mode=mode,
        backend="onnxruntime",
        device="cuda",
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"      Read {len(frames)} frames.")

    results = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_frame, f, wholebody) for f in frames]
        for fut in tqdm(futures, desc="      pose"):
            results.append(fut.result())

    data = {"keypoints": [], "scores": []}
    for keypoints, scores, wh in results:
        data["keypoints"].append(keypoints / np.array(wh)[None, None])
        data["scores"].append(scores)
    return data


def build_model(args):
    print(f"[2/3] Build model and load checkpoint: {args.checkpoint}")
    model = Uni_Sign(args=args)
    model.cuda()
    for _, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    ret = model.load_state_dict(state, strict=True)
    if ret.missing_keys:
        print("      missing keys:", len(ret.missing_keys))
    if ret.unexpected_keys:
        print("      unexpected keys:", len(ret.unexpected_keys))

    model.eval()
    model.to(torch.bfloat16)
    return model


def run_inference(args, pose_data, model):
    print(f"[3/3] Inference")
    dataset = S2T_Dataset_online(args=args)
    dataset.rgb_data = args.video
    dataset.pose_data = pose_data

    loader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn,
                        sampler=torch.utils.data.SequentialSampler(dataset))

    preds_ids = []
    with torch.no_grad():
        for src_input, tgt_input in loader:
            for k in src_input.keys():
                if isinstance(src_input[k], torch.Tensor):
                    src_input[k] = src_input[k].to(torch.bfloat16).cuda()
            stack_out = model(src_input, tgt_input)
            out = model.generate(stack_out,
                                 max_new_tokens=args.max_new_tokens,
                                 num_beams=args.num_beams)
            for i in range(len(out)):
                preds_ids.append(out[i])

    tokenizer = model.mt5_tokenizer
    pad_value = tokenizer.eos_token_id
    pad_tensor = torch.ones(150 - len(preds_ids[0])).cuda() * pad_value
    preds_ids[0] = torch.cat((preds_ids[0], pad_tensor.long()), dim=0)
    padded = pad_sequence(preds_ids, batch_first=True, padding_value=pad_value)
    decoded = tokenizer.batch_decode(padded, skip_special_tokens=True)
    return decoded[0]


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    utils.set_seed(args.seed)

    t0 = time.time()
    pose_data = extract_pose(args.video, mode=args.rtmpose_mode)
    t1 = time.time()
    model = build_model(args)
    t2 = time.time()
    prediction = run_inference(args, pose_data, model)
    t3 = time.time()

    print()
    print("=" * 60)
    print(f"Video      : {args.video}")
    print(f"Prediction : {prediction}")
    print("=" * 60)
    print(f"Timing | pose: {t1-t0:.1f}s | model load: {t2-t1:.1f}s | inference: {t3-t2:.1f}s | total: {t3-t0:.1f}s")


if __name__ == "__main__":
    main()
