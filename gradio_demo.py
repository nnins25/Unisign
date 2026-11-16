"""
AIHub_KSL 학습 모델로 수어 영상을 인식해 한국어 텍스트를 출력하는 Gradio 데모.

사용 예:
    python gradio_demo.py --checkpoint out/aihub_ksl_cslr_pose_only_stgcn_v2/best_checkpoint.pth
"""
import argparse
import copy
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RTMLIB_LOCAL = Path(__file__).parent / "demo" / "rtmlib-main"
if RTMLIB_LOCAL.exists() and str(RTMLIB_LOCAL) not in sys.path:
    sys.path.insert(0, str(RTMLIB_LOCAL))

import cv2
import gradio as gr
import numpy as np
import torch
from tqdm import tqdm

from rtmlib import Wholebody
import utils
from demo_ksl_video import build_model, run_inference


# ── 전역 리소스 (앱 시작 시 한 번만 로드) ──────────────────────────────────
_model = None
_args = None
_wholebody = None


def load_resources(args):
    global _model, _args, _wholebody
    _args = args
    utils.set_seed(args.seed)

    print("Wholebody pose estimator 로딩 중...")
    _wholebody = Wholebody(
        to_openpose=False,
        mode=args.rtmpose_mode,
        backend="onnxruntime",
        device="cuda",
    )

    print("모델 로딩 중...")
    _model = build_model(args)
    print("로딩 완료. Gradio 앱을 시작합니다.\n")


# ── 포즈 추출 + 시각화 ──────────────────────────────────────────────────────
def _process_frame(frame):
    frame = np.uint8(frame)
    keypoints, scores = _wholebody(frame)
    h, w, _ = frame.shape
    return keypoints, scores, [w, h]


def extract_pose_with_vis(video_path: str, vis_out_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_process_frame, f) for f in frames]
        for fut in tqdm(futures, desc="포즈 추출"):
            results.append(fut.result())

    # 시각화 영상 저장
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(vis_out_path, fourcc, fps, (width, height))

    pose_data = {"keypoints": [], "scores": []}
    for (keypoints, scores, wh), frame in zip(results, frames):
        vis_frame = frame.copy()
        try:
            if keypoints is not None and keypoints.shape[0] > 0:
                kps = keypoints[0]   # 첫 번째 사람 [num_kp, 2]
                sc = scores[0] if scores is not None and scores.shape[0] > 0 else np.ones(len(kps))
                for (x, y), conf in zip(kps, sc):
                    if conf > 0.3:
                        cv2.circle(vis_frame, (int(x), int(y)), 3, (0, 255, 80), -1)
        except Exception:
            pass
        writer.write(vis_frame)

        pose_data["keypoints"].append(keypoints / np.array(wh)[None, None])
        pose_data["scores"].append(scores)

    writer.release()
    return pose_data


# ── Gradio 추론 함수 ─────────────────────────────────────────────────────────
def predict(video_path):
    if video_path is None:
        return "", "", None

    vis_path = tempfile.mktemp(suffix=".mp4")
    try:
        t0 = time.time()
        pose_data = extract_pose_with_vis(video_path, vis_path)
        t1 = time.time()

        args_copy = copy.copy(_args)
        args_copy.video = video_path
        prediction = run_inference(args_copy, pose_data, _model)
        t2 = time.time()

        timing = (
            f"포즈 추출: {t1 - t0:.1f}s  |  "
            f"추론: {t2 - t1:.1f}s  |  "
            f"총 소요: {t2 - t0:.1f}s"
        )
        return prediction, timing, vis_path

    except Exception as e:
        return f"[오류] {e}", "", None


# ── Gradio UI ────────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(title="KSL CSLR Recognition Demo") as demo:
        gr.Markdown("# KSL Recognition Demo")
        gr.Markdown("수어 영상을 업로드하고 **인식 시작** 버튼을 누르면 텍스트로 변환합니다.")

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="수어 영상 업로드", sources=["upload"])
                run_btn = gr.Button("START", variant="primary", size="lg")

            with gr.Column(scale=1):
                pose_output = gr.Video(label="Pose Results", interactive=False)
                text_output = gr.Textbox(
                    label="Results",
                    lines=3,
                    interactive=False,
                    placeholder="인식된 수어 텍스트가 여기에 표시됩니다.",
                )
                timing_output = gr.Textbox(label="Processing Time", interactive=False)

        run_btn.click(
            fn=predict,
            inputs=[video_input],
            outputs=[text_output, timing_output, pose_output],
        )

    return demo


# ── 진입점 ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",
                   default="out/aihub_ksl_cslr_pose_only_stgcn_v2/best_checkpoint.pth")
    p.add_argument("--rtmpose_mode", default="lightweight",
                   choices=["lightweight", "balanced", "performance"])
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
    p.add_argument("--max_length", default=256, type=int)
    p.add_argument("--max_new_tokens", default=100, type=int)
    p.add_argument("--num_beams", default=4, type=int)
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--share", action="store_true", help="Gradio public link 생성")
    return p.parse_args()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    load_resources(args)
    demo = build_ui()
    demo.launch(share=args.share, show_api=False)
