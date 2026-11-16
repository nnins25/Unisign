"""
실시간 카메라로 한국 수어(KSL) 인식 GUI 데모.

사용법:
    python ksl_gui.py --checkpoint out/aihub_ksl_cslr_pose_only_stgcn_v2/best_checkpoint.pth
"""

import sys, os

print("[1] 기본 import 시작", flush=True)

import argparse
import copy
import threading
import time
from pathlib import Path
from typing import List, Optional

print("[2] 경로 설정", flush=True)
RTMLIB_LOCAL = Path(__file__).parent / "demo" / "rtmlib-main"
if RTMLIB_LOCAL.exists() and str(RTMLIB_LOCAL) not in sys.path:
    sys.path.insert(0, str(RTMLIB_LOCAL))

print("[3] cv2/numpy/torch import", flush=True)
import cv2
import numpy as np
import torch

print("[4] PySide6 import", flush=True)
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

print("[5] rtmlib import", flush=True)
from rtmlib import Wholebody

print("[6] utils import", flush=True)
import utils

print("[7] demo_ksl_video import", flush=True)
from demo_ksl_video import build_model, run_inference

print("[9] 모든 import 완료 (mediapipe는 카메라 시작 시 지연 로드)", flush=True)

# ── 전역 리소스 ────────────────────────────────────────────────────────────────
_model = None
_wholebody = None
_args = None
_wholebody_lock = threading.Lock()  # RTMPose 동시 호출 방지


def load_resources(args):
    global _model, _wholebody, _args
    _args = args
    utils.set_seed(args.seed)

    print("Wholebody pose estimator 로딩 중...", flush=True)
    _wholebody = Wholebody(
        to_openpose=False,
        mode=args.rtmpose_mode,
        backend="onnxruntime",
        device="cuda",
    )

    print("모델 로딩 중...", flush=True)
    _model = build_model(args)
    print("로딩 완료.\n", flush=True)


def _run_wholebody(frame: np.ndarray):
    with _wholebody_lock:
        return _wholebody(np.uint8(frame))


# ── Load Worker (백그라운드 모델 로딩) ────────────────────────────────────────────
class LoadWorker(QObject):
    progress = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, args) -> None:
        super().__init__()
        self._args = args

    @Slot()
    def run(self) -> None:
        try:
            self.progress.emit("Wholebody pose estimator 로딩 중...")
            load_resources(self._args)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


# ── VideoLabel ─────────────────────────────────────────────────────────────────
class VideoLabel(QLabel):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self._pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(820, 462)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return int(width * 9 / 16)

    def set_video_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._fit()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        if self._pixmap:
            self.setPixmap(
                self._pixmap.scaled(
                    self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )


# ── Camera Worker ──────────────────────────────────────────────────────────────
class CameraWorker(QObject):
    frame_ready = Signal(QImage, float)
    error = Signal(str)
    status = Signal(str)
    finished = Signal()

    def __init__(self, camera_index: int) -> None:
        super().__init__()
        self.camera_index = camera_index
        self.running = False
        self.recording = False
        self.show_pose = False
        self._buffer: List[np.ndarray] = []
        self._lock = threading.Lock()
        self._last_kps: Optional[np.ndarray] = None  # RTMPose 미리보기 캐시
        self._last_scs: Optional[np.ndarray] = None
        self._pose_frame_cnt = 0

    @Slot()
    def run(self) -> None:
        self.status.emit(f"카메라 {self.camera_index}번 연결 중...")
        cap = _open_camera(self.camera_index)
        if not cap.isOpened():
            self.error.emit(f"카메라 {self.camera_index}번을 열 수 없습니다.")
            self.finished.emit()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        self.running = True
        last_time = time.perf_counter()
        deadline = time.perf_counter() + 3.0
        self.status.emit("카메라 실행 중")

        while self.running:
            ok, frame = cap.read()
            if not ok or frame is None:
                if time.perf_counter() < deadline:
                    time.sleep(0.03)
                    continue
                self.error.emit("카메라 프레임을 읽지 못했습니다.")
                break

            now = time.perf_counter()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now

            if self.recording:
                with self._lock:
                    self._buffer.append(frame.copy())
                # 우상단 빨간 REC 표시
                h, w = frame.shape[:2]
                cv2.circle(frame, (w - 36, 30), 11, (0, 0, 220), -1)
                cv2.putText(
                    frame,
                    "REC",
                    (w - 100, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 220),
                    2,
                    cv2.LINE_AA,
                )

            # 포즈 미리보기: RTMPose(Wholebody) 키포인트 표시 (5프레임마다 갱신)
            if self.show_pose and _wholebody is not None:
                self._pose_frame_cnt += 1
                if self._pose_frame_cnt % 5 == 1:
                    try:
                        kps, scs = _run_wholebody(frame)
                        self._last_kps = kps
                        self._last_scs = scs
                    except Exception:
                        pass
                if self._last_kps is not None and self._last_kps.shape[0] > 0:
                    kps0 = self._last_kps[0]
                    scs0 = (
                        self._last_scs[0]
                        if self._last_scs is not None
                        else np.ones(len(kps0))
                    )
                    for (x, y), conf in zip(kps0, scs0):
                        if conf > 0.3:
                            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 80), -1)

            cv2.putText(
                frame,
                f"FPS {fps:.1f}",
                (18, frame.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
            self.frame_ready.emit(img, fps)

        cap.release()
        self.finished.emit()

    @Slot()
    def stop(self) -> None:
        self.running = False

    def start_recording(self) -> None:
        with self._lock:
            self._buffer = []
        self.recording = True

    def stop_recording(self) -> List[np.ndarray]:
        self.recording = False
        with self._lock:
            frames = list(self._buffer)
            self._buffer = []
        return frames


# ── Inference Worker ───────────────────────────────────────────────────────────
class InferenceWorker(QObject):
    result_ready = Signal(str, str)  # text, timing_str
    progress = Signal(str)
    error = Signal(str)
    finished = Signal()

    def __init__(self, frames: List[np.ndarray]) -> None:
        super().__init__()
        self.frames = frames

    @Slot()
    def run(self) -> None:
        try:
            t0 = time.time()
            n = len(self.frames)
            self.progress.emit(f"포즈 추출 중... ({n}프레임)")

            pose_data: dict = {"keypoints": [], "scores": []}
            for i, frame in enumerate(self.frames):
                if (i + 1) % max(1, n // 10) == 0:
                    self.progress.emit(f"포즈 추출 중... ({i + 1}/{n})")
                keypoints, scores = _run_wholebody(frame)
                h, w, _ = frame.shape
                pose_data["keypoints"].append(keypoints / np.array([w, h])[None, None])
                pose_data["scores"].append(scores)

            t1 = time.time()
            self.progress.emit("모델 추론 중...")

            args_copy = copy.copy(_args)
            args_copy.video = ""
            text = run_inference(args_copy, pose_data, _model)

            t2 = time.time()
            timing = (
                f"포즈 추출: {t1 - t0:.1f}s  |  "
                f"추론: {t2 - t1:.1f}s  |  "
                f"총 소요: {t2 - t0:.1f}s"
            )
            self.result_ready.emit(text, timing)

        except Exception as e:
            self.error.emit(f"[오류] {e}")
        finally:
            self.finished.emit()


# ── LLM Worker (Ollama 자연어 변환) ───────────────────────────────────────────
class LLMWorker(QObject):
    result_ready = Signal(str)
    error = Signal(str)
    finished = Signal()

    def __init__(self, gloss: str) -> None:
        super().__init__()
        self.gloss = gloss

    @Slot()
    def run(self) -> None:
        try:
            import re
            import requests

            # few-shot turns: model learns to output ONLY a short sentence
            messages = [
                {"role": "system", "content": (
                    "너는 한국 수어 글로스를 자연스러운 한국어 문장으로 번역하는 번역기야. "
                    "번역된 문장 하나만 출력해. 설명·사과·제안·추가 문장은 절대 출력하지 마."
                )},
                {"role": "user",      "content": "지갑 실종 없다"},
                {"role": "assistant", "content": "지갑을 잃어버린 것이 아닙니다."},
                {"role": "user",      "content": "차 유리 내리다"},
                {"role": "assistant", "content": "차 유리를 내려 주세요."},
                {"role": "user",      "content": "돈 계산 잘못하다"},
                {"role": "assistant", "content": "돈 계산을 잘못했습니다."},
                {"role": "user",      "content": "시청 내리다"},
                {"role": "assistant", "content": "시청역에서 내리세요."},
                {"role": "user",      "content": "카드 단말기 터치 다시"},
                {"role": "assistant", "content": "카드를 단말기에 다시 터치해 주세요."},
                {"role": "user",      "content": "손바닥찍다 차내리다 가다"},
                {"role": "assistant", "content": "손바닥을 찍고 차에서 내려서 가세요."},
                {"role": "user",      "content": "지하철 옷가게 있다"},
                {"role": "assistant", "content": "지하철 근처에 옷가게가 있습니다."},
                {"role": "user",      "content": self.gloss},
            ]
            resp = requests.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "qwen2.5:3b",
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 60},
                },
                timeout=60,
            )
            resp.raise_for_status()
            text = resp.json().get("message", {}).get("content", "").strip()
            # 첫 문장만 (두 번째 문장부터 설명이 붙는 경우 방지)
            for sep in (".", "。", "\n"):
                if sep in text:
                    text = text.split(sep)[0] + ("." if sep == "." else "")
                    break
            # 비한글 문자 제거 (한글·공백·문장부호만 유지)
            text = re.sub(r"[^가-힣ᄀ-ᇿ㄰-㆏\s\.,!?]", "", text).strip()
            self.result_ready.emit(text)
        except Exception as e:
            self.error.emit(f"[LLM 오류] {e}")
        finally:
            self.finished.emit()


# ── Main Window ────────────────────────────────────────────────────────────────
class SignLanguageWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("KSL CSLR Recognition Demo")
        self.resize(1360, 820)

        self._camera_worker: Optional[CameraWorker] = None
        self._camera_thread: Optional[QThread] = None
        self._inf_worker: Optional[InferenceWorker] = None
        self._inf_thread: Optional[QThread] = None
        self._llm_worker: Optional[LLMWorker] = None
        self._llm_thread: Optional[QThread] = None
        self._pending_frames: List[np.ndarray] = []

        self._build_widgets()
        self._build_layout()
        self._connect_signals()
        self._apply_style()

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("")

    # ── widgets ─────────────────────────────────────────────────────────────

    def _build_widgets(self) -> None:
        self.video_label = VideoLabel("카메라 대기 중")
        self.video_label.setObjectName("videoView")

        self.model_badge = QLabel("MODEL LOADING...")
        self.model_badge.setObjectName("badgeLoading")

        self.fps_label = QLabel("FPS")
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("")
        self.timing_label = QLabel("-")
        self.timing_label.setObjectName("timingLabel")
        self.timing_label.setWordWrap(True)

        self.history_list = QListWidget()
        self.history_list.setMinimumHeight(140)

        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumHeight(36)
        self.refresh_cam_btn = QPushButton("새로고침")
        self.refresh_cam_btn.setMinimumHeight(36)
        self._populate_cameras()

        self.start_btn = QPushButton("START")
        self.start_btn.setEnabled(False)  # 모델 로딩 완료 후 활성화
        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setEnabled(False)

        self.pose_btn = QPushButton("포즈 미리보기 OFF")
        self.pose_btn.setCheckable(True)
        self.pose_btn.setEnabled(False)

        self.record_btn = QPushButton("인식 시작")
        self.record_btn.setEnabled(False)

        self.infer_btn = QPushButton("Recognition Start")
        self.infer_btn.setEnabled(False)
        self.infer_btn.setObjectName("inferBtn")

        self.natural_text = QTextEdit()
        self.natural_text.setReadOnly(True)
        self.natural_text.setPlaceholderText("")

        self.nlp_btn = QPushButton("자연어 변환")
        self.nlp_btn.setEnabled(False)
        self.nlp_btn.setObjectName("nlpBtn")

    # ── layout ──────────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        main = QGridLayout(root)
        main.setContentsMargins(18, 18, 18, 18)
        main.setHorizontalSpacing(18)
        main.setVerticalSpacing(14)

        # header
        hdr = QHBoxLayout()
        title = QLabel("KSL Recognition Demo")
        title.setObjectName("title")
        hdr.addWidget(title)
        hdr.addStretch()
        hdr.addWidget(self.model_badge)

        # video panel
        vid_panel = QFrame()
        vid_panel.setObjectName("panel")
        vid_layout = QVBoxLayout(vid_panel)
        vid_layout.setContentsMargins(12, 12, 12, 12)
        vid_layout.addWidget(self.video_label)

        # result group
        result_grp = QGroupBox("Results")
        result_layout = QVBoxLayout(result_grp)
        result_layout.addWidget(QLabel("글로스 (Gloss)"))
        result_layout.addWidget(self.result_text, 1)
        result_layout.addWidget(self.nlp_btn)
        result_layout.addWidget(QLabel("자연어 (Korean)"))
        result_layout.addWidget(self.natural_text, 1)
        result_layout.addWidget(QLabel("Processing Time"))
        result_layout.addWidget(self.timing_label)

        # control group
        ctrl_grp = QGroupBox("Camera Control")
        ctrl_layout = QGridLayout(ctrl_grp)
        ctrl_layout.addWidget(QLabel("Camera"), 0, 0)
        ctrl_layout.addWidget(self.camera_combo, 0, 1)
        ctrl_layout.addWidget(self.refresh_cam_btn, 0, 2)
        ctrl_layout.addWidget(self.start_btn, 1, 1)
        ctrl_layout.addWidget(self.stop_btn, 1, 2)
        ctrl_layout.addWidget(self.pose_btn, 2, 1, 1, 2)
        ctrl_layout.addWidget(self.record_btn, 3, 1, 1, 2)
        ctrl_layout.addWidget(self.infer_btn, 4, 1, 1, 2)

        # history group
        hist_grp = QGroupBox("History")
        hist_layout = QVBoxLayout(hist_grp)
        hist_layout.addWidget(self.fps_label)
        hist_layout.addWidget(self.history_list)

        # side panel
        side = QFrame()
        side.setObjectName("sidePanel")
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(14, 14, 14, 14)
        side_layout.setSpacing(14)
        side_layout.addWidget(result_grp, 3)
        side_layout.addWidget(ctrl_grp)
        side_layout.addWidget(hist_grp, 2)

        main.addLayout(hdr, 0, 0, 1, 2)
        main.addWidget(vid_panel, 1, 0)
        main.addWidget(side, 1, 1)
        main.setColumnStretch(0, 3)
        main.setColumnStretch(1, 1)

    # ── signals ─────────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self.start_btn.clicked.connect(self._start_camera)
        self.stop_btn.clicked.connect(self._stop_camera)
        self.refresh_cam_btn.clicked.connect(self._populate_cameras)
        self.pose_btn.toggled.connect(self._toggle_pose)
        self.record_btn.clicked.connect(self._toggle_recording)
        self.infer_btn.clicked.connect(self._start_inference)
        self.nlp_btn.clicked.connect(self._start_nlp)

    # ── model loading ────────────────────────────────────────────────────────

    def start_loading(self, args) -> None:
        self._load_worker = LoadWorker(args)
        self._load_thread = QThread(self)
        self._load_worker.moveToThread(self._load_thread)

        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.progress.connect(self.statusBar().showMessage)
        self._load_worker.finished.connect(self._on_load_done)
        self._load_worker.error.connect(self._on_load_error)
        self._load_worker.finished.connect(self._load_thread.quit)
        self._load_worker.finished.connect(self._load_worker.deleteLater)
        self._load_thread.finished.connect(self._load_thread.deleteLater)

        self._load_thread.start()

    @Slot()
    def _on_load_done(self) -> None:
        self.model_badge.setText("MODEL READY")
        self.model_badge.setObjectName("badgeReady")
        self.model_badge.setStyleSheet("")  # 스타일 강제 갱신
        self.start_btn.setEnabled(True)
        self.statusBar().showMessage("모델 로딩 완료. 카메라를 시작하세요.")

    @Slot(str)
    def _on_load_error(self, msg: str) -> None:
        self.model_badge.setText("LOAD ERROR")
        self.statusBar().showMessage(f"[로딩 오류] {msg}")
        self.result_text.setPlainText(f"[로딩 오류]\n{msg}")

    # ── camera ──────────────────────────────────────────────────────────────

    def _populate_cameras(self) -> None:
        prev = self.camera_combo.currentData()
        self.camera_combo.clear()
        for i in range(5):
            self.camera_combo.addItem(f"Camera {i}", i)
        if prev is not None:
            idx = self.camera_combo.findData(prev)
            if idx >= 0:
                self.camera_combo.setCurrentIndex(idx)

    @Slot()
    def _start_camera(self) -> None:
        if self._camera_thread is not None:
            return
        cam_idx = self.camera_combo.currentData() or 0
        self._camera_worker = CameraWorker(cam_idx)
        self._camera_thread = QThread(self)
        self._camera_worker.moveToThread(self._camera_thread)

        self._camera_thread.started.connect(self._camera_worker.run)
        self._camera_worker.frame_ready.connect(self._on_frame)
        self._camera_worker.error.connect(self._on_error)
        self._camera_worker.status.connect(self.statusBar().showMessage)
        self._camera_worker.finished.connect(self._camera_thread.quit)
        self._camera_worker.finished.connect(self._camera_worker.deleteLater)
        self._camera_thread.finished.connect(self._camera_thread.deleteLater)
        self._camera_thread.finished.connect(self._on_camera_stopped)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.pose_btn.setEnabled(True)
        self.record_btn.setEnabled(True)
        self.camera_combo.setEnabled(False)
        self.refresh_cam_btn.setEnabled(False)
        self.video_label.clear()
        self.video_label.setText("카메라 연결 중...")
        self._camera_thread.start()

    @Slot()
    def _stop_camera(self) -> None:
        if self._camera_worker is not None:
            if self._camera_worker.recording:
                self._camera_worker.stop_recording()
            self._camera_worker.stop()
        self.stop_btn.setEnabled(False)
        self.record_btn.setEnabled(False)
        self.infer_btn.setEnabled(False)
        self.pose_btn.setEnabled(False)

    @Slot()
    def _on_camera_stopped(self) -> None:
        self._camera_worker = None
        self._camera_thread = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.record_btn.setEnabled(False)
        self.infer_btn.setEnabled(False)
        self.pose_btn.setEnabled(False)
        self.pose_btn.setChecked(False)
        self.statusBar().showMessage("카메라 정지")

    @Slot(QImage, float)
    def _on_frame(self, image: QImage, fps: float) -> None:
        self.video_label.set_video_pixmap(QPixmap.fromImage(image))
        self.fps_label.setText(f"FPS {fps:.1f}")

    # ── pose toggle ─────────────────────────────────────────────────────────

    @Slot(bool)
    def _toggle_pose(self, checked: bool) -> None:
        self.pose_btn.setText("포즈 미리보기 ON" if checked else "포즈 미리보기 OFF")
        if self._camera_worker is not None:
            self._camera_worker.show_pose = checked

    # ── recording ───────────────────────────────────────────────────────────

    @Slot()
    def _toggle_recording(self) -> None:
        if self._camera_worker is None:
            return
        if not self._camera_worker.recording:
            self._camera_worker.start_recording()
            self.record_btn.setText("녹화 중지")
            self.infer_btn.setEnabled(False)
            self.statusBar().showMessage("녹화 중... 수어를 수행하세요.")
        else:
            frames = self._camera_worker.stop_recording()
            self.record_btn.setText("녹화 시작")
            if frames:
                self._pending_frames = frames
                self.infer_btn.setEnabled(True)
                self.statusBar().showMessage(
                    f"녹화 완료: {len(frames)}프레임. '인식하기' 버튼을 누르세요."
                )
            else:
                self.statusBar().showMessage("녹화된 프레임이 없습니다.")

    # ── inference ────────────────────────────────────────────────────────────

    @Slot()
    def _start_inference(self) -> None:
        if not self._pending_frames or self._inf_thread is not None:
            return

        self.infer_btn.setEnabled(False)
        self.record_btn.setEnabled(False)
        self.result_text.setPlainText("처리 중...")
        self.timing_label.setText("-")

        self._inf_worker = InferenceWorker(self._pending_frames)
        self._inf_thread = QThread(self)
        self._inf_worker.moveToThread(self._inf_thread)

        self._inf_thread.started.connect(self._inf_worker.run)
        self._inf_worker.result_ready.connect(self._on_result)
        self._inf_worker.error.connect(self._on_error)
        self._inf_worker.progress.connect(self.statusBar().showMessage)
        self._inf_worker.finished.connect(self._inf_thread.quit)
        self._inf_worker.finished.connect(self._inf_worker.deleteLater)
        self._inf_thread.finished.connect(self._inf_thread.deleteLater)
        self._inf_thread.finished.connect(self._on_inference_done)

        self._inf_thread.start()

    @Slot(str, str)
    def _on_result(self, text: str, timing: str) -> None:
        self.result_text.setPlainText(text)
        self.natural_text.clear()
        self.timing_label.setText(timing)
        self.nlp_btn.setEnabled(True)
        self.statusBar().showMessage(f"인식 완료: {text}")

        item = QListWidgetItem(f"{time.strftime('%H:%M:%S')}  {text}")
        self.history_list.insertItem(0, item)
        while self.history_list.count() > 20:
            self.history_list.takeItem(self.history_list.count() - 1)

    @Slot()
    def _on_inference_done(self) -> None:
        self._inf_worker = None
        self._inf_thread = None
        if self._camera_worker is not None:
            self.record_btn.setEnabled(True)

    # ── nlp ──────────────────────────────────────────────────────────────────

    @Slot()
    def _start_nlp(self) -> None:
        if self._llm_thread is not None:
            return
        gloss = self.result_text.toPlainText().strip()
        if not gloss:
            return

        self.nlp_btn.setEnabled(False)
        self.natural_text.setPlainText("변환 중...")
        self.statusBar().showMessage("자연어 변환 중...")

        self._llm_worker = LLMWorker(gloss)
        self._llm_thread = QThread(self)
        self._llm_worker.moveToThread(self._llm_thread)

        self._llm_thread.started.connect(self._llm_worker.run)
        self._llm_worker.result_ready.connect(self._on_nlp_result)
        self._llm_worker.error.connect(self._on_nlp_error)
        self._llm_worker.finished.connect(self._llm_thread.quit)
        self._llm_worker.finished.connect(self._llm_worker.deleteLater)
        self._llm_thread.finished.connect(self._llm_thread.deleteLater)
        self._llm_thread.finished.connect(self._on_nlp_done)

        self._llm_thread.start()

    @Slot(str)
    def _on_nlp_result(self, text: str) -> None:
        self.natural_text.setPlainText(text)
        self.statusBar().showMessage(f"자연어 변환 완료: {text}")

    @Slot(str)
    def _on_nlp_error(self, msg: str) -> None:
        self.natural_text.setPlainText(msg)
        self.statusBar().showMessage(msg)

    @Slot()
    def _on_nlp_done(self) -> None:
        self._llm_worker = None
        self._llm_thread = None
        self.nlp_btn.setEnabled(True)

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self.statusBar().showMessage(msg)
        self.result_text.setPlainText(msg)

    # ── close ────────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        self._stop_camera()
        if self._camera_thread is not None:
            self._camera_thread.quit()
            self._camera_thread.wait(1500)
        if self._inf_thread is not None:
            self._inf_thread.quit()
            self._inf_thread.wait(3000)
        if self._llm_thread is not None:
            self._llm_thread.quit()
            self._llm_thread.wait(3000)
        event.accept()

    # ── style ────────────────────────────────────────────────────────────────

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #111318;
                color: #eef1f6;
                font-family: "Malgun Gothic";
                font-size: 14px;
            }
            #title { font-size: 26px; font-weight: 800; }
            #panel, #sidePanel, QGroupBox {
                background: #181c23;
                border: 1px solid #2a303a;
                border-radius: 8px;
            }
            #videoView {
                background: #050609;
                border-radius: 6px;
                color: #8f98a8;
                font-size: 20px;
            }
            #badgeReady {
                background: #173b2f;
                color: #91f2c9;
                border-radius: 6px;
                padding: 7px 14px;
                font-weight: 700;
            }
            #badgeLoading {
                background: #2b2a14;
                color: #f2d161;
                border-radius: 6px;
                padding: 7px 14px;
                font-weight: 700;
            }
            QGroupBox {
                margin-top: 12px;
                padding: 12px;
                font-weight: 700;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
                color: #aeb7c8;
            }
            QPushButton {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #3f82ff, stop:1 #255fcf);
                border: 1px solid #5a92ff;
                border-bottom: 3px solid #173f91;
                border-radius: 6px;
                padding: 10px 14px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #4b8cff, stop:1 #2d6de6);
            }
            QPushButton:pressed, QPushButton:checked {
                background: #173f91;
                border-top: 3px solid #0d2658;
                border-bottom: 1px solid #5a92ff;
                padding-top: 12px;
                padding-bottom: 8px;
            }
            QPushButton:disabled {
                background: #303640;
                color: #777f8c;
                border: 1px solid #303640;
            }
            #inferBtn {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #22b86e, stop:1 #147a47);
                border: 1px solid #2fd480;
                border-bottom: 3px solid #0c4f2e;
                font-size: 16px;
                padding: 14px;
            }
            #inferBtn:hover {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #27d47f, stop:1 #189c58);
            }
            #inferBtn:disabled {
                background: #303640;
                color: #777f8c;
                border: 1px solid #303640;
            }
            #nlpBtn {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #9b59b6, stop:1 #6c3483);
                border: 1px solid #bb8fce;
                border-bottom: 3px solid #4a235a;
                font-size: 14px;
                padding: 10px;
            }
            #nlpBtn:hover {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #af7ac5, stop:1 #7d3c98);
            }
            #nlpBtn:disabled {
                background: #303640;
                color: #777f8c;
                border: 1px solid #303640;
            }
            QComboBox, QTextEdit, QListWidget {
                background: #0d1016;
                border: 1px solid #2a303a;
                border-radius: 6px;
                padding: 8px;
                selection-background-color: #2f6fed;
            }
            QTextEdit { font-size: 22px; font-weight: 700; }
            #timingLabel { color: #7a8494; font-size: 12px; }
        """)


# ── helpers ───────────────────────────────────────────────────────────────────
_CAM_BACKENDS = (cv2.CAP_ANY, cv2.CAP_MSMF, cv2.CAP_DSHOW)


def _open_camera(index: int):
    for backend in _CAM_BACKENDS:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture()


# ── argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default="out/aihub_ksl_cslr_pose_only_stgcn_v2/best_checkpoint.pth",
    )
    p.add_argument(
        "--rtmpose_mode",
        default="lightweight",
        choices=["lightweight", "balanced", "performance"],
    )
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
    return p.parse_args()


# ── entry ─────────────────────────────────────────────────────────────────────
def main() -> int:
    import traceback as _tb

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print("ksl_gui.py 시작", flush=True)

    try:
        args = parse_args()
        print("args 파싱 완료", flush=True)

        app = QApplication(sys.argv)
        print("QApplication 생성 완료", flush=True)

        window = SignLanguageWindow()
        print("Window 생성 완료", flush=True)

        window.show()
        print("window.show() 완료 — 창이 보여야 합니다", flush=True)

        window.start_loading(args)
        print("백그라운드 로딩 시작됨", flush=True)

        return app.exec()

    except Exception:
        _tb.print_exc()
        try:
            from PySide6.QtWidgets import QMessageBox

            _app = QApplication.instance() or QApplication(sys.argv)
            msg = QMessageBox()
            msg.setWindowTitle("ksl_gui 시작 오류")
            msg.setText(_tb.format_exc())
            msg.exec()
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
