"""
실시간 카메라로 한국 수어(KSL) 인식 GUI 데모.

사용법:
    python ksl_gui_copy.py --checkpoint out/aihub_ksl_cslr_pose_only_stgcn_v2/best_checkpoint.pth
"""

import argparse
import copy
import importlib
import os
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from PySide6.QtCore import QObject, QRectF, QThread, Qt, Signal, Slot
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

RTMLIB_LOCAL = Path(__file__).parent / "demo" / "rtmlib-main"
if RTMLIB_LOCAL.exists() and str(RTMLIB_LOCAL) not in sys.path:
    sys.path.insert(0, str(RTMLIB_LOCAL))

Wholebody = importlib.import_module("rtmlib").Wholebody
draw_skeleton = importlib.import_module("rtmlib").draw_skeleton
utils = importlib.import_module("utils")
video_module = importlib.import_module("demo_ksl_video")
build_model = video_module.build_model
run_inference = video_module.run_inference

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


# ── VideoLabel ──
class VideoLabel(QLabel):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self._pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 180)
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
            target = self.contentsRect().size()
            if target.isEmpty():
                return
            scaled = self._pixmap.scaled(
                target,
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation,
            )
            left = (scaled.width() - target.width()) // 2
            top = (scaled.height() - target.height()) // 2
            self.setPixmap(
                scaled.copy(left, top, target.width(), target.height())
            )


class SkeletonSwitch(QPushButton):
    def __init__(self) -> None:
        super().__init__()
        self.setCheckable(True)
        self.setAccessibleName("스켈레톤 표시")
        self.setToolTip("스켈레톤 표시 켜기/끄기")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(84, 36)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        track = QRectF(1, 1, self.width() - 2, self.height() - 2)
        knob_size = track.height() - 6
        if self.isChecked():
            track_color = QColor("#16c86b")
            label_color = QColor("#075c35")
            label = "ON"
            knob_x = track.right() - knob_size - 3
            label_rect = QRectF(
                track.left() + 5,
                track.top(),
                track.width() - knob_size - 9,
                track.height(),
            )
        else:
            track_color = QColor("#e6e8eb")
            label_color = QColor("#858b91")
            label = "OFF"
            knob_x = track.left() + 3
            label_rect = QRectF(
                knob_x + knob_size + 2,
                track.top(),
                track.width() - knob_size - 9,
                track.height(),
            )

        if not self.isEnabled():
            track_color = QColor("#eceff1")
            label_color = QColor("#a3a9ae")

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(track_color)
        painter.drawRoundedRect(track, track.height() / 2, track.height() / 2)

        knob_rect = QRectF(
            knob_x,
            track.top() + 3,
            knob_size,
            knob_size,
        )
        painter.setBrush(QColor(0, 0, 0, 28))
        painter.drawEllipse(knob_rect.translated(0, 1.5))
        painter.setPen(QPen(QColor("#d4d8dc"), 1))
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(knob_rect)

        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(label_color)
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label)

        if self.hasFocus() and self.isEnabled():
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#397fbd"), 1, Qt.PenStyle.DashLine))
            focus_rect = track.adjusted(1.5, 1.5, -1.5, -1.5)
            painter.drawRoundedRect(
                focus_rect, focus_rect.height() / 2, focus_rect.height() / 2
            )
        painter.end()


# ── Camera Worker ──
class CameraWorker(QObject):
    frame_ready = Signal(QImage)
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
        deadline = time.perf_counter() + 3.0

        while self.running:
            ok, frame = cap.read()
            if not ok or frame is None:
                if time.perf_counter() < deadline:
                    time.sleep(0.03)
                    continue
                self.error.emit("카메라 프레임을 읽지 못했습니다.")
                break

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
                if (
                    self._last_kps is not None
                    and self._last_kps.shape[0] > 0
                    and self._last_scs is not None
                ):
                    frame = draw_skeleton(
                        frame,
                        self._last_kps,
                        self._last_scs,
                        openpose_skeleton=False,
                        kpt_thr=0.3,
                    )

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
            self.frame_ready.emit(img)

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


# ── Inference Worker ──
class InferenceWorker(QObject):
    result_ready = Signal(str)
    progress = Signal(str)
    error = Signal(str)
    finished = Signal()

    def __init__(self, frames: List[np.ndarray]) -> None:
        super().__init__()
        self.frames = frames

    @Slot()
    def run(self) -> None:
        try:
            n = len(self.frames)
            self.progress.emit(f"포즈 추출 중... ({n}프레임)")

            pose_data: dict = {"keypoints": [], "scores": []}
            for i, frame in enumerate(self.frames):
                if (i + 1) % max(1, n // 10) == 0:
                    self.progress.emit(f"포즈 추출 중... ({i + 1}/{n})")
                keypoints, scores = _run_wholebody(frame)
                h, w, _ = frame.shape
                normalized = keypoints / np.array([w, h])[None, None]
                pose_data["keypoints"].append(normalized)
                pose_data["scores"].append(scores)

            self.progress.emit("모델 추론 중...")

            args_copy = copy.copy(_args)
            args_copy.video = ""
            text = run_inference(args_copy, pose_data, _model)

            self.result_ready.emit(text)

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


# ── Main Window ──
class SignLanguageWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("KSL Translator")
        self.resize(1440, 1008)

        self._camera_worker: Optional[CameraWorker] = None
        self._camera_thread: Optional[QThread] = None
        self._inf_worker: Optional[InferenceWorker] = None
        self._inf_thread: Optional[QThread] = None
        self._llm_worker: Optional[LLMWorker] = None
        self._llm_thread: Optional[QThread] = None
        self._pending_frames: List[np.ndarray] = []
        self._camera_ready = False
        self.camera_status_bar = QStatusBar()
        self.camera_status_bar.setObjectName("cameraStatusBar")
        self.camera_status_bar.setSizeGripEnabled(False)
        self.camera_status_bar.setFixedHeight(24)
        self.camera_status_bar.setStyleSheet(
            "QStatusBar#cameraStatusBar { "
            "background: rgba(255, 255, 255, 225); "
            "color: #52616b; border: none; } "
            "QStatusBar#cameraStatusBar::item { border: none; }"
        )

        self._build_widgets()
        self._build_layout()
        self._connect_signals()
        self._apply_style()

        self.camera_status_bar.showMessage("")

    # ── widgets ─────────────────────────────────────────────────────────────

    def _build_widgets(self) -> None:
        self.video_label = VideoLabel("카메라 대기 중")
        self.video_label.setObjectName("videoView")

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("")
        self.result_text.setMinimumHeight(80)
        self.result_text.setMaximumHeight(220)

        self.record_btn = QPushButton("●  녹화 시작")
        self.record_btn.setEnabled(False)
        self.record_btn.setObjectName("recordBtn")
        self.record_btn.setFixedHeight(38)

        self.stop_record_btn = QPushButton("■  녹화 종료")
        self.stop_record_btn.setEnabled(False)
        self.stop_record_btn.setObjectName("stopRecordBtn")
        self.stop_record_btn.setFixedHeight(38)

        self.pose_btn = SkeletonSwitch()
        self.pose_btn.setEnabled(False)

        self.natural_text = QTextEdit()
        self.natural_text.setReadOnly(True)
        self.natural_text.setObjectName("translationText")
        self.natural_text.setPlaceholderText("")
        self.natural_text.setMinimumHeight(80)
        self.natural_text.setMaximumHeight(220)

    # ── layout ──────────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        hdr = QHBoxLayout()
        hdr.setContentsMargins(24, 12, 24, 8)
        title_stack = QVBoxLayout()
        title_stack.setSpacing(0)
        brand = QHBoxLayout()
        brand.setSpacing(6)
        ksl_title = QLabel("KSL")
        ksl_title.setStyleSheet(
            "font-size: 48px; font-weight: 800; color: #08bf68;"
        )
        translator_title = QLabel("Translator")
        translator_title.setObjectName("title")
        translator_title.setStyleSheet(
            "font-size: 48px; font-weight: 800; color: #263238;"
        )
        brand.addWidget(ksl_title)
        brand.addWidget(translator_title)
        brand.addStretch()
        title_stack.addLayout(brand)
        subtitle = QLabel("한국 수어를 자연스러운 한국어로")
        subtitle.setStyleSheet("font-size: 14px; color: #737b82;")
        title_stack.addWidget(subtitle)
        hdr.addLayout(title_stack)
        hdr.addStretch()
        outer.addLayout(hdr)

        content = QWidget(root)
        main = QGridLayout(content)
        main.setContentsMargins(250, 140, 250, 140)
        main.setHorizontalSpacing(10)
        main.setVerticalSpacing(8)

        # result group
        result_panel = QWidget()
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(6)

        gloss_label = QLabel("글로스")
        gloss_label.setStyleSheet("font-size: 15pt;")
        result_layout.addWidget(gloss_label)
        result_layout.addWidget(self.result_text, 1)

        korean_label = QLabel("한글 문장")
        korean_label.setStyleSheet("font-size: 15pt;")
        result_layout.addWidget(korean_label)
        result_layout.addWidget(self.natural_text, 1)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        actions.addWidget(self.record_btn, 1)
        actions.addWidget(self.stop_record_btn, 1)
        result_layout.addLayout(actions)

        camera_panel = QWidget()
        camera_panel.setMinimumSize(self.video_label.minimumSize())
        camera_layout = QGridLayout(camera_panel)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        camera_layout.setSpacing(0)
        camera_layout.addWidget(self.video_label, 0, 0)

        pose_controls = QWidget(camera_panel)
        pose_controls.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        pose_controls.setStyleSheet("background: transparent;")
        pose_layout = QHBoxLayout(pose_controls)
        pose_layout.setContentsMargins(8, 8, 8, 8)
        pose_layout.setSpacing(6)
        joint_label = QLabel("관절 표시")
        joint_label.setStyleSheet(
            "color: #263238; background: rgba(255, 255, 255, 220); "
            "border-radius: 4px; padding: 3px 6px;"
        )
        pose_layout.addWidget(joint_label)
        pose_layout.addWidget(self.pose_btn)
        camera_layout.addWidget(
            pose_controls,
            0,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )
        camera_layout.addWidget(
            self.camera_status_bar,
            0,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
        )

        main.addWidget(camera_panel, 0, 0)
        main.addWidget(result_panel, 0, 1)
        main.setRowStretch(0, 1)
        main.setColumnStretch(0, 3)
        main.setColumnStretch(1, 2)
        outer.addWidget(content, 1)

        footer = QWidget(root)
        # footer.setStyleSheet("border-top: 1px solid #e6e9ec;")
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(32, 12, 32, 14)
        footer_layout.setSpacing(4)

        # footer_title = QLabel("KSL Translator")
        # footer_title.setStyleSheet(
        #     "font-size: 15px; font-weight: 700; color: #626c74;"
        # )


        footer_description = QLabel(
                "<hr>"
                "카메라 영상에서 수어 동작을 인식해 글로스로 변환하고,"
                "자연스러운 한국어 문장으로 번역합니다."
            )
        footer_description.setWordWrap(True)
        footer_description.setStyleSheet("font-size: 12px; color: #818a91;")
        footer_gloss_note = QLabel(
            """<font style="color: #52616b;">글로스(gloss)</font>란 수어의 최소 의미 단위를 음성 언어로 1:1 대응시킨 """
            "것으로, 수어를 기계가 다루기 위한 중간 표기 입니다."
        )
        footer_gloss_note.setWordWrap(True)
        footer_gloss_note.setStyleSheet("font-size: 12px; color: #818a91;")
        footer_note = QLabel(
            "인식 결과는 촬영 환경과 수어 표현에 따라 달라질 수 있습니다."
        )
        footer_note.setWordWrap(True)
        footer_note.setStyleSheet("font-size: 12px; color: #969da3;")

        # footer_layout.addWidget(footer_title)
        footer_layout.addWidget(footer_description)
        footer_layout.addWidget(footer_gloss_note)
        footer_layout.addWidget(footer_note)
        outer.addWidget(footer)

    # ── signals ─────────────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        self.record_btn.clicked.connect(self._toggle_recording)
        self.stop_record_btn.clicked.connect(self._toggle_recording)
        self.pose_btn.toggled.connect(self._toggle_pose)

    # ── model loading ────────────────────────────────────────────────────────

    def start_loading(self, args) -> None:
        self._load_worker = LoadWorker(args)
        self._load_thread = QThread(self)
        self._load_worker.moveToThread(self._load_thread)

        self._load_thread.started.connect(self._load_worker.run)
        self._load_worker.progress.connect(self.camera_status_bar.showMessage)
        self._load_worker.finished.connect(self._on_load_done)
        self._load_worker.error.connect(self._on_load_error)
        self._load_worker.finished.connect(self._load_thread.quit)
        self._load_worker.finished.connect(self._load_worker.deleteLater)
        self._load_thread.finished.connect(self._load_thread.deleteLater)

        self._load_thread.start()

    @Slot()
    def _on_load_done(self) -> None:
        self.camera_status_bar.showMessage("모델 준비 완료. 카메라를 연결합니다.")
        self._start_camera()

    @Slot(str)
    def _on_load_error(self, msg: str) -> None:
        self.camera_status_bar.showMessage(f"[로딩 오류] {msg}")
        self.result_text.setPlainText(f"[로딩 오류]\n{msg}")

    # ── camera ──────────────────────────────────────────────────────────────

    @Slot()
    def _start_camera(self) -> None:
        if self._camera_thread is not None:
            return
        self._camera_worker = CameraWorker(0)
        self._camera_thread = QThread(self)
        self._camera_worker.moveToThread(self._camera_thread)

        self._camera_thread.started.connect(self._camera_worker.run)
        self._camera_worker.frame_ready.connect(self._on_frame)
        self._camera_worker.error.connect(self._on_error)
        self._camera_worker.status.connect(self.camera_status_bar.showMessage)
        self._camera_worker.finished.connect(self._camera_thread.quit)
        self._camera_worker.finished.connect(self._camera_worker.deleteLater)
        self._camera_thread.finished.connect(self._camera_thread.deleteLater)
        self._camera_thread.finished.connect(self._on_camera_stopped)

        self._camera_ready = False
        self.record_btn.setEnabled(False)
        self.stop_record_btn.setEnabled(False)
        self.pose_btn.setChecked(False)
        self.pose_btn.setEnabled(False)
        self.video_label.clear()
        self.video_label.setText("카메라 연결 중...")
        self._camera_thread.start()

    @Slot()
    def _stop_camera(self) -> None:
        if self._camera_worker is not None:
            if self._camera_worker.recording:
                self._camera_worker.stop_recording()
            self._camera_worker.stop()
        self._camera_ready = False
        self.record_btn.setEnabled(False)
        self.stop_record_btn.setEnabled(False)
        self.pose_btn.setChecked(False)
        self.pose_btn.setEnabled(False)

    @Slot()
    def _on_camera_stopped(self) -> None:
        self._camera_worker = None
        self._camera_thread = None
        self._camera_ready = False
        self.record_btn.setEnabled(False)
        self.stop_record_btn.setEnabled(False)
        self.pose_btn.setChecked(False)
        self.pose_btn.setEnabled(False)
        self.camera_status_bar.showMessage("카메라 연결이 종료되었습니다.")

    @Slot(QImage)
    def _on_frame(self, image: QImage) -> None:
        self.video_label.set_video_pixmap(QPixmap.fromImage(image))
        if not self._camera_ready:
            self._camera_ready = True
            self.pose_btn.setEnabled(True)
            if self._inf_thread is None:
                self.record_btn.setEnabled(True)

    @Slot(bool)
    def _toggle_pose(self, enabled: bool) -> None:
        if self._camera_worker is not None:
            self._camera_worker.show_pose = enabled
            if enabled:
                self._camera_worker._pose_frame_cnt = 0
                self._camera_worker._last_kps = None
                self._camera_worker._last_scs = None

    # ── recording ───────────────────────────────────────────────────────────

    @Slot()
    def _toggle_recording(self) -> None:
        if self._camera_worker is None:
            return
        if not self._camera_worker.recording:
            self._camera_worker.start_recording()
            self.record_btn.setEnabled(False)
            self.stop_record_btn.setEnabled(True)
            self.result_text.clear()
            self.natural_text.clear()
            self.camera_status_bar.showMessage("녹화 중... 수어를 수행하세요.")
        else:
            frames = self._camera_worker.stop_recording()
            self.stop_record_btn.setEnabled(False)
            if frames:
                self._pending_frames = frames
                self.camera_status_bar.showMessage("녹화 완료. 수어를 인식합니다.")
                self._start_inference()
            else:
                self.record_btn.setEnabled(self._camera_ready)
                self.camera_status_bar.showMessage(
                    "녹화된 프레임이 없습니다. 다시 녹화해 주세요."
                )

    # ── inference ────────────────────────────────────────────────────────────

    @Slot()
    def _start_inference(self) -> None:
        if not self._pending_frames or self._inf_thread is not None:
            return

        self.record_btn.setEnabled(False)
        self.stop_record_btn.setEnabled(False)
        self.result_text.setPlainText("처리 중...")

        self._inf_worker = InferenceWorker(self._pending_frames)
        self._inf_thread = QThread(self)
        self._inf_worker.moveToThread(self._inf_thread)

        self._inf_thread.started.connect(self._inf_worker.run)
        self._inf_worker.result_ready.connect(self._on_result)
        self._inf_worker.error.connect(self._on_error)
        self._inf_worker.progress.connect(self.camera_status_bar.showMessage)
        self._inf_worker.finished.connect(self._inf_thread.quit)
        self._inf_worker.finished.connect(self._inf_worker.deleteLater)
        self._inf_thread.finished.connect(self._inf_thread.deleteLater)
        self._inf_thread.finished.connect(self._on_inference_done)

        self._inf_thread.start()

    @Slot(str)
    def _on_result(self, text: str) -> None:
        self.result_text.setPlainText(text)
        self.natural_text.clear()
        self.camera_status_bar.showMessage(f"인식 완료: {text}")
        if text.strip():
            self._start_nlp()

    @Slot()
    def _on_inference_done(self) -> None:
        self._inf_worker = None
        self._inf_thread = None
        if self._camera_worker is not None:
            self.record_btn.setEnabled(self._camera_ready)

    # ── nlp ──────────────────────────────────────────────────────────────────

    @Slot()
    def _start_nlp(self) -> None:
        if self._llm_thread is not None:
            return
        gloss = self.result_text.toPlainText().strip()
        if not gloss:
            return

        self.natural_text.setPlainText("변환 중...")
        self.camera_status_bar.showMessage("자연어 변환 중...")

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
        self.camera_status_bar.showMessage(f"자연어 변환 완료: {text}")

    @Slot(str)
    def _on_nlp_error(self, msg: str) -> None:
        self.natural_text.setPlainText(msg)
        self.camera_status_bar.showMessage(msg)

    @Slot()
    def _on_nlp_done(self) -> None:
        self._llm_worker = None
        self._llm_thread = None

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self.camera_status_bar.showMessage(msg)
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
                background: #ffffff;
                color: #263238;
                font-family: "Malgun Gothic";
                font-size: 14px;
            }
            #title { font-size: 20px; font-weight: 800; color: #263238; }
            #videoView {
                background: #f1f3f5;
                border-radius: 6px;
                color: #64748b;
                font-size: 20px;
            }
            QPushButton {
                background: #397fbd;
                color: #ffffff;
                border: 1px solid #397fbd;
                border-radius: 6px;
                padding: 6px 8px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #2d689d;
            }
            #recordBtn {
                background: #0caf43;
                border: none;
                border-radius: 0px;
                padding: 8px 12px;
                font-size: 14px;
            }
            #recordBtn:hover {
                background: #078f36;
            }
            #stopRecordBtn {
                background: #f21824;
                border: none;
                border-radius: 0px;
                padding: 8px 12px;
                font-size: 14px;
            }
            #stopRecordBtn:hover {
                background: #d60d19;
            }
            #recordBtn:disabled, #stopRecordBtn:disabled {
                background: #e3e7ea;
                color: #818a91;
            }
            QPushButton:disabled {
                background: #eceff1;
                color: #89939b;
                border: 1px solid #d8dde1;
            }
            QTextEdit {
                background: #ffffff;
                color: #263238;
                border: 1px solid #d5dbe0;
                border-radius: 6px;
                padding: 8px;
                selection-background-color: #d7e9f8;
                selection-color: #1f4e79;
            }
            QTextEdit { font-size: 22px; font-weight: 700; }
            #translationText { font-size: 22px; font-weight: 700; }
        """)


# ── helpers ──
_CAM_BACKENDS = (cv2.CAP_ANY, cv2.CAP_MSMF, cv2.CAP_DSHOW)


def _open_camera(index: int):
    for backend in _CAM_BACKENDS:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture()


# ── argument parsing ──
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


# ── entry ──
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

        window.showMaximized()
        print("window.showMaximized() 완료 — 화면 작업 영역에 표시", flush=True)

        window.start_loading(args)
        print("백그라운드 로딩 시작됨", flush=True)

        return app.exec()

    except Exception:
        _tb.print_exc()
        try:
            from PySide6.QtWidgets import QMessageBox

            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            app.processEvents()
            msg = QMessageBox()
            msg.setWindowTitle("ksl_gui 시작 오류")
            msg.setText(_tb.format_exc())
            msg.exec()
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
