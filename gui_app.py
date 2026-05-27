import sys
import time
from dataclasses import dataclass
from typing import Optional

import cv2
from PySide6.QtCore import QObject, QThread, Qt, QTimer, Signal, Slot
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

try:
    import mediapipe as mp
except Exception:
    mp = None


MODEL_PATH = r"out\aihub_ksl_cslr_pose_only_tiny_eval\checkpoint_174.pth"
CAMERA_BACKENDS = (
    cv2.CAP_ANY,
    cv2.CAP_MSMF,
    cv2.CAP_DSHOW,
)


@dataclass
class Prediction:
    label: str
    confidence: float
    translation: str


class GestureTranslator:
    """Replace predict() with the trained sign-language model later."""

    def __init__(self) -> None:
        self.model_path: Optional[str] = MODEL_PATH or None
        self.ready = bool(self.model_path)
        self.label_to_korean = {
            "hello": "안녕하세요",
            "thanks": "감사합니다",
            "sorry": "미안합니다",
            "no_gesture": "",
        }

    def predict(self, frame) -> Optional[Prediction]:
        if not self.ready:
            return None

        # TODO: Replace this block with real model inference.
        # Expected return: Prediction(label="...", confidence=0.0~1.0, translation="...")
        return Prediction(label="model_ready", confidence=0.0, translation=f"연결된 checkpoint: {self.model_path}")


class VideoLabel(QLabel):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.current_pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(820, 462)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return int(width * 9 / 16)

    def set_video_pixmap(self, pixmap: QPixmap) -> None:
        self.current_pixmap = pixmap
        self._fit_pixmap()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_pixmap()

    def _fit_pixmap(self) -> None:
        if self.current_pixmap is None:
            return
        scaled = self.current_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)


class CameraWorker(QObject):
    frame_ready = Signal(QImage, float)
    error = Signal(str)
    status = Signal(str)
    finished = Signal()

    def __init__(self, camera_index: int, translator: GestureTranslator, draw_landmarks: bool) -> None:
        super().__init__()
        self.camera_index = camera_index
        self.translator = translator
        self.draw_landmarks = draw_landmarks
        self.running = False
        self.latest_prediction: Optional[Prediction] = None

    @Slot()
    def run(self) -> None:
        self.status.emit(f"카메라 {self.camera_index}번 연결 중...")
        cap = open_camera(self.camera_index)
        if not cap.isOpened():
            self.error.emit(f"카메라 {self.camera_index}번을 열 수 없습니다.")
            self.finished.emit()
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        holistic = None
        if self.draw_landmarks:
            if mp is None:
                self.error.emit("Landmarks를 사용하려면 mediapipe 설치가 필요합니다: pip install mediapipe")
            else:
                holistic = mp.solutions.holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    refine_face_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )c

        self.running = True
        last_time = time.perf_counter()
        first_frame_deadline = time.perf_counter() + 3.0
        self.status.emit("카메라 프레임 수신 중")

        while self.running:
            ok, frame = cap.read()
            if not ok or frame is None:
                if time.perf_counter() < first_frame_deadline:
                    time.sleep(0.03)
                    continue
                self.error.emit("카메라 프레임을 읽지 못했습니다.")
                break

            now = time.perf_counter()
            elapsed = max(now - last_time, 1e-6)
            last_time = now
            fps = 1.0 / elapsed

            if holistic is not None:
                rgb_for_landmarks = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb_for_landmarks)
                self._draw_holistic_landmarks(frame, results)

            prediction = self.translator.predict(frame)
            if prediction:
                self.latest_prediction = prediction
                cv2.putText(
                    frame,
                    f"{prediction.label} {prediction.confidence:.2f}",
                    (24, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (32, 220, 128),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                f"FPS {fps:.1f}",
                (24, frame.shape[0] - 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
            self.frame_ready.emit(image, fps)

        cap.release()
        if holistic is not None:
            holistic.close()
        self.finished.emit()

    @Slot()
    def stop(self) -> None:
        self.running = False

    def _draw_holistic_landmarks(self, frame, results) -> None:
        drawing = mp.solutions.drawing_utils
        styles = mp.solutions.drawing_styles
        holistic_module = mp.solutions.holistic

        if results.pose_landmarks:
            drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                holistic_module.POSE_CONNECTIONS,
                landmark_drawing_spec=styles.get_default_pose_landmarks_style(),
            )
        if results.face_landmarks:
            drawing.draw_landmarks(
                frame,
                results.face_landmarks,
                holistic_module.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=styles.get_default_face_mesh_tesselation_style(),
            )
        if results.left_hand_landmarks:
            drawing.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                holistic_module.HAND_CONNECTIONS,
                styles.get_default_hand_landmarks_style(),
                styles.get_default_hand_connections_style(),
            )
        if results.right_hand_landmarks:
            drawing.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                holistic_module.HAND_CONNECTIONS,
                styles.get_default_hand_landmarks_style(),
                styles.get_default_hand_connections_style(),
            )


class SignLanguageWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Sign Language Translation")
        self.resize(1360, 820)

        self.translator = GestureTranslator()
        self.worker: Optional[CameraWorker] = None
        self.thread: Optional[QThread] = None
        self.last_prediction: Optional[Prediction] = None

        self.video_label = VideoLabel("카메라 대기 중")
        self.video_label.setObjectName("videoView")

        self.status_badge = QLabel("MODEL OFFLINE")
        self.status_badge.setObjectName("statusBadge")

        self.fps_label = QLabel("FPS 0.0")
        self.prediction_label = QLabel("대기 중")
        self.prediction_label.setObjectName("predictionLabel")
        self.confidence_label = QLabel("-")
        self.confidence_label.setObjectName("confidenceLabel")
        self.translation_text = QTextEdit()
        self.translation_text.setReadOnly(True)
        self.translation_text.setPlaceholderText("번역 문장이 여기에 표시됩니다.")

        self.history_list = QListWidget()
        self.history_list.setMinimumHeight(180)

        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumHeight(38)
        self.refresh_camera_button = QPushButton("새로고침")
        self.refresh_camera_button.setMinimumHeight(38)
        self.refresh_cameras()

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.landmarks_button = QPushButton("Landmarks OFF")
        self.landmarks_button.setCheckable(True)

        self._build_layout()
        self._connect_signals()
        self._apply_style()

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._refresh_prediction_panel)
        self.refresh_timer.start(250)

        self.setStatusBar(QStatusBar())
        if self.translator.ready:
            self.status_badge.setText("MODEL READY")
            self.status_badge.setStyleSheet("background: #173b2f; color: #91f2c9;")
            self.statusBar().showMessage("모델 경로가 코드에 설정되어 있습니다.")
        else:
            self.statusBar().showMessage("GUI 준비 완료. 모델 없이도 카메라 화면 테스트가 가능합니다.")

    def _build_layout(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        main_layout = QGridLayout(root)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setHorizontalSpacing(18)
        main_layout.setVerticalSpacing(14)

        header = QHBoxLayout()
        title = QLabel("Sign Language Translation")
        title.setObjectName("title")
        header.addWidget(title)
        header.addStretch()
        header.addWidget(self.status_badge)

        video_panel = QFrame()
        video_panel.setObjectName("panel")
        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(12, 12, 12, 12)
        video_layout.addWidget(self.video_label)

        recognition_group = QGroupBox("Recognition")
        recognition_group.setObjectName("recognitionGroup")
        recognition_layout = QVBoxLayout(recognition_group)
        recognition_layout.addWidget(QLabel("Current Sign"))
        recognition_layout.addWidget(self.prediction_label)
        recognition_layout.addWidget(QLabel("Confidence"))
        recognition_layout.addWidget(self.confidence_label)
        recognition_layout.addWidget(QLabel("Translated Sentence"))
        recognition_layout.addWidget(self.translation_text, 1)

        control_group = QGroupBox("Control")
        control_layout = QGridLayout(control_group)
        control_layout.addWidget(QLabel("Camera"), 0, 0)
        control_layout.addWidget(self.camera_combo, 0, 1)
        control_layout.addWidget(self.refresh_camera_button, 0, 2)
        control_layout.addWidget(self.landmarks_button, 1, 1, 1, 2)
        control_layout.addWidget(self.start_button, 2, 1)
        control_layout.addWidget(self.stop_button, 2, 2)

        history_group = QGroupBox("History")
        history_layout = QVBoxLayout(history_group)
        history_layout.addWidget(self.fps_label)
        history_layout.addWidget(self.history_list)

        side_panel = QFrame()
        side_panel.setObjectName("sidePanel")
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(14, 14, 14, 14)
        side_layout.setSpacing(14)
        side_layout.addWidget(recognition_group, 3)
        side_layout.addWidget(control_group)
        side_layout.addWidget(history_group, 2)

        main_layout.addLayout(header, 0, 0, 1, 2)
        main_layout.addWidget(video_panel, 1, 0)
        main_layout.addWidget(side_panel, 1, 1)
        main_layout.setColumnStretch(0, 3)
        main_layout.setColumnStretch(1, 1)

    def _connect_signals(self) -> None:
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.refresh_camera_button.clicked.connect(self.refresh_cameras)
        self.landmarks_button.toggled.connect(self.toggle_landmarks_label)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #111318;
                color: #eef1f6;
                font-family: "Malgun Gothic";
                font-size: 14px;
            }
            #title {
                font-size: 28px;
                font-weight: 800;
            }
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
            #statusBadge {
                background: #4a1e28;
                color: #ffb8c4;
                border-radius: 6px;
                padding: 7px 12px;
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
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3f82ff, stop:1 #255fcf);
                border: 1px solid #5a92ff;
                border-bottom: 3px solid #173f91;
                border-radius: 6px;
                padding: 10px 14px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
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
            QComboBox, QTextEdit, QListWidget {
                background: #0d1016;
                border: 1px solid #2a303a;
                border-radius: 6px;
                padding: 8px;
                selection-background-color: #2f6fed;
            }
            QTextEdit {
                font-size: 24px;
                font-weight: 700;
            }
            #recognitionGroup {
                border-color: #3e7c65;
            }
            #predictionLabel {
                color: #65d6ad;
                font-size: 38px;
                font-weight: 800;
                padding: 8px 0;
            }
            #confidenceLabel {
                color: #d9dee8;
                font-size: 22px;
                font-weight: 700;
            }
            """
        )

    @Slot()
    def toggle_landmarks_label(self) -> None:
        text = "Landmarks ON" if self.landmarks_button.isChecked() else "Landmarks OFF"
        self.landmarks_button.setText(text)

    @Slot()
    def refresh_cameras(self) -> None:
        current = self.camera_combo.currentData()
        self.camera_combo.clear()

        # Keep this lightweight. Some Windows camera backends block while probing.
        found = list(range(5))

        for index in found:
            self.camera_combo.addItem(f"Camera {index}", index)

        if current in found:
            self.camera_combo.setCurrentIndex(found.index(current))

    @Slot()
    def start_camera(self) -> None:
        if self.thread is not None:
            return

        camera_index = self.camera_combo.currentData()
        if camera_index is None:
            camera_index = 0

        self.worker = CameraWorker(camera_index, self.translator, self.landmarks_button.isChecked())
        self.thread = QThread(self)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.error.connect(self.show_error)
        self.worker.status.connect(self.show_status)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self._camera_stopped)

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.camera_combo.setEnabled(False)
        self.refresh_camera_button.setEnabled(False)
        self.landmarks_button.setEnabled(False)
        self.video_label.clear()
        self.video_label.setText("카메라 연결 중...")
        self.thread.start()
        self.statusBar().showMessage("카메라 실행 중")

    @Slot()
    def stop_camera(self) -> None:
        if self.worker is not None:
            self.worker.stop()
        self.stop_button.setEnabled(False)

    @Slot(QImage, float)
    def update_frame(self, image: QImage, fps: float) -> None:
        pixmap = QPixmap.fromImage(image)
        self.video_label.set_video_pixmap(pixmap)
        self.fps_label.setText(f"FPS {fps:.1f}")

    @Slot(str)
    def show_error(self, message: str) -> None:
        self.statusBar().showMessage(message)
        self.video_label.clear()
        self.video_label.setText(message)

    @Slot(str)
    def show_status(self, message: str) -> None:
        self.statusBar().showMessage(message)

    @Slot()
    def _camera_stopped(self) -> None:
        self.worker = None
        self.thread = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.camera_combo.setEnabled(True)
        self.refresh_camera_button.setEnabled(True)
        self.landmarks_button.setEnabled(True)
        self.statusBar().showMessage("카메라 정지")

    def _refresh_prediction_panel(self) -> None:
        if self.worker is None:
            return

        prediction = self.worker.latest_prediction
        if prediction is None or prediction == self.last_prediction:
            return

        self.last_prediction = prediction
        self.prediction_label.setText(prediction.label)
        self.confidence_label.setText(f"{prediction.confidence:.2f}")
        self.translation_text.setPlainText(prediction.translation)

        item = QListWidgetItem(
            f"{time.strftime('%H:%M:%S')}  {prediction.label}  "
            f"{prediction.confidence:.2f}  {prediction.translation}"
        )
        self.history_list.insertItem(0, item)
        while self.history_list.count() > 20:
            self.history_list.takeItem(self.history_list.count() - 1)

    def closeEvent(self, event) -> None:
        self.stop_camera()
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(1500)
        event.accept()


def open_camera(index: int):
    for backend in CAMERA_BACKENDS:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture()


def main() -> int:
    app = QApplication(sys.argv)
    window = SignLanguageWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
