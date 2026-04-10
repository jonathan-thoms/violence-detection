"""
Violence Detection System — Control Panel GUI
Directive: directives/07_gui_control_panel.md

Premium dark-themed PyQt6 control center for the violence detection pipeline.
Manages face registration, launches detection modes, and shows live status.

Launch with:  python execution/gui_app.py
"""

import ctypes
import os
import sys
import shutil
import threading
import time

# ---- PyTorch MUST be imported before PyQt6 on Windows (c10.dll fix) ----
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import cv2
import numpy as np
from dotenv import load_dotenv

from face_gatekeeper import FaceGatekeeper
from screen_capture import ScreenCapturer
from violence_filter import ViolenceFilter
from overlay import ScreenOverlay, CaptureWorker as OverlayCaptureWorker

from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize, QPropertyAnimation,
    QEasingCurve, pyqtProperty, QRect
)
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QColor, QFont, QFontDatabase, QIcon,
    QLinearGradient, QBrush, QPen, QPainterPath
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QFileDialog, QScrollArea,
    QFrame, QGraphicsDropShadowEffect, QSystemTrayIcon, QMenu,
    QSizePolicy, QSpacerItem
)

# ---- Windows DPI awareness ----
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except (AttributeError, OSError):
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


# ═══════════════════════════════════════════════════════════════════════════
# COLOUR PALETTE & STYLE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
COLORS = {
    "bg_dark":       "#0a0e1a",
    "bg_card":       "#121829",
    "bg_card_hover": "#1a2240",
    "border":        "#1e2a4a",
    "border_glow":   "#3b5fe0",
    "accent_blue":   "#4a7dff",
    "accent_purple": "#7c4dff",
    "accent_green":  "#00e676",
    "accent_red":    "#ff1744",
    "accent_amber":  "#ffab00",
    "text_primary":  "#e8eaf6",
    "text_secondary":"#7986cb",
    "text_dim":      "#3d4977",
    "btn_start":     "#1b5e20",
    "btn_start_h":   "#2e7d32",
    "btn_violence":  "#4a148c",
    "btn_violence_h":"#6a1b9a",
    "btn_stop":      "#b71c1c",
    "btn_stop_h":    "#d32f2f",
    "btn_add":       "#1a237e",
    "btn_add_h":     "#283593",
}

STYLESHEET = f"""
QMainWindow {{
    background-color: {COLORS["bg_dark"]};
}}
QLabel {{
    color: {COLORS["text_primary"]};
    font-family: 'Segoe UI', 'Inter', sans-serif;
}}
QPushButton {{
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    padding: 10px 20px;
    color: {COLORS["text_primary"]};
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 600;
}}
QPushButton:hover {{
    border: 1px solid {COLORS["border_glow"]};
}}
QPushButton:disabled {{
    background-color: #1a1f30;
    color: #3d4060;
    border: 1px solid #1e2240;
}}
QScrollArea {{
    background: transparent;
    border: none;
}}
QScrollArea > QWidget > QWidget {{
    background: transparent;
}}
QScrollBar:vertical {{
    background: {COLORS["bg_dark"]};
    width: 6px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {COLORS["text_dim"]};
    border-radius: 3px;
    min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
"""


# ═══════════════════════════════════════════════════════════════════════════
# SHARED STATE (same pattern as main_orchestrator.py)
# ═══════════════════════════════════════════════════════════════════════════
TARGET_PRESENT: bool = False
_shutdown_event = threading.Event()


# ═══════════════════════════════════════════════════════════════════════════
# FACE WORKER (daemon thread — same as main_orchestrator)
# ═══════════════════════════════════════════════════════════════════════════
def face_worker(gatekeeper: FaceGatekeeper, shutdown_event: threading.Event) -> None:
    global TARGET_PRESENT
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FaceWorker] WARNING: Cannot open webcam.")
        return
    print("[FaceWorker] Webcam opened.")
    last_check = 0.0
    while not shutdown_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        interval = 3.0 if TARGET_PRESENT else 0.5
        now = time.time()
        if now - last_check > interval:
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            match, _ = gatekeeper.is_target_present(small)
            TARGET_PRESENT = match
            last_check = now
    cap.release()
    print("[FaceWorker] Thread exiting.")


# ═══════════════════════════════════════════════════════════════════════════
# FULL-SYSTEM CAPTURE WORKER (QThread — YOLO + face gate)
# ═══════════════════════════════════════════════════════════════════════════
class FullSystemWorker(QThread):
    """Screen capture + YOLO inference, gated by TARGET_PRESENT."""
    data_ready = pyqtSignal(object, list)
    fps_report = pyqtSignal(float, str)  # (fps, status_string)

    def __init__(self, model_path: str):
        super().__init__()
        self.running = True
        self.model_path = model_path

    def _merge_boxes(self, boxes):
        if not boxes:
            return []
        clusters = []
        for box in boxes:
            x1, y1, x2, y2 = box
            matched = False
            for cluster in clusters:
                cx1, cy1, cx2, cy2 = cluster
                pad = 30
                if not (x2 < cx1 - pad or x1 > cx2 + pad or y2 < cy1 - pad or y1 > cy2 + pad):
                    cluster[0] = min(x1, cx1)
                    cluster[1] = min(y1, cy1)
                    cluster[2] = max(x2, cx2)
                    cluster[3] = max(y2, cy2)
                    matched = True
                    break
            if not matched:
                clusters.append(box)
        return clusters

    def run(self):
        capturer = ScreenCapturer(monitor_index=1)
        violence_filter = ViolenceFilter(self.model_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        warmup = capturer.get_frame()
        violence_filter.model(warmup, device=device, imgsz=640, verbose=False)

        # Build target class list
        target_classes = []
        blur_keywords = ['violence', 'knife', 'gun', 'weapon', 'pistol', 'rifle']
        for cls_id, cls_name in violence_filter.model.names.items():
            name_lower = cls_name.lower()
            if "non" in name_lower or "safe" in name_lower:
                continue
            if any(kw in name_lower for kw in blur_keywords):
                target_classes.append(cls_id)

        prev_time = time.time()
        frame_count = 0

        while self.running:
            frame = capturer.get_frame()
            if TARGET_PRESENT:
                results = violence_filter.model(frame, device=device, imgsz=640, verbose=False)
                raw_boxes = []
                for result in results:
                    if result.boxes is None:
                        continue
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        if conf < 0.15:
                            continue
                        if target_classes and cls_id not in target_classes:
                            continue
                        x1, y1, x2, y2 = box.xyxy[0]
                        raw_boxes.append([int(x1), int(y1), int(x2), int(y2)])
                merged = self._merge_boxes(raw_boxes)
                self.data_ready.emit(frame, merged)
            else:
                self.data_ready.emit(frame, [])

            frame_count += 1
            now = time.time()
            if now - prev_time >= 1.0:
                fps = frame_count / (now - prev_time)
                status = "ACTIVE" if TARGET_PRESENT else "STANDBY"
                self.fps_report.emit(fps, status)
                frame_count = 0
                prev_time = now

    def stop(self):
        self.running = False
        self.wait(3000)


# ═══════════════════════════════════════════════════════════════════════════
# VIOLENCE-ONLY CAPTURE WORKER (QThread — always active, no face gate)
# ═══════════════════════════════════════════════════════════════════════════
class ViolenceOnlyWorker(QThread):
    """Screen capture + YOLO inference, always active (no face gate)."""
    data_ready = pyqtSignal(object, list)
    fps_report = pyqtSignal(float, str)

    def __init__(self, model_path: str):
        super().__init__()
        self.running = True
        self.model_path = model_path

    def _merge_boxes(self, boxes):
        if not boxes:
            return []
        clusters = []
        for box in boxes:
            x1, y1, x2, y2 = box
            matched = False
            for cluster in clusters:
                cx1, cy1, cx2, cy2 = cluster
                pad = 30
                if not (x2 < cx1 - pad or x1 > cx2 + pad or y2 < cy1 - pad or y1 > cy2 + pad):
                    cluster[0] = min(x1, cx1)
                    cluster[1] = min(y1, cy1)
                    cluster[2] = max(x2, cx2)
                    cluster[3] = max(y2, cy2)
                    matched = True
                    break
            if not matched:
                clusters.append(box)
        return clusters

    def run(self):
        capturer = ScreenCapturer(monitor_index=1)
        violence_filter = ViolenceFilter(self.model_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        warmup = capturer.get_frame()
        violence_filter.model(warmup, device=device, imgsz=640, verbose=False)

        target_classes = []
        blur_keywords = ['violence', 'knife', 'gun', 'weapon', 'pistol', 'rifle']
        for cls_id, cls_name in violence_filter.model.names.items():
            name_lower = cls_name.lower()
            if "non" in name_lower or "safe" in name_lower:
                continue
            if any(kw in name_lower for kw in blur_keywords):
                target_classes.append(cls_id)

        prev_time = time.time()
        frame_count = 0

        while self.running:
            frame = capturer.get_frame()
            results = violence_filter.model(frame, device=device, imgsz=640, verbose=False)
            raw_boxes = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    if conf < 0.15:
                        continue
                    if target_classes and cls_id not in target_classes:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0]
                    raw_boxes.append([int(x1), int(y1), int(x2), int(y2)])
            merged = self._merge_boxes(raw_boxes)
            self.data_ready.emit(frame, merged)

            frame_count += 1
            now = time.time()
            if now - prev_time >= 1.0:
                fps = frame_count / (now - prev_time)
                self.fps_report.emit(fps, "ACTIVE")
                frame_count = 0
                prev_time = now

    def stop(self):
        self.running = False
        self.wait(3000)


# ═══════════════════════════════════════════════════════════════════════════
# FACE CARD WIDGET
# ═══════════════════════════════════════════════════════════════════════════
class FaceCard(QFrame):
    """A thumbnail card for a registered face with a delete button."""
    removed = pyqtSignal(str)  # emits filename

    def __init__(self, filename: str, filepath: str, parent=None):
        super().__init__(parent)
        self.filename = filename
        self.filepath = filepath
        self.setFixedSize(110, 130)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self.setStyleSheet(f"""
            FaceCard {{
                background: {COLORS["bg_card"]};
                border: 1px solid {COLORS["border"]};
                border-radius: 10px;
            }}
            FaceCard:hover {{
                border: 1px solid {COLORS["accent_blue"]};
                background: {COLORS["bg_card_hover"]};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 4)
        layout.setSpacing(4)

        # Thumbnail
        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(96, 80)
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb_label.setStyleSheet("border-radius: 6px; background: #0d1020;")

        pixmap = QPixmap(filepath)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(96, 80, Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
        self.thumb_label.setPixmap(pixmap)
        layout.addWidget(self.thumb_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Bottom row: name + delete
        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)

        name_label = QLabel(filename[:10])
        name_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 9px;")
        name_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        bottom.addWidget(name_label)

        del_btn = QPushButton("✕")
        del_btn.setFixedSize(22, 22)
        del_btn.setStyleSheet(f"""
            QPushButton {{
                background: {COLORS["accent_red"]};
                color: white;
                border-radius: 11px;
                font-size: 11px;
                font-weight: bold;
                border: none;
                padding: 0;
            }}
            QPushButton:hover {{
                background: #ff5252;
            }}
        """)
        del_btn.clicked.connect(lambda: self.removed.emit(self.filename))
        bottom.addWidget(del_btn)

        layout.addLayout(bottom)


class AddFaceCard(QFrame):
    """A '+' card to add a new face image."""
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(110, 130)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(f"""
            AddFaceCard {{
                background: {COLORS["bg_card"]};
                border: 2px dashed {COLORS["text_dim"]};
                border-radius: 10px;
            }}
            AddFaceCard:hover {{
                border: 2px dashed {COLORS["accent_blue"]};
                background: {COLORS["bg_card_hover"]};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        plus = QLabel("+")
        plus.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 36px; font-weight: 300;")
        plus.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(plus)

        txt = QLabel("Add Face")
        txt.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px;")
        txt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(txt)

    def mousePressEvent(self, event):
        self.clicked.emit()


# ═══════════════════════════════════════════════════════════════════════════
# STATUS INDICATOR (pulsing dot)
# ═══════════════════════════════════════════════════════════════════════════
class StatusDot(QWidget):
    """A small pulsing colored dot."""
    def __init__(self, color: str = COLORS["text_dim"], parent=None):
        super().__init__(parent)
        self.setFixedSize(12, 12)
        self._color = QColor(color)

    def set_color(self, color: str):
        self._color = QColor(color)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)

        # Outer glow
        glow = QColor(self._color)
        glow.setAlpha(60)
        painter.setBrush(QBrush(glow))
        painter.drawEllipse(0, 0, 12, 12)

        # Inner core
        painter.setBrush(QBrush(self._color))
        painter.drawEllipse(2, 2, 8, 8)
        painter.end()


# ═══════════════════════════════════════════════════════════════════════════
# GLASS CARD CONTAINER
# ═══════════════════════════════════════════════════════════════════════════
class GlassCard(QFrame):
    """A card with glassmorphism-style background."""
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            GlassCard {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(18, 24, 41, 220),
                    stop:1 rgba(26, 34, 64, 200)
                );
                border: 1px solid {COLORS["border"]};
                border-radius: 14px;
            }}
        """)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(18, 14, 18, 14)
        self._layout.setSpacing(10)

        if title:
            lbl = QLabel(title)
            lbl.setStyleSheet(f"""
                color: {COLORS["text_secondary"]};
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 2px;
            """)
            self._layout.addWidget(lbl)

    def content_layout(self):
        return self._layout


# ═══════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════
class ControlPanel(QMainWindow):
    def __init__(self):
        super().__init__()

        load_dotenv(dotenv_path=os.path.join(_PROJECT_ROOT, ".env"))

        self.faces_dir = os.path.join(_PROJECT_ROOT, "data", "faces")
        os.makedirs(self.faces_dir, exist_ok=True)

        self.model_path = os.getenv("YOLO_MODEL_PATH", "models/violence_best.pt")
        if not os.path.isabs(self.model_path):
            self.model_path = os.path.join(_PROJECT_ROOT, self.model_path)

        # State
        self._gatekeeper = None
        self._face_thread = None
        self._worker = None
        self._overlay = None
        self._mode = "idle"  # idle | full | violence_only

        self._setup_ui()
        self._setup_tray()
        self._refresh_faces()

    # ───────────────────────────────────────────────────────────────────
    # UI SETUP
    # ───────────────────────────────────────────────────────────────────
    def _setup_ui(self):
        self.setWindowTitle("Violence Detection System")
        self.setFixedSize(480, 680)
        self.setStyleSheet(STYLESHEET)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 16, 20, 20)
        root.setSpacing(16)

        # ── HEADER ──
        header = QHBoxLayout()
        header.setSpacing(10)

        shield = QLabel("🛡️")
        shield.setStyleSheet("font-size: 28px;")
        header.addWidget(shield)

        title = QLabel("VIOLENCE DETECTION")
        title.setStyleSheet(f"""
            color: {COLORS["text_primary"]};
            font-size: 20px;
            font-weight: 700;
            letter-spacing: 3px;
        """)
        header.addWidget(title)
        header.addStretch()

        # Minimize-to-tray hint
        tray_hint = QLabel("🔽 Minimizes to tray")
        tray_hint.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
        header.addWidget(tray_hint)

        root.addLayout(header)

        # Accent line
        accent_line = QFrame()
        accent_line.setFixedHeight(2)
        accent_line.setStyleSheet(f"""
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 {COLORS["accent_blue"]},
                stop:0.5 {COLORS["accent_purple"]},
                stop:1 transparent
            );
        """)
        root.addWidget(accent_line)

        # ── FACE GALLERY CARD ──
        self.face_card = GlassCard("REGISTERED FACES")
        face_layout = self.face_card.content_layout()

        self.face_scroll = QScrollArea()
        self.face_scroll.setWidgetResizable(True)
        self.face_scroll.setFixedHeight(160)
        self.face_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.face_container = QWidget()
        self.face_grid = QHBoxLayout(self.face_container)
        self.face_grid.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.face_grid.setContentsMargins(0, 0, 0, 0)
        self.face_grid.setSpacing(10)

        self.face_scroll.setWidget(self.face_container)
        face_layout.addWidget(self.face_scroll)

        root.addWidget(self.face_card)

        # ── DETECTION CONTROLS CARD ──
        controls_card = GlassCard("DETECTION CONTROLS")
        ctrl = controls_card.content_layout()

        # Start Full System
        self.btn_full = QPushButton("▶   Start Full System")
        self.btn_full.setMinimumHeight(48)
        self.btn_full.setStyleSheet(
            "background-color: #1b5e20; color: #ffffff; font-size: 14px;"
            "font-weight: 600; padding: 12px 20px; border: 1px solid #2e7d32;"
            "border-radius: 8px;"
        )
        self.btn_full.clicked.connect(self._start_full_system)
        ctrl.addWidget(self.btn_full)

        full_desc = QLabel("Face Detection + Violence Blur")
        full_desc.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px; margin-left: 10px;")
        ctrl.addWidget(full_desc)

        ctrl.addSpacing(6)

        # Violence Only
        self.btn_violence = QPushButton("▶   Violence Only")
        self.btn_violence.setMinimumHeight(48)
        self.btn_violence.setStyleSheet(
            "background-color: #4a148c; color: #ffffff; font-size: 14px;"
            "font-weight: 600; padding: 12px 20px; border: 1px solid #6a1b9a;"
            "border-radius: 8px;"
        )
        self.btn_violence.clicked.connect(self._start_violence_only)
        ctrl.addWidget(self.btn_violence)

        violence_desc = QLabel("Screen Blur without Face Gate")
        violence_desc.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 11px; margin-left: 10px;")
        ctrl.addWidget(violence_desc)

        ctrl.addSpacing(6)

        # Stop
        self.btn_stop = QPushButton("■   Stop System")
        self.btn_stop.setMinimumHeight(48)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(
            "background-color: #1a1f30; color: #3d4060; font-size: 14px;"
            "font-weight: 600; padding: 12px 20px; border: 1px solid #1e2240;"
            "border-radius: 8px;"
        )
        self.btn_stop.clicked.connect(self._stop_system)
        ctrl.addWidget(self.btn_stop)

        root.addWidget(controls_card)

        # ── STATUS CARD ──
        status_card = GlassCard("STATUS")
        status_layout = status_card.content_layout()

        # Status rows
        self._status_rows = {}
        for key, label_text, default_val in [
            ("system",  "System",           "Idle"),
            ("mode",    "Mode",             "—"),
            ("fps",     "FPS",              "—"),
            ("faces",   "Faces Registered", "0"),
        ]:
            row = QHBoxLayout()
            row.setSpacing(8)

            dot = StatusDot(COLORS["text_dim"])
            row.addWidget(dot)

            lbl = QLabel(label_text)
            lbl.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
            row.addWidget(lbl)

            row.addStretch()

            val = QLabel(default_val)
            val.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 12px; font-weight: 600;")
            row.addWidget(val)

            status_layout.addLayout(row)
            self._status_rows[key] = (dot, val)

        root.addWidget(status_card)
        root.addStretch()

        # Footer
        footer = QLabel("Built with YOLOv8 + PyQt6  •  GPU Accelerated")
        footer.setStyleSheet(f"color: {COLORS['text_dim']}; font-size: 10px;")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(footer)

    # ───────────────────────────────────────────────────────────────────
    # SYSTEM TRAY
    # ───────────────────────────────────────────────────────────────────
    def _setup_tray(self):
        self.tray_icon = QSystemTrayIcon(self)
        # Use a built-in icon - create a small shield pixmap
        pix = QPixmap(32, 32)
        pix.fill(QColor(0, 0, 0, 0))
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QBrush(QColor(COLORS["accent_blue"])))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(2, 2, 28, 28)
        p.setBrush(QBrush(QColor(COLORS["bg_dark"])))
        p.drawEllipse(6, 6, 20, 20)
        p.setBrush(QBrush(QColor(COLORS["accent_green"])))
        p.drawEllipse(10, 10, 12, 12)
        p.end()

        self.tray_icon.setIcon(QIcon(pix))
        self.tray_icon.setToolTip("Violence Detection System")

        tray_menu = QMenu()
        tray_menu.setStyleSheet(f"""
            QMenu {{
                background: {COLORS["bg_card"]};
                color: {COLORS["text_primary"]};
                border: 1px solid {COLORS["border"]};
                padding: 4px;
            }}
            QMenu::item:selected {{
                background: {COLORS["bg_card_hover"]};
            }}
        """)

        show_action = tray_menu.addAction("Show Control Panel")
        show_action.triggered.connect(self._restore_from_tray)

        stop_action = tray_menu.addAction("Stop System")
        stop_action.triggered.connect(self._stop_system)

        tray_menu.addSeparator()
        quit_action = tray_menu.addAction("Quit")
        quit_action.triggered.connect(self._quit_app)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self._tray_activated)

    def _tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._restore_from_tray()

    def _minimize_to_tray(self):
        """Minimize to taskbar + show tray icon so user can always find us."""
        self.tray_icon.show()
        self.showMinimized()
        self.tray_icon.showMessage(
            "Violence Detection",
            "System running. Click taskbar icon or tray to restore.",
            QSystemTrayIcon.MessageIcon.Information,
            2000
        )

    def _restore_from_tray(self):
        self.showNormal()
        self.activateWindow()
        self.raise_()

    def _quit_app(self):
        self._stop_system()
        self.tray_icon.hide()
        QApplication.quit()

    # ───────────────────────────────────────────────────────────────────
    # FACE GALLERY
    # ───────────────────────────────────────────────────────────────────
    def _refresh_faces(self):
        """Rebuild the face gallery from the data/faces/ directory."""
        # Clear existing
        while self.face_grid.count():
            item = self.face_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add existing faces
        extensions = ("*.jpg", "*.jpeg", "*.png")
        import glob
        face_files = []
        for ext in extensions:
            face_files.extend(glob.glob(os.path.join(self.faces_dir, ext)))
        face_files.sort()

        for filepath in face_files:
            fname = os.path.basename(filepath)
            card = FaceCard(fname, filepath)
            card.removed.connect(self._remove_face)
            self.face_grid.addWidget(card)

        # Add the "+" card
        add_card = AddFaceCard()
        add_card.clicked.connect(self._add_face)
        self.face_grid.addWidget(add_card)

        # Update status
        count = len(face_files)
        dot, val = self._status_rows["faces"]
        val.setText(str(count))
        dot.set_color(COLORS["accent_green"] if count > 0 else COLORS["text_dim"])

    def _add_face(self):
        """Open file picker to add a face image."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Face Image(s)", "",
            "Images (*.jpg *.jpeg *.png)"
        )
        if not paths:
            return

        for path in paths:
            fname = os.path.basename(path)
            dest = os.path.join(self.faces_dir, fname)

            base, ext = os.path.splitext(fname)
            counter = 1
            while os.path.exists(dest):
                dest = os.path.join(self.faces_dir, f"{base}_{counter}{ext}")
                counter += 1

            shutil.copy2(path, dest)

        self._refresh_faces()

    def _remove_face(self, filename: str):
        """Remove a face image file and refresh."""
        filepath = os.path.join(self.faces_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        self._refresh_faces()

    # ───────────────────────────────────────────────────────────────────
    # LAUNCH: FULL SYSTEM (face + violence)
    # ───────────────────────────────────────────────────────────────────
    def _start_full_system(self):
        global TARGET_PRESENT, _shutdown_event

        # Check for faces
        import glob
        face_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            face_files.extend(glob.glob(os.path.join(self.faces_dir, ext)))

        if not face_files:
            self._set_status("system", "Error: No faces registered", COLORS["accent_red"])
            return

        self._mode = "full"
        _shutdown_event = threading.Event()
        TARGET_PRESENT = False

        # Face gatekeeper
        self._gatekeeper = FaceGatekeeper(faces_dir=self.faces_dir)
        self._face_thread = threading.Thread(
            target=face_worker,
            args=(self._gatekeeper, _shutdown_event),
            daemon=True,
            name="FaceGatekeeper"
        )
        self._face_thread.start()

        # Overlay
        self._overlay = ScreenOverlay()
        self._overlay.show()

        # Worker
        self._worker = FullSystemWorker(self.model_path)
        self._worker.data_ready.connect(self._overlay.update_data)
        self._worker.fps_report.connect(self._on_fps_report)
        self._worker.start()

        self._set_status("system", "Running", COLORS["accent_green"])
        self._set_status("mode", "Full System", COLORS["accent_blue"])

        self._update_button_styles(running=True)

        self._minimize_to_tray()

    # ───────────────────────────────────────────────────────────────────
    # LAUNCH: VIOLENCE ONLY
    # ───────────────────────────────────────────────────────────────────
    def _start_violence_only(self):
        self._mode = "violence_only"

        # Overlay
        self._overlay = ScreenOverlay()
        self._overlay.show()

        # Worker
        self._worker = ViolenceOnlyWorker(self.model_path)
        self._worker.data_ready.connect(self._overlay.update_data)
        self._worker.fps_report.connect(self._on_fps_report)
        self._worker.start()

        self._set_status("system", "Running", COLORS["accent_green"])
        self._set_status("mode", "Violence Only", COLORS["accent_purple"])

        self._update_button_styles(running=True)

        self._minimize_to_tray()

    # ───────────────────────────────────────────────────────────────────
    # STOP
    # ───────────────────────────────────────────────────────────────────
    def _stop_system(self):
        global _shutdown_event

        if self._worker:
            self._worker.stop()
            self._worker = None

        if self._face_thread:
            _shutdown_event.set()
            self._face_thread.join(timeout=2.0)
            self._face_thread = None

        if self._overlay:
            self._overlay.close()
            self._overlay = None

        self._mode = "idle"

        self._set_status("system", "Idle", COLORS["text_dim"])
        self._set_status("mode", "—", COLORS["text_dim"])
        self._set_status("fps", "—", COLORS["text_dim"])

        self._update_button_styles(running=False)

        if not self.isVisible() or self.isMinimized():
            self._restore_from_tray()

    # ───────────────────────────────────────────────────────────────────
    # STATUS HELPERS
    # ───────────────────────────────────────────────────────────────────
    def _set_status(self, key: str, text: str, color: str = None):
        dot, val = self._status_rows[key]
        val.setText(text)
        if color:
            dot.set_color(color)

    def _on_fps_report(self, fps: float, status: str):
        color = COLORS["accent_green"] if status == "ACTIVE" else COLORS["accent_amber"]
        self._set_status("fps", f"{fps:.1f}", color)

    def _update_button_styles(self, running: bool):
        """Swap button styles based on whether the system is running."""
        if running:
            self.btn_full.setEnabled(False)
            self.btn_violence.setEnabled(False)
            self.btn_stop.setEnabled(True)
            # Dim the start buttons
            dim = (
                "background-color: #1a1f30; color: #3d4060; font-size: 14px;"
                "font-weight: 600; padding: 12px 20px; border: 1px solid #1e2240;"
                "border-radius: 8px;"
            )
            self.btn_full.setStyleSheet(dim)
            self.btn_violence.setStyleSheet(dim)
            # Light up the stop button
            self.btn_stop.setStyleSheet(
                "background-color: #b71c1c; color: #ffffff; font-size: 14px;"
                "font-weight: 600; padding: 12px 20px; border: 1px solid #d32f2f;"
                "border-radius: 8px;"
            )
        else:
            self.btn_full.setEnabled(True)
            self.btn_violence.setEnabled(True)
            self.btn_stop.setEnabled(False)
            # Light up start buttons
            self.btn_full.setStyleSheet(
                "background-color: #1b5e20; color: #ffffff; font-size: 14px;"
                "font-weight: 600; padding: 12px 20px; border: 1px solid #2e7d32;"
                "border-radius: 8px;"
            )
            self.btn_violence.setStyleSheet(
                "background-color: #4a148c; color: #ffffff; font-size: 14px;"
                "font-weight: 600; padding: 12px 20px; border: 1px solid #6a1b9a;"
                "border-radius: 8px;"
            )
            # Dim the stop button
            dim = (
                "background-color: #1a1f30; color: #3d4060; font-size: 14px;"
                "font-weight: 600; padding: 12px 20px; border: 1px solid #1e2240;"
                "border-radius: 8px;"
            )
            self.btn_stop.setStyleSheet(dim)

    # ───────────────────────────────────────────────────────────────────
    # WINDOW EVENTS
    # ───────────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        if self._mode != "idle":
            # Minimize to tray instead of closing
            event.ignore()
            self._minimize_to_tray()
        else:
            self._stop_system()
            self.tray_icon.hide()
            event.accept()

    def paintEvent(self, event):
        """Draw gradient background."""
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor(COLORS["bg_dark"]))
        gradient.setColorAt(1, QColor("#060a14"))
        painter.fillRect(self.rect(), gradient)
        painter.end()
        super().paintEvent(event)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # Keep running in tray

    window = ControlPanel()
    window.show()

    try:
        sys.exit(app.exec())
    except (KeyboardInterrupt, SystemExit):
        window._stop_system()
        window.tray_icon.hide()


if __name__ == "__main__":
    main()
