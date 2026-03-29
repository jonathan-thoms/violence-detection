"""
Transparent Windows Overlay Module
Directive: directives/05_transparent_overlay.md

A full-screen PyQt6 window that is completely transparent and click-through.
It receives raw frames and bounding boxes from a worker thread via signals,
extracts the ROI, blurs it natively, and paints it onto the screen.

CRITICAL:
- Main thread MUST run the PyQt6 QApplication event loop.
- Background thread handles mss (screen capture) and YOLO inference.
- Uses WDA_EXCLUDEFROMCAPTURE so mss never sees the overlay's own blur.
"""

import ctypes
import os
import sys
import time

import torch

# ---- Fix for Windows DLL load order crash: ----
# PyTorch (via violence_filter) must be imported BEFORE PyQt6
# otherwise c10.dll fails to load.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from screen_capture import ScreenCapturer
from violence_filter import ViolenceFilter

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor
from PyQt6.QtWidgets import QApplication, QMainWindow


# ---- FIX WINDOWS DPI SCALING OFFSET ----
# This ensures PyQt6 uses physical pixels, matching the coordinates from mss.
# Without this, blur boxes will be offset on displays with >100% scaling.
import ctypes
import os

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"

try:
    # 2 = PROCESS_PER_MONITOR_DPI_AWARE
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except AttributeError:
    try:
         ctypes.windll.user32.SetProcessDPIAware()
    except AttributeError:
         pass


class CaptureWorker(QThread):
    """
    Background worker thread doing the heavy lifting:
    Screen capture + YOLO inference.
    Emits (frame, boxes) to the GUI thread for drawing.
    """
    data_ready = pyqtSignal(object, list)

    def __init__(self, model_path: str = "yolov8n.pt"):
        super().__init__()
        self.running = True
        self.model_path = model_path

    def _merge_boxes(self, boxes):
        """Merge overlapping boxes to prevent flicker from double-detections."""
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
        print("[CaptureWorker] Initialising ScreenCapturer and ViolenceFilter...")
        capturer = ScreenCapturer(monitor_index=1)
        violence_filter = ViolenceFilter(self.model_path)
        
        print("[CaptureWorker] Loop started.")
        prev_time = time.time()
        frame_count = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[CaptureWorker] Using device: {device}")

        # Warm-up: first inference is slow due to model setup/fusion
        warmup_frame = capturer.get_frame()
        violence_filter.model(warmup_frame, device=device, imgsz=640, verbose=False)
        print("[CaptureWorker] Warm-up inference done.")
        
        # Dynamically determine which class IDs to blur
        target_classes = []
        blur_keywords = ['violence', 'knife', 'gun', 'weapon', 'pistol', 'rifle', 'person', 'cell phone']
        for cls_id, cls_name in violence_filter.model.names.items():
            name_lower = cls_name.lower()
            if "non" in name_lower or "safe" in name_lower:
                continue
            if any(kw in name_lower for kw in blur_keywords):
                target_classes.append(cls_id)
                
        if target_classes:
            names = [violence_filter.model.names[i] for i in target_classes]
            print(f"[CaptureWorker] Targeting classes: {names} (IDs: {target_classes})")
        else:
            print("[CaptureWorker] WARNING: No target classes found. Blurring ALL detections.")
        
        while self.running:
            frame = capturer.get_frame()
            
            results = violence_filter.model(frame, device=device, imgsz=640, verbose=False)
            
            frame_count += 1
            curr_time = time.time()
            if curr_time - prev_time >= 1.0:
                fps = frame_count / (curr_time - prev_time)
                print(f"[Worker] Capture FPS: {fps:.1f}")
                frame_count = 0
                prev_time = curr_time
            
            raw_boxes = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    if conf < 0.15: # Lowered from 0.30 to catch everything
                        continue
                        
                    if target_classes and cls_id not in target_classes:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0]
                    raw_boxes.append([int(x1), int(y1), int(x2), int(y2)])
            
            merged_boxes = self._merge_boxes(raw_boxes)
            self.data_ready.emit(frame, merged_boxes)

    def stop(self):
        self.running = False
        self.wait()


# Windows constant: exclude this window from all screen captures
WDA_EXCLUDEFROMCAPTURE = 0x00000011


class ScreenOverlay(QMainWindow):
    """
    Windows click-through transparent overlay.

    FLICKER PREVENTION:
    1. SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE) — the overlay is
       INVISIBLE to mss/BitBlt/PrintWindow. YOLO always sees the raw screen
       content, never its own blur. This breaks the detection feedback loop.
    2. WA_NoSystemBackground — prevents Qt/DWM from clearing the window
       to transparent between paint events.
    3. Persistent QPixmap buffer — allocated once, reused every frame.
    """
    def __init__(self):
        super().__init__()
        
        screen = QApplication.primaryScreen()
        geometry = screen.geometry()
        self._width = geometry.width()
        self._height = geometry.height()
        self.setGeometry(geometry)
        
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.WindowTransparentForInput
        )
        
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        
        # Persistent full-screen buffer (allocated once, reused every frame)
        self._buffer = QPixmap(self._width, self._height)
        self._buffer.fill(QColor(0, 0, 0, 0))
        
        # Temporal smoothing state
        self.current_boxes = []
        self.frames_since_last_detection = 0
        
        # FPS tracking
        self._ui_frame_count = 0
        self._ui_prev_time = time.time()

    def showEvent(self, event):
        """
        After the window is shown and has a valid HWND, call
        SetWindowDisplayAffinity to exclude this overlay from all
        screen captures (mss, BitBlt, PrintWindow, etc.).
        This is the KEY fix: YOLO never sees its own blur output.
        """
        super().showEvent(event)
        hwnd = int(self.winId())
        result = ctypes.windll.user32.SetWindowDisplayAffinity(
            hwnd, WDA_EXCLUDEFROMCAPTURE
        )
        if result:
            print("[Overlay] SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE) OK — overlay hidden from screen capture.")
        else:
            print("[Overlay] WARNING: SetWindowDisplayAffinity failed. Flicker may occur.")

    def update_data(self, frame: np.ndarray, boxes: list):
        # ---- Temporal Smoothing ----
        if boxes:
            if not self.current_boxes or len(boxes) != len(self.current_boxes):
                self.current_boxes = boxes
            else:
                ALPHA = 0.4
                boxes.sort(key=lambda b: b[0])
                self.current_boxes.sort(key=lambda b: b[0])
                
                smoothed = []
                for (nx1, ny1, nx2, ny2), (cx1, cy1, cx2, cy2) in zip(boxes, self.current_boxes):
                    sx1 = int(cx1 + ALPHA * (nx1 - cx1))
                    sy1 = int(cy1 + ALPHA * (ny1 - cy1))
                    sx2 = int(cx2 + ALPHA * (nx2 - cx2))
                    sy2 = int(cy2 + ALPHA * (ny2 - cy2))
                    smoothed.append([sx1, sy1, sx2, sy2])
                self.current_boxes = smoothed

            self.frames_since_last_detection = 0
        else:
            self.frames_since_last_detection += 1
            if self.frames_since_last_detection > 15:  # ~0.5s grace period
                self.current_boxes = []

        # ---- Render blur into persistent buffer ----
        if frame is not None and self.current_boxes:
            h, w = frame.shape[:2]
            
            self._buffer.fill(QColor(0, 0, 0, 0))
            painter = QPainter(self._buffer)
            
            for (x1, y1, x2, y2) in self.current_boxes:
                pad = 25
                x1c, y1c = max(0, x1 - pad), max(0, y1 - pad)
                x2c, y2c = min(w, x2 + pad), min(h, y2 + pad)
                
                if x2c <= x1c or y2c <= y1c:
                    continue

                roi = frame[y1c:y2c, x1c:x2c]
                roi_h, roi_w = roi.shape[:2]
                
                if roi_h == 0 or roi_w == 0:
                    continue
                
                # Fast downsampled blur
                scale = 4
                sw, sh = max(1, roi_w // scale), max(1, roi_h // scale)
                small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_LINEAR)
                blurred = cv2.blur(small, (25, 25))
                blurred_roi = cv2.resize(blurred, (roi_w, roi_h), interpolation=cv2.INTER_LINEAR)
                
                bytes_per_line = 3 * roi_w
                q_img = QImage(
                    blurred_roi.data,
                    roi_w, roi_h,
                    bytes_per_line,
                    QImage.Format.Format_BGR888
                ).copy()
                
                painter.drawImage(x1c, y1c, q_img)
            
            painter.end()
        elif not self.current_boxes:
            self._buffer.fill(QColor(0, 0, 0, 0))
        
        self.update()

    def paintEvent(self, event):
        self._ui_frame_count += 1
        curr_time = time.time()
        if (curr_time - self._ui_prev_time) >= 1.0:
            fps = self._ui_frame_count / (curr_time - self._ui_prev_time)
            print(f"[UI] Paint FPS: {fps:.1f}")
            self._ui_frame_count = 0
            self._ui_prev_time = curr_time
        
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._buffer)
        painter.end()


def main():
    print("==================================================")
    print("  PyQt6 Transparent Violence Overlay")
    print("==================================================")
    print("WARNING: This is a click-through overlay. To exit,")
    print("         you MUST use the console running this script,")
    print("         and press Ctrl+C.")
    print("==================================================")
    
    app = QApplication(sys.argv)
    
    overlay = ScreenOverlay()
    overlay.show()
    
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(_PROJECT_ROOT, ".env"))
    model_path = os.getenv("YOLO_MODEL_PATH", "models/violence_best.pt")
    if not os.path.isabs(model_path):
        model_path = os.path.join(_PROJECT_ROOT, model_path)
    
    worker = CaptureWorker(model_path=model_path)
    worker.data_ready.connect(overlay.update_data)
    worker.start()
    
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("[Main] Exiting...")
    finally:
        worker.stop()


if __name__ == "__main__":
    main()
