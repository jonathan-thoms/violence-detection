"""
Main Orchestrator — Integration Entry Point
Directive: directives/04_main_orchestrator.md, directives/06_upgrade_violence_model.md

Integrates FaceGatekeeper (daemon thread), ScreenCapturer + ViolenceFilter
(QThread worker), and PyQt6 ScreenOverlay (main thread) into a single
real-time pipeline.

CONCURRENCY MODEL (Directive 06 — PyQt6 Overlay)
─────────────────
┌──────────────────────────────────────┐
│  DAEMON THREAD (face_worker)         │
│  - cv2.VideoCapture(0) webcam loop   │
│  - face_gatekeeper.is_target_present │
│  - Updates TARGET_PRESENT (bool)     │
│  - Resize to 25% for speed           │
└──────────────────────────────────────┘
         ↓  reads shared bool
┌──────────────────────────────────────┐
│  QThread (CaptureWorker)             │
│  - mss screen capture                │
│  - if TARGET_PRESENT → YOLO infer    │
│  - Emits pyqtSignal(frame, boxes)    │
└──────────────────────────────────────┘
         ↓  signal → slot
┌──────────────────────────────────────┐
│  MAIN THREAD (PyQt6 event loop)      │
│  - QApplication.exec()               │
│  - ScreenOverlay.update_data(f, b)   │
│  - Double-buffered paintEvent        │
└──────────────────────────────────────┘

CRITICAL: app.exec() MUST run in the main thread.
"""

import os
import sys
import threading
import time

import torch

# ---- Fix for Windows DLL load order crash: ----
# PyTorch must be imported BEFORE PyQt6 otherwise c10.dll fails to load.
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
from overlay import ScreenOverlay

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication


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


# ---------------------------------------------------------------------------
# Shared state (thread-safe for a single-writer / single-reader bool)
# ---------------------------------------------------------------------------
TARGET_PRESENT: bool = False
_shutdown_event = threading.Event()


# ---------------------------------------------------------------------------
# Background thread: Face Gatekeeper worker (unchanged — daemon thread)
# ---------------------------------------------------------------------------
def face_worker(gatekeeper: FaceGatekeeper) -> None:
    """
    Continuously grab webcam frames, run face matching, and update
    the global TARGET_PRESENT flag.
    """
    global TARGET_PRESENT

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FaceWorker] WARNING: Cannot open webcam. "
              "TARGET_PRESENT will stay False.")
        return

    print("[FaceWorker] Webcam opened in background thread.")

    while not _shutdown_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        match, _ = gatekeeper.is_target_present(small)
        TARGET_PRESENT = match

    cap.release()
    print("[FaceWorker] Webcam released. Thread exiting.")


# ---------------------------------------------------------------------------
# QThread: Screen capture + YOLO inference worker
# ---------------------------------------------------------------------------
class CaptureWorker(QThread):
    """
    Background QThread: screen capture + conditional YOLO inference.
    Emits (frame, boxes) via pyqtSignal to the ScreenOverlay on the
    main thread.
    """
    data_ready = pyqtSignal(object, list)

    def __init__(self, model_path: str, confidence: float = 0.45):
        super().__init__()
        self.running = True
        self.model_path = model_path
        self.confidence = confidence

    def _merge_boxes(self, boxes):
        """Merge overlapping boxes to reduce flickering."""
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

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[CaptureWorker] Using device: {device}")
        if device == 'cpu':
            torch.set_num_threads(max(1, os.cpu_count() // 2))

        # Warm-up: first inference is slow due to model setup/fusion
        warmup_frame = capturer.get_frame()
        violence_filter.model(warmup_frame, device=device, imgsz=640, verbose=False)
        print("[CaptureWorker] Warm-up inference done.")

        # Dynamically determine which class IDs to blur based on model's names
        target_classes = []
        # Added 'person' and 'cell phone' just so you can test the blurring works immediately!
        blur_keywords = ['violence', 'knife', 'gun', 'weapon', 'pistol', 'rifle', 'person', 'cell phone']
        for cls_id, cls_name in violence_filter.model.names.items():
            if any(kw in cls_name.lower() for kw in blur_keywords):
                target_classes.append(cls_id)
                
        if target_classes:
            names = [violence_filter.model.names[i] for i in target_classes]
            print(f"[CaptureWorker] Dynamically targeting classes to blur: {names} (IDs: {target_classes})")
        else:
            print("[CaptureWorker] WARNING: No violence/weapon classes found in model. Blurring ALL detections.")

        prev_time = time.time()
        frame_count = 0

        print("[CaptureWorker] Loop started.")

        while self.running:
            frame = capturer.get_frame()

            # ---- Conditional processing based on face presence ----
            if TARGET_PRESENT:
                results = violence_filter.model(
                    frame, device=device, imgsz=640, verbose=False,
                )

                raw_boxes = []
                for result in results:
                    if result.boxes is None:
                        continue
                    for box in result.boxes:
                        conf = float(box.conf[0])
                        if conf < 0.30:  # Lowered so knives/phones are detected easier
                            continue
                        cls_id = int(box.cls[0])
                        # Filter based on dynamic target classes
                        if target_classes and cls_id not in target_classes:
                            continue
                        x1, y1, x2, y2 = box.xyxy[0]
                        raw_boxes.append([int(x1), int(y1), int(x2), int(y2)])

                merged = self._merge_boxes(raw_boxes)
                self.data_ready.emit(frame, merged)
            else:
                # No target present — send frame with empty boxes
                self.data_ready.emit(frame, [])

            # FPS tracking
            frame_count += 1
            curr_time = time.time()
            if curr_time - prev_time >= 1.0:
                fps = frame_count / (curr_time - prev_time)
                status = "ACTIVE" if TARGET_PRESENT else "STANDBY"
                print(f"[Worker] {status} | FPS: {fps:.1f}")
                frame_count = 0
                prev_time = curr_time

    def stop(self):
        self.running = False
        self.wait()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    # ---- Load .env ----
    load_dotenv(dotenv_path=os.path.join(_PROJECT_ROOT, ".env"))

    target_face_path = os.getenv("TARGET_FACE_PATH", "data/test_target.jpeg")
    if not os.path.isabs(target_face_path):
        target_face_path = os.path.join(_PROJECT_ROOT, target_face_path)

    model_path = os.getenv(
        "YOLO_MODEL_PATH",
        os.path.join(_PROJECT_ROOT, "models", "violence_best.pt"),
    )

    # ---- Print banner ----
    print("=" * 60)
    print("  MAIN ORCHESTRATOR — PyQt6 Violence Overlay Pipeline")
    print("=" * 60)
    print("WARNING: This runs a click-through overlay.")
    print("         To exit, press Ctrl+C in this console.")
    print("=" * 60)

    # ---- Instantiate face gatekeeper ----
    gatekeeper = FaceGatekeeper(target_face_path)

    # ---- Launch face gatekeeper in a daemon thread ----
    face_thread = threading.Thread(
        target=face_worker,
        args=(gatekeeper,),
        daemon=True,
        name="FaceGatekeeper",
    )
    face_thread.start()
    print("[Main] Face gatekeeper thread started.")

    # ---- PyQt6 GUI on main thread (CRITICAL) ----
    app = QApplication(sys.argv)

    overlay = ScreenOverlay()
    overlay.show()

    # ---- QThread worker for screen capture + YOLO ----
    worker = CaptureWorker(model_path=model_path, confidence=0.45)
    worker.data_ready.connect(overlay.update_data)
    worker.start()
    print("[Main] Capture worker thread started.")

    # ---- Enter main event loop ----
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt received.")
    finally:
        worker.stop()
        _shutdown_event.set()
        face_thread.join(timeout=2.0)
        print("[Main] Shutdown complete.")


if __name__ == "__main__":
    main()
