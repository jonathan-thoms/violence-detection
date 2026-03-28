"""
YOLO Inference & Blurring Module
Directive: directives/03_violence_filter.md, directives/06_upgrade_violence_model.md

Takes a frame, runs YOLOv8 inference, and applies blur to
detected bounding boxes.

Per directive 06:
  - Uses custom violence detection model (models/violence_best.pt)
  - No class ID filtering — any detection above threshold triggers blur

Per SKILL.md §2:
  - ALWAYS use Ultralytics YOLOv8 syntax.
  - Inference: model(frame, stream=True)
  - Box parsing: box.xyxy[0] -> int coords
"""

import cv2
import numpy as np
from ultralytics import YOLO


class ViolenceFilter:
    """Runs YOLOv8 inference and blurs detected regions."""

    def __init__(self, model_path: str = "models/violence_best.pt"):
        """
        Load a YOLOv8 model.

        Args:
            model_path: Path to a .pt weights file.  If the file does not
                        exist locally, Ultralytics will auto-download it.
        """
        self.model = YOLO(model_path)
        print(f"[ViolenceFilter] Model loaded: {model_path}")

    def process_frame(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.45,
        target_classes: list[int] | None = None,
    ) -> np.ndarray:
        """
        Run inference and blur every detection that passes the threshold.

        Args:
            frame: BGR numpy array (OpenCV format).
            confidence_threshold: Minimum confidence to trigger blur.
            target_classes: List of class IDs to blur.
                            None = blur all detected classes (default for
                            the custom violence model).

        Returns:
            The frame with blur applied over each detected ROI.
        """
        # stream=True returns a generator — memory-efficient (SKILL.md §2)
        results = self.model(frame, stream=True, verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                # ---- Filter by confidence ----
                conf = float(box.conf[0])
                if conf < confidence_threshold:
                    continue

                # ---- Filter by class (if specified) ----
                cls_id = int(box.cls[0])
                if target_classes is not None and cls_id not in target_classes:
                    continue

                # ---- Extract integer coords (SKILL.md §2) ----
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # ---- Clamp to frame bounds (SKILL.md §3: verify shapes) ----
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue  # degenerate box

                # ---- Apply rapid blur to the ROI ----
                roi = frame[y1:y2, x1:x2]
                # cv2.blur is significantly faster than GaussianBlur for large kernels
                blurred_roi = cv2.blur(roi, (51, 51))
                frame[y1:y2, x1:x2] = blurred_roi

        return frame


# ---------------------------------------------------------------------------
# Test block — run with:  python execution/violence_filter.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    import os

    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
    model_path = os.path.join(_PROJECT_ROOT, "models", "violence_best.pt")

    vf = ViolenceFilter(model_path)

    # Use webcam for testing
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam (index 0).")
        exit(1)

    print("[ViolenceFilter] Webcam opened. Press 'q' to quit.")
    print("  Blurring ALL detections from custom violence model")

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame.")
            break

        # Blur all violence detections (no class filter)
        processed = vf.process_frame(
            frame, confidence_threshold=0.45
        )

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        cv2.putText(
            processed,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Violence Filter Test", processed)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[ViolenceFilter] Shutdown complete.")
