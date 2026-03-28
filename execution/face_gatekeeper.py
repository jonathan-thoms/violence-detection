"""
Face Gatekeeper Module
Directive: directives/01_face_gatekeeper.md

Detects whether a specific target face is present in a webcam frame
using the face_recognition library (dlib-backed).
"""

import os
import time

import cv2
import face_recognition
import numpy as np
from dotenv import load_dotenv


class FaceGatekeeper:
    """Loads a reference face encoding and checks live frames for a match."""

    def __init__(self, reference_image_path: str):
        """
        Load and encode the reference (target) face image.

        Args:
            reference_image_path: Absolute or project-relative path to the
                                  target face JPEG/PNG.
        """
        if not os.path.isfile(reference_image_path):
            raise FileNotFoundError(
                f"Reference image not found: {reference_image_path}"
            )

        ref_image = face_recognition.load_image_file(reference_image_path)
        encodings = face_recognition.face_encodings(ref_image)

        if len(encodings) == 0:
            raise ValueError(
                "No face detected in the reference image. "
                "Please provide a clear, front-facing photo."
            )

        self.target_encoding = encodings[0]
        print(f"[FaceGatekeeper] Reference face loaded from: {reference_image_path}")

    def is_target_present(self, frame: np.ndarray) -> tuple[bool, list]:
        """
        Check whether the target face appears in *frame*.

        Args:
            frame: A BGR numpy array (standard OpenCV format).

        Returns:
            (match_found, face_locations)
            - match_found: True if at least one face matches the target.
            - face_locations: List of (top, right, bottom, left) tuples for
              every detected face so the caller can draw bounding boxes.
        """
        # face_recognition expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        if not face_locations:
            return False, []

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        match_found = False
        for encoding in face_encodings:
            results = face_recognition.compare_faces(
                [self.target_encoding], encoding, tolerance=0.6
            )
            if results[0]:
                match_found = True
                break

        return match_found, face_locations


# ---------------------------------------------------------------------------
# Test block — run with:  python execution/face_gatekeeper.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Load .env from project root (one level up from execution/)
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(dotenv_path=env_path)

    target_face_path = os.getenv("TARGET_FACE_PATH", "data/test_target.jpeg")
    # Resolve relative to project root
    if not os.path.isabs(target_face_path):
        target_face_path = os.path.join(
            os.path.dirname(__file__), "..", target_face_path
        )

    gatekeeper = FaceGatekeeper(target_face_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam (index 0).")
        exit(1)

    print("[FaceGatekeeper] Webcam opened. Press 'q' to quit.")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame.")
            break

        # ---- Resize to 1/4 for faster face_recognition processing ----
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        match, locations = gatekeeper.is_target_present(small_frame)

        # ---- Draw bounding boxes scaled back to original size ----
        for top, right, bottom, left in locations:
            # Scale back up by 4x
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            color = (0, 255, 0) if match else (0, 0, 255)  # green / red
            label = "TARGET" if match else "UNKNOWN"

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(
                frame,
                label,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        # ---- FPS calculation ----
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2,
        )

        # ---- Display (MUST be main thread per SKILL.md §3) ----
        cv2.imshow("Face Gatekeeper Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[FaceGatekeeper] Shutdown complete.")
