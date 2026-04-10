"""
Face Gatekeeper Module
Directive: directives/01_face_gatekeeper.md

Detects whether ANY registered target face is present in a webcam frame
using the face_recognition library (dlib-backed).

Supports multiple reference faces loaded from a directory.
"""

import os
import time
import glob
import shutil

import cv2
import face_recognition
import numpy as np
from dotenv import load_dotenv


class FaceGatekeeper:
    """Loads multiple reference face encodings and checks live frames for a match."""

    def __init__(self, faces_dir: str = None, reference_image_path: str = None):
        """
        Load and encode reference (target) face images.

        Args:
            faces_dir: Path to a directory containing face images.
                       All .jpg/.jpeg/.png files will be loaded.
            reference_image_path: Legacy single-image path (used if faces_dir
                                  is None or empty).
        """
        self.target_encodings = []  # list of (encoding, filename)
        self.faces_dir = faces_dir

        if faces_dir and os.path.isdir(faces_dir):
            self._load_directory(faces_dir)
        elif reference_image_path:
            self._load_single(reference_image_path)

        print(f"[FaceGatekeeper] {len(self.target_encodings)} face(s) loaded.")

    def _load_directory(self, faces_dir: str):
        """Scan a directory and encode every face image found."""
        extensions = ("*.jpg", "*.jpeg", "*.png")
        for ext in extensions:
            for filepath in glob.glob(os.path.join(faces_dir, ext)):
                self._encode_file(filepath)

    def _load_single(self, path: str):
        """Encode a single reference image (legacy compat)."""
        if os.path.isfile(path):
            self._encode_file(path)
        else:
            print(f"[FaceGatekeeper] WARNING: File not found: {path}")

    def _encode_file(self, filepath: str):
        """Load an image, extract the first face encoding, store it."""
        try:
            img = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(img)
            if encodings:
                self.target_encodings.append((encodings[0], os.path.basename(filepath)))
                print(f"[FaceGatekeeper] ✓ Encoded: {os.path.basename(filepath)}")
            else:
                print(f"[FaceGatekeeper] ✗ No face found in: {os.path.basename(filepath)}")
        except Exception as e:
            print(f"[FaceGatekeeper] ✗ Error loading {os.path.basename(filepath)}: {e}")

    def add_face(self, source_path: str) -> bool:
        """
        Copy a face image into the faces directory and encode it.

        Returns True if a face was successfully detected and added.
        """
        if not self.faces_dir or not os.path.isdir(self.faces_dir):
            print("[FaceGatekeeper] No faces directory configured.")
            return False

        filename = os.path.basename(source_path)
        dest = os.path.join(self.faces_dir, filename)

        # Avoid overwriting — add a suffix if needed
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(dest):
            dest = os.path.join(self.faces_dir, f"{base}_{counter}{ext}")
            counter += 1

        shutil.copy2(source_path, dest)

        # Try to encode
        try:
            img = face_recognition.load_image_file(dest)
            encodings = face_recognition.face_encodings(img)
            if encodings:
                self.target_encodings.append((encodings[0], os.path.basename(dest)))
                print(f"[FaceGatekeeper] ✓ Added: {os.path.basename(dest)}")
                return True
            else:
                os.remove(dest)
                print(f"[FaceGatekeeper] ✗ No face detected in {filename}. File removed.")
                return False
        except Exception as e:
            if os.path.exists(dest):
                os.remove(dest)
            print(f"[FaceGatekeeper] ✗ Error: {e}")
            return False

    def remove_face(self, filename: str) -> bool:
        """Remove a face by filename from encodings and delete the file."""
        self.target_encodings = [
            (enc, name) for enc, name in self.target_encodings if name != filename
        ]
        if self.faces_dir:
            filepath = os.path.join(self.faces_dir, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"[FaceGatekeeper] ✓ Removed: {filename}")
                return True
        return False

    def get_face_files(self) -> list:
        """Return list of (filename, full_path) for all registered faces."""
        if not self.faces_dir or not os.path.isdir(self.faces_dir):
            return []
        result = []
        for enc, name in self.target_encodings:
            full = os.path.join(self.faces_dir, name)
            if os.path.exists(full):
                result.append((name, full))
        return result

    def is_target_present(self, frame: np.ndarray) -> tuple[bool, list]:
        """
        Check whether ANY registered face appears in *frame*.

        Args:
            frame: A BGR numpy array (standard OpenCV format).

        Returns:
            (match_found, face_locations)
        """
        if not self.target_encodings:
            return False, []

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        if not face_locations:
            return False, []

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        known_encodings = [enc for enc, _ in self.target_encodings]

        for encoding in face_encodings:
            results = face_recognition.compare_faces(
                known_encodings, encoding, tolerance=0.6
            )
            if any(results):
                return True, face_locations

        return False, face_locations


# ---------------------------------------------------------------------------
# Test block — run with:  python execution/face_gatekeeper.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(dotenv_path=env_path)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    faces_dir = os.path.join(project_root, "data", "faces")

    gatekeeper = FaceGatekeeper(faces_dir=faces_dir)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam (index 0).")
        exit(1)

    print("[FaceGatekeeper] Webcam opened. Press 'q' to quit.")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        match, locations = gatekeeper.is_target_present(small_frame)

        for top, right, bottom, left in locations:
            top *= 4; right *= 4; bottom *= 4; left *= 4
            color = (0, 255, 0) if match else (0, 0, 255)
            label = "TARGET" if match else "UNKNOWN"
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Face Gatekeeper Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[FaceGatekeeper] Shutdown complete.")
