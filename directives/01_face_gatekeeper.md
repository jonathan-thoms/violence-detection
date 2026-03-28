# Directive: Face Gatekeeper Module

## Goal
Create a standalone Python script in `execution/face_gatekeeper.py` that utilizes the webcam to detect if a specific target user is present.

## Inputs
1. A live webcam feed using `cv2.VideoCapture(0)`.
2. A reference image path loaded from the `.env` file (`TARGET_FACE_PATH`).

## Tools & Libraries
- `cv2` (OpenCV)
- `face_recognition`
- `python-dotenv`

## Required Outputs
A class named `FaceGatekeeper` with:
1. An `__init__` method that loads and encodes the reference image using `face_recognition.face_encodings`.
2. A method `is_target_present(frame)` that takes a BGR numpy array (webcam frame), checks for faces, and returns `True` if a face matches the reference encoding, otherwise `False`.
3. A `if __name__ == "__main__":` test block that opens the webcam, draws a green box around the face if `True` and red if `False`, and displays the FPS.

## Constraints
- Frame processing for face recognition can be slow. In the test block, resize the webcam frame to 1/4 size before passing it to `face_recognition` to maintain a reasonable FPS, but scale the bounding box coordinates back up for drawing.