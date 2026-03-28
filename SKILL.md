---
name: Real-Time Computer Vision & YOLOv8 Rules
description: Strict guidelines for writing high-FPS video processing code using OpenCV, YOLOv8, and mss.
risk: low
---

# Computer Vision Development Rules

When writing or modifying any computer vision, face recognition, or object detection code in this project, you MUST adhere to the following deterministic rules:

## 1. Screen Capture
* **NEVER** use `pyautogui`, `ImageGrab`, or `mss.shot()`. They are too slow for real-time video.
* **ALWAYS** use the `mss` library with a persistent `sct = mss.mss()` instance.
* **ALWAYS** drop the Alpha channel immediately after capture using `cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)` before passing the frame to OpenCV or YOLO.

## 2. YOLO Inference
* **ALWAYS** use Ultralytics YOLOv8 syntax. Do not use YOLOv5 or YOLOv7 syntax.
* Inference call: `results = model(frame, stream=True)` (use stream generator for speed).
* Bounding box parsing: Extract coordinates using `box.xyxy[0]` and convert them to integers `int(x1), int(y1), int(x2), int(y2)`.

## 3. OpenCV Window Management
* **ALWAYS** run `cv2.imshow()` and `cv2.waitKey(1)` inside the **main thread**. Running GUI commands in sub-threads will crash the application.
* Verify `numpy` array shapes before applying visual filters (like `cv2.GaussianBlur`) to prevent out-of-bounds dimension errors.

## 4. Concurrency & Performance
* Face recognition (`face_recognition` library) is CPU-heavy. **ALWAYS** run the face detection loop in a separate `threading.Thread` or `multiprocessing.Process` so it does not block the main `mss` screen capture loop.
* Resize frames to 25% or 50% scale before running face recognition to maintain high FPS, then scale the bounding boxes back up for the UI.