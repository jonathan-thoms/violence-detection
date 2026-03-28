# Directive: YOLO Inference & Blurring Module

## Goal
Create a script in `execution/violence_filter.py` that takes a frame, runs YOLOv8 inference, and applies a Gaussian blur to detected bounding boxes.

## Tools & Libraries
- `ultralytics` (YOLOv8)
- `cv2`

## Required Outputs
A class named `ViolenceFilter` with:
1. An `__init__(model_path)` method that loads a YOLOv8 model.
2. A method `process_frame(frame, confidence_threshold=0.5)` that:
    - Runs YOLO inference on the input `frame`.
    - Iterates through the detected bounding boxes.
    - Extracts the Region of Interest (ROI) for each box.
    - Applies `cv2.GaussianBlur` (e.g., kernel size 51x51) to the ROI.
    - Replaces the original frame's ROI with the blurred ROI.
    - Returns the modified frame.
3. A `if __name__ == "__main__":` test block that initializes the class with the standard `yolov8n.pt` model, reads a test video or the webcam, and displays the processed output.

## Constraints
- For the test block, since we haven't trained a custom violence model yet, just use the base `yolov8n.pt` model and trigger the blur if the detected class is "person" (class ID 0) to verify the blurring logic works.