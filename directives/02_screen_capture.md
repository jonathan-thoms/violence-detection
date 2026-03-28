# Directive: High-Speed Screen Capture Module

## Goal
Create a script in `execution/screen_capture.py` dedicated to capturing the primary monitor with the lowest possible latency.

## Tools & Libraries
- `mss`
- `cv2`
- `numpy`
- `time`

## Required Outputs
A class named `ScreenCapturer` with:
1. An `__init__` method that initializes the `mss` instance and defines the monitor bounding box (use monitor 1).
2. A method `get_frame()` that grabs the screen, converts the raw pixels to a numpy array, drops the Alpha channel (BGRA to BGR), and returns the frame.
3. A `if __name__ == "__main__":` test block that runs a continuous `while True` loop, fetches the frame, displays it in an OpenCV window, calculates the FPS, and prints the FPS to the console.

## Constraints
- Do NOT use `pyautogui` or `ImageGrab`; they are too slow. Rely strictly on `mss`.
- The target is > 30 FPS in the test block.