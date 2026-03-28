# Directive: Main Orchestrator (Integration)

## Goal
Create the main entry point for the application in `execution/main_orchestrator.py`. This script must integrate the Face Gatekeeper, Screen Capturer, and Violence Filter modules, ensuring the application runs smoothly in real-time without UI blocking or lag.

## Tools & Libraries
- `threading`
- `cv2`
- `time`
- Custom modules: `face_gatekeeper`, `screen_capture`, `violence_filter`

## Required Outputs
A script that performs the following:
1.  **Initialization:**
    - Instantiate the `FaceGatekeeper`, `ScreenCapturer`, and `ViolenceFilter` classes.
    - Set up a thread-safe global variable or class attribute `TARGET_PRESENT = False`.
2.  **Background Thread (The Gatekeeper):**
    - Create a worker function that continuously grabs frames from the webcam and runs the `is_target_present()` method.
    - Update the `TARGET_PRESENT` variable based on the result.
    - Run this function in a daemon `threading.Thread` so it doesn't block the main process and exits when the program closes.
3.  **Main Loop (The Display):**
    - Continuously grab frames using the `ScreenCapturer`.
    - Check the state of `TARGET_PRESENT`.
    - **IF TRUE:** Pass the screen frame through `ViolenceFilter.process_frame()` to apply the YOLO inference and blurring. Display the processed frame.
    - **IF FALSE:** Display the raw, unfiltered screen frame (or apply a green "Standby / Unlocked" overlay text to indicate the system is idle).
    - Calculate and display the overall pipeline FPS on the screen using `cv2.putText`.
    - Handle safe exit (e.g., closing windows and releasing resources when 'q' is pressed).

## Constraints
- **Absolute Rule:** `cv2.imshow()` and `cv2.waitKey()` MUST be called in the main thread. Do not attempt to create or update OpenCV windows from within the background face recognition thread.
- Prioritize high FPS for the screen capture. If the background face recognition thread lags, it should only delay the update of the `TARGET_PRESENT` status, not the frame rate of the video feed.