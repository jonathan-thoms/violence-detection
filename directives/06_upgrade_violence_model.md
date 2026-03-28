Goal
Update execution/violence_filter.py to utilize a custom trained violence detection model, and refactor execution/main_orchestrator.py to feed the output to the new PyQt6 transparent overlay instead of an OpenCV window.

Instructions

Model Swap: In violence_filter.py, modify the initialization to load models/violence_best.pt instead of the default yolov8n.pt.

Logic Update: Remove the hardcoded class ID filter (which previously looked for "person" / class ID 0). Since the new model is specifically trained for violence or weapons, any positive detection above the confidence threshold (e.g., 0.45 or 0.5) should trigger the bounding box to be returned.

Orchestrator Refactor: In main_orchestrator.py, completely remove all cv2.imshow() and cv2.waitKey() logic.

GUI Initialization: Refactor main_orchestrator.py to initialize the PyQt6 QApplication and the ScreenOverlay (from execution/overlay.py) in the main thread.

Threading: Move the continuous mss screen capture and ViolenceFilter processing into a QThread. This worker thread must emit a pyqtSignal containing the raw frame and the detected bounding boxes to the ScreenOverlay so it can paint the blurred regions.

Constraints

CRITICAL: Ensure the PyQt QApplication event loop (app.exec()) runs strictly in the main thread. If the AI tries to run the GUI in a sub-thread, the application will crash.

If no violence is detected (empty bounding box list), the ScreenOverlay must update to clear the previous paint events, remaining completely transparent.