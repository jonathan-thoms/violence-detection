"""
High-Speed Screen Capture Module
Directive: directives/02_screen_capture.md

Captures the primary monitor with the lowest possible latency using mss.
Per SKILL.md: NEVER use pyautogui, ImageGrab, or mss.shot().
"""

import time

import cv2
import mss
import numpy as np


class ScreenCapturer:
    """Persistent mss-based screen capturer optimised for FPS."""

    def __init__(self, monitor_index: int = 1):
        """
        Initialise the mss instance and define the capture region.

        Args:
            monitor_index: Which monitor to capture (1 = primary).
        """
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[monitor_index]
        print(
            f"[ScreenCapturer] Capturing monitor {monitor_index}: "
            f"{self.monitor['width']}x{self.monitor['height']}"
        )

    def get_frame(self) -> np.ndarray:
        """
        Grab a single screen frame and return it as a BGR numpy array.

        Per SKILL.md §1: drop the Alpha channel immediately after capture.
        Uses cv2.cvtColor(BGRA→BGR) which is the fastest contiguous path.
        """
        raw = self.sct.grab(self.monitor)
        # np.array on an mss Screenshot gives a (H, W, 4) BGRA array
        frame = np.array(raw, dtype=np.uint8)
        # cvtColor produces a contiguous BGR array (fast C-level copy)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


# ---------------------------------------------------------------------------
# Test block — run with:  python execution/screen_capture.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    capturer = ScreenCapturer(monitor_index=1)

    prev_time = time.time()
    frame_count = 0
    fps = 0.0
    display_interval = 3  # only render to window every Nth frame

    print("[ScreenCapturer] Running capture loop. Press 'q' to quit.")
    print(f"  (Displaying every {display_interval} frames to reduce GUI overhead)")

    while True:
        frame = capturer.get_frame()

        # ---- FPS calculation (rolling per-second update) ----
        frame_count += 1
        curr_time = time.time()
        elapsed = curr_time - prev_time

        if elapsed >= 1.0:
            fps = frame_count / elapsed
            print(f"  FPS: {fps:.1f}")
            frame_count = 0
            prev_time = curr_time

        # ---- Display every Nth frame to keep GUI responsive
        #      without tanking capture FPS ----
        if frame_count % display_interval == 0:
            display = cv2.resize(frame, (960, 540))
            cv2.putText(
                display,
                f"FPS: {fps:.1f}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2,
            )
            # MUST be main thread per SKILL.md §3
            cv2.imshow("Screen Capture Test", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    print(f"[ScreenCapturer] Shutdown complete. Last FPS: {fps:.1f}")
