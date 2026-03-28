Goal
Create a system-wide, full-screen transparent overlay using PyQt6 in execution/overlay.py. This overlay must sit invisibly on top of the Windows desktop, allow all mouse clicks to pass through to the apps below, and only paint blurred rectangles over coordinates where violence is detected.

Tools & Libraries
PyQt6 (specifically QMainWindow, QApplication, Qt, QPainter, QImage, pyqtSignal, QThread)

cv2

numpy

Required Outputs
A script containing a ScreenOverlay class that inherits from QMainWindow and achieves the following:

Window Setup:

Matches the primary monitor's resolution exactly.

Applies the following PyQt6 flags to ensure it is a Windows click-through overlay:

Qt.WindowType.FramelessWindowHint (Removes borders/title bar)

Qt.WindowType.WindowStaysOnTopHint (Keeps it above all other apps)

Qt.WindowType.WindowTransparentForInput (Ignores mouse/keyboard input)

Applies self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground) to make the base window completely invisible.

Data Reception (Thread Safety):

Includes a pyqtSignal(object, list) to safely receive the raw screen frame (numpy array) and the YOLO bounding boxes from a background worker thread.

A slot method update_data(frame, boxes) connected to this signal that stores the latest frame and boxes, and then calls self.update() to trigger a repaint.

Painting the Blurs (The core visual logic):

Overrides the paintEvent(self, event) method.

When triggered, it checks if there are boxes to draw.

For each bounding box, it extracts that specific Region of Interest (ROI) from the stored raw frame.

Applies cv2.blur (e.g., with a large kernel like 51x51 or 99x99) to the ROI for O(1) paint performance. (Do NOT use cv2.GaussianBlur as it is too slow on the CPU for large windows).

Converts the blurred numpy ROI into a QImage and then a QPixmap.

Uses QPainter to draw this blurred QPixmap onto the transparent window at the exact (x1, y1) coordinates of the bounding box.

Constraints
CRITICAL: Do NOT use cv2.imshow. OpenCV cannot handle native Windows click-through transparency.

CRITICAL THREADING: PyQt6's app.exec() MUST run in the main application thread. The ScreenCapturer and ViolenceFilter (YOLO) loop must be moved into a separate QThread (or standard threading.Thread) that emits the pyqtSignal to the overlay.

Ensure memory is managed well. Do not let QImage or QPixmap objects leak memory during the rapid paintEvent calls.