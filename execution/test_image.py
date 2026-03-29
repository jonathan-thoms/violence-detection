import urllib.request
from ultralytics import YOLO
import cv2
import os

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "violence_custom", "weights", "best.pt")
model = YOLO(model_path)

# Download a sample image of a handgun
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/1911_Pistol.jpg/800px-1911_Pistol.jpg"
img_path = "test_gun.jpg"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response, open(img_path, 'wb') as out_file:
    out_file.write(response.read())

results = model(img_path, conf=0.10)

print("\n--- RESULTS ---")
for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.names[cls_id]
        print(f"Detected: {name} (ID {cls_id}) with confidence: {conf*100:.1f}%")

print("Finished evaluating test image.")
