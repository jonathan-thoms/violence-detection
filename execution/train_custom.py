"""
Script to train a custom YOLOv8 model locally on your GPU.

1. Go to Roboflow Universe (e.g., https://universe.roboflow.com/violence-detection-fbe46/violence-detection-nbx24/dataset/1)
2. Click "Download Dataset" -> Select "YOLOv8" format.
3. Extract the downloaded ZIP into a folder named `datasets/violence-detection` in the project root.
4. Run this script!
"""

import os
import sys

# Ensure Ultralytics can run from the execution folder
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from ultralytics import YOLO

def main():
    # 1. Path to your extracted Roboflow dataset's data.yaml
    data_yaml_path = os.path.join(_PROJECT_ROOT, "datasets", "violence-detection", "data.yaml")
    
    if not os.path.exists(data_yaml_path):
        print(f"[ERROR] Could not find data.yaml at: {data_yaml_path}")
        print("Please download the YOLOv8 dataset from Roboflow and extract it there.")
        sys.exit(1)
        
    print(f"[Train] Found dataset: {data_yaml_path}")
    print("[Train] Loading base YOLOv8 Nano model...")
    
    # 2. Load the base YOLOv8 model
    model = YOLO("yolov8n.pt") 

    # 3. Train the model on your dataset using your CUDA GPU
    print("\n" + "="*50)
    print("  Starting Training on CUDA GPU")
    print("="*50 + "\n")
    
    # epochs=25 is usually a good quick test.
    # imgsz=640 is the standard YOLO resolution.
    results = model.train(
        data=data_yaml_path,
        epochs=25,
        imgsz=640,
        device="cuda",  # Uses your NVIDIA GPU
        project=os.path.join(_PROJECT_ROOT, "models"),
        name="violence_custom",
        exist_ok=True,
        workers=0,  # FIX: Prevents "resource already mapped" Windows CUDA crash
        batch=8     # FIX: Lowers memory footprint to prevent out-of-memory errors
    )
    
    print("\n" + "="*50)
    print("  Training Complete!")
    print(f"  Your new .pt file is located at: models/violence_custom/weights/best.pt")
    print("  Update your .env to use this new path: YOLO_MODEL_PATH=models/violence_custom/weights/best.pt")
    print("="*50)

if __name__ == "__main__":
    main()
