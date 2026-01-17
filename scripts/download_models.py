"""Download pre-trained models for projects."""

import os
from pathlib import Path
from contextlib import contextmanager
import shutil
from ultralytics import YOLO
import mediapipe as mp


@contextmanager
def _pushd(path: Path):
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)

def setup_models():
    """Download and cache models."""
    
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("📥 Downloading pose estimation models...")

    # Move any previously downloaded weights from repo root into the cache dir.
    for f in Path.cwd().glob("yolo11*-pose.pt"):
        dest = model_dir / f.name
        if dest.exists():
            f.unlink()
        else:
            shutil.move(str(f), str(dest))
    
    # YOLOv11 variants
    with _pushd(model_dir):
        for variant in ['n', 's', 'm']:
            print(f"  - YOLOv11{variant}-pose.pt...")
            YOLO(f"yolo11{variant}-pose.pt")  # Auto-downloads pretrained weights
    
    print("✅ Models downloaded successfully!")
    print(f"📁 Models saved to: {model_dir}")

if __name__ == "__main__":
    setup_models()
