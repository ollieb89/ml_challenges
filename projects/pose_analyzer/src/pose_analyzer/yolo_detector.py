from ultralytics import YOLO
import cv2

class YOLOPoseDetector:
    def __init__(self, model_variant: str = "n"):  # n, s, m, l, x
        """Initialize YOLOv11 pose detector."""
        from pathlib import Path
        # Resolve data/models directory relative to this file
        # src/pose_analyzer/yolo_detector.py -> src/pose_analyzer/ -> src/ -> projects/pose_analyzer/ -> ai-ml-pipeline/
        model_name = f"yolo11{model_variant}-pose.pt"
        data_dir = Path(__file__).resolve().parents[4] / "data" / "models"
        model_path = data_dir / model_name
        
        # Fallback to current working directory if not found in data/models
        if not model_path.exists():
            model_path = Path(model_name)
            
        self.model = YOLO(str(model_path))
    
    def detect(self, frame):
        """Detect pose keypoints using YOLOv11."""
        results = self.model(frame, conf=0.5, iou=0.45)
        return results[0].keypoints
