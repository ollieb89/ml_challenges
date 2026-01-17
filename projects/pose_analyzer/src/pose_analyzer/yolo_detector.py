from ultralytics import YOLO
import cv2

class YOLOPoseDetector:
    def __init__(self, model_variant: str = "n"):  # n, s, m, l, x
        """Initialize YOLOv11 pose detector."""
        self.model = YOLO(f"yolo11{model_variant}-pose.pt")
    
    def detect(self, frame):
        """Detect pose keypoints using YOLOv11."""
        results = self.model(frame, conf=0.5, iou=0.45)
        return results[0].keypoints
