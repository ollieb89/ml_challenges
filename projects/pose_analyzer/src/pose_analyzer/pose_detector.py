import mediapipe as mp
from mediapipe.tasks import python as task_python
from mediapipe.tasks.python import vision

BaseOptions = task_python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

class MediaPipePoseDetector:
    def __init__(self, model_path: str = None):
        """Initialize MediaPipe pose landmarker."""
        if model_path is None:
            # Auto-download from MediaPipe models
            model_path = "path/to/pose_landmarker_full.task"
        
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_poses=2,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        self.landmarker = PoseLandmarker.create_from_options(options)
    
    def detect(self, frame):
        """Detect pose landmarks in frame."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self.landmarker.detect(mp_image)
        return result.pose_landmarks
