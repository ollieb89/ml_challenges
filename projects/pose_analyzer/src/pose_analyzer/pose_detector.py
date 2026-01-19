from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, List, Optional, Sequence
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python as task_python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

from .biomechanics import JointAngles

BaseOptions = task_python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = vision.RunningMode

MEDIAPIPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)


@dataclass
class DetectionMetrics:
    """Latency and resource metrics captured per inference."""

    latency_ms: float
    fps: float
    vram_mb: float
    num_poses: int


@dataclass
class DetectionResult:
    """Container for detector output and metrics."""

    keypoints: List[np.ndarray]
    metrics: DetectionMetrics
    raw_output: Any
    joint_angles: Optional[JointAngles] = None


class _VRAMTracker:
    """Utility context manager to track CUDA memory usage safely."""

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda:0" if torch.cuda.is_available() else None)
        self.vram_mb: float = 0.0

    def __enter__(self) -> "_VRAMTracker":
        if self.device and "cuda" in self.device:
            torch.cuda.reset_peak_memory_stats(self.device)
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:  # type: ignore[override]
        if self.device and "cuda" in self.device:
            peak_bytes = torch.cuda.max_memory_allocated(self.device)
            self.vram_mb = peak_bytes / (1024**2)


class MediaPipePoseDetector:
    """MediaPipe Pose landmarker wrapper with instrumentation."""

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        *,
        cache_dir: Optional[str | Path] = None,
        max_poses: int = 2,
        min_confidence: float = 0.5,
        target_fps: float = 30.0,
    ) -> None:
        self.target_fps = target_fps
        resolved_path = self._resolve_model_path(model_path, cache_dir)
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(resolved_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=max_poses,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        self._frame_index = 0

    @staticmethod
    def _resolve_model_path(
        model_path: Optional[str | Path], cache_dir: Optional[str | Path]
    ) -> Path:
        if model_path:
            return Path(model_path)

        cache_root = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "pose_analyzer"
        cache_root.mkdir(parents=True, exist_ok=True)
        destination = cache_root / "pose_landmarker_full.task"
        if not destination.exists():
            urlretrieve(MEDIAPIPE_MODEL_URL, destination)
        return destination

    @staticmethod
    def _convert_landmarks(
        landmarks: Optional[Sequence[Sequence[mp.NormalizedLandmark]]]
    ) -> List[np.ndarray]:
        if not landmarks:
            return []
        converted: List[np.ndarray] = []
        for person in landmarks:
            coords = np.array([[lm.x, lm.y, lm.z] for lm in person], dtype=np.float32)
            converted.append(coords)
        return converted

    def detect(self, frame: np.ndarray) -> DetectionResult:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((self._frame_index / self.target_fps) * 1000)
        start = perf_counter()
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        latency_ms = (perf_counter() - start) * 1000
        self._frame_index += 1

        keypoints = self._convert_landmarks(result.pose_landmarks)
        metrics = DetectionMetrics(
            latency_ms=latency_ms,
            fps=(1000.0 / latency_ms) if latency_ms > 0 else 0.0,
            vram_mb=0.0,
            num_poses=len(keypoints),
        )
        return DetectionResult(keypoints=keypoints, metrics=metrics, raw_output=result)


class YOLOPosev11Detector:
    """YOLOv11 pose detector with latency and VRAM measurements.
    
    Supports FP16 (half precision) mode for faster inference on GPUs
    with tensor cores. Use a smaller imgsz (e.g., 640) for faster
    inference when TensorRT is not available.
    """

    def __init__(
        self,
        model_variant: str = "n",
        *,
        device: Optional[str] = None,
        imgsz: int = 640,  # Reduced from 1080 for faster inference
        confidence: float = 0.5,
        iou: float = 0.45,
        engine_path: Optional[str | Path] = None,
        half: bool = True,  # Enable FP16 by default on CUDA
        use_compile: bool = False,
    ) -> None:
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.confidence = confidence
        self.iou = iou
        self.use_compile = use_compile
        
        # FP16 only on CUDA
        self.half = half and torch.cuda.is_available() and "cuda" in self.device
        
        # Resolve weights path
        if engine_path:
            weights = Path(engine_path)
        else:
            weights = Path(__file__).resolve().parents[4] / "data" / "models" / f"yolo11{model_variant}-pose.pt"
            # Fallback to current directory for backward compatibility
            if not weights.exists():
                weights = Path(f"yolo11{model_variant}-pose.pt")
                
        self.model = YOLO(str(weights), task="pose")
        self.is_tensorrt = str(weights).endswith(".engine")
        if self.is_tensorrt:
            self.imgsz = 1088
            self.runtime_device: str | int = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.imgsz = imgsz
            self.model.to(self.device)
            # Apply torch.compile optimization (PyTorch 2.0+)
            if self.use_compile and not self.is_tensorrt and hasattr(torch, "compile"):
                try:
                    # Optimize the inner nn.Module
                    # use reduce-overhead for low latency, but it requires static shapes usually.
                    # Given adaptive batching, shapes might vary. Using dynamic=True or default mode is safer.
                    self.model.model = torch.compile(self.model.model, dynamic=True)
                except Exception as e:
                    print(f"Warning: torch.compile failed, falling back to eager mode. Error: {e}")
            # Convert model to FP16 for faster inference
            if self.half:
                self.model.model.half()
            self.runtime_device = self.device

    @staticmethod
    def _extract_keypoints(prediction) -> List[np.ndarray]:
        if prediction.keypoints is None or prediction.keypoints.data is None:
            return []
        keypoint_tensor = prediction.keypoints.data
        return [kp.cpu().numpy() for kp in keypoint_tensor]

    def detect(self, frame: np.ndarray) -> DetectionResult:
        with _VRAMTracker(self.device) as tracker:
            start = perf_counter()
            predictions = self.model(
                frame,
                imgsz=self.imgsz,
                conf=self.confidence,
                iou=self.iou,
                device=self.runtime_device,
                half=self.half,
                verbose=False,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latency_ms = (perf_counter() - start) * 1000

        result = predictions[0]
        keypoints = self._extract_keypoints(result)
        metrics = DetectionMetrics(
            latency_ms=latency_ms,
            fps=(1000.0 / latency_ms) if latency_ms > 0 else 0.0,
            vram_mb=tracker.vram_mb,
            num_poses=len(keypoints),
        )
        return DetectionResult(keypoints=keypoints, metrics=metrics, raw_output=result)


__all__ = [
    "DetectionMetrics",
    "DetectionResult",
    "MediaPipePoseDetector",
    "YOLOPosev11Detector",
]
