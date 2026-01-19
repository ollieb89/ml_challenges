"""
Pose Analyzer Package

Provides pose detection, biomechanics analysis, form scoring,
and concurrent multi-stream processing capabilities.
"""

from .pose_detector import (
    DetectionMetrics,
    DetectionResult,
    MediaPipePoseDetector,
    YOLOPosev11Detector,
)
from .biomechanics import JointAngleCalculator
from .form_scorer import FormScorer
from .form_anomaly_detector import FormAnomalyDetector
from .video_processor import MultiStreamProcessor
from .concurrent_stream_processor import (
    ConcurrentStreamProcessor,
    FramePacket,
    StreamMetrics,
    BenchmarkResult,
    SyntheticFrameSource,
    VideoFrameSource,
    VRAMMonitor,
)

__all__ = [
    # Detectors
    "DetectionMetrics",
    "DetectionResult",
    "MediaPipePoseDetector",
    "YOLOPosev11Detector",
    # Biomechanics
    "JointAngleCalculator",
    # Scoring
    "FormScorer",
    "FormAnomalyDetector",
    # Processing
    "MultiStreamProcessor",
    "ConcurrentStreamProcessor",
    "FramePacket",
    "StreamMetrics",
    "BenchmarkResult",
    "SyntheticFrameSource",
    "VideoFrameSource",
    "VRAMMonitor",
]
