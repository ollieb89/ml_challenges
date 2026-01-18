"""Multi-stream video processing with stage-aware detector selection."""

from __future__ import annotations

import argparse
import json
import queue
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
import torch

from .pose_detector import DetectionResult, MediaPipePoseDetector, YOLOPosev11Detector
from .biomechanics import JointAngleCalculator


@dataclass
class StreamStats:
    """Runtime metrics per stream."""

    frames_processed: int = 0
    dropped_frames: int = 0
    avg_latency_ms: float = 0.0
    peak_latency_ms: float = 0.0


class MultiStreamProcessor:
    """Queue-based concurrent processor for up to 4 streams on a single GPU."""

    def __init__(
        self,
        video_paths: Iterable[Path],
        *,
        max_streams: int = 4,
        detector: str = "yolo",
        batch_size: int = 2,
        max_latency_ms: float = 100.0,
        gpu_mem_drop_threshold: float = 0.9,
        frame_stride: int = 1,
        engine_path: Optional[str | Path] = None,
        enable_joint_angles: bool = True,
        confidence_threshold: float = 0.5,
        smoothing_window: int = 3,
    ) -> None:
        self.video_paths = [Path(p) for p in video_paths]
        self.max_streams = max_streams
        self.batch_size = batch_size
        self.max_latency_ms = max_latency_ms
        self.gpu_mem_drop_threshold = gpu_mem_drop_threshold
        self.frame_stride = frame_stride
        self.detector_name = detector
        self.detector = self._create_detector(detector, engine_path=engine_path)
        self.stats: Dict[int, StreamStats] = {}
        self._stop_event = threading.Event()
        
        # Joint angle calculation
        self.enable_joint_angles = enable_joint_angles
        if self.enable_joint_angles:
            self.angle_calculator = JointAngleCalculator(
                confidence_threshold=confidence_threshold,
                smoothing_window=smoothing_window
            )

    @staticmethod
    def _create_detector(name: str, *, engine_path: Optional[str | Path]):
        if name.lower() == "mediapipe":
            return MediaPipePoseDetector()
        if name.lower().startswith("yolo"):
            variant = name.split(":", maxsplit=1)[1] if ":" in name else "n"
            return YOLOPosev11Detector(model_variant=variant, engine_path=engine_path)
        raise ValueError(f"Unsupported detector: {name}")

    def _should_drop_frame(self) -> bool:
        if not torch.cuda.is_available():
            return False
        stats = torch.cuda.memory_stats()
        used = stats.get("active_bytes.all.current", 0)
        total = torch.cuda.get_device_properties(0).total_memory
        return used / total > self.gpu_mem_drop_threshold

    def _process_frame(self, stream_id: int, frame: np.ndarray) -> DetectionResult:
        start = perf_counter()
        result: DetectionResult = self.detector.detect(frame)
        
        # Calculate joint angles if enabled and keypoints are detected
        if self.enable_joint_angles and result.keypoints:
            try:
                # Use first detected person for angle calculation
                keypoints = result.keypoints[0]
                
                # Extract confidence scores if available from detector
                confidences = None
                if hasattr(result.raw_output, 'keypoints') and hasattr(result.raw_output.keypoints, 'conf'):
                    # YOLO confidence scores
                    confidences = result.raw_output.keypoints.conf[0].cpu().numpy()
                elif hasattr(result.raw_output, 'pose_landmarks') and result.raw_output.pose_landmarks:
                    # MediaPipe visibility scores
                    confidences = np.array([lm.visibility for lm in result.raw_output.pose_landmarks[0]])
                
                # Calculate angles
                joint_angles = self.angle_calculator.calculate_angles(keypoints, confidences)
                result.joint_angles = joint_angles
                
            except Exception as e:
                # Log error but don't fail the processing
                print(f"Warning: Joint angle calculation failed for stream {stream_id}: {e}")
                result.joint_angles = None
        
        latency_ms = (perf_counter() - start) * 1000
        stream_stats = self.stats.setdefault(stream_id, StreamStats())
        stream_stats.frames_processed += 1
        stream_stats.avg_latency_ms += (latency_ms - stream_stats.avg_latency_ms) / stream_stats.frames_processed
        stream_stats.peak_latency_ms = max(stream_stats.peak_latency_ms, latency_ms)
        if latency_ms > self.max_latency_ms:
            stream_stats.dropped_frames += 1
            
        return result

    def run(self) -> Dict[int, StreamStats]:
        stream_threads: List[threading.Thread] = []
        for stream_id, video_path in enumerate(self.video_paths[: self.max_streams]):
            thread = threading.Thread(target=self._process_stream, args=(stream_id, video_path), daemon=True)
            stream_threads.append(thread)
            thread.start()

        for thread in stream_threads:
            thread.join()
        return self.stats

    def _process_stream(self, stream_id: int, video_path: Path) -> None:
        cap = cv2.VideoCapture(str(video_path))
        frame_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=self.batch_size * 2)
        reader = threading.Thread(target=self._reader_loop, args=(cap, frame_queue), daemon=True)
        reader.start()
        stride_index = 0

        try:
            while not self._stop_event.is_set():
                batch: List[np.ndarray] = []
                while len(batch) < self.batch_size:
                    try:
                        frame = frame_queue.get(timeout=0.5)
                    except queue.Empty:
                        break
                    stride_index += 1
                    if stride_index % self.frame_stride != 0:
                        continue
                    if self._should_drop_frame():
                        self.stats.setdefault(stream_id, StreamStats()).dropped_frames += 1
                        continue
                    batch.append(frame)
                if not batch:
                    if not reader.is_alive() and frame_queue.empty():
                        break
                    continue
                for frame in batch:
                    self._process_frame(stream_id, frame)
        finally:
            self._stop_event.set()
            cap.release()

    @staticmethod
    def _reader_loop(cap: cv2.VideoCapture, frame_queue: "queue.Queue[np.ndarray]") -> None:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                frame_queue.put(frame, timeout=0.5)
            except queue.Full:
                continue


def _default_videos() -> List[Path]:
    repo_root = Path(__file__).parents[3]
    return [
        repo_root / "data/pose_references/videos/squat_form_001.mp4",
        repo_root / "data/pose_references/videos/pushup_form_004.mp4",
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-stream pose processing demo")
    parser.add_argument("--videos", type=Path, nargs="+", default=None, help="Paths to video files")
    parser.add_argument("--detector", type=str, default="yolo:n", help="Detector identifier (yolo:<variant> or mediapipe)")
    parser.add_argument("--engine-path", type=Path, default=None, help="Optional TensorRT engine path for YOLO")
    parser.add_argument("--max-streams", type=int, default=2, help="Maximum concurrent streams")
    parser.add_argument("--batch-size", type=int, default=2, help="Frames processed per batch")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--gpu-threshold", type=float, default=0.9, help="Drop frames when GPU memory exceeds this fraction")
    parser.add_argument("--max-latency-ms", type=float, default=100.0, help="Latency budget triggering drop accounting")
    parser.add_argument("--stats-output", type=Path, default=None, help="Optional JSON stats output path")
    parser.add_argument("--enable-joint-angles", action="store_true", default=True, help="Enable joint angle calculations")
    parser.add_argument("--disable-joint-angles", dest="enable_joint_angles", action="store_false", help="Disable joint angle calculations")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Confidence threshold for joint angle calculations")
    parser.add_argument("--smoothing-window", type=int, default=3, help="Temporal smoothing window for joint angles")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    videos = args.videos if args.videos else _default_videos()
    processor = MultiStreamProcessor(
        videos,
        max_streams=min(args.max_streams, len(videos)),
        detector=args.detector,
        batch_size=args.batch_size,
        max_latency_ms=args.max_latency_ms,
        gpu_mem_drop_threshold=args.gpu_threshold,
        frame_stride=args.frame_stride,
        engine_path=args.engine_path,
        enable_joint_angles=args.enable_joint_angles,
        confidence_threshold=args.confidence_threshold,
        smoothing_window=args.smoothing_window,
    )
    stats = processor.run()
    stats_dict = {stream_id: asdict(stat) for stream_id, stat in stats.items()}
    print(json.dumps(stats_dict, indent=2))
    if args.stats_output:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.stats_output, "w", encoding="utf-8") as fh:
            json.dump(stats_dict, fh, indent=2)


if __name__ == "__main__":
    main()


__all__ = ["MultiStreamProcessor", "StreamStats", "main"]
