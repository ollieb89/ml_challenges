from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Protocol

import cv2
import numpy as np
from tqdm import tqdm

from .pose_detector import DetectionResult


class PoseDetector(Protocol):
    """Typed interface implemented by detector wrappers."""

    def detect(self, frame: np.ndarray) -> DetectionResult:  # pragma: no cover - Protocol
        ...


@dataclass
class VideoBenchmarkSample:
    """Metadata describing a single benchmark video."""

    name: str
    path: Path
    expected_multi_person: bool = False
    stride: int = 1
    max_frames: Optional[int] = None


@dataclass
class FrameMeasurement:
    latency_ms: float
    vram_mb: float
    num_poses: int


@dataclass
class DetectorBenchmarkResult:
    name: str
    avg_latency_ms: float
    latency_p95_ms: float
    avg_fps: float
    avg_vram_mb: float
    multi_person_success_rate: float
    total_frames: int
    raw_metrics: List[FrameMeasurement] = field(default_factory=list)


class PoseDetectorBenchmark:
    """Benchmark harness that evaluates detectors against sample videos."""

    def __init__(
        self,
        *,
        max_frames: int = 180,
        warmup_frames: int = 5,
    ) -> None:
        self.max_frames = max_frames
        self.warmup_frames = warmup_frames

    def run(
        self,
        *,
        name: str,
        detector: PoseDetector,
        samples: Iterable[VideoBenchmarkSample],
    ) -> DetectorBenchmarkResult:
        measurements: List[FrameMeasurement] = []
        multi_person_hits = 0
        total_multi_frames = 0

        for sample in samples:
            video_metrics = self._run_on_video(detector, sample)
            measurements.extend(video_metrics["frames"])
            multi_person_hits += video_metrics["multi_hits"]
            total_multi_frames += video_metrics["multi_frames"]

        latencies = [m.latency_ms for m in measurements]
        vram = [m.vram_mb for m in measurements]
        fps_values = [1000.0 / l for l in latencies if l > 0]
        latency_p95 = np.percentile(latencies, 95).item() if latencies else 0.0
        multi_rate = (multi_person_hits / total_multi_frames) if total_multi_frames else 0.0

        return DetectorBenchmarkResult(
            name=name,
            avg_latency_ms=mean(latencies) if latencies else 0.0,
            latency_p95_ms=latency_p95,
            avg_fps=mean(fps_values) if fps_values else 0.0,
            avg_vram_mb=mean(vram) if vram else 0.0,
            multi_person_success_rate=multi_rate,
            total_frames=len(measurements),
            raw_metrics=measurements,
        )

    def _run_on_video(
        self,
        detector: PoseDetector,
        sample: VideoBenchmarkSample,
    ) -> Dict[str, object]:
        cap = cv2.VideoCapture(str(sample.path))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {sample.path}")

        frames: List[FrameMeasurement] = []
        multi_hits = 0
        multi_frames = 0
        captured = 0
        target_frames = sample.max_frames or self.max_frames

        try:
            with tqdm(total=target_frames, desc=f"{sample.name}", leave=False) as pbar:
                while captured < target_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if captured % sample.stride != 0:
                        captured += 1
                        continue

                    result = detector.detect(frame)
                    measurement = FrameMeasurement(
                        latency_ms=result.metrics.latency_ms,
                        vram_mb=result.metrics.vram_mb,
                        num_poses=result.metrics.num_poses,
                    )
                    if captured >= self.warmup_frames:
                        frames.append(measurement)
                        if sample.expected_multi_person:
                            multi_frames += 1
                            if result.metrics.num_poses >= 2:
                                multi_hits += 1

                    captured += 1
                    pbar.update(1)
        finally:
            cap.release()

        return {
            "frames": frames,
            "multi_hits": multi_hits,
            "multi_frames": multi_frames,
        }


__all__ = [
    "PoseDetectorBenchmark",
    "VideoBenchmarkSample",
    "DetectorBenchmarkResult",
]
