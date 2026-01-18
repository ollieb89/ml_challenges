#!/usr/bin/env python3
"""CLI entry for benchmarking pose detectors on curated 1080p videos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pose_analyzer.benchmark import (
    DetectorBenchmarkResult,
    PoseDetectorBenchmark,
    VideoBenchmarkSample,
)
from pose_analyzer.pose_detector import MediaPipePoseDetector, YOLOPosev11Detector


def load_samples(config_path: Path) -> list[VideoBenchmarkSample]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    repo_root = config_path.parents[3]
    samples: list[VideoBenchmarkSample] = []
    for entry in config:
        entry_path = Path(entry["path"])
        resolved_path = entry_path if entry_path.is_absolute() else (repo_root / entry_path)
        samples.append(
            VideoBenchmarkSample(
                name=entry["name"],
                path=resolved_path.resolve(),
                expected_multi_person=entry.get("expected_multi_person", False),
                stride=entry.get("stride", 1),
            )
        )
    return samples


def summarize(result: DetectorBenchmarkResult) -> dict:
    return {
        "name": result.name,
        "avg_latency_ms": result.avg_latency_ms,
        "latency_p95_ms": result.latency_p95_ms,
        "avg_fps": result.avg_fps,
        "avg_vram_mb": result.avg_vram_mb,
        "multi_person_success_rate": result.multi_person_success_rate,
        "total_frames": result.total_frames,
    }


def run() -> None:
    parser = argparse.ArgumentParser(description="Pose detector benchmark runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parents[1] / "config" / "benchmark_videos.json",
        help="Path to benchmark video config JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parents[2] / "reports" / "benchmark_raw.json",
        help="Where to store raw benchmark results",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=240,
        help="Frames per sample video",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=10,
        help="Frames ignored for warmup",
    )
    parser.add_argument(
        "--yolo-variant",
        type=str,
        default="n",
        help="YOLOv11 variant (n, s, m, l, x)",
    )
    args = parser.parse_args()

    samples = load_samples(args.config)
    benchmark = PoseDetectorBenchmark(max_frames=args.max_frames, warmup_frames=args.warmup_frames)

    detectors = {
        "mediapipe": MediaPipePoseDetector(),
        "yolov11": YOLOPosev11Detector(model_variant=args.yolo_variant),
    }

    output_data: dict[str, dict] = {}
    for name, detector in detectors.items():
        result = benchmark.run(name=name, detector=detector, samples=samples)
        output_data[name] = summarize(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(json.dumps(output_data, indent=2))


if __name__ == "__main__":
    run()
