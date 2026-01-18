# Detector Comparison: MediaPipe vs YOLOv11 (Day 2 Morning)

Benchmark configuration: `projects/pose_analyzer/scripts/run_detector_benchmark.py`
- Videos: `config/benchmark_videos.json` (squat, push-up multi-person, deadlift)
- Frames per video: 120 (10 warmup discarded)
- Resolution: 1080p source (YOLO auto-upsamples to stride-compatible 1088)
- Hardware: pixi `cuda` env with GPU acceleration

## Summary Metrics

| Detector   | Avg Latency (ms) | P95 Latency (ms) | Avg FPS | Avg VRAM (MB) | Multi-person Success | Frames |
|------------|-----------------|------------------|--------:|---------------|----------------------|--------|
| MediaPipe  | 12.48           | 15.11            | 90.90   | 0.00          | 78.2%                | 275    |
| YOLOv11-n  | **4.75**        | **5.52**         | **212.53** | 82.97     | **80.9%**            | 275    |

_Source: `reports/benchmark_raw.json`_

## Observations

### MediaPipe Pose Landmarker
- ✅ Extremely light footprint (0 MB GPU VRAM; runs fully on CPU)
- ✅ High FPS (>90) sufficient for single-stream real-time
- ⚠️ Latency ~12-15 ms -> still acceptable but 3× slower than YOLO
- ⚠️ Slightly worse multi-person recall (78%) on crowded push-up clips
- ✅ Easier deployment on CPU-only systems, no CUDA dependency

### YOLOv11n Pose
- ✅ Fastest inference: 4.75 ms avg (<5.6 ms P95) -> >200 FPS
- ✅ Best multi-person coverage (81%) thanks to multi-instance predictions
- ⚠️ Consumes ~83 MB GPU VRAM per stream; scales linearly with concurrent streams
- ⚠️ Requires CUDA-capable GPU; auto-adjusts resolution to 1088 due to stride constraint
- ✅ Aligns better with 4-stream, <100 ms aggregate latency requirement

## Decision for 4-Stream Goal

Given the 4-stream requirement (<100 ms end-to-end, <11 GB VRAM), YOLOv11n provides the best performance headroom: even with per-stream VRAM ~83 MB, total detector VRAM fits comfortably (<350 MB) leaving room for preprocessing and downstream analytics. Latency per frame (~5 ms) ensures that even with batching or queueing, we remain well under the 100 ms envelope.

MediaPipe remains a strong fallback for CPU-only or ultra-low-power deployments, but for the flagship RTX 5070 Ti multi-stream target, **YOLOv11n is recommended**.

## Next Actions
1. Integrate YOLOv11n detector into `video_processor` multi-stream pipeline with batch-friendly API.
2. Keep MediaPipe implementation accessible for CPU fallback mode and on-device inference scenarios.
3. Document detector toggle options (CLI flag or config) for future benchmarking.
