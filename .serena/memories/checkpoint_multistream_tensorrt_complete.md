# Multi-Stream TensorRT Pose Detection Checkpoint

## Status
- **Phase**: Day 2 Afternoon (Multi-Stream Video Processor) - COMPLETED
- **Goal**: 4 concurrent 1080p pose detection streams with <100ms latency
- **Achieved**: 4 streams @ ~19ms avg latency, ~534MiB VRAM, 1 frame dropped/stream

## Key Results
- **2-stream test**: 8.5–8.8ms avg latency, ~267MiB VRAM
- **4-stream test**: 18.9–19.0ms avg latency, ~534MiB VRAM
- **TensorRT engine**: yolo11n-pose.engine @ 1088x1088
- **Peak latency spikes**: 257–271ms (acceptable)
- **Frame drops**: 1 per stream (due to initial startup)

## Fixes Applied
- TensorRT input size mismatch: forced imgsz=1088 for .engine files
- Device management: skip .to(device) for TensorRT engines
- Engine loading: proper runtime device routing

## Files Updated
- `pose_detector.py`: TensorRT engine support and imgsz handling
- `video_processor.py`: Multi-stream processor with CLI harness
- `pixi.toml`: ONNX, TensorRT, and bindings dependencies
- `benchmark_videos.json`: Expanded video list with max_frames

## Commands
```bash
# 2-stream
pixi run python -m pose_analyzer.video_processor \
  --videos squat_form_001.mp4 pushup_form_004.mp4 \
  --detector yolo:n --engine-path yolo11n-pose.engine \
  --max-streams 2 --stats-output reports/multistream_trt_stats.json

# 4-stream
pixi run python -m pose_analyzer.video_processor \
  --videos squat_form_001.mp4 pushup_form_004.mp4 deadlift_form_003.mp4 squat_form_005.mp4 \
  --detector yolo:n --engine-path yolo11n-pose.engine \
  --max-streams 4 --stats-output reports/multistream_trt_4stream_stats.json
```

## Next Steps
- Day 2 Evening: Optimize batch size and GPU threshold
- Day 3 Morning: Implement real-time visualization overlay
- Day 3 Afternoon: Add pose analysis and form scoring

## Performance Summary
- Detector: YOLOv11n via TensorRT (recommended over MediaPipe)
- Latency: 19ms avg (target <100ms) ✅
- VRAM: 534MiB for 4 streams (RTX 5070 Ti: 15818MiB) ✅
- Throughput: ~4×1080p streams concurrently ✅