#!/usr/bin/env python3
"""
Day 7 Challenge: 4-Stream Concurrent Pose Detection Benchmark

Success Criteria:
- 4x 1080p streams on RTX 5070 Ti
- <100ms average latency
- <11GB VRAM usage

Run with:
    pixi run -e cuda python scripts/benchmark_4stream.py
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "projects" / "pose_analyzer" / "src"))

from pose_analyzer.pose_detector import YOLOPosev11Detector
from pose_analyzer.concurrent_stream_processor import (
    ConcurrentStreamProcessor,
    SyntheticFrameSource,
    VideoFrameSource,
)


def find_test_videos(num_needed: int = 4) -> list[Path]:
    """Find available test videos in the project."""
    video_dirs = [
        PROJECT_ROOT / "data" / "pose_references" / "videos",
        PROJECT_ROOT / "data" / "videos",
        PROJECT_ROOT / "data",
    ]
    
    videos = []
    for vdir in video_dirs:
        if vdir.exists():
            videos.extend(vdir.glob("*.mp4"))
            videos.extend(vdir.glob("*.webm"))
            videos.extend(vdir.glob("*.avi"))
    
    return videos[:num_needed]


def main():
    parser = argparse.ArgumentParser(
        description="4-Stream Concurrent Pose Detection Benchmark (Day 7 Challenge)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with synthetic frames (no video files needed)
    python scripts/benchmark_4stream.py --synthetic
    
    # Run with specific videos
    python scripts/benchmark_4stream.py --videos video1.mp4 video2.mp4 video3.mp4 video4.mp4
    
    # Use TensorRT for faster inference
    python scripts/benchmark_4stream.py --synthetic --engine-path data/models/yolo11n-pose.engine
""",
    )
    
    parser.add_argument(
        "--videos",
        nargs="*",
        help="Video files to process",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic frames for testing (no video files needed)",
    )
    parser.add_argument(
        "--synthetic-frames",
        type=int,
        default=300,
        help="Number of synthetic frames per stream (default: 300)",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="1920x1080",
        help="Synthetic frame resolution (default: 1920x1080)",
    )
    parser.add_argument(
        "--num-streams",
        type=int,
        default=4,
        help="Number of concurrent streams (default: 4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Max batch size for inference (default: 8)",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=0.0,
        help="Limit input stream FPS to simulate real-time source (0=unlimited)",
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        default="n",
        choices=["n", "s", "m", "l"],
        help="YOLO model variant (default: n = nano)",
    )
    parser.add_argument(
        "--engine-path",
        type=str,
        help="TensorRT engine path for faster inference",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="projects/pose_analyzer/reports/day7_4stream_benchmark.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--use-compile",
        action="store_true",
        help="Enable torch.compile optimization (PyTorch 2.0+)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed per-stream output",
    )
    
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split("x"))
    
    print("=" * 70)
    print("4-STREAM CONCURRENT POSE DETECTION BENCHMARK")
    print("=" * 70)
    print()
    
    # Initialize detector
    print(f"[1/3] Initializing YOLO-Pose detector (variant={args.model_variant})...")
    engine_path = Path(args.engine_path) if args.engine_path else None
    if engine_path and engine_path.exists():
        print(f"      Using TensorRT engine: {engine_path}")
    if args.use_compile:
        print("      Using torch.compile(dynamic=True) optimization")
    
    detector = YOLOPosev11Detector(
        model_variant=args.model_variant,
        engine_path=engine_path,
        use_compile=args.use_compile,
    )
    print(f"      Device: {detector.device}")
    print(f"      Is TensorRT: {detector.is_tensorrt}")
    print()
    
    # Create frame sources
    print(f"[2/3] Creating {args.num_streams} frame sources...")
    sources = []
    
    if args.videos:
        # Use provided videos
        for i, video in enumerate(args.videos[:args.num_streams]):
            vpath = Path(video)
            if vpath.exists():
                print(f"      Stream {i}: {vpath.name}")
                sources.append(VideoFrameSource(
                    vpath, 
                    max_frames=args.synthetic_frames,
                    target_fps=args.target_fps
                ))
            else:
                print(f"      Stream {i}: [MISSING] {video} → Using synthetic")
                sources.append(SyntheticFrameSource(
                    num_frames=args.synthetic_frames,
                    resolution=(width, height),
                    target_fps=args.target_fps,
                ))
    elif not args.synthetic:
        # Try to find videos automatically
        found_videos = find_test_videos(args.num_streams)
        if found_videos:
            print(f"      Found {len(found_videos)} test videos")
            for i, vpath in enumerate(found_videos):
                print(f"      Stream {i}: {vpath.name}")
                sources.append(VideoFrameSource(
                    vpath, 
                    max_frames=args.synthetic_frames,
                    target_fps=args.target_fps
                ))
    
    # Fill remaining with synthetic
    while len(sources) < args.num_streams:
        i = len(sources)
        print(f"      Stream {i}: Synthetic {width}x{height} ({args.synthetic_frames} frames)")
        sources.append(SyntheticFrameSource(
            num_frames=args.synthetic_frames,
            resolution=(width, height),
            target_fps=args.target_fps,
        ))
    
    print()
    
    # Run benchmark
    print(f"[3/3] Running {len(sources)}-stream benchmark...")
    print(f"      Batch size: {args.batch_size}")
    print(f"      Resolution: {width}x{height}")
    print(f"      Frames per stream: {args.synthetic_frames}")
    print()
    
    processor = ConcurrentStreamProcessor(
        detector,
        max_batch_size=args.batch_size,
    )
    
    result = processor.run(sources)
    
    # Print results
    print()
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    
    # Overall status
    if result.success:
        print("Status: ✅ ALL TARGETS MET")
    else:
        print("Status: ❌ SOME TARGETS MISSED")
    print()
    
    # Performance metrics
    print("PERFORMANCE:")
    lat_status = "✅" if result.latency_target_met else "❌"
    print(f"  Avg latency:  {result.avg_latency_ms:7.2f}ms  {lat_status} (target: <100ms)")
    print(f"  P95 latency:  {result.p95_latency_ms:7.2f}ms")
    print(f"  P99 latency:  {result.p99_latency_ms:7.2f}ms")
    print(f"  Max latency:  {result.max_latency_ms:7.2f}ms")
    print(f"  Throughput:   {result.total_throughput_fps:7.1f} FPS (total across all streams)")
    print()
    
    # Memory metrics
    print("MEMORY:")
    vram_status = "✅" if result.vram_target_met else "❌"
    print(f"  Peak VRAM:    {result.vram_peak_mb:7.1f}MB  {vram_status} (target: <11000MB)")
    print(f"  Avg VRAM:     {result.vram_avg_mb:7.1f}MB")
    print()
    
    # Streams
    print("STREAMS:")
    streams_status = "✅" if result.streams_target_met else "❌"
    print(f"  Active:       {result.total_streams:7}     {streams_status} (target: 4)")
    print(f"  Total frames: {result.total_frames_processed:7}")
    print(f"  Duration:     {result.duration_sec:7.2f}s")
    print()
    
    # Per-stream breakdown
    if args.verbose:
        print("PER-STREAM BREAKDOWN:")
        print("-" * 50)
        for sid in sorted(result.per_stream_metrics.keys()):
            m = result.per_stream_metrics[sid]
            drop_pct = (m.frames_dropped / (m.frames_processed + m.frames_dropped) * 100) if (m.frames_processed + m.frames_dropped) > 0 else 0
            print(f"  Stream {sid}:")
            print(f"    Processed:  {m.frames_processed}, Dropped: {m.frames_dropped} ({drop_pct:.1f}%)")
            print(f"    GPU lat:    {m.avg_latency_ms:.2f}ms, P99: {m.p99_latency_ms:.2f}ms")
            print(f"    E2E lat:    {m.avg_e2e_latency_ms:.2f}ms (includes queue wait)")
            print(f"    Throughput: {m.throughput_fps:.1f} FPS")
        print()
    
    # Success criteria summary
    print("=" * 70)
    print("SUCCESS CRITERIA VALIDATION")
    print("=" * 70)
    print(f"  {'✅' if result.streams_target_met else '❌'} 4 streams running concurrently")
    print(f"  {'✅' if result.latency_target_met else '❌'} Average latency <100ms ({result.avg_latency_ms:.2f}ms)")
    print(f"  {'✅' if result.vram_target_met else '❌'} Peak VRAM <11GB ({result.vram_peak_mb/1000:.2f}GB)")
    print()
    
    # Final verdict
    if result.success:
        print("🎉 DAY 7 CHALLENGE: PASSED!")
    else:
        print("⚠️  DAY 7 CHALLENGE: NEEDS OPTIMIZATION")
        if not result.latency_target_met:
            print("    → Consider: TensorRT engine, smaller batch size, faster model variant")
        if not result.vram_target_met:
            print("    → Consider: Smaller model, reduce batch size, enable gradient checkpointing")
    print()
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    
    print(f"Results saved to: {output_path}")
    
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
