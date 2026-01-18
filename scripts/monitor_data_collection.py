#!/usr/bin/env python3
"""
Monitor data collection progress and verify results.
"""

import json
import subprocess
from pathlib import Path


def monitor_progress():
    """Monitor the data collection progress."""
    data_dir = Path("data/pose_references")
    
    print("="*60)
    print("DATA COLLECTION MONITOR")
    print("="*60)
    
    # Check total videos in config
    with open(data_dir / "video_list.json") as f:
        video_list = json.load(f)
    total_videos = len(video_list)
    
    # Check processed videos
    videos_dir = data_dir / "videos"
    processed_videos = list(videos_dir.glob("*.mp4"))
    processed_count = len(processed_videos)
    
    print(f"\n📊 Progress Overview:")
    print(f"  Total videos: {total_videos}")
    print(f"  Processed: {processed_count}")
    print(f"  Remaining: {total_videos - processed_count}")
    print(f"  Progress: {processed_count/total_videos*100:.1f}%")
    
    # Check frames
    frames_dir = data_dir / "frames"
    if frames_dir.exists():
        frame_dirs = [d for d in frames_dir.iterdir() if d.is_dir()]
        total_frames = 0
        
        print(f"\n🖼️  Frame Extraction:")
        for frame_dir in sorted(frame_dirs):
            frame_count = len(list(frame_dir.glob("*.jpg")))
            total_frames += frame_count
            print(f"  {frame_dir.name}: {frame_count:,} frames")
        
        print(f"  Total frames: {total_frames:,}")
    
    # Check file sizes
    print(f"\n💾 Storage Usage:")
    if videos_dir.exists():
        video_size = sum(f.stat().st_size for f in processed_videos) / (1024**3)
        print(f"  Videos: {video_size:.2f} GB")
    
    if frames_dir.exists():
        frame_files = list(frames_dir.rglob("*.jpg"))
        frame_size = sum(f.stat().st_size for f in frame_files) / (1024**3)
        print(f"  Frames: {frame_size:.2f} GB")
    
    total_size = video_size + frame_size if 'video_size' in locals() and 'frame_size' in locals() else 0
    print(f"  Total: {total_size:.2f} GB")
    
    # Check if script is still running
    try:
        result = subprocess.run(
            ["pgrep", "-f", "collect_fitness_data"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"\n🔄 Status: Script still running (PID: {result.stdout.strip()})")
        else:
            print(f"\n✅ Status: Script completed")
            
            # Check for metadata.csv
            metadata_file = data_dir / "metadata.csv"
            if metadata_file.exists():
                print(f"📋 Metadata: Available")
                with open(metadata_file) as f:
                    lines = f.readlines()
                    print(f"  Entries: {len(lines)-1} videos")
            else:
                print(f"⚠️  Metadata: Not yet created")
    except Exception as e:
        print(f"\n❌ Error checking script status: {e}")
    
    print(f"\n" + "="*60)


if __name__ == "__main__":
    monitor_progress()
