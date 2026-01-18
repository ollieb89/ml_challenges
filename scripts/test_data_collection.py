#!/usr/bin/env python3
"""
Test script for fitness data collection workflow.
Tests with a single video to verify the pipeline works.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.collect_fitness_data import FitnessDataCollector


def test_single_video():
    """Test the data collection workflow with a single video."""
    
    print("="*60)
    print("Testing Fitness Data Collection Workflow")
    print("="*60)
    print()
    
    # Create test output directory
    test_output = project_root / "data" / "pose_references_test"
    test_output.mkdir(exist_ok=True)
    
    # Initialize collector
    collector = FitnessDataCollector(output_dir=test_output, fps=30)
    
    # Check dependencies
    print("1. Checking dependencies...")
    if not collector.check_dependencies():
        print("✗ Dependencies check failed")
        print("Install yt-dlp: pip install yt-dlp")
        return False
    print("✓ Dependencies OK")
    print()
    
    # Test with a single short video
    test_video = {
        "url": "https://www.youtube.com/watch?v=ultWZbUMPL8",
        "video_id": "test_squat_001",
        "exercise_type": "squat",
        "form_quality": "good",
        "description": "Test video - squat form"
    }
    
    print("2. Testing video download...")
    video_path = collector.download_video(
        url=test_video["url"],
        video_id=test_video["video_id"],
        exercise_type=test_video["exercise_type"],
        form_quality=test_video["form_quality"],
    )
    
    if video_path is None:
        print("✗ Video download failed")
        return False
    print(f"✓ Video downloaded: {video_path}")
    print()
    
    print("3. Testing frame extraction...")
    frame_paths = collector.extract_frames(
        video_path=video_path,
        video_id=test_video["video_id"],
        max_frames=100,  # Limit to 100 frames for testing
    )
    
    if not frame_paths:
        print("✗ Frame extraction failed")
        return False
    print(f"✓ Extracted {len(frame_paths)} frames")
    print()
    
    print("4. Testing metadata creation...")
    metadata = collector.create_metadata_entry(
        video_id=test_video["video_id"],
        exercise_type=test_video["exercise_type"],
        form_quality=test_video["form_quality"],
        url=test_video["url"],
        frame_paths=frame_paths,
    )
    
    collector.save_metadata([metadata])
    print(f"✓ Metadata saved: {collector.metadata_file}")
    print()
    
    print("="*60)
    print("✓ All tests passed!")
    print("="*60)
    print()
    print("Test output directory:", test_output)
    print("  - Videos:", test_output / "videos")
    print("  - Frames:", test_output / "frames")
    print("  - Metadata:", test_output / "metadata.csv")
    print()
    print("You can now run the full collection:")
    print("  bash scripts/quick_collect_data.sh")
    print()
    
    return True


if __name__ == "__main__":
    try:
        success = test_single_video()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
