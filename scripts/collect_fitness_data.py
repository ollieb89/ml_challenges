#!/usr/bin/env python3
"""
Fitness Video Data Collection Script
Downloads fitness videos and extracts frames at 30fps for pose analysis reference dataset.

Usage:
    python collect_fitness_data.py --urls urls.txt --output data/pose_references/
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from tqdm import tqdm


class FitnessDataCollector:
    """Collects fitness videos and extracts frames for pose analysis."""

    def __init__(self, output_dir: Path, fps: int = 30):
        """
        Initialize the data collector.

        Args:
            output_dir: Directory to store videos and frames
            fps: Target frames per second for extraction
        """
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.videos_dir = self.output_dir / "videos"
        self.frames_dir = self.output_dir / "frames"
        self.metadata_file = self.output_dir / "metadata.csv"

        # Create directories
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed."""
        try:
            subprocess.run(
                ["yt-dlp", "--version"],
                capture_output=True,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: yt-dlp is not installed.")
            print("Install with: pip install yt-dlp")
            return False

    def download_video(
        self,
        url: str,
        video_id: str,
        exercise_type: str,
        form_quality: str = "good",
    ) -> Optional[Path]:
        """
        Download a video using yt-dlp.

        Args:
            url: Video URL (YouTube, etc.)
            video_id: Unique identifier for the video
            exercise_type: Type of exercise (squat, pushup, deadlift)
            form_quality: Quality of form (good, bad, mixed)

        Returns:
            Path to downloaded video or None if failed
        """
        output_path = self.videos_dir / f"{video_id}.mp4"

        if output_path.exists():
            print(f"Video {video_id} already exists, skipping download.")
            return output_path

        print(f"Downloading {video_id} ({exercise_type})...")

        try:
            cmd = [
                "yt-dlp",
                "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
                "--merge-output-format", "mp4",
                "-o", str(output_path),
                url,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            if output_path.exists():
                print(f"✓ Downloaded: {video_id}")
                return output_path
            else:
                print(f"✗ Download failed: {video_id}")
                return None

        except subprocess.CalledProcessError as e:
            print(f"✗ Error downloading {video_id}: {e.stderr}")
            return None

    def extract_frames(
        self,
        video_path: Path,
        video_id: str,
        max_frames: Optional[int] = None,
    ) -> List[str]:
        """
        Extract frames from video at target FPS.

        Args:
            video_path: Path to video file
            video_id: Unique identifier for the video
            max_frames: Maximum number of frames to extract (None = all)

        Returns:
            List of extracted frame paths
        """
        print(f"Extracting frames from {video_id}...")

        # Create directory for this video's frames
        video_frames_dir = self.frames_dir / video_id
        video_frames_dir.mkdir(exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"✗ Failed to open video: {video_path}")
            return []

        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps if original_fps > 0 else 0

        print(f"  Video: {original_fps:.2f} fps, {total_frames} frames, {duration:.2f}s")

        # Calculate frame skip interval
        frame_interval = max(1, int(original_fps / self.fps))

        extracted_frames = []
        frame_count = 0
        extracted_count = 0

        with tqdm(total=total_frames, desc=f"  Extracting") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Extract frame at target FPS
                if frame_count % frame_interval == 0:
                    frame_path = video_frames_dir / f"frame_{extracted_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    extracted_frames.append(str(frame_path.relative_to(self.output_dir)))
                    extracted_count += 1

                    if max_frames and extracted_count >= max_frames:
                        break

                frame_count += 1
                pbar.update(1)

        cap.release()
        print(f"✓ Extracted {extracted_count} frames at ~{self.fps} fps")

        return extracted_frames

    def create_metadata_entry(
        self,
        video_id: str,
        exercise_type: str,
        form_quality: str,
        url: str,
        frame_paths: List[str],
        keyframes: Optional[List[int]] = None,
    ) -> Dict:
        """
        Create metadata entry for a video.

        Args:
            video_id: Unique identifier
            exercise_type: Type of exercise
            form_quality: Quality of form
            url: Original video URL
            frame_paths: List of extracted frame paths
            keyframes: List of keyframe indices (optional)

        Returns:
            Metadata dictionary
        """
        return {
            "video_id": video_id,
            "exercise_type": exercise_type,
            "form_quality": form_quality,
            "url": url,
            "num_frames": len(frame_paths),
            "keyframes": keyframes or [],
            "frames_dir": f"frames/{video_id}",
        }

    def save_metadata(self, metadata_entries: List[Dict]) -> None:
        """Save metadata to CSV file."""
        if not metadata_entries:
            print("No metadata to save.")
            return

        fieldnames = [
            "video_id",
            "exercise_type",
            "form_quality",
            "url",
            "num_frames",
            "keyframes",
            "frames_dir",
        ]

        with open(self.metadata_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in metadata_entries:
                # Convert keyframes list to string
                entry_copy = entry.copy()
                entry_copy["keyframes"] = json.dumps(entry_copy["keyframes"])
                writer.writerow(entry_copy)

        print(f"\n✓ Metadata saved to: {self.metadata_file}")

    def process_video_list(self, video_list: List[Dict]) -> None:
        """
        Process a list of videos: download and extract frames.

        Args:
            video_list: List of dicts with keys: url, video_id, exercise_type, form_quality
        """
        metadata_entries = []

        for i, video_info in enumerate(video_list, 1):
            print(f"\n[{i}/{len(video_list)}] Processing: {video_info['video_id']}")

            # Download video
            video_path = self.download_video(
                url=video_info["url"],
                video_id=video_info["video_id"],
                exercise_type=video_info["exercise_type"],
                form_quality=video_info.get("form_quality", "good"),
            )

            if video_path is None:
                print(f"Skipping frame extraction for {video_info['video_id']}")
                continue

            # Extract frames
            frame_paths = self.extract_frames(
                video_path=video_path,
                video_id=video_info["video_id"],
            )

            if frame_paths:
                # Create metadata entry
                metadata = self.create_metadata_entry(
                    video_id=video_info["video_id"],
                    exercise_type=video_info["exercise_type"],
                    form_quality=video_info.get("form_quality", "good"),
                    url=video_info["url"],
                    frame_paths=frame_paths,
                    keyframes=video_info.get("keyframes"),
                )
                metadata_entries.append(metadata)

        # Save all metadata
        self.save_metadata(metadata_entries)

        print(f"\n{'='*60}")
        print(f"✓ Processing complete!")
        print(f"  Videos: {len(metadata_entries)}")
        print(f"  Output: {self.output_dir}")
        print(f"{'='*60}")


def load_video_list_from_file(file_path: Path) -> List[Dict]:
    """
    Load video list from JSON file.

    Expected format:
    [
        {
            "url": "https://youtube.com/watch?v=...",
            "video_id": "squat_001",
            "exercise_type": "squat",
            "form_quality": "good",
            "keyframes": [10, 50, 100]  # optional
        },
        ...
    ]
    """
    with open(file_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Collect fitness videos and extract frames for pose analysis"
    )
    parser.add_argument(
        "--videos",
        type=Path,
        required=True,
        help="JSON file with video list",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/pose_references"),
        help="Output directory for videos and frames",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target FPS for frame extraction (default: 30)",
    )

    args = parser.parse_args()

    # Initialize collector
    collector = FitnessDataCollector(output_dir=args.output, fps=args.fps)

    # Check dependencies
    if not collector.check_dependencies():
        sys.exit(1)

    # Load video list
    try:
        video_list = load_video_list_from_file(args.videos)
        print(f"Loaded {len(video_list)} videos from {args.videos}")
    except Exception as e:
        print(f"Error loading video list: {e}")
        sys.exit(1)

    # Process videos
    collector.process_video_list(video_list)


if __name__ == "__main__":
    main()
