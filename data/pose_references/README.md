# Pose Reference Dataset

This directory contains fitness videos and extracted frames for pose analysis reference dataset.

## Directory Structure

```
pose_references/
├── videos/              # Downloaded fitness videos (MP4)
├── frames/              # Extracted frames at 30fps
│   ├── squat_form_001/
│   │   ├── frame_000000.jpg
│   │   ├── frame_000001.jpg
│   │   └── ...
│   ├── pushup_form_001/
│   └── ...
├── metadata.csv         # Video metadata and frame information
├── video_list.json      # Source video URLs and configuration
└── README.md           # This file
```

## Metadata CSV Format

The `metadata.csv` file contains the following columns:

- **video_id**: Unique identifier for the video
- **exercise_type**: Type of exercise (squat, pushup, deadlift)
- **form_quality**: Quality of form (good, bad, mixed)
- **url**: Original video URL
- **num_frames**: Number of extracted frames
- **keyframes**: JSON array of important frame indices
- **frames_dir**: Relative path to frames directory

## Usage

### 1. Install Dependencies

```bash
# Install yt-dlp for video downloading
pip install yt-dlp

# Install OpenCV for frame extraction
pip install opencv-python

# Or use pixi (recommended)
pixi add yt-dlp opencv-python
```

### 2. Collect Data

Run the data collection script:

```bash
python scripts/collect_fitness_data.py \
    --videos data/pose_references/video_list.json \
    --output data/pose_references \
    --fps 30
```

### 3. Verify Data

Check the metadata file:

```bash
cat data/pose_references/metadata.csv
```

Count extracted frames:

```bash
find data/pose_references/frames -name "*.jpg" | wc -l
```

## Video Sources

The `video_list.json` contains 15 curated fitness videos:

- **Squats**: 5 videos (proper form demonstrations)
- **Push-ups**: 5 videos (technique breakdowns)
- **Deadlifts**: 5 videos (form guides)

All videos are from YouTube and demonstrate proper exercise form.

## Frame Extraction Details

- **Target FPS**: 30 frames per second
- **Format**: JPEG (95% quality)
- **Resolution**: Original video resolution (up to 1080p)
- **Naming**: `frame_{index:06d}.jpg` (zero-padded 6 digits)

## Adding New Videos

To add new videos, edit `video_list.json`:

```json
{
  "url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "video_id": "exercise_type_###",
  "exercise_type": "squat|pushup|deadlift",
  "form_quality": "good|bad|mixed",
  "description": "Brief description",
  "keyframes": [10, 50, 100]  // Optional: important frame indices
}
```

Then re-run the collection script.

## Notes

- Videos are downloaded at up to 1080p resolution
- Frame extraction preserves aspect ratio
- Duplicate downloads are skipped automatically
- Failed downloads are logged and skipped
- Progress bars show extraction status

## Next Steps

After collecting data:

1. Review extracted frames for quality
2. Annotate keyframes (important poses)
3. Label form quality (good/bad examples)
4. Use for pose detection model training/validation
5. Create reference pose templates for form scoring
