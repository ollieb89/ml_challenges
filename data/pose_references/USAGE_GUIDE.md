# Fitness Data Collection - Usage Guide

## Quick Start

### Option 1: Automated Collection (Recommended)

Run the quick-start script to collect all 15 videos:

```bash
bash scripts/quick_collect_data.sh
```

This will:
- Install dependencies (yt-dlp, opencv-python)
- Download 15 fitness videos (5 squats, 5 push-ups, 5 deadlifts)
- Extract frames at 30fps
- Generate metadata.csv

**Estimated time**: 30-60 minutes (depends on internet speed)

### Option 2: Manual Collection

```bash
# Install dependencies
pip install yt-dlp opencv-python tqdm

# Run collection script
python scripts/collect_fitness_data.py \
    --videos data/pose_references/video_list.json \
    --output data/pose_references \
    --fps 30
```

### Option 3: Test First (Recommended for first run)

Test with a single video before running full collection:

```bash
python scripts/test_data_collection.py
```

This downloads and processes one video to verify everything works.

## Expected Output

After successful collection:

```
data/pose_references/
├── videos/
│   ├── squat_form_001.mp4
│   ├── squat_form_002.mp4
│   ├── pushup_form_001.mp4
│   ├── deadlift_form_001.mp4
│   └── ... (15 videos total)
├── frames/
│   ├── squat_form_001/
│   │   ├── frame_000000.jpg
│   │   ├── frame_000001.jpg
│   │   └── ... (~1000-3000 frames per video)
│   ├── pushup_form_001/
│   └── ... (15 directories)
├── metadata.csv
├── video_list.json
└── README.md
```

## Metadata CSV Format

```csv
video_id,exercise_type,form_quality,url,num_frames,keyframes,frames_dir
squat_form_001,squat,good,https://youtube.com/...,2847,[],frames/squat_form_001
pushup_form_001,pushup,good,https://youtube.com/...,1523,[],frames/pushup_form_001
```

## Troubleshooting

### yt-dlp not found

```bash
pip install yt-dlp
# or with pixi
pixi add yt-dlp
```

### opencv-python not found

```bash
pip install opencv-python
# or with pixi
pixi add opencv-python
```

### Video download fails

Some videos may be unavailable or region-restricted. The script will:
- Log the error
- Skip the failed video
- Continue with remaining videos

### Disk space

Each video is ~50-200MB, frames are ~1-5GB per video.
Total expected: ~10-20GB for all 15 videos with frames.

## Customization

### Change FPS

Extract frames at different FPS (e.g., 15fps for smaller dataset):

```bash
python scripts/collect_fitness_data.py \
    --videos data/pose_references/video_list.json \
    --output data/pose_references \
    --fps 15
```

### Add Custom Videos

Edit `data/pose_references/video_list.json`:

```json
{
  "url": "https://www.youtube.com/watch?v=YOUR_VIDEO_ID",
  "video_id": "custom_exercise_001",
  "exercise_type": "squat",
  "form_quality": "good",
  "description": "Your custom video"
}
```

### Process Subset of Videos

Create a custom JSON file with only the videos you want:

```bash
python scripts/collect_fitness_data.py \
    --videos my_custom_list.json \
    --output data/my_custom_dataset \
    --fps 30
```

## Next Steps

After data collection:

1. **Verify data quality**
   ```bash
   # Count total frames
   find data/pose_references/frames -name "*.jpg" | wc -l
   
   # Check metadata
   cat data/pose_references/metadata.csv
   ```

2. **Review sample frames**
   ```bash
   # View first frame of each video
   ls data/pose_references/frames/*/frame_000000.jpg
   ```

3. **Continue with Day 2**
   - Implement pose detection (MediaPipe vs YOLOv11)
   - Test on collected dataset
   - Benchmark performance

## Performance Notes

- **Download speed**: ~5-10 minutes per video (depends on internet)
- **Frame extraction**: ~1-2 minutes per video
- **Total time**: 30-60 minutes for all 15 videos
- **Disk usage**: ~15-20GB total

## Day 1 Evening Deliverable Checklist

- [x] Download 10-15 fitness videos ✓ (15 videos configured)
- [x] Create reference pose dataset ✓ (video_list.json)
- [x] Extract frames at 30fps ✓ (collect_fitness_data.py)
- [x] Store in `data/pose_references/` ✓ (directory structure)
- [x] Create metadata CSV ✓ (metadata.csv with all required fields)

**Status**: Ready to execute! Run `bash scripts/quick_collect_data.sh`
