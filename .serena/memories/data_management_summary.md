# Data Management Summary

## Gitignore Updates Made
Added the following directories to .gitignore to prevent pushing large/generated data to GitHub:
- `data/results/` - Generated profiling results
- `data/pose_references/videos/` - Large binary video files  
- `data/pose_references/frames/` - Thousands of extracted frame images
- `data/test_videos/` - Test video directory

## What's Now Tracked (Small, Important Files)
- Documentation files (*.md) in pose_references/
- Metadata files (*.csv) - video metadata and frame information
- Configuration files (*.json) - video list and dataset configuration
- YOLO models (whitelisted) - essential ML models

## What Can Be Safely Deleted
### Videos & Frames (Large Storage Usage)
- **Location**: `data/pose_references/videos/` and `data/pose_references/frames/`
- **Size**: Significant (15 videos + thousands of extracted frames)
- **Regeneration**: Can be recreated using:
  ```bash
  python scripts/collect_fitness_data.py \
      --videos data/pose_references/video_list.json \
      --output data/pose_references \
      --fps 30
  ```

### Results Data
- **Location**: `data/results/baseline_memory_profile.csv`
- **Size**: Small
- **Regeneration**: Can be recreated by running profiling scripts

## Recommendations
1. **If storage is a concern**: Delete the videos and frames directories
2. **If you need the data soon**: Keep it locally (now properly ignored)
3. **For collaboration**: The metadata and config files are tracked, so others can regenerate the dataset

## Commands to Clean Up (Optional)
```bash
# Remove large video files (can be regenerated)
rm -rf data/pose_references/videos/
rm -rf data/pose_references/frames/

# Remove generated results (can be regenerated)  
rm -rf data/results/
```