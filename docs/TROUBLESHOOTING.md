# AI/ML Pipeline Troubleshooting Guide

This guide helps you troubleshoot common issues with the AI/ML Pipeline.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Database Problems](#database-problems)
- [GPU/CUDA Issues](#gpucuda-issues)
- [Pose Analyzer Issues](#pose-analyzer-issues)
- [Performance Issues](#performance-issues)
- [Common Errors](#common-errors)
- [Getting Help](#getting-help)

## Installation Issues

### Pixi Installation Failed

**Problem:** Pixi installation fails or pixi command not found

**Solutions:**
```bash
# Try alternative installation method
curl -L https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-unknown-linux-musl.tar.gz | tar xz
sudo mv pixi /usr/local/bin/

# Or install via cargo
cargo install pixi

# Verify installation
pixi --version
```

### Python Version Mismatch

**Problem:** Python version is not 3.10+

**Solutions:**
```bash
# Check current Python version
python --version
python3 --version

# Install Python 3.10 on Ubuntu
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev

# Set Python 3.10 as default
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Verify
python --version
```

### Dependencies Installation Failed

**Problem:** `pixi install` fails with dependency conflicts

**Solutions:**
```bash
# Clear pixi cache
pixi clean

# Reinstall dependencies
pixi install --force

# Check for conflicting packages
pixi tree

# Update pixi itself
pixi self-update
```

## Database Problems

### PostgreSQL Connection Failed

**Problem:** Cannot connect to PostgreSQL database

**Symptoms:**
- `Connection refused` error
- `FATAL: database "ai_ml_pipeline" does not exist`
- `FATAL: password authentication failed for user "ai_ml_user"`

**Solutions:**

1. **Check PostgreSQL status:**
```bash
sudo systemctl status postgresql
sudo systemctl start postgresql
```

2. **Verify database exists:**
```bash
sudo -u postgres psql -l
```

3. **Recreate database:**
```bash
# Drop and recreate
sudo -u postgres psql -c "DROP DATABASE IF EXISTS ai_ml_pipeline;"
sudo -u postgres psql -c "CREATE DATABASE ai_ml_pipeline OWNER ai_ml_user;"

# Reset permissions
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE ai_ml_pipeline TO ai_ml_user;"
```

4. **Test connection:**
```bash
psql -h localhost -U ai_ml_user -d ai_ml_pipeline -c "SELECT version();"
```

### Redis Connection Failed

**Problem:** Cannot connect to Redis server

**Symptoms:**
- `Connection refused` error
- `Redis connection timeout`

**Solutions:**

1. **Check Redis status:**
```bash
sudo systemctl status redis
# or
sudo systemctl status redis-server
```

2. **Start Redis:**
```bash
sudo systemctl start redis
# or
sudo systemctl start redis-server
```

3. **Test connection:**
```bash
redis-cli ping
# Should return: PONG
```

4. **Check Redis configuration:**
```bash
redis-cli info server
```

### Database Migration Failed

**Problem:** Database migrations fail to run

**Solutions:**
```bash
# Check migration status
pixi run python -c "
import sys
sys.path.append('projects/shared_utils/src')
from shared_utils.database import check_migration_status
check_migration_status()
"

# Run migrations manually
pixi run python -c "
import sys
sys.path.append('projects/shared_utils/src')
from shared_utils.database import run_migrations
run_migrations()
"

# Reset database (WARNING: This deletes all data)
./scripts/setup_databases.sh --reset
```

## GPU/CUDA Issues

### CUDA Not Found

**Problem:** `CUDA not available` or `CUDA out of memory`

**Solutions:**

1. **Check NVIDIA drivers:**
```bash
nvidia-smi
```

2. **Check CUDA installation:**
```bash
nvcc --version
```

3. **Install CUDA (Ubuntu):**
```bash
# Download and install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

4. **Set environment variables:**
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### GPU Memory Issues

**Problem:** `CUDA out of memory` errors

**Solutions:**

1. **Monitor memory usage:**
```python
import torch
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
```

2. **Clear cache:**
```python
torch.cuda.empty_cache()
```

3. **Reduce batch size:**
```python
# In your configuration
config['batch_size'] = 16  # Reduce from 32 or 64
```

4. **Use gradient checkpointing:**
```python
model.gradient_checkpointing_enable()
```

### Multiple GPU Issues

**Problem:** Code not using the correct GPU

**Solutions:**

1. **List available GPUs:**
```python
import torch
print(f"Available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

2. **Set specific GPU:**
```bash
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export CUDA_VISIBLE_DEVICES=1  # Use second GPU
```

3. **Check current device:**
```python
print(f"Current device: {torch.cuda.current_device()}")
```

## Pose Analyzer Issues

### Model Loading Failed

**Problem:** Cannot load pose analysis model

**Symptoms:**
- `FileNotFoundError: models/pose_model.pth`
- `RuntimeError: Invalid model file`

**Solutions:**

1. **Download models:**
```bash
./scripts/download_models.py
```

2. **Check model file:**
```bash
ls -la data/models/
file data/models/pose_model.pth
```

3. **Verify model path in config:**
```yaml
pose_analyzer:
  model_path: "data/models/pose_model.pth"  # Ensure correct path
```

### Pose Data Loading Failed

**Problem:** Cannot load pose reference data

**Solutions:**

1. **Check data directory:**
```bash
ls -la data/pose_references/
```

2. **Verify data format:**
```python
import numpy as np
# Check if data files are valid numpy arrays
try:
    data = np.load('data/pose_references/sample_pose.npy')
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
except Exception as e:
    print(f"Error loading data: {e}")
```

3. **Regenerate test data:**
```bash
pixi run python scripts/generate_test_data.py
```

### Low Pose Quality Results

**Problem:** Pose analysis returns low quality scores

**Solutions:**

1. **Check confidence thresholds:**
```yaml
pose_analyzer:
  confidence_threshold: 0.5  # Try lowering this
  keypoint_threshold: 0.3   # Try lowering this
```

2. **Verify input data quality:**
```python
# Check keypoint values
pose_data = analyzer.load_pose_data('data/pose_references/')
for pose in pose_data[:3]:
    print(f"Keypoints range: [{pose.keypoints.min():.2f}, {pose.keypoints.max():.2f}]")
    print(f"Confidence: {pose.confidence:.2f}")
```

3. **Calibrate the analyzer:**
```python
# Run calibration
analyzer.calibrate(pose_data)
```

## Performance Issues

### Slow Performance

**Problem:** Code runs slowly

**Solutions:**

1. **Profile the code:**
```bash
# Run with profiling
pixi run python -m cProfile -o profile.stats your_script.py

# Analyze results
pixi run python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"
```

2. **Check GPU utilization:**
```bash
watch -n 1 nvidia-smi
```

3. **Optimize batch size:**
```python
# Experiment with different batch sizes
for batch_size in [8, 16, 32, 64]:
    start_time = time.time()
    # Run your processing
    end_time = time.time()
    print(f"Batch size {batch_size}: {end_time-start_time:.2f}s")
```

4. **Enable mixed precision:**
```python
# Use automatic mixed precision
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    # Your model code here
```

### Memory Leaks

**Problem:** Memory usage increases over time

**Solutions:**

1. **Monitor memory:**
```python
import psutil
import torch

def print_memory_usage():
    process = psutil.Process()
    print(f"RAM: {process.memory_info().rss / 1024**2:.2f} MB")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
```

2. **Clear caches regularly:**
```python
import gc
import torch

gc.collect()
torch.cuda.empty_cache()
```

3. **Check for circular references:**
```python
# Use weak references for callbacks
import weakref

class MyComponent:
    def __init__(self, callback):
        self.callback = weakref.ref(callback)
```

## Common Errors

### Import Errors

**Problem:** `ModuleNotFoundError` or `ImportError`

**Solutions:**

1. **Check Python path:**
```python
import sys
print(sys.path)
```

2. **Install missing dependencies:**
```bash
pixi install <package_name>
```

3. **Check project structure:**
```bash
# Verify you're in the right directory
pwd
ls -la projects/
```

### Permission Errors

**Problem:** `Permission denied` errors

**Solutions:**

1. **Fix script permissions:**
```bash
chmod +x scripts/*.sh
```

2. **Fix data directory permissions:**
```bash
chmod -R 755 data/
```

3. **Run with appropriate user:**
```bash
# Don't use sudo unless necessary
./scripts/setup_databases.sh  # Instead of sudo ./scripts/setup_databases.sh
```

### Configuration Errors

**Problem:** Invalid configuration files

**Solutions:**

1. **Validate YAML syntax:**
```bash
python -c "import yaml; yaml.safe_load(open('config/machines.yml'))"
```

2. **Check configuration schema:**
```python
from shared_utils.config import validate_config
validate_config('config/machines.yml')
```

3. **Use default configuration:**
```python
from shared_utils.config import get_default_config
config = get_default_config()
```

## Getting Help

### Debug Mode

Enable debug logging for more information:

```python
from shared_utils.logging import setup_logging
logger = setup_logging('debug_mode', level='DEBUG')
```

### Validation Scripts

Run validation to check your setup:

```bash
# Environment validation
./scripts/validate_env.py

# Database validation
./scripts/validate_db.py

# GPU validation
./scripts/validate_gpu.py
```

### Log Files

Check log files for detailed error information:

```bash
# View recent logs
tail -f logs/app.log

# Search for errors
grep -i error logs/app.log

# View all log files
ls -la logs/
```

### Common Debug Commands

```bash
# Check system resources
htop
df -h
free -h

# Check GPU status
nvidia-smi
nvidia-smi -q

# Check database connections
psql -h localhost -U ai_ml_user -d ai_ml_pipeline -c "SELECT version();"
redis-cli info

# Check Python environment
pixi run python --version
pixi run pip list
```

### When to Ask for Help

If you've tried the above solutions and still have issues:

1. **Gather information:**
   - Error messages (full traceback)
   - System information (OS, Python version, GPU)
   - Configuration files
   - Log files

2. **Create minimal reproduction:**
   - Simplify your code to the smallest example that shows the problem
   - Test with different data/configurations

3. **Check existing issues:**
   - Review this troubleshooting guide
   - Check the API documentation
   - Look at example code

4. **Provide context:**
   - What you were trying to do
   - What you expected to happen
   - What actually happened
   - Steps you've already tried

## Quick Reference

### Environment Check Commands

```bash
# System info
uname -a
python --version
nvidia-smi

# Services
sudo systemctl status postgresql
sudo systemctl status redis

# Database
psql -h localhost -U ai_ml_user -d ai_ml_pipeline -c "SELECT 1;"
redis-cli ping

# Project
./scripts/validate_env.py
pixi run pytest
```

### Common Fixes

```bash
# Reset environment
pixi clean
pixi install

# Reset database
./scripts/setup_databases.sh

# Clear GPU memory
pixi run python -c "import torch; torch.cuda.empty_cache()"

# Fix permissions
chmod +x scripts/*.sh
chmod -R 755 data/
```

### Performance Optimization

```python
# GPU memory
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

# Mixed precision
with torch.cuda.amp.autocast():
    # model code

# Batch processing
for batch in data_loader:
    # process batch
```
