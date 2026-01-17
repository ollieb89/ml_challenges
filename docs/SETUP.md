# AI/ML Pipeline Setup Guide

This guide will help you set up the AI/ML Pipeline development environment.

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended) or macOS
- **Python**: 3.10 or higher
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: Minimum 50GB free space
- **GPU**: NVIDIA GPU with CUDA support (recommended for GPU optimization features)

### Required Software

1. **Python 3.10+**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3.10 python3.10-venv python3.10-dev

   # macOS (using Homebrew)
   brew install python@3.10
   ```

2. **PostgreSQL**
   ```bash
   # Ubuntu/Debian
   sudo apt install postgresql postgresql-contrib

   # macOS
   brew install postgresql
   brew services start postgresql
   ```

3. **Redis**
   ```bash
   # Ubuntu/Debian
   sudo apt install redis-server

   # macOS
   brew install redis
   brew services start redis
   ```

4. **NVIDIA CUDA Toolkit** (for GPU support)
   - Download from [NVIDIA website](https://developer.nvidia.com/cuda-downloads)
   - Follow installation instructions for your OS

5. **Git**
   ```bash
   # Ubuntu/Debian
   sudo apt install git

   # macOS
   brew install git
   ```

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ai-ml-pipeline
```

### 2. Install Pixi (Package Manager)

```bash
# Install pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Restart your terminal or source your shell profile
source ~/.bashrc  # or ~/.zshrc
```

### 3. Install Dependencies

```bash
# Install all project dependencies
pixi install

# Verify installation
pixi run python --version
```

### 4. Setup Databases

```bash
# Run the database setup script
./scripts/setup_databases.sh
```

This script will:
- Create PostgreSQL database and user
- Start Redis service
- Run database migrations

### 5. Download Required Models

```bash
# Download pre-trained models
./scripts/download_models.py
```

### 6. Verify Installation

```bash
# Run the verification script
./scripts/validate_env.py

# Run benchmarks to test performance
./scripts/benchmark.sh
```

## Project Structure

```
ai-ml-pipeline/
├── projects/                    # Main project packages
│   ├── gpu_optimizer/          # GPU optimization tools
│   ├── pose_analyzer/          # Pose analysis components
│   └── shared_utils/           # Shared utilities
├── notebooks/                  # Jupyter notebooks for exploration
├── data/                       # Shared data directory
│   ├── pose_references/       # Reference pose datasets
│   ├── test_videos/           # Test video files
│   └── cache/                 # Cached data
├── scripts/                    # Utility scripts
├── docs/                      # Documentation
└── config/                    # Configuration files
```

## Development Workflow

### 1. Start Development Session

```bash
# Start the development environment
./scripts/dev_session.sh
```

### 2. Run Tests

```bash
# Run all tests
pixi run pytest

# Run tests for specific package
pixi run pytest projects/gpu_optimizer/tests/
```

### 3. Run Notebooks

```bash
# Start Jupyter notebook server
pixi run jupyter notebook

# Or use JupyterLab
pixi run jupyter lab
```

### 4. Code Quality

```bash
# Format code
pixi run ruff format .

# Lint code
pixi run ruff check .

# Type checking
pixi run mypy .
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Database configuration
DATABASE_URL=postgresql://ai_ml_user:ai_ml_password@localhost:5432/ai_ml_pipeline
REDIS_URL=redis://localhost:6379

# GPU configuration
CUDA_VISIBLE_DEVICES=0

# Logging
LOG_LEVEL=INFO
```

### Machine Configuration

Edit `config/machines.yml` to specify your hardware configuration:

```yaml
machines:
  workstation:
    gpu:
      available: true
      memory_gb: 12
      compute_capability: "7.5"
    cpu:
      cores: 8
      memory_gb: 32
```

## Troubleshooting

### Common Issues

1. **CUDA not found**
   - Ensure NVIDIA drivers are installed
   - Check CUDA installation: `nvcc --version`
   - Verify GPU is visible: `nvidia-smi`

2. **Database connection failed**
   - Check PostgreSQL is running: `sudo systemctl status postgresql`
   - Verify database exists: `psql -h localhost -U ai_ml_user -d ai_ml_pipeline`

3. **Redis connection failed**
   - Check Redis is running: `redis-cli ping`
   - Verify Redis configuration: `redis-cli info`

4. **Permission errors**
   - Ensure scripts are executable: `chmod +x scripts/*.sh`
   - Check file permissions in data directories

### Getting Help

- Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
- Review the [API Documentation](API.md)
- Check logs in the `logs/` directory
- Run the validation script: `./scripts/validate_env.py`

## Next Steps

1. **Explore the notebooks** in the `notebooks/` directory
2. **Run the examples** in `projects/*/examples/`
3. **Read the API documentation** in `docs/API.md`
4. **Check the implementation guide** in `IMPLEMENTATION_PACKAGE_SUMMARY.md`

## Support

For additional support:
- Check the project documentation
- Review the code comments and docstrings
- Run the help script: `./scripts/help.sh`
- Check the troubleshooting guide for common issues
