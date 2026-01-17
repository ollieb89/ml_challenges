# AI/ML Pipeline: Pose Analysis & GPU Optimization

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive AI/ML pipeline combining real-time pose analysis with intelligent GPU optimization. This project provides production-ready tools for computer vision applications and efficient deep learning model deployment.

## 🎯 Overview

The AI/ML Pipeline is a modular, scalable system designed for:

- **Real-time Pose Analysis**: Advanced human pose detection and analysis using YOLO models
- **GPU Optimization**: Intelligent VRAM profiling and optimization for PyTorch models
- **Production APIs**: RESTful services for seamless integration
- **Developer Tools**: Comprehensive testing, monitoring, and development utilities

### Key Benefits

- 🚀 **Performance Optimized**: Intelligent GPU memory management and optimization
- 🎯 **Production Ready**: Robust APIs with comprehensive error handling
- 🔧 **Developer Friendly**: Extensive documentation, testing, and development tools
- 📊 **Monitoring Built-in**: Real-time performance metrics and health checks
- 🧩 **Modular Design**: Independent components that can be used separately or together

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AI/ML Pipeline                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   Pose Analyzer │    │  GPU Optimizer  │                │
│  │                 │    │                 │                │
│  │ • YOLO Models   │    │ • VRAM Profiler │                │
│  │ • MediaPipe     │    │ • Model Opt.    │                │
│  │ • FastAPI       │    │ • FastAPI       │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                       │                        │
│           └───────────┬───────────┘                        │
│                       │                                    │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Shared Utils & Infrastructure              │ │
│  │ • Configuration • Logging • Database • Monitoring      │ │
│  └─────────────────────────────────────────────────────────┘ │
│                           │                                │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              Data Storage & Caching                     │ │
│  │ • PostgreSQL • Redis • File Storage • Model Cache      │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## ✨ Features

### Pose Analyzer
- **Real-time Detection**: YOLO-based human pose detection with high accuracy
- **Multi-model Support**: YOLO11n/s/m models for different performance requirements
- **Pose Analysis**: Quality assessment, keypoint accuracy, and improvement recommendations
- **Batch Processing**: Efficient processing of multiple poses
- **Comparison Tools**: Pose similarity and alignment analysis

### GPU Optimizer
- **VRAM Profiling**: Real-time GPU memory usage monitoring
- **Model Optimization**: Automatic optimization recommendations and application
- **Memory Management**: Intelligent cache management and cleanup
- **Performance Metrics**: Detailed performance analytics and reporting
- **Multi-GPU Support**: Scalable across multiple GPU configurations

### Infrastructure
- **REST APIs**: Comprehensive FastAPI-based services
- **Database Integration**: PostgreSQL for persistent data, Redis for caching
- **Monitoring**: Prometheus metrics and health checks
- **Logging**: Structured logging with multiple levels
- **Configuration**: Flexible YAML-based configuration system

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA 12.0+ (optional, CPU mode available)
- PostgreSQL and Redis
- [Pixi](https://pixi.sh) package manager

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd ai-ml-pipeline

# 2. Install Pixi (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash

# 3. Install dependencies
pixi install

# 4. Setup databases and download models
./scripts/setup_databases.sh
pixi run download-models

# 5. Verify installation
pixi run validate-env
```

### Running the Services

```bash
# Start Pose Analyzer API (port 8001)
pixi run pose-api

# Start GPU Optimizer API (port 8002)
pixi run vram-api

# Or run both services in development mode
./scripts/dev_session.sh
```

### Basic Usage

```python
# Pose Analysis
import requests

# Analyze a pose
response = requests.post(
    "http://localhost:8001/analyze",
    json={"image_path": "path/to/image.jpg"}
)
result = response.json()
print(f"Pose quality: {result['pose_quality']}")

# GPU Optimization
response = requests.get("http://localhost:8002/gpu/info")
gpu_info = response.json()
print(f"GPU Memory: {gpu_info['memory_used_mb']}MB / {gpu_info['memory_total_mb']}MB")
```

## 📁 Project Structure

```
ai-ml-pipeline/
├── projects/                    # Main project packages
│   ├── gpu_optimizer/          # GPU optimization tools
│   │   ├── api/                # FastAPI application
│   │   ├── src/                # Core optimization logic
│   │   ├── tests/              # Unit and integration tests
│   │   └── examples/           # Usage examples
│   ├── pose_analyzer/          # Pose analysis components
│   │   ├── api/                # FastAPI application
│   │   ├── src/                # Core analysis logic
│   │   ├── tests/              # Unit and integration tests
│   │   └── examples/           # Usage examples
│   └── shared_utils/           # Shared utilities
│       └── src/                # Common functionality
├── notebooks/                  # Jupyter notebooks
│   ├── pose_demo.ipynb         # Pose analysis demo
│   └── vram_profiling_demo.ipynb # GPU profiling demo
├── data/                       # Data directories
│   ├── models/                 # Pre-trained models
│   ├── pose_references/        # Reference pose datasets
│   ├── test_videos/            # Test video files
│   └── cache/                  # Cached data
├── scripts/                    # Utility scripts
│   ├── setup_databases.sh      # Database setup
│   ├── download_models.py      # Model downloader
│   ├── benchmark.sh            # Performance benchmarks
│   └── dev_session.sh          # Development environment
├── docs/                       # Documentation
│   ├── SETUP.md                # Detailed setup guide
│   ├── API.md                  # API documentation
│   └── TROUBLESHOOTING.md      # Troubleshooting guide
├── config/                     # Configuration files
│   └── machines.yml            # Hardware configuration
└── tests/                      # Integration tests
```

## 🔧 Development

### Environment Setup

```bash
# Activate development environment
pixi shell -e dev

# Or use CPU-only environment
pixi shell -e cpu
```

### Code Quality

```bash
# Format code
pixi run format

# Lint code
pixi run lint

# Type checking
pixi run pyright

# Run tests
pixi run test

# Run tests with coverage
pixi run test --cov=projects
```

### Development Workflow

```bash
# Start development session with all services
./scripts/dev_session.sh

# Run notebooks
pixi run jupyter lab

# Run benchmarks
./scripts/benchmark.sh

# Sync project dependencies
pixi run sync-projects
```

## 📚 Documentation

- **[Setup Guide](docs/SETUP.md)** - Detailed installation and configuration
- **[API Documentation](docs/API.md)** - Complete API reference and examples
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[Implementation Guide](IMPLEMENTATION_PACKAGE_SUMMARY.md)** - Architecture and design decisions

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pixi run test

# Run specific package tests
pixi run pytest projects/gpu_optimizer/tests/
pixi run pytest projects/pose_analyzer/tests/

# Run with coverage
pixi run test --cov=projects --cov-report=html

# Run integration tests
pixi run pytest tests/integration/
```

### Test Coverage

The project maintains comprehensive test coverage:
- Unit tests for all core components
- Integration tests for API endpoints
- Performance benchmarks
- GPU/CPU compatibility tests

## 📊 Performance

### Benchmarks

```bash
# Run performance benchmarks
./scripts/benchmark.sh

# GPU profiling demo
pixi run jupyter notebook notebooks/vram_profiling_demo.ipynb

# Pose analysis demo
pixi run jupyter notebook notebooks/pose_demo.ipynb
```

### Expected Performance

- **Pose Detection**: < 50ms per frame (GPU), < 200ms per frame (CPU)
- **GPU Profiling**: < 1ms overhead for memory queries
- **API Response**: < 100ms for standard endpoints
- **Memory Usage**: Optimized models use 30-50% less VRAM

## 🔌 API Endpoints

### Pose Analyzer (Port 8001)

```bash
# Health check
GET /health

# Analyze pose from image
POST /analyze
Content-Type: application/json
{
  "image_path": "path/to/image.jpg",
  "model": "yolo11n"
}

# Batch analysis
POST /batch-analyze
Content-Type: application/json
{
  "image_paths": ["img1.jpg", "img2.jpg"],
  "model": "yolo11s"
}

# Compare poses
POST /compare
Content-Type: application/json
{
  "pose1_id": "pose_001",
  "pose2_id": "pose_002"
}
```

### GPU Optimizer (Port 8002)

```bash
# Health check
GET /health

# GPU information
GET /gpu/info

# Profile model memory
POST /profile
Content-Type: application/json
{
  "model_path": "path/to/model.pth"
}

# Get optimization recommendations
POST /recommendations
Content-Type: application/json
{
  "memory_profile": {...}
}

# Apply optimizations
POST /optimize
Content-Type: application/json
{
  "model_path": "path/to/model.pth",
  "optimizations": ["quantization", "pruning"]
}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pixi run test`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style

- Use `ruff` for formatting and linting
- Follow PEP 8 style guidelines
- Add type hints for all functions
- Include docstrings for public APIs
- Write tests for new functionality

## 📋 Requirements

### System Requirements

- **OS**: Linux (Ubuntu 20.04+) or macOS
- **Python**: 3.10 or higher
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: Minimum 50GB free space
- **GPU**: NVIDIA GPU with CUDA support (recommended)

### Software Dependencies

- **Python 3.10+**
- **PostgreSQL** (for persistent storage)
- **Redis** (for caching)
- **NVIDIA CUDA Toolkit 12.0+** (optional, for GPU support)

### Python Packages

Key dependencies include:
- PyTorch 2.7.1 (with CUDA/CPU support)
- FastAPI 0.104+
- OpenCV 4.10+
- MediaPipe 0.10+
- Ultralytics 8.4+
- NumPy, SciPy, Pandas
- pytest, ruff, pyright (development)

## 🐛 Troubleshooting

### Common Issues

1. **CUDA not found**
   ```bash
   # Check NVIDIA drivers
   nvidia-smi
   
   # Check CUDA installation
   nvcc --version
   
   # Use CPU mode if no GPU
   pixi shell -e cpu
   ```

2. **Database connection failed**
   ```bash
   # Check PostgreSQL status
   sudo systemctl status postgresql
   
   # Test connection
   psql -h localhost -U ai_ml_user -d ai_ml_pipeline
   ```

3. **Memory issues**
   ```bash
   # Clear GPU cache
   pixi run python -c "import torch; torch.cuda.empty_cache()"
   
   # Check memory usage
   pixi run detect-gpu
   ```

For more troubleshooting, see the [Troubleshooting Guide](docs/TROUBLESHOOTING.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [YOLO](https://ultralytics.com/yolo) for pose detection models
- [MediaPipe](https://mediapipe.dev) for computer vision utilities
- [FastAPI](https://fastapi.tiangolo.com) for the web framework
- [Pixi](https://pixi.sh) for package management

## 📞 Support

- 📧 Email: buitelaarolivier@gmail.com
- 📖 Documentation: [docs/](docs/)
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/your-repo/discussions)

---

**Built with ❤️ for the AI/ML community**