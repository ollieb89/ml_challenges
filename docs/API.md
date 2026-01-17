# AI/ML Pipeline API Documentation

This document provides comprehensive API documentation for the AI/ML Pipeline components.

## Table of Contents

- [Shared Utils](#shared-utils)
- [GPU Optimizer](#gpu-optimizer)
- [Pose Analyzer](#pose-analyzer)
- [Examples](#examples)

## Shared Utils

### Configuration Module

#### `load_config(config_path: str) -> Dict`

Load configuration from YAML file.

**Parameters:**
- `config_path` (str): Path to configuration file

**Returns:**
- `Dict`: Configuration dictionary

**Example:**
```python
from shared_utils.config import load_config

config = load_config('config/machines.yml')
```

#### `get_machine_config(machine_name: str) -> Dict`

Get configuration for specific machine.

**Parameters:**
- `machine_name` (str): Name of the machine configuration

**Returns:**
- `Dict`: Machine-specific configuration

### Logging Module

#### `setup_logging(name: str, level: str = "INFO") -> Logger`

Setup logging with consistent formatting.

**Parameters:**
- `name` (str): Logger name
- `level` (str): Logging level (DEBUG, INFO, WARNING, ERROR)

**Returns:**
- `Logger`: Configured logger instance

**Example:**
```python
from shared_utils.logging import setup_logging

logger = setup_logging('my_component')
logger.info("Component initialized")
```

### Database Module

#### `setup_database(config: Dict) -> None`

Setup database connections and tables.

**Parameters:**
- `config` (Dict): Database configuration

#### `get_db_connection(config: Dict) -> Connection`

Get database connection.

**Parameters:**
- `config` (Dict): Database configuration

**Returns:**
- `Connection`: Database connection object

### Data Models

#### `PoseData`

Data model for pose information.

**Attributes:**
- `keypoints` (np.ndarray): 3D keypoints array
- `confidence` (float): Confidence score
- `timestamp` (datetime): Capture timestamp

#### `GPUInfo`

Data model for GPU information.

**Attributes:**
- `name` (str): GPU name
- `memory_total` (int): Total memory in bytes
- `memory_used` (int): Used memory in bytes
- `temperature` (float): GPU temperature

## GPU Optimizer

### VRAMProfiler

#### `__init__(config: Dict)`

Initialize VRAM profiler.

**Parameters:**
- `config` (Dict): Configuration dictionary

#### `profile_model(model: torch.nn.Module) -> Dict`

Profile model memory usage.

**Parameters:**
- `model` (torch.nn.Module): PyTorch model to profile

**Returns:**
- `Dict`: Memory usage statistics

**Example:**
```python
from gpu_optimizer import VRAMProfiler

profiler = VRAMProfiler(config)
memory_stats = profiler.profile_model(my_model)
print(f"Model uses {memory_stats['memory_mb']} MB")
```

#### `get_memory_info() -> Dict`

Get current GPU memory information.

**Returns:**
- `Dict`: Memory usage statistics

#### `optimize_memory(model: torch.nn.Module) -> torch.nn.Module`

Optimize model memory usage.

**Parameters:**
- `model` (torch.nn.Module): Model to optimize

**Returns:**
- `torch.nn.Module`: Optimized model

### ModelOptimizer

#### `__init__(config: Dict)`

Initialize model optimizer.

**Parameters:**
- `config` (Dict): Configuration dictionary

#### `get_optimization_recommendations(memory_profile: Dict) -> List[str]`

Get optimization recommendations based on memory profile.

**Parameters:**
- `memory_profile` (Dict): Memory usage profile

**Returns:**
- `List[str]`: List of optimization recommendations

#### `apply_optimizations(model: torch.nn.Module, optimizations: List[str]) -> torch.nn.Module`

Apply specified optimizations to model.

**Parameters:**
- `model` (torch.nn.Module): Model to optimize
- `optimizations` (List[str]): List of optimizations to apply

**Returns:**
- `torch.nn.Module`: Optimized model

## Pose Analyzer

### PoseAnalyzer

#### `__init__(config: Dict)`

Initialize pose analyzer.

**Parameters:**
- `config` (Dict): Configuration dictionary

#### `load_pose_data(data_path: str) -> List[PoseData]`

Load pose data from directory.

**Parameters:**
- `data_path` (str): Path to pose data directory

**Returns:**
- `List[PoseData]`: List of pose data objects

**Example:**
```python
from pose_analyzer import PoseAnalyzer

analyzer = PoseAnalyzer(config)
pose_data = analyzer.load_pose_data('data/pose_references/')
print(f"Loaded {len(pose_data)} poses")
```

#### `analyze_pose(pose_data: PoseData) -> Dict`

Analyze individual pose.

**Parameters:**
- `pose_data` (PoseData): Pose data to analyze

**Returns:**
- `Dict`: Analysis results including:
  - `pose_quality` (float): Quality score
  - `keypoint_accuracy` (Dict[str, float]): Per-keypoint accuracy
  - `recommendations` (List[str]): Improvement suggestions

#### `compare_poses(pose1: PoseData, pose2: PoseData) -> Dict`

Compare two poses.

**Parameters:**
- `pose1` (PoseData): First pose
- `pose2` (PoseData): Second pose

**Returns:**
- `Dict`: Comparison results including:
  - `similarity_score` (float): Similarity score (0-1)
  - `differences` (Dict[str, float]): Per-keypoint differences
  - `alignment_score` (float): Alignment quality

#### `batch_analyze(pose_data_list: List[PoseData]) -> List[Dict]`

Analyze multiple poses in batch.

**Parameters:**
- `pose_data_list` (List[PoseData]): List of poses to analyze

**Returns:**
- `List[Dict]`: List of analysis results

### PoseProcessor

#### `__init__(config: Dict)`

Initialize pose processor.

**Parameters:**
- `config` (Dict): Configuration dictionary

#### `preprocess_pose(raw_pose: np.ndarray) -> PoseData`

Preprocess raw pose data.

**Parameters:**
- `raw_pose` (np.ndarray): Raw pose keypoints

**Returns:**
- `PoseData`: Processed pose data

#### `extract_features(pose_data: PoseData) -> np.ndarray`

Extract features from pose data.

**Parameters:**
- `pose_data` (PoseData): Pose data

**Returns:**
- `np.ndarray`: Feature vector

## Examples

### Basic GPU Optimization

```python
import torch
from gpu_optimizer import VRAMProfiler, ModelOptimizer

# Initialize components
config = load_config('config/machines.yml')
profiler = VRAMProfiler(config)
optimizer = ModelOptimizer(config)

# Create model
model = torch.nn.Sequential(
    torch.nn.Linear(784, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10)
).cuda()

# Profile memory
memory_stats = profiler.profile_model(model)
print(f"Model memory usage: {memory_stats['memory_mb']} MB")

# Get recommendations
recommendations = optimizer.get_optimization_recommendations(memory_stats)
for rec in recommendations:
    print(f"- {rec}")

# Apply optimizations
optimized_model = optimizer.apply_optimizations(model, recommendations)
```

### Pose Analysis Workflow

```python
from pose_analyzer import PoseAnalyzer

# Initialize analyzer
config = load_config('config/machines.yml')
analyzer = PoseAnalyzer(config)

# Load pose data
pose_data = analyzer.load_pose_data('data/pose_references/')

# Analyze poses
for i, pose in enumerate(pose_data[:5]):  # Analyze first 5 poses
    result = analyzer.analyze_pose(pose)
    print(f"Pose {i+1}: Quality={result['pose_quality']:.2f}")
    
    if result['pose_quality'] < 0.8:
        print("  Recommendations:")
        for rec in result['recommendations']:
            print(f"    - {rec}")

# Compare poses
if len(pose_data) >= 2:
    comparison = analyzer.compare_poses(pose_data[0], pose_data[1])
    print(f"Similarity: {comparison['similarity_score']:.2f}")
```

### Batch Processing

```python
from pose_analyzer import PoseAnalyzer
import time

# Initialize analyzer
analyzer = PoseAnalyzer(config)

# Load data
pose_data = analyzer.load_pose_data('data/pose_references/')

# Batch analysis
start_time = time.time()
results = analyzer.batch_analyze(pose_data)
end_time = time.time()

print(f"Analyzed {len(pose_data)} poses in {end_time-start_time:.2f} seconds")
print(f"Average quality: {sum(r['pose_quality'] for r in results) / len(results):.2f}")
```

## Error Handling

### Common Exceptions

- `ConfigurationError`: Invalid configuration
- `GPUError`: GPU-related issues
- `PoseDataError`: Invalid pose data
- `DatabaseError`: Database connection issues

### Error Handling Example

```python
from shared_utils.exceptions import ConfigurationError, GPUError
from shared_utils.logging import setup_logging

logger = setup_logging('my_app')

try:
    config = load_config('config/machines.yml')
    profiler = VRAMProfiler(config)
except ConfigurationError as e:
    logger.error(f"Configuration error: {e}")
    sys.exit(1)
except GPUError as e:
    logger.error(f"GPU error: {e}")
    sys.exit(1)
```

## Configuration

### GPU Optimizer Configuration

```yaml
gpu_optimizer:
  memory_threshold: 0.8  # 80% memory usage threshold
  optimization_level: "medium"  # low, medium, high
  cache_size: 1000  # Number of models to cache
```

### Pose Analyzer Configuration

```yaml
pose_analyzer:
  confidence_threshold: 0.5
  keypoint_threshold: 0.3
  batch_size: 32
  model_path: "models/pose_model.pth"
```

## Performance Tips

1. **GPU Memory Management**
   - Use `torch.cuda.empty_cache()` regularly
   - Monitor memory usage with VRAMProfiler
   - Apply model optimizations when possible

2. **Pose Analysis**
   - Use batch processing for multiple poses
   - Cache processed results
   - Preprocess data consistently

3. **General**
   - Use appropriate logging levels
   - Monitor database connections
   - Implement proper error handling

## Testing

### Unit Tests

```bash
# Run all tests
pixi run pytest

# Run specific module tests
pixi run pytest projects/gpu_optimizer/tests/
pixi run pytest projects/pose_analyzer/tests/
pixi run pytest projects/shared_utils/tests/
```

### Integration Tests

```bash
# Run integration tests
pixi run pytest tests/integration/
```

### Performance Tests

```bash
# Run benchmarks
./scripts/benchmark.sh
```
