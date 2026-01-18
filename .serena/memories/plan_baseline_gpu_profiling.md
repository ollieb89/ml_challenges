# Plan: Baseline GPU Memory Profiling Script Implementation

## Objective
Implement a baseline GPU memory profiling script that measures memory per layer for:
- ResNet50 (inference)
- ViT-Base (inference) 
- Llama-7B (inference, no quantization)

## Output Format
layer_name → VRAM_MB table

## Project Context
- Working within existing gpu_optimizer package
- Project uses pixi for dependency management
- Has CUDA support via pixi features
- Existing memory_profiler.py file (702 bytes) - need to examine

## Implementation Phases

### Phase 1: Analysis and Setup
- Examine existing memory_profiler.py implementation
- Check available dependencies and models
- Verify CUDA environment setup
- Plan the profiling approach

### Phase 2: Core Implementation
- Design layer-wise memory profiling mechanism
- Implement model loading and inference setup
- Create memory measurement utilities
- Build output formatting for layer_name → VRAM_MB table

### Phase 3: Model-Specific Implementations
- Implement ResNet50 profiling
- Implement ViT-Base profiling  
- Implement Llama-7B profiling (no quantization)
- Ensure consistent measurement approach across models

### Phase 4: Testing and Validation
- Test script with available GPU
- Validate memory measurements
- Verify output format
- Add error handling and logging

## Key Technical Considerations
- Use PyTorch for model loading and inference
- Leverage nvidia-ml-py for GPU memory monitoring
- Implement proper memory cleanup between measurements
- Handle different model architectures consistently
- Ensure reproducible measurements

## Success Criteria
- Script runs without errors on CUDA-enabled system
- Produces layer_name → VRAM_MB table for all three models
- Measurements are consistent and repeatable
- Code follows Python AI/ML development best practices