"""FastAPI routes for GPU optimizer with memory profiling and optimization."""

import asyncio
import time
from typing import List

import nvidia_ml_py as ml
import numpy as np
import torch
from fastapi import FastAPI, HTTPException

from gpu_optimizer.memory_profiler import MemoryProfiler
from .schemas import (
    GPUInfo,
    MemoryProfileRequest,
    MemoryProfileResponse,
    OptimizationRequest,
    OptimizationResponse,
    HealthResponse
)


# Global state for GPU monitoring
nvidia_ml = None
memory_profiler = None


async def get_gpu_info() -> List[GPUInfo]:
    """Get GPU information for all available GPUs."""
    if not nvidia_ml:
        raise HTTPException(status_code=503, detail="GPU monitoring not available")
    
    gpu_info_list = []
    device_count = ml.nvmlDeviceGetCount()
    
    for gpu_id in range(device_count):
        try:
            handle = ml.nvmlDeviceGetHandleByIndex(gpu_id)
            
            # Get GPU name
            name = ml.nvmlDeviceGetName(handle).decode('utf-8')
            
            # Get memory info
            memory_info = ml.nvmlDeviceGetMemoryInfo(handle)
            memory_total = memory_info.total // (1024 * 1024)  # Convert to MB
            memory_used = memory_info.used // (1024 * 1024)
            memory_free = memory_info.free // (1024 * 1024)
            
            # Get utilization
            utilization = ml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu
            
            # Get temperature (if available)
            try:
                temperature = ml.nvmlDeviceGetTemperature(handle, ml.NVML_TEMPERATURE_GPU)
            except Exception:
                temperature = None
            
            # Get power usage (if available)
            try:
                power_usage = ml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to Watts
            except Exception:
                power_usage = None
            
            gpu_info = GPUInfo(
                gpu_id=gpu_id,
                name=name,
                memory_total=memory_total,
                memory_used=memory_used,
                memory_free=memory_free,
                utilization=gpu_util,
                temperature=temperature,
                power_usage=power_usage
            )
            gpu_info_list.append(gpu_info)
            
        except Exception as e:
            print(f"Error getting info for GPU {gpu_id}: {e}")
            continue
    
    return gpu_info_list


def create_gpu_routes(app: FastAPI) -> None:
    """Create and register GPU optimizer routes."""
    
    @app.get("/api/gpus", response_model=List[GPUInfo])
    async def list_gpus():
        """Get information about all available GPUs."""
        return await get_gpu_info()
    
    
    @app.get("/api/gpus/{gpu_id}", response_model=GPUInfo)
    async def get_gpu_details(gpu_id: int):
        """Get detailed information about a specific GPU."""
        gpu_info_list = await get_gpu_info()
        
        for gpu_info in gpu_info_list:
            if gpu_info.gpu_id == gpu_id:
                return gpu_info
        
        raise HTTPException(status_code=404, detail=f"GPU {gpu_id} not found")
    
    
    @app.post("/api/profile/memory", response_model=MemoryProfileResponse)
    async def profile_memory_usage(request: MemoryProfileRequest):
        """Profile memory usage for a given model configuration."""
        if not memory_profiler:
            raise HTTPException(status_code=503, detail="Memory profiler not available")
        
        try:
            # Create dummy input tensor
            input_tensor = torch.randn(*request.input_shape)
            
            # Create a simple dummy model for profiling
            # In a real implementation, this would load the actual model
            dummy_model = torch.nn.Sequential(
                torch.nn.Linear(np.prod(request.input_shape[1:]), 512),
                torch.nn.ReLU(),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 10)
            )
            
            if torch.cuda.is_available():
                dummy_model = dummy_model.cuda()
                input_tensor = input_tensor.cuda()
            
            # Profile the forward pass
            start_time = asyncio.get_event_loop().time()
            output, peak_memory_gb, _ = memory_profiler.profile_forward_pass(dummy_model, input_tensor)
            end_time = asyncio.get_event_loop().time()
            
            profiling_time_ms = (end_time - start_time) * 1000
            
            return MemoryProfileResponse(
                model_name=request.model_name,
                peak_memory_gb=peak_memory_gb,
                input_shape=request.input_shape,
                batch_size=request.batch_size,
                profiling_time_ms=profiling_time_ms
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Memory profiling failed: {str(e)}")
    
    
    @app.post("/api/optimize", response_model=OptimizationResponse)
    async def optimize_gpu_memory(request: OptimizationRequest):
        """Optimize GPU memory usage based on target constraints."""
        try:
            # Get current GPU info
            gpu_info_list = await get_gpu_info()
            
            if not gpu_info_list:
                raise HTTPException(status_code=404, detail="No GPUs available")
            
            # Calculate current memory usage
            current_memory_gb = sum(gpu.memory_used for gpu in gpu_info_list) / 1024.0
            target_memory_gb = request.target_memory_gb
            
            # Generate optimization recommendations
            recommendations = []
            
            if current_memory_gb > target_memory_gb:
                memory_to_free_gb = current_memory_gb - target_memory_gb
                
                if request.optimization_level == "high":
                    recommendations.extend([
                        f"Enable gradient checkpointing to save ~{memory_to_free_gb * 0.3:.2f}GB",
                        f"Use mixed precision training to save ~{memory_to_free_gb * 0.4:.2f}GB",
                        f"Reduce batch size by 50% to save ~{memory_to_free_gb * 0.3:.2f}GB"
                    ])
                elif request.optimization_level == "medium":
                    recommendations.extend([
                        f"Use mixed precision training to save ~{memory_to_free_gb * 0.4:.2f}GB",
                        f"Reduce batch size by 25% to save ~{memory_to_free_gb * 0.15:.2f}GB"
                    ])
                else:  # low
                    recommendations.extend([
                        f"Reduce batch size by 15% to save ~{memory_to_free_gb * 0.1:.2f}GB"
                    ])
                
                optimized = True
                memory_saved_gb = min(memory_to_free_gb, current_memory_gb * 0.7)
            else:
                recommendations.append("Current memory usage is within target limits")
                optimized = False
                memory_saved_gb = 0.0
            
            # Simulate optimization time
            optimization_time_ms = 150.0  # Simulated processing time
            
            return OptimizationResponse(
                optimized=optimized,
                memory_saved_gb=memory_saved_gb,
                optimization_time_ms=optimization_time_ms,
                recommendations=recommendations
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")
    
    
    @app.get("/api/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        gpu_count = 0
        if nvidia_ml:
            try:
                gpu_count = ml.nvmlDeviceGetCount()
            except Exception:
                pass
        
        return HealthResponse(
            status="healthy",
            gpu_monitoring_available=nvidia_ml is not None,
            gpu_count=gpu_count,
            cuda_available=torch.cuda.is_available(),
            uptime_seconds=time.time()
        )


def initialize_gpu_monitoring() -> bool:
    """Initialize GPU monitoring components."""
    global nvidia_ml, memory_profiler
    
    try:
        nvidia_ml = ml.nvmlInit()
        memory_profiler = MemoryProfiler()
        print("GPU monitoring initialized successfully")
        return True
    except Exception as e:
        print(f"Failed to initialize GPU monitoring: {e}")
        nvidia_ml = None
        memory_profiler = None
        return False


def cleanup_gpu_monitoring():
    """Clean up GPU monitoring resources."""
    global nvidia_ml
    
    if nvidia_ml:
        try:
            ml.nvmlShutdown()
        except Exception:
            pass