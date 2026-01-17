"""Pydantic schemas for GPU optimizer API requests and responses."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class GPUInfo(BaseModel):
    """GPU information response model."""
    
    gpu_id: int = Field(..., description="GPU device ID")
    name: str = Field(..., description="GPU device name")
    memory_total: int = Field(..., description="Total memory in MB")
    memory_used: int = Field(..., description="Used memory in MB")
    memory_free: int = Field(..., description="Free memory in MB")
    utilization: float = Field(..., description="GPU utilization percentage")
    temperature: Optional[float] = Field(None, description="GPU temperature in Celsius")
    power_usage: Optional[float] = Field(None, description="Power usage in Watts")


class MemoryProfileRequest(BaseModel):
    """Memory profiling request model."""
    
    model_name: str = Field(..., description="Name of the model to profile")
    input_shape: List[int] = Field(..., description="Input tensor shape")
    batch_size: int = Field(default=1, description="Batch size for profiling")


class MemoryProfileResponse(BaseModel):
    """Memory profiling response model."""
    
    model_name: str = Field(..., description="Model name that was profiled")
    peak_memory_gb: float = Field(..., description="Peak memory usage in GB")
    input_shape: List[int] = Field(..., description="Input tensor shape used")
    batch_size: int = Field(..., description="Batch size used")
    profiling_time_ms: float = Field(..., description="Profiling duration in milliseconds")


class OptimizationRequest(BaseModel):
    """GPU optimization request model."""
    
    target_memory_gb: float = Field(..., description="Target memory usage in GB")
    optimization_level: str = Field(default="medium", description="Optimization level: low, medium, high")
    preserve_accuracy: bool = Field(default=True, description="Whether to preserve model accuracy")


class OptimizationResponse(BaseModel):
    """GPU optimization response model."""
    
    optimized: bool = Field(..., description="Whether optimization was successful")
    memory_saved_gb: float = Field(..., description="Memory saved in GB")
    optimization_time_ms: float = Field(..., description="Optimization duration in milliseconds")
    recommendations: List[str] = Field(..., description="Optimization recommendations")


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Service status")
    gpu_monitoring_available: bool = Field(..., description="GPU monitoring availability")
    gpu_count: int = Field(..., description="Number of available GPUs")
    cuda_available: bool = Field(..., description="CUDA availability")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: float = Field(..., description="Error timestamp")