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


class MemoryTraceRequest(BaseModel):
    """Memory trace request model."""
    
    model_name: str = Field(..., description="Name of the model to trace")
    input_shape: List[int] = Field(..., description="Input tensor shape")
    batch_size: int = Field(default=1, description="Batch size for tracing")
    max_events: int = Field(default=10000, description="Maximum events to capture")
    enable_fragmentation: bool = Field(default=True, description="Enable fragmentation tracking")


class LayerMemoryStats(BaseModel):
    """Layer memory statistics model."""
    
    layer_name: str = Field(..., description="Layer name")
    peak_allocated: int = Field(..., description="Peak allocated memory in bytes")
    peak_reserved: int = Field(..., description="Peak reserved memory in bytes")
    total_allocations: int = Field(..., description="Total number of allocations")
    total_deallocations: int = Field(..., description="Total number of deallocations")
    avg_fragmentation: float = Field(..., description="Average fragmentation ratio (0-1)")
    duration_ms: float = Field(..., description="Layer execution duration in milliseconds")


class MemoryTraceResponse(BaseModel):
    """Memory trace response model."""
    
    model_name: str = Field(..., description="Model name that was traced")
    total_events: int = Field(..., description="Total events captured")
    total_layers: int = Field(..., description="Total layers traced")
    peak_allocated_mb: float = Field(..., description="Peak allocated memory in MB")
    peak_reserved_mb: float = Field(..., description="Peak reserved memory in MB")
    avg_fragmentation: float = Field(..., description="Average fragmentation ratio")
    tracing_time_ms: float = Field(..., description="Tracing duration in milliseconds")
    layer_stats: Dict[str, LayerMemoryStats] = Field(..., description="Per-layer memory statistics")


class FlameGraphExportRequest(BaseModel):
    """Flame graph export request model."""
    
    trace_id: str = Field(..., description="ID of the trace to export")
    format: str = Field(default="speedscope", description="Export format (speedscope, timeline)")


class FlameGraphExportResponse(BaseModel):
    """Flame graph export response model."""
    
    trace_id: str = Field(..., description="Trace ID")
    format: str = Field(..., description="Export format")
    download_url: str = Field(..., description="URL to download the exported file")
    file_size_bytes: int = Field(..., description="File size in bytes")