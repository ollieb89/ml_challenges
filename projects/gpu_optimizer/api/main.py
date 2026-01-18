"""FastAPI application for GPU VRAM optimization suite with tensor management."""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import create_gpu_routes, initialize_gpu_monitoring, cleanup_gpu_monitoring


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle for GPU monitoring setup."""
    # Startup
    initialize_gpu_monitoring()
    
    yield
    
    # Shutdown
    cleanup_gpu_monitoring()


def create_gpu_optimizer_app() -> FastAPI:
    """Create and configure the GPU Optimizer FastAPI application."""
    
    # Create FastAPI app
    app = FastAPI(
        title="GPU Optimizer API",
        description="GPU VRAM optimization suite with tensor management",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register GPU optimizer routes
    create_gpu_routes(app)
    
    # Add root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "GPU Optimizer API",
            "version": "0.1.0",
            "description": "GPU VRAM optimization suite with tensor management and memory tracing",
            "endpoints": {
                "gpu_list": "/api/gpus",
                "gpu_details": "/api/gpus/{gpu_id}",
                "memory_profiling": "/api/profile/memory",
                "memory_tracing": "/api/trace/memory",
                "trace_export": "/api/trace/export",
                "trace_list": "/api/trace/list",
                "trace_delete": "/api/trace/{trace_id}",
                "optimization": "/api/optimize",
                "health": "/api/health",
                "documentation": "/api/docs"
            }
        }
    
    # Add API info endpoint
    @app.get("/api/info")
    async def api_info():
        """Get detailed API information."""
        return {
            "api_version": "0.1.0",
            "features": [
                "GPU monitoring and profiling",
                "Memory usage analysis",
                "Real-time memory tracing with layer attribution",
                "Fragmentation analysis",
                "Flame graph export (speedscope format)",
                "Optimization recommendations",
                "Real-time GPU metrics",
                "Tensor memory tracking"
            ],
            "supported_operations": [
                "GPU information retrieval",
                "Memory profiling",
                "Memory tracing with <1% overhead",
                "Per-layer memory statistics",
                "Flame graph visualization export",
                "Performance optimization",
                "Resource monitoring"
            ]
        }
    
    return app


# Create the app instance
app = create_gpu_optimizer_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)