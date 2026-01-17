"""Pose Analyzer API package initialization."""

from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .routes import create_pose_routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle for pose analysis services."""
    # Startup
    print("Starting Pose Analyzer API...")
    
    yield
    
    # Shutdown
    print("Shutting down Pose Analyzer API...")


def create_pose_analyzer_app() -> FastAPI:
    """Create and configure the Pose Analyzer FastAPI application."""
    
    # Create FastAPI app
    app = FastAPI(
        title="Pose Analyzer API",
        description="Real-time fitness form detector with multi-stream pose estimation",
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
    
    # Register pose analysis routes
    create_pose_routes(app)
    
    # Add root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Pose Analyzer API",
            "version": "0.1.0",
            "description": "Real-time fitness form detector with multi-stream pose estimation",
            "endpoints": {
                "pose_detection": "/api/pose/detect",
                "form_analysis": "/api/pose/analyze-form",
                "video_analysis": "/api/pose/analyze-video",
                "realtime": "/api/pose/realtime",
                "health": "/api/pose/health",
                "documentation": "/api/docs"
            }
        }
    
    # Add API info endpoint
    @app.get("/api/info")
    async def api_info():
        """Get detailed API information."""
        return {
            "api_version": "0.1.0",
            "supported_models": ["MediaPipe", "YOLO"],
            "features": [
                "Real-time pose detection",
                "Fitness form analysis",
                "Video processing",
                "WebSocket streaming",
                "Multi-person tracking"
            ],
            "supported_exercises": [
                "squat",
                "deadlift", 
                "pushup",
                "pullup",
                "bench_press",
                "overhead_press",
                "lunge",
                "plank"
            ]
        }
    
    return app


# Create the app instance
app = create_pose_analyzer_app()


if __name__ == "__main__":
    uvicorn.run(
        "pose_analyzer.api.app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )