from __future__ import annotations

import logging
from typing import Dict, Optional

from prometheus_client import Gauge, Summary, start_http_server

logger = logging.getLogger(__name__)


class MetricsManager:
    """Central manager for Prometheus metrics across the pipeline.
    
    This class follows the singleton pattern to ensure consistent 
    metric registration and access.
    """

    _instance: Optional[MetricsManager] = None
    _initialized: bool = False

    def __new__(cls) -> MetricsManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, port: int = 8000) -> None:
        if self._initialized:
            return
            
        self.port = port
        self.gauges: Dict[str, Gauge] = {}
        self.summaries: Dict[str, Summary] = {}
        
        # --- Pose Analysis Metrics ---
        self.gauges["pose_fps"] = Gauge(
            "pose_fps", "Pose detector throughput in frames per second", ["stream_id"]
        )
        self.summaries["pose_latency_ms"] = Summary(
            "pose_latency_ms", "Pose detector processing latency in milliseconds", ["stream_id"]
        )
        self.gauges["anomaly_detection_rate"] = Gauge(
            "anomaly_detection_rate", "Number of anomalies detected per second", ["stream_id"]
        )

        # --- GPU Optimization Metrics ---
        self.gauges["gpu_utilization"] = Gauge(
            "gpu_utilization_percent", "Current GPU utilization percentage", ["device_id"]
        )
        self.gauges["vram_usage_mb"] = Gauge(
            "vram_usage_mb", "Current VRAM usage in megabytes", ["device_id"]
        )
        self.gauges["vram_fragmentation_ratio"] = Gauge(
            "vram_fragmentation_ratio", "Memory fragmentation ratio (inactive_split / reserved)", ["device_id"]
        )
        self.gauges["vram_reduction_percent"] = Gauge(
            "vram_reduction_percent", "VRAM reduction percentage from optimization", ["device_id", "optimization_type"]
        )
        self.gauges["gpu_batch_size"] = Gauge(
            "gpu_batch_size", "Current model batch size", ["device_id", "model_name"]
        )
        self.gauges["gpu_throughput"] = Gauge(
            "gpu_throughput_fps", "Overall GPU throughput in frames per second", ["device_id", "model_name"]
        )

        try:
            start_http_server(self.port, addr='0.0.0.0')
            logger.info(f"Prometheus metrics server started on port {self.port} (0.0.0.0)")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server on port {self.port}: {e}")

        self._initialized = True

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Update a gauge value with optional labels."""
        if name in self.gauges:
            if labels:
                self.gauges[name].labels(**labels).set(value)
            else:
                self.gauges[name].set(value)

    def observe_summary(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a value in a summary with optional labels."""
        if name in self.summaries:
            if labels:
                self.summaries[name].labels(**labels).observe(value)
            else:
                self.summaries[name].observe(value)


# Global instance
metrics = MetricsManager()
