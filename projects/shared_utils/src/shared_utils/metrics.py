"""
Metrics utilities for shared_utils package.

Provides lightweight metrics collection and reporting functionality
for monitoring and performance tracking across the ai-ml-pipeline.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time


@dataclass
class MetricPoint:
    """Single metric data point with timestamp."""
    timestamp: float
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


class SimpleMetrics:
    """Simple in-memory metrics collector."""
    
    def __init__(self, max_points: int = 1000):
        self.max_points = max_points
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
    
    def increment(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        self._counters[name] += value
        self._record_point(name, self._counters[name], tags)
    
    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        self._gauges[name] = value
        self._record_point(name, value, tags)
    
    def timing(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a timing metric."""
        self._record_point(name, value, tags)
    
    def _record_point(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric point."""
        point = MetricPoint(timestamp=time.time(), value=value, tags=tags or {})
        self._metrics[name].append(point)
    
    def get_points(self, name: str) -> List[MetricPoint]:
        """Get all points for a metric."""
        return list(self._metrics[name])
    
    def get_latest(self, name: str) -> Optional[MetricPoint]:
        """Get the latest point for a metric."""
        points = self._metrics[name]
        return points[-1] if points else None
    
    def clear(self, name: Optional[str] = None):
        """Clear metrics. If name is None, clear all metrics."""
        if name:
            self._metrics[name].clear()
            self._counters.pop(name, None)
            self._gauges.pop(name, None)
        else:
            self._metrics.clear()
            self._counters.clear()
            self._gauges.clear()


# Global metrics instance
metrics = SimpleMetrics()


# Convenience functions
def increment(name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
    """Increment a counter metric."""
    metrics.increment(name, value, tags)


def gauge(name: str, value: float, tags: Optional[Dict[str, str]] = None):
    """Set a gauge metric."""
    metrics.gauge(name, value, tags)


def timing(name: str, value: float, tags: Optional[Dict[str, str]] = None):
    """Record a timing metric."""
    metrics.timing(name, value, tags)
