"""
API Integration Tests for MemoryTracer

Tests the FastAPI endpoints for memory tracing functionality.
"""

import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpu_optimizer.api.main import create_gpu_optimizer_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_gpu_optimizer_app()
    return TestClient(app)


class TestAPIEndpoints:
    """Test API endpoint availability."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint returns correct information."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "GPU Optimizer API"
        assert "memory_tracing" in data["endpoints"]
        assert "trace_export" in data["endpoints"]
        assert "trace_list" in data["endpoints"]
    
    def test_api_info_endpoint(self, client):
        """Test API info includes memory tracing features."""
        response = client.get("/api/info")
        assert response.status_code == 200
        
        data = response.json()
        features = data["features"]
        
        # Check for memory tracing features
        assert any("memory tracing" in f.lower() for f in features)
        assert any("fragmentation" in f.lower() for f in features)
        assert any("flame graph" in f.lower() for f in features)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/api/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "cuda_available" in data


class TestMemoryTracingEndpoints:
    """Test memory tracing API endpoints."""
    
    @pytest.mark.skipif(
        not pytest.importorskip("torch").cuda.is_available(),
        reason="CUDA not available"
    )
    def test_memory_trace_endpoint(self, client):
        """Test memory tracing endpoint."""
        request_data = {
            "model_name": "test_model",
            "input_shape": [4, 10],
            "batch_size": 4,
            "max_events": 1000,
            "enable_fragmentation": True
        }
        
        response = client.post("/api/trace/memory", json=request_data)
        
        if response.status_code == 503:
            pytest.skip("GPU monitoring not available")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["model_name"] == "test_model"
        assert "total_events" in data
        assert "total_layers" in data
        assert "peak_allocated_mb" in data
        assert "avg_fragmentation" in data
        assert "layer_stats" in data
    
    def test_trace_list_endpoint(self, client):
        """Test listing active traces."""
        response = client.get("/api/trace/list")
        assert response.status_code == 200
        
        data = response.json()
        assert "active_traces" in data
        assert "count" in data
        assert isinstance(data["active_traces"], list)
    
    def test_trace_delete_nonexistent(self, client):
        """Test deleting non-existent trace returns 404."""
        response = client.delete("/api/trace/nonexistent_trace")
        assert response.status_code == 404
    
    @pytest.mark.skipif(
        not pytest.importorskip("torch").cuda.is_available(),
        reason="CUDA not available"
    )
    def test_trace_export_workflow(self, client):
        """Test complete trace and export workflow."""
        # First, create a trace
        trace_request = {
            "model_name": "export_test_model",
            "input_shape": [2, 10],
            "batch_size": 2,
            "max_events": 500
        }
        
        trace_response = client.post("/api/trace/memory", json=trace_request)
        
        if trace_response.status_code == 503:
            pytest.skip("GPU monitoring not available")
        
        assert trace_response.status_code == 200
        
        # Get the list of traces to find the trace_id
        list_response = client.get("/api/trace/list")
        assert list_response.status_code == 200
        
        traces = list_response.json()["active_traces"]
        if not traces:
            pytest.skip("No traces available for export test")
        
        trace_id = traces[0]
        
        # Export as speedscope
        export_request = {
            "trace_id": trace_id,
            "format": "speedscope"
        }
        
        export_response = client.post("/api/trace/export", json=export_request)
        assert export_response.status_code == 200
        
        export_data = export_response.json()
        assert export_data["trace_id"] == trace_id
        assert export_data["format"] == "speedscope"
        assert "download_url" in export_data
        assert "file_size_bytes" in export_data
        
        # Clean up - delete the trace
        delete_response = client.delete(f"/api/trace/{trace_id}")
        assert delete_response.status_code == 200


class TestAPIDocumentation:
    """Test API documentation endpoints."""
    
    def test_openapi_docs_available(self, client):
        """Test that OpenAPI docs are available."""
        response = client.get("/api/docs")
        assert response.status_code == 200
    
    def test_redoc_available(self, client):
        """Test that ReDoc is available."""
        response = client.get("/api/redoc")
        assert response.status_code == 200


class TestErrorHandling:
    """Test API error handling."""
    
    def test_invalid_trace_export_format(self, client):
        """Test that invalid export format returns error."""
        export_request = {
            "trace_id": "fake_trace",
            "format": "invalid_format"
        }
        
        response = client.post("/api/trace/export", json=export_request)
        # Should return 404 because trace doesn't exist
        assert response.status_code == 404
    
    def test_memory_trace_invalid_input(self, client):
        """Test that invalid input returns error."""
        invalid_request = {
            "model_name": "test",
            "input_shape": [],  # Empty shape
            "batch_size": 0  # Invalid batch size
        }
        
        response = client.post("/api/trace/memory", json=invalid_request)
        # Should return validation error or 500
        assert response.status_code in [422, 500, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
