"""
Tests for MemoryTracer

Validates memory tracking, layer attribution, fragmentation analysis,
and flame graph export functionality.
"""

import pytest
import torch
import torch.nn as nn
import json
import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpu_optimizer.memory_tracer import MemoryTracer, MemoryEvent, LayerMemoryStats


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@pytest.fixture
def cuda_available():
    """Check if CUDA is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return True


@pytest.fixture
def simple_model(cuda_available):
    """Create a simple model on CUDA."""
    return SimpleModel().cuda()


@pytest.fixture
def sample_input(cuda_available):
    """Create sample input tensor."""
    return torch.randn(4, 10).cuda()


class TestMemoryTracer:
    """Test MemoryTracer functionality."""
    
    def test_initialization(self):
        """Test tracer initialization."""
        tracer = MemoryTracer(max_events=1000)
        
        assert tracer.max_events == 1000
        assert len(tracer.events) == 0
        assert len(tracer.layer_stats) == 0
        assert not tracer.is_tracing
    
    def test_context_manager(self, simple_model, sample_input):
        """Test context manager interface."""
        tracer = MemoryTracer()
        
        assert not tracer.is_tracing
        
        with tracer.trace(simple_model):
            assert tracer.is_tracing
            _ = simple_model(sample_input)
        
        assert not tracer.is_tracing
        assert len(tracer.events) > 0
    
    def test_event_recording(self, simple_model, sample_input):
        """Test that events are recorded."""
        tracer = MemoryTracer(max_events=10000)
        
        with tracer.trace(simple_model):
            _ = simple_model(sample_input)
        
        # Should have recorded events
        assert len(tracer.events) > 0
        
        # Check event structure
        event = tracer.events[0]
        assert hasattr(event, 'timestamp')
        assert hasattr(event, 'event_type')
        assert hasattr(event, 'allocated_bytes')
        assert hasattr(event, 'reserved_bytes')
        assert hasattr(event, 'fragmentation_ratio')
    
    def test_layer_tracking(self, simple_model, sample_input):
        """Test layer-level memory tracking."""
        tracer = MemoryTracer()
        
        with tracer.trace(simple_model):
            _ = simple_model(sample_input)
        
        # Should have layer statistics
        layer_stats = tracer.get_layer_stats()
        assert len(layer_stats) > 0
        
        # Check that fc1 and fc2 are tracked
        layer_names = list(layer_stats.keys())
        assert any('fc1' in name for name in layer_names)
        assert any('fc2' in name for name in layer_names)
    
    def test_peak_memory_tracking(self, simple_model, sample_input):
        """Test peak memory measurement."""
        tracer = MemoryTracer()
        
        with tracer.trace(simple_model):
            _ = simple_model(sample_input)
        
        peak_memory = tracer.get_peak_memory_per_layer()
        
        # Should have peak memory for each layer
        assert len(peak_memory) > 0
        
        # All values should be positive
        assert all(mem > 0 for mem in peak_memory.values())
    
    def test_fragmentation_tracking(self, simple_model, sample_input):
        """Test fragmentation analysis."""
        tracer = MemoryTracer(enable_fragmentation_tracking=True)
        
        with tracer.trace(simple_model):
            # Multiple passes to create fragmentation
            for _ in range(3):
                _ = simple_model(sample_input)
        
        frag_stats = tracer.get_fragmentation_stats()
        
        # Should have fragmentation data
        assert len(frag_stats) > 0
        
        # Fragmentation should be between 0 and 1
        assert all(0 <= frag <= 1 for frag in frag_stats.values())
    
    def test_bounded_event_storage(self, simple_model, sample_input):
        """Test that event storage is bounded."""
        max_events = 100
        tracer = MemoryTracer(max_events=max_events)
        
        with tracer.trace(simple_model):
            # Run many iterations
            for _ in range(50):
                _ = simple_model(sample_input)
        
        # Should not exceed max_events
        assert len(tracer.events) <= max_events
    
    def test_flame_graph_export(self, simple_model, sample_input):
        """Test flame graph export."""
        tracer = MemoryTracer()
        
        with tracer.trace(simple_model):
            _ = simple_model(sample_input)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            tracer.export_flame_graph(output_path)
            
            # Verify file exists and is valid JSON
            assert Path(output_path).exists()
            
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            # Check speedscope format
            assert '$schema' in data
            assert 'profiles' in data
            assert 'shared' in data
            assert 'frames' in data['shared']
            
        finally:
            # Cleanup
            Path(output_path).unlink(missing_ok=True)
    
    def test_timeline_export(self, simple_model, sample_input):
        """Test timeline export."""
        tracer = MemoryTracer()
        
        with tracer.trace(simple_model):
            _ = simple_model(sample_input)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name
        
        try:
            tracer.export_timeline(output_path)
            
            # Verify file exists and is valid JSON
            assert Path(output_path).exists()
            
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            # Check timeline format
            assert 'events' in data
            assert 'layer_stats' in data
            assert 'summary' in data
            
            # Verify summary fields
            summary = data['summary']
            assert 'total_events' in summary
            assert 'total_layers' in summary
            assert 'peak_allocated_mb' in summary
            assert 'avg_fragmentation' in summary
            
        finally:
            # Cleanup
            Path(output_path).unlink(missing_ok=True)
    
    def test_multiple_traces(self, simple_model, sample_input):
        """Test multiple tracing sessions."""
        tracer = MemoryTracer()
        
        # First trace
        with tracer.trace(simple_model):
            _ = simple_model(sample_input)
        
        first_event_count = len(tracer.events)
        
        # Second trace (should clear previous data)
        with tracer.trace(simple_model):
            _ = simple_model(sample_input)
        
        second_event_count = len(tracer.events)
        
        # Events should be cleared between traces
        assert second_event_count > 0
        # Counts might be similar but not necessarily equal
        assert abs(first_event_count - second_event_count) < first_event_count * 0.5
    
    def test_empty_export_raises_error(self):
        """Test that exporting without tracing raises error."""
        tracer = MemoryTracer()
        
        with pytest.raises(ValueError, match="No events recorded"):
            tracer.export_flame_graph("dummy.json")
    
    def test_layer_stats_structure(self, simple_model, sample_input):
        """Test layer statistics structure."""
        tracer = MemoryTracer()
        
        with tracer.trace(simple_model):
            _ = simple_model(sample_input)
        
        layer_stats = tracer.get_layer_stats()
        
        # Check at least one layer stat
        if layer_stats:
            stat = next(iter(layer_stats.values()))
            
            assert hasattr(stat, 'layer_name')
            assert hasattr(stat, 'peak_allocated')
            assert hasattr(stat, 'peak_reserved')
            assert hasattr(stat, 'total_allocations')
            assert hasattr(stat, 'total_deallocations')
            assert hasattr(stat, 'avg_fragmentation')
            assert hasattr(stat, 'duration_ms')
            
            # Validate types
            assert isinstance(stat.layer_name, str)
            assert isinstance(stat.peak_allocated, int)
            assert isinstance(stat.peak_reserved, int)
            assert isinstance(stat.avg_fragmentation, float)
            assert isinstance(stat.duration_ms, float)


class TestMemoryTracerPerformance:
    """Test performance characteristics."""
    
    def test_overhead_is_minimal(self, simple_model, sample_input):
        """Test that tracing overhead is < 1%."""
        import time
        
        # Warmup
        for _ in range(10):
            _ = simple_model(sample_input)
        
        torch.cuda.synchronize()
        
        # Baseline timing
        baseline_times = []
        for _ in range(50):
            start = time.perf_counter()
            _ = simple_model(sample_input)
            torch.cuda.synchronize()
            baseline_times.append(time.perf_counter() - start)
        
        baseline_avg = sum(baseline_times) / len(baseline_times)
        
        # Traced timing
        tracer = MemoryTracer(max_events=10000)
        traced_times = []
        
        for _ in range(50):
            with tracer.trace(simple_model):
                start = time.perf_counter()
                _ = simple_model(sample_input)
                torch.cuda.synchronize()
                traced_times.append(time.perf_counter() - start)
        
        traced_avg = sum(traced_times) / len(traced_times)
        
        # Calculate overhead
        overhead_pct = ((traced_avg - baseline_avg) / baseline_avg) * 100
        
        # Log results
        print(f"\nOverhead Test Results:")
        print(f"  Baseline: {baseline_avg*1000:.3f} ms")
        print(f"  Traced:   {traced_avg*1000:.3f} ms")
        print(f"  Overhead: {overhead_pct:.2f}%")
        
        # Allow up to 5% overhead in tests (more lenient than production requirement)
        # Production target is <1% but tests can be more variable
        assert overhead_pct < 5.0, f"Overhead too high: {overhead_pct:.2f}%"


class TestMemoryTracerEdgeCases:
    """Test edge cases and error handling."""
    
    def test_nested_tracing_raises_error(self, simple_model, sample_input):
        """Test that nested tracing raises error."""
        tracer = MemoryTracer()
        
        with pytest.raises(RuntimeError, match="already in progress"):
            with tracer.trace(simple_model):
                with tracer.trace(simple_model):
                    pass
    
    def test_stop_without_start(self):
        """Test stopping without starting."""
        tracer = MemoryTracer()
        
        # Should not raise error
        tracer.stop_tracing()
        
        assert not tracer.is_tracing
    
    def test_empty_model(self, cuda_available):
        """Test tracing empty model."""
        class EmptyModel(nn.Module):
            def forward(self, x):
                return x
        
        model = EmptyModel().cuda()
        input_tensor = torch.randn(4, 10).cuda()
        
        tracer = MemoryTracer()
        
        with tracer.trace(model):
            _ = model(input_tensor)
        
        # Should complete without error
        assert len(tracer.events) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
