"""
Memory Tracer Demo

Demonstrates the usage of MemoryTracer for tracking PyTorch model memory usage
with layer-level attribution, fragmentation analysis, and flame graph export.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpu_optimizer.memory_tracer import MemoryTracer


class SimpleConvNet(nn.Module):
    """Simple CNN for demonstration."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Simple transformer block for demonstration."""
    
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


def demo_basic_usage():
    """Demonstrate basic memory tracing."""
    print("\n" + "="*70)
    print("Demo 1: Basic Memory Tracing")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
        return
    
    # Create model and input
    model = SimpleConvNet().cuda()
    input_tensor = torch.randn(8, 3, 32, 32).cuda()
    
    # Create tracer
    tracer = MemoryTracer(max_events=10000)
    
    # Trace execution
    print("\nTracing model execution...")
    with tracer.trace(model):
        output = model(input_tensor)
    
    # Print summary
    tracer.print_summary()
    
    # Get layer statistics
    print("\nLayer Statistics:")
    print("-" * 70)
    layer_stats = tracer.get_layer_stats()
    for name, stats in sorted(
        layer_stats.items(),
        key=lambda x: x[1].peak_allocated,
        reverse=True
    )[:10]:
        print(f"  {name[:50]:<50} {stats.peak_allocated/(1024**2):>8.2f} MB")
    
    return tracer


def demo_fragmentation_analysis():
    """Demonstrate fragmentation tracking."""
    print("\n" + "="*70)
    print("Demo 2: Fragmentation Analysis")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
        return
    
    # Create model with varying tensor sizes (causes fragmentation)
    model = SimpleConvNet().cuda()
    
    # Create tracer with fragmentation tracking
    tracer = MemoryTracer(
        max_events=10000,
        enable_fragmentation_tracking=True
    )
    
    print("\nTracing with fragmentation analysis...")
    with tracer.trace(model):
        # Multiple forward passes with different batch sizes
        for batch_size in [4, 8, 16, 8, 4]:
            input_tensor = torch.randn(batch_size, 3, 32, 32).cuda()
            _ = model(input_tensor)
    
    # Get fragmentation statistics
    print("\nFragmentation Statistics:")
    print("-" * 70)
    frag_stats = tracer.get_fragmentation_stats()
    for name, frag_ratio in sorted(
        frag_stats.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]:
        print(f"  {name[:50]:<50} {frag_ratio:>6.1%}")
    
    return tracer


def demo_transformer_tracing():
    """Demonstrate tracing a transformer model."""
    print("\n" + "="*70)
    print("Demo 3: Transformer Model Tracing")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
        return
    
    # Create transformer model
    model = TransformerBlock(d_model=512, nhead=8).cuda()
    input_tensor = torch.randn(4, 128, 512).cuda()  # (batch, seq_len, d_model)
    
    # Create tracer
    tracer = MemoryTracer(max_events=20000)
    
    print("\nTracing transformer execution...")
    with tracer.trace(model):
        output = model(input_tensor)
    
    # Print summary
    tracer.print_summary()
    
    # Show attention vs FFN memory usage
    print("\nMemory Usage by Component:")
    print("-" * 70)
    layer_stats = tracer.get_layer_stats()
    
    attention_layers = {k: v for k, v in layer_stats.items() if 'attention' in k.lower()}
    ffn_layers = {k: v for k, v in layer_stats.items() if 'ffn' in k.lower()}
    
    if attention_layers:
        attn_peak = sum(s.peak_allocated for s in attention_layers.values())
        print(f"  Attention Peak: {attn_peak/(1024**2):.2f} MB")
    
    if ffn_layers:
        ffn_peak = sum(s.peak_allocated for s in ffn_layers.values())
        print(f"  FFN Peak: {ffn_peak/(1024**2):.2f} MB")
    
    return tracer


def demo_flame_graph_export():
    """Demonstrate flame graph export."""
    print("\n" + "="*70)
    print("Demo 4: Flame Graph Export")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
        return
    
    # Create model
    model = SimpleConvNet().cuda()
    input_tensor = torch.randn(8, 3, 32, 32).cuda()
    
    # Create tracer
    tracer = MemoryTracer(max_events=10000)
    
    print("\nTracing model execution...")
    with tracer.trace(model):
        output = model(input_tensor)
    
    # Export flame graph
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    flame_graph_path = output_dir / "memory_flame_graph.json"
    timeline_path = output_dir / "memory_timeline.json"
    
    print("\nExporting visualizations...")
    tracer.export_flame_graph(str(flame_graph_path))
    tracer.export_timeline(str(timeline_path))
    
    print(f"\nFiles created:")
    print(f"  - Flame graph: {flame_graph_path}")
    print(f"  - Timeline: {timeline_path}")
    print(f"\nView flame graph at: https://www.speedscope.app/")


def demo_overhead_measurement():
    """Measure tracer overhead."""
    print("\n" + "="*70)
    print("Demo 5: Overhead Measurement")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping demo.")
        return
    
    import time
    
    model = SimpleConvNet().cuda()
    input_tensor = torch.randn(8, 3, 32, 32).cuda()
    
    # Warmup
    for _ in range(10):
        _ = model(input_tensor)
    
    torch.cuda.synchronize()
    
    # Baseline timing (no tracing)
    print("\nMeasuring baseline performance...")
    baseline_times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = model(input_tensor)
        torch.cuda.synchronize()
        baseline_times.append(time.perf_counter() - start)
    
    baseline_avg = sum(baseline_times) / len(baseline_times)
    
    # Traced timing
    print("Measuring traced performance...")
    tracer = MemoryTracer(max_events=100000)
    traced_times = []
    
    for _ in range(100):
        with tracer.trace(model):
            start = time.perf_counter()
            _ = model(input_tensor)
            torch.cuda.synchronize()
            traced_times.append(time.perf_counter() - start)
    
    traced_avg = sum(traced_times) / len(traced_times)
    
    # Calculate overhead
    overhead_pct = ((traced_avg - baseline_avg) / baseline_avg) * 100
    
    print("\nPerformance Results:")
    print("-" * 70)
    print(f"  Baseline: {baseline_avg*1000:.3f} ms")
    print(f"  Traced:   {traced_avg*1000:.3f} ms")
    print(f"  Overhead: {overhead_pct:.2f}%")
    
    if overhead_pct < 1.0:
        print(f"\n  ✓ SUCCESS: Overhead < 1% ({overhead_pct:.2f}%)")
    else:
        print(f"\n  ✗ WARNING: Overhead > 1% ({overhead_pct:.2f}%)")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("Memory Tracer Demonstration")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("\nERROR: CUDA not available. Please run on a GPU-enabled system.")
        return
    
    print(f"\nCUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch Version: {torch.__version__}")
    
    # Run demos
    demo_basic_usage()
    demo_fragmentation_analysis()
    demo_transformer_tracing()
    demo_flame_graph_export()
    demo_overhead_measurement()
    
    print("\n" + "="*70)
    print("All demos completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
