import torch
import nvidia_ml_py as ml

class MemoryProfiler:
    """Track tensor memory usage across layers."""
    
    def __init__(self):
        self.snapshots = []
    
    def profile_forward_pass(self, model, input_tensor):
        """Capture memory during forward pass."""
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            output = model(input_tensor)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        return output, peak_memory, prof
