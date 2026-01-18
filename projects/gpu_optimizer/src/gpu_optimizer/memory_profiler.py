import torch
import nvidia_ml_py as ml
from typing import Dict, List, Tuple, Any, Callable
import pandas as pd
from collections import defaultdict
import gc

class MemoryProfiler:
    """Track tensor memory usage across layers."""
    
    def __init__(self):
        self.snapshots = []
        self.layer_memory: Dict[str, float] = {}
        self.hooks: List[Callable] = []
        self.current_layer = 0
    
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
    
    def profile_layer_wise(self, model, input_tensor, model_name: str) -> Dict[str, float]:
        """Profile memory usage layer by layer."""
        # Clear previous measurements
        self.layer_memory.clear()
        self._remove_hooks()
        
        # Register forward hooks
        self._register_forward_hooks(model, model_name)
        
        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()
        
        # Run forward pass
        with torch.no_grad():
            _ = model(input_tensor)
        
        torch.cuda.synchronize()
        
        # Clean up hooks
        self._remove_hooks()
        
        return self.layer_memory
    
    def _register_forward_hooks(self, model, model_name: str):
        """Register forward hooks for all layers."""
        self.hooks.clear()
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(
                    self._create_forward_hook(name, model_name)
                )
                self.hooks.append(hook)
    
    def _create_forward_hook(self, layer_name: str, model_name: str):
        """Create a forward hook for memory measurement."""
        def hook(module, input, output):
            # Get current memory usage
            current_memory = torch.cuda.memory_allocated() / 1e6  # Convert to MB
            
            # Store layer memory usage
            clean_name = self._clean_layer_name(layer_name, model_name)
            self.layer_memory[clean_name] = current_memory
            
        return hook
    
    def _clean_layer_name(self, layer_name: str, model_name: str) -> str:
        """Clean and standardize layer names."""
        # Remove common prefixes and make more readable
        name = layer_name.replace(f"{model_name}.", "")
        
        # Handle different model architectures
        if "resnet" in model_name.lower():
            name = name.replace("layer", "ResNet_Layer")
        elif "vit" in model_name.lower():
            name = name.replace("encoder.layer", "ViT_Block")
            name = name.replace("embeddings", "ViT_Embed")
        elif "llama" in model_name.lower():
            name = name.replace("model.layers", "Llama_Block")
            name = name.replace("embed_tokens", "Llama_Embed")
        
        return name
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def format_memory_table(self, layer_memory: Dict[str, float], model_name: str) -> pd.DataFrame:
        """Format memory measurements as a clean table."""
        df = pd.DataFrame([
            {"layer_name": layer, "vram_mb": round(memory, 2)}
            for layer, memory in layer_memory.items()
        ])
        
        # Sort by memory usage (descending)
        df = df.sort_values("vram_mb", ascending=False)
        
        return df
