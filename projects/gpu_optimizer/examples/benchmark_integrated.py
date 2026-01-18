import torch
import torchvision
import argparse
import sys
import os
import logging
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from gpu_optimizer.tensor_swapper import TensorSwapper
from gpu_optimizer.checkpoint_manager import CheckpointManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IntegrationTest")

def run_integration_test(model_name="resnet50", batch_size=32, steps=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Integration Test: {model_name} | BS={batch_size} | Device={device}")
    
    # Setup Model
    if model_name == "resnet50":
        model = torchvision.models.resnet50()
    elif model_name == "vit":
        model = torchvision.models.vit_b_16()
        
    model.to(device)
    model.train()
    
    # Setup Optimizer Components
    
    # 1. Gradient Checkpointing
    logger.info("Initializing CheckpointManager...")
    ckpt_manager = CheckpointManager(model, device=device)
    inputs = torch.randn(batch_size, 3, 224, 224, device=device, requires_grad=True)
    
    ckpt_manager.profile_and_optimize(
        inputs, 
        target_memory_reduction=0.3, # 30% reduction from checkpointing
        max_compute_overhead=0.2
    )
    
    # 2. Tensor Swapping
    logger.info("Initializing TensorSwapper...")
    # High threshold to only trigger if we still need space, or 0.0 to force test
    swapper = TensorSwapper(threshold_percent=0.7, device=device) 
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    targets = torch.randint(0, 1000, (batch_size,), device=device)
    
    logger.info("Starting training loop with BOTH optimizations...")
    
    start_time = time.time()
    max_mem = 0
    
    for i in range(steps):
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad()
        
        # Swapper Context
        with swapper.enable():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
        optimizer.step()
        
        current_max = torch.cuda.max_memory_allocated() / (1024**3)
        max_mem = max(max_mem, current_max)
        logger.info(f"Step {i+1}/{steps} complete. Peak Mem: {current_max:.2f} GB")

    duration = time.time() - start_time
    logger.info(f"Integration Test Complete.")
    logger.info(f"  Avg Latency: {(duration/steps)*1000:.2f} ms")
    logger.info(f"  Peak Memory: {max_mem:.2f} GB")
    logger.info(f"  Swapper Stats: {swapper.get_stats()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    
    run_integration_test(args.model, args.batch_size)
