import torch
import torchvision
import argparse
import sys
import os
import logging
from torch.autograd import Variable

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from gpu_optimizer.checkpoint_manager import CheckpointManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BenchmarkCheckpoint")

def run_benchmark(model_name="resnet50", batch_size=32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Benchmarking {model_name} on {device} with BS={batch_size}")
    
    if model_name == "resnet50":
        model = torchvision.models.resnet50()
    elif model_name == "vit":
        model = torchvision.models.vit_b_16()
    else:
        raise ValueError("Unknown model")
        
    model.to(device)
    model.train() # Checkpointing relies on require_grad in backward
    
    # Input requiring grad to simulate training
    inputs = torch.randn(batch_size, 3, 224, 224, device=device, requires_grad=True)
    
    manager = CheckpointManager(model, device=device)
    
    # Step 1: Profile and Optimize
    logger.info("Starting Profile & Optimize...")
    selected = manager.profile_and_optimize(
        inputs, 
        target_memory_reduction=0.4, # 40%
        max_compute_overhead=0.2     # 20%
    )
    
    logger.info(f"Checkpointed layers: {len(selected)}")
    # logger.info(f"Layers: {selected}")

    # Step 2: Verify Execution (Just running forward/backward to check for crashes)
    logger.info("Verifying execution with checkpointing...")
    try:
        output = model(inputs)
        loss = output.sum()
        loss.backward()
        logger.info("Verification Successful: Backward pass completed.")
    except Exception as e:
        logger.error(f"Verification Failed: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "vit"])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    
    run_benchmark(args.model, args.batch_size)
