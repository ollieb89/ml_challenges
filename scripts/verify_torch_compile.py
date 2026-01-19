
import time
import torch
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestCompile")

def main():
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Device: {torch.cuda.get_device_name(0)}")
        
    try:
        compiler_version = torch._dynamo.list_backends()
        logger.info(f"Available Backends: {compiler_version}")
    except:
        logger.info("Could not list backends via _dynamo")

    # Load Model
    logger.info("Loading YOLO model...")
    model = YOLO("yolo11n-pose.pt") # Using existing file or download
    
    # Warmup standard
    logger.info("Running standard warmup...")
    img = torch.zeros((1, 3, 640, 640), device='cuda').float()
    
    # Move model to URL
    # YOLO handles device usually, but let's force it
    # note: Ultralytics handles .to() internally usually when passed tensor?
    # No, model.to('cuda') is safer.
    
    # Create dummy frame
    import numpy as np
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Wait, Ultralytics auto-loads to device if input is on device usually? 
    # Let's just call predict
    res = model.predict(source=dummy_frame, device="cuda", verbose=False, max_det=1)
    # next(res) # Trigger 1 - list returned if not stream=True
    
    logger.info("Applying torch.compile(mode='reduce-overhead')...")
    start_c = time.monotonic()
    
    try:
        # We compile the inner nn.Module
        model.model = torch.compile(model.model, mode="reduce-overhead")
        logger.info(f"Compilation declaration took {time.monotonic() - start_c:.4f}s")
        
        logger.info("Running Compiled Warmup (Should trigger JIT)...")
        t0 = time.monotonic()
        # Trigger inference
        model.predict(source=dummy_frame, device="cuda", verbose=False, max_det=1)
        duration = time.monotonic() - t0
        logger.info(f"First inference (JIT compile) took {duration:.4f}s")
        
        logger.info("Running Benchmark (100 frames)...")
        # Reuse same generator? No, stream=True
        
        t0 = time.monotonic()
        count = 0
        for _ in range(100):
            model.predict(source=dummy_frame, device="cuda", verbose=False, max_det=1)
            count += 1
            
        dur = time.monotonic() - t0
        fps = count / dur
        logger.info(f"Compiled Throughput: {fps:.2f} FPS")
        logger.info("SUCCESS: torch.compile worked!")
        
    except Exception as e:
        logger.error(f"FAILURE: torch.compile failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
