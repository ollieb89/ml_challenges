'''VRAM Stress Test for PoseDetector (YOLO)

This script loads the YOLO pose detection model and iteratively runs forward passes with increasing batch sizes
to determine the maximum batch size that fits within the GPU's 8 GB VRAM.

Usage:
    pixi run python scripts/vram_stress_test.py --model data/models/yolo11n-pose.pt

The script logs results to `reports/vram_stress.csv` and optionally creates a memory‑vs‑batch plot saved as `reports/vram_curve.png`.
''' 

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

# Optional: matplotlib for plotting (imported lazily)
try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

def load_model(model_path: Path, device: torch.device):
    """Load the YOLO pose detector model.

    Args:
        model_path: Path to the serialized model file (e.g., .pt).
        device: Torch device where the model will be moved.

    Returns:
        The loaded YOLOPosev11Detector instance.
    """
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Add the pose_analyzer package source directory to PYTHONPATH
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    pose_analyzer_src = project_root / "projects" / "pose_analyzer" / "src"
    if str(pose_analyzer_src) not in sys.path:
        sys.path.insert(0, str(pose_analyzer_src))
    
    try:
        from pose_analyzer.pose_detector import YOLOPosev11Detector
    except Exception as e:
        raise ImportError(f"Failed to import YOLOPosev11Detector: {e}")
    
    # Determine model variant from filename (e.g., yolo11n-pose.pt -> 'n')
    model_name = model_path.stem  # e.g., 'yolo11n-pose'
    variant = 'n'  # default
    if 'yolo11' in model_name:
        # Extract variant letter after 'yolo11' (e.g., 'n', 's', 'm', 'l', 'x')
        suffix = model_name.replace('yolo11', '').split('-')[0]
        if suffix:
            variant = suffix
    
    # Create the detector with the model file path
    device_str = str(device) if device.type == 'cuda' else 'cpu'
    detector = YOLOPosev11Detector(
        model_variant=variant,
        device=device_str,
        engine_path=model_path,
    )
    return detector

def dummy_frame(img_size: int = 640) -> np.ndarray:
    """Create a dummy BGR frame for single-image testing."""
    return np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

def measure_batch(detector, batch_size: int, device: torch.device, img_size: int = 640) -> tuple[float, bool]:
    """Run a forward pass for a given batch size and return memory usage.

    Uses YOLO's underlying model directly for batch inference.
    Returns a tuple ``(memory_used_MB, oom_flag)``.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        # Create batch of dummy frames for YOLO inference
        # YOLO's __call__ accepts a list of frames or paths
        frames = [np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8) for _ in range(batch_size)]
        
        with torch.no_grad():
            # Use YOLO's native batch inference
            _ = detector.model(frames, imgsz=img_size, device=detector.device, verbose=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
        mem_bytes = torch.cuda.max_memory_allocated(device)
        mem_mb = mem_bytes / (1024 ** 2)
        return mem_mb, False
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return 0.0, True

def run_stress_test(
    model_path: Path,
    device_idx: int = 0,
    batch_sizes: list[int] | None = None,
    report_path: Path = Path("reports/vram_stress.csv"),
    plot_path: Path | None = Path("reports/vram_curve.png"),
) -> None:
    """Main driver for the VRAM stress test.

    Args:
        model_path: Path to the YOLO model file.
        device_idx: CUDA device index (default 0).
        batch_sizes: List of batch sizes to test; defaults to exponential list.
        report_path: CSV file where results are written.
        plot_path: Optional path for a memory‑vs‑batch plot.
    """
    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA device not available – cannot perform VRAM stress test.")

    detector = load_model(model_path, device)

    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["batch_size", "memory_used_MB", "oom_flag"])
        for bs in batch_sizes:
            mem_mb, oom = measure_batch(detector, bs, device)
            writer.writerow([bs, f"{mem_mb:.2f}", int(oom)])
            print(f"Batch {bs}: {'OOM' if oom else f'{mem_mb:.2f} MB'}")
            if oom:
                break

    if plot_path and plt is not None:
        import pandas as pd
        df = pd.read_csv(report_path)
        plt.figure(figsize=(8, 5))
        plt.plot(df["batch_size"], df["memory_used_MB"], marker="o")
        plt.title("VRAM Usage vs Batch Size")
        plt.xlabel("Batch Size")
        plt.ylabel("Memory Used (MB)")
        plt.grid(True)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VRAM stress test for YOLO PoseDetector")
    parser.add_argument("--model", type=str, required=True, help="Path to the YOLO .pt model file")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (default 0)")
    parser.add_argument("--batches", type=int, nargs="+", help="Explicit list of batch sizes to test")
    parser.add_argument("--no-plot", action="store_true", help="Skip generating the memory‑vs‑batch plot")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        run_stress_test(
            model_path=Path(args.model),
            device_idx=args.device,
            batch_sizes=args.batches,
            plot_path=None if args.no_plot else Path("reports/vram_curve.png"),
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
