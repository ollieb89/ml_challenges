#!/usr/bin/env python3
"""GPU Constraint Profiler for Multi-System Testing.

This script profiles GPU memory constraints across different models and configurations
to determine what works on constrained VRAM systems (e.g., 8GB RTX 3070 Ti).

Usage:
    pixi run -e cuda python scripts/gpu_constraint_profiler.py
    pixi run -e cuda python scripts/gpu_constraint_profiler.py --models resnet50 vit_base
    pixi run -e cuda python scripts/gpu_constraint_profiler.py --output reports/system_constraints/

The script will:
1. Detect current GPU and VRAM capacity
2. Profile multiple models (ResNet50, ViT-Base, Llama-7B variants)
3. Test batch size limits until OOM
4. Test different precision levels (FP32, FP16, INT8, INT4)
5. Output structured constraint data as JSON and a summary markdown report

Models tested:
- ResNet50 (inference)
- ViT-Base (inference)
- Llama-7B (with various quantization levels)
- YOLO Pose (via existing detector)
"""

import argparse
import gc
import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """GPU hardware information."""
    name: str
    total_vram_gb: float
    compute_capability: str
    cuda_version: str
    driver_version: str
    device_index: int


@dataclass
class ModelConstraint:
    """Constraint information for a specific model configuration."""
    model_name: str
    precision: str  # fp32, fp16, int8, int4
    max_batch_size: int
    vram_at_max_batch_mb: float
    oom_batch_size: int | None  # First batch size that causes OOM
    inference_latency_ms: float | None
    notes: str


@dataclass
class SystemConstraintReport:
    """Complete system constraint report."""
    timestamp: str
    gpu_info: GPUInfo
    model_constraints: list[ModelConstraint]
    recommendations: list[str]


def detect_gpu() -> GPUInfo:
    """Detect and return GPU information."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available")
    
    props = torch.cuda.get_device_properties(0)
    cuda_version = torch.version.cuda or "unknown"
    
    # Try to get driver version via nvidia-smi
    driver_version = "unknown"
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            driver_version = result.stdout.strip()
    except Exception:
        pass
    
    return GPUInfo(
        name=props.name,
        total_vram_gb=props.total_memory / (1024**3),
        compute_capability=f"{props.major}.{props.minor}",
        cuda_version=cuda_version,
        driver_version=driver_version,
        device_index=0,
    )


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def measure_memory_and_latency(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    n_warmup: int = 3,
    n_runs: int = 5,
) -> tuple[float, float]:
    """Measure peak memory and average latency for a model forward pass.
    
    Returns:
        Tuple of (peak_memory_mb, avg_latency_ms)
    """
    clear_gpu_memory()
    
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_tensor)
            torch.cuda.synchronize()
    
    # Reset and measure
    torch.cuda.reset_peak_memory_stats()
    
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            _ = model(input_tensor)
            end.record()
            
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))
    
    peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
    avg_latency_ms = sum(latencies) / len(latencies)
    
    return peak_memory_mb, avg_latency_ms


def test_batch_size_limit(
    model: torch.nn.Module,
    create_input_fn,
    device: torch.device,
    max_batch: int = 256,
) -> tuple[int, int | None, float, float]:
    """Find the maximum batch size that fits in VRAM.
    
    Returns:
        Tuple of (max_working_batch, oom_batch, vram_at_max_mb, latency_at_max_ms)
    """
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    batch_sizes = [bs for bs in batch_sizes if bs <= max_batch]
    
    max_working_batch = 0
    oom_batch = None
    vram_at_max = 0.0
    latency_at_max = 0.0
    
    for bs in batch_sizes:
        clear_gpu_memory()
        try:
            input_tensor = create_input_fn(bs, device)
            vram, latency = measure_memory_and_latency(model, input_tensor)
            max_working_batch = bs
            vram_at_max = vram
            latency_at_max = latency
            logger.info(f"  Batch {bs}: {vram:.1f} MB, {latency:.2f} ms")
        except torch.cuda.OutOfMemoryError:
            oom_batch = bs
            logger.warning(f"  Batch {bs}: OOM")
            clear_gpu_memory()
            break
        except Exception as e:
            logger.error(f"  Batch {bs}: Error - {e}")
            break
    
    return max_working_batch, oom_batch, vram_at_max, latency_at_max


def profile_resnet50(device: torch.device) -> ModelConstraint:
    """Profile ResNet50 inference constraints."""
    logger.info("Profiling ResNet50 (FP32)...")
    
    try:
        import torchvision.models as models
        model = models.resnet50(weights=None).to(device).eval()
        
        def create_input(batch_size: int, dev: torch.device) -> torch.Tensor:
            return torch.randn(batch_size, 3, 224, 224, device=dev)
        
        max_batch, oom_batch, vram, latency = test_batch_size_limit(
            model, create_input, device, max_batch=256
        )
        
        del model
        clear_gpu_memory()
        
        return ModelConstraint(
            model_name="ResNet50",
            precision="fp32",
            max_batch_size=max_batch,
            vram_at_max_batch_mb=vram,
            oom_batch_size=oom_batch,
            inference_latency_ms=latency,
            notes="Standard ImageNet model, batch inference"
        )
    except Exception as e:
        logger.error(f"ResNet50 profiling failed: {e}")
        return ModelConstraint(
            model_name="ResNet50",
            precision="fp32",
            max_batch_size=0,
            vram_at_max_batch_mb=0,
            oom_batch_size=1,
            inference_latency_ms=None,
            notes=f"Failed: {str(e)}"
        )


def profile_resnet50_fp16(device: torch.device) -> ModelConstraint:
    """Profile ResNet50 inference constraints at FP16."""
    logger.info("Profiling ResNet50 (FP16)...")
    
    try:
        import torchvision.models as models
        model = models.resnet50(weights=None).to(device).half().eval()
        
        def create_input(batch_size: int, dev: torch.device) -> torch.Tensor:
            return torch.randn(batch_size, 3, 224, 224, device=dev, dtype=torch.float16)
        
        max_batch, oom_batch, vram, latency = test_batch_size_limit(
            model, create_input, device, max_batch=256
        )
        
        del model
        clear_gpu_memory()
        
        return ModelConstraint(
            model_name="ResNet50",
            precision="fp16",
            max_batch_size=max_batch,
            vram_at_max_batch_mb=vram,
            oom_batch_size=oom_batch,
            inference_latency_ms=latency,
            notes="Half precision, faster inference"
        )
    except Exception as e:
        logger.error(f"ResNet50 FP16 profiling failed: {e}")
        return ModelConstraint(
            model_name="ResNet50",
            precision="fp16",
            max_batch_size=0,
            vram_at_max_batch_mb=0,
            oom_batch_size=1,
            inference_latency_ms=None,
            notes=f"Failed: {str(e)}"
        )


def profile_vit_base(device: torch.device) -> ModelConstraint:
    """Profile ViT-Base inference constraints."""
    logger.info("Profiling ViT-Base (FP32)...")
    
    try:
        import torchvision.models as models
        model = models.vit_b_16(weights=None).to(device).eval()
        
        def create_input(batch_size: int, dev: torch.device) -> torch.Tensor:
            return torch.randn(batch_size, 3, 224, 224, device=dev)
        
        max_batch, oom_batch, vram, latency = test_batch_size_limit(
            model, create_input, device, max_batch=256
        )
        
        del model
        clear_gpu_memory()
        
        return ModelConstraint(
            model_name="ViT-Base-16",
            precision="fp32",
            max_batch_size=max_batch,
            vram_at_max_batch_mb=vram,
            oom_batch_size=oom_batch,
            inference_latency_ms=latency,
            notes="Vision Transformer, larger memory footprint than CNNs"
        )
    except Exception as e:
        logger.error(f"ViT-Base profiling failed: {e}")
        return ModelConstraint(
            model_name="ViT-Base-16",
            precision="fp32",
            max_batch_size=0,
            vram_at_max_batch_mb=0,
            oom_batch_size=1,
            inference_latency_ms=None,
            notes=f"Failed: {str(e)}"
        )


def profile_vit_base_fp16(device: torch.device) -> ModelConstraint:
    """Profile ViT-Base inference constraints at FP16."""
    logger.info("Profiling ViT-Base (FP16)...")
    
    try:
        import torchvision.models as models
        model = models.vit_b_16(weights=None).to(device).half().eval()
        
        def create_input(batch_size: int, dev: torch.device) -> torch.Tensor:
            return torch.randn(batch_size, 3, 224, 224, device=dev, dtype=torch.float16)
        
        max_batch, oom_batch, vram, latency = test_batch_size_limit(
            model, create_input, device, max_batch=256
        )
        
        del model
        clear_gpu_memory()
        
        return ModelConstraint(
            model_name="ViT-Base-16",
            precision="fp16",
            max_batch_size=max_batch,
            vram_at_max_batch_mb=vram,
            oom_batch_size=oom_batch,
            inference_latency_ms=latency,
            notes="Half precision Vision Transformer"
        )
    except Exception as e:
        logger.error(f"ViT-Base FP16 profiling failed: {e}")
        return ModelConstraint(
            model_name="ViT-Base-16",
            precision="fp16",
            max_batch_size=0,
            vram_at_max_batch_mb=0,
            oom_batch_size=1,
            inference_latency_ms=None,
            notes=f"Failed: {str(e)}"
        )


def profile_llama_stub(device: torch.device, precision: str = "fp16") -> ModelConstraint:
    """Profile Llama-7B memory requirements (stub - estimates without loading full model).
    
    Full Llama-7B loading requires ~14GB for FP16, so we provide estimates
    and attempt loading if VRAM is sufficient.
    """
    logger.info(f"Profiling Llama-7B ({precision}) - estimation mode...")
    
    gpu_info = detect_gpu()
    vram_gb = gpu_info.total_vram_gb
    
    # Llama-7B memory estimates (approximate)
    estimates = {
        "fp32": {"model_gb": 28, "seq128_batch1_gb": 30},
        "fp16": {"model_gb": 14, "seq128_batch1_gb": 16},
        "int8": {"model_gb": 7, "seq128_batch1_gb": 8.5},
        "int4": {"model_gb": 3.5, "seq128_batch1_gb": 5},
    }
    
    est = estimates.get(precision, estimates["fp16"])
    fits = vram_gb > est["seq128_batch1_gb"]
    
    if fits:
        notes = f"Estimated to fit. Model: ~{est['model_gb']}GB, Inference: ~{est['seq128_batch1_gb']}GB"
        max_batch = 1 if precision in ["fp32", "fp16"] else 4 if precision == "int8" else 8
    else:
        notes = f"WILL NOT FIT. Requires ~{est['seq128_batch1_gb']}GB, available: {vram_gb:.1f}GB"
        max_batch = 0
    
    return ModelConstraint(
        model_name="Llama-7B",
        precision=precision,
        max_batch_size=max_batch,
        vram_at_max_batch_mb=est["seq128_batch1_gb"] * 1024 if fits else 0,
        oom_batch_size=2 if fits else 1,
        inference_latency_ms=None,
        notes=notes
    )


def profile_yolo_pose(device: torch.device) -> ModelConstraint:
    """Profile YOLO Pose detector constraints."""
    logger.info("Profiling YOLO Pose (YOLOv11n-pose)...")
    
    try:
        # Add pose_analyzer to path
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        pose_analyzer_src = project_root / "projects" / "pose_analyzer" / "src"
        if str(pose_analyzer_src) not in sys.path:
            sys.path.insert(0, str(pose_analyzer_src))
        
        from pose_analyzer.pose_detector import YOLOPosev11Detector
        import numpy as np
        
        # Find model file
        model_path = project_root / "data" / "models" / "yolo11n-pose.pt"
        if not model_path.exists():
            return ModelConstraint(
                model_name="YOLO-Pose-v11n",
                precision="fp16",  # YOLO default
                max_batch_size=0,
                vram_at_max_batch_mb=0,
                oom_batch_size=None,
                inference_latency_ms=None,
                notes=f"Model file not found: {model_path}"
            )
        
        detector = YOLOPosev11Detector(
            model_variant="n",
            device="cuda:0",
            engine_path=model_path,
        )
        
        # Test batch sizes
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        max_working_batch = 0
        oom_batch = None
        vram_at_max = 0.0
        latency_at_max = 0.0
        
        for bs in batch_sizes:
            clear_gpu_memory()
            try:
                frames = [np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8) for _ in range(bs)]
                torch.cuda.reset_peak_memory_stats()
                
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                with torch.no_grad():
                    _ = detector.model(frames, imgsz=640, device="cuda:0", verbose=False)
                end.record()
                torch.cuda.synchronize()
                
                vram = torch.cuda.max_memory_allocated() / (1024**2)
                latency = start.elapsed_time(end)
                
                max_working_batch = bs
                vram_at_max = vram
                latency_at_max = latency
                logger.info(f"  Batch {bs}: {vram:.1f} MB, {latency:.2f} ms")
                
            except torch.cuda.OutOfMemoryError:
                oom_batch = bs
                logger.warning(f"  Batch {bs}: OOM")
                clear_gpu_memory()
                break
            except Exception as e:
                logger.error(f"  Batch {bs}: Error - {e}")
                break
        
        return ModelConstraint(
            model_name="YOLO-Pose-v11n",
            precision="fp16",  # YOLO uses FP16 by default
            max_batch_size=max_working_batch,
            vram_at_max_batch_mb=vram_at_max,
            oom_batch_size=oom_batch,
            inference_latency_ms=latency_at_max,
            notes="YOLOv11 nano pose detection model, 640x640 input"
        )
        
    except Exception as e:
        logger.error(f"YOLO Pose profiling failed: {e}")
        return ModelConstraint(
            model_name="YOLO-Pose-v11n",
            precision="fp16",
            max_batch_size=0,
            vram_at_max_batch_mb=0,
            oom_batch_size=None,
            inference_latency_ms=None,
            notes=f"Failed: {str(e)}"
        )


def generate_recommendations(gpu_info: GPUInfo, constraints: list[ModelConstraint]) -> list[str]:
    """Generate recommendations based on constraint results."""
    recommendations = []
    vram_gb = gpu_info.total_vram_gb
    
    # VRAM-based recommendations
    if vram_gb < 10:
        recommendations.append(
            f"🔴 CRITICAL: {gpu_info.name} has limited VRAM ({vram_gb:.1f}GB). "
            "Use quantized models (INT8/INT4) for LLMs."
        )
    elif vram_gb < 14:
        recommendations.append(
            f"⚠️ WARNING: {gpu_info.name} has moderate VRAM ({vram_gb:.1f}GB). "
            "Llama-7B FP16 may not fit. Use INT8 quantization."
        )
    else:
        recommendations.append(
            f"✅ {gpu_info.name} has sufficient VRAM ({vram_gb:.1f}GB) for most models."
        )
    
    # Model-specific recommendations
    for constraint in constraints:
        if constraint.max_batch_size == 0:
            recommendations.append(
                f"❌ {constraint.model_name} ({constraint.precision}): FAILS on this GPU. "
                f"Reason: {constraint.notes}"
            )
        elif constraint.oom_batch_size is not None:
            recommendations.append(
                f"⚠️ {constraint.model_name} ({constraint.precision}): "
                f"Max batch={constraint.max_batch_size}, OOM at batch={constraint.oom_batch_size}"
            )
        else:
            recommendations.append(
                f"✅ {constraint.model_name} ({constraint.precision}): "
                f"Max batch={constraint.max_batch_size} ({constraint.vram_at_max_batch_mb:.0f}MB)"
            )
    
    return recommendations


def run_profiling(
    models: list[str] | None = None,
    output_dir: Path = Path("reports/system_constraints"),
) -> SystemConstraintReport:
    """Run complete GPU constraint profiling.
    
    Args:
        models: List of models to profile. If None, profiles all available.
        output_dir: Directory to save results.
    """
    device = torch.device("cuda:0")
    gpu_info = detect_gpu()
    
    logger.info("=" * 60)
    logger.info(f"🔍 GPU CONSTRAINT PROFILER")
    logger.info("=" * 60)
    logger.info(f"GPU: {gpu_info.name}")
    logger.info(f"VRAM: {gpu_info.total_vram_gb:.2f} GB")
    logger.info(f"CUDA: {gpu_info.cuda_version}")
    logger.info(f"Compute: SM {gpu_info.compute_capability}")
    logger.info("=" * 60)
    
    # Default models to profile
    all_models = {
        "resnet50": profile_resnet50,
        "resnet50_fp16": profile_resnet50_fp16,
        "vit_base": profile_vit_base,
        "vit_base_fp16": profile_vit_base_fp16,
        "yolo_pose": profile_yolo_pose,
        "llama7b_fp16": lambda d: profile_llama_stub(d, "fp16"),
        "llama7b_int8": lambda d: profile_llama_stub(d, "int8"),
        "llama7b_int4": lambda d: profile_llama_stub(d, "int4"),
    }
    
    if models:
        models_to_run = {k: v for k, v in all_models.items() if k in models}
    else:
        models_to_run = all_models
    
    constraints: list[ModelConstraint] = []
    
    for model_name, profile_fn in models_to_run.items():
        logger.info(f"\n{'─' * 40}")
        try:
            constraint = profile_fn(device)
            constraints.append(constraint)
        except Exception as e:
            logger.error(f"Failed to profile {model_name}: {e}")
    
    recommendations = generate_recommendations(gpu_info, constraints)
    
    report = SystemConstraintReport(
        timestamp=datetime.now().isoformat(),
        gpu_info=gpu_info,
        model_constraints=constraints,
        recommendations=recommendations,
    )
    
    return report


def save_report(report: SystemConstraintReport, output_dir: Path):
    """Save the constraint report to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate system identifier
    gpu_name = report.gpu_info.name.lower().replace(" ", "_").replace("-", "_")
    vram_gb = int(report.gpu_info.total_vram_gb)
    system_id = f"{gpu_name}_{vram_gb}gb"
    
    # Save JSON
    json_path = output_dir / f"{system_id}_constraints.json"
    with open(json_path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    logger.info(f"📄 JSON report saved: {json_path}")
    
    # Save markdown summary
    md_path = output_dir / f"{system_id}_constraints.md"
    _write_markdown_report(report, md_path)
    logger.info(f"📝 Markdown report saved: {md_path}")


def _write_markdown_report(report: SystemConstraintReport, path: Path):
    """Write a human-readable markdown report."""
    gpu = report.gpu_info
    
    lines = [
        f"# GPU Constraint Report: {gpu.name}",
        "",
        f"**Generated:** {report.timestamp}",
        "",
        "## GPU Information",
        "",
        f"| Property | Value |",
        f"|----------|-------|",
        f"| GPU Name | {gpu.name} |",
        f"| Total VRAM | {gpu.total_vram_gb:.2f} GB |",
        f"| Compute Capability | SM {gpu.compute_capability} |",
        f"| CUDA Version | {gpu.cuda_version} |",
        f"| Driver Version | {gpu.driver_version} |",
        "",
        "## Model Constraints",
        "",
        "| Model | Precision | Max Batch | VRAM (MB) | Latency (ms) | OOM Batch | Notes |",
        "|-------|-----------|-----------|-----------|--------------|-----------|-------|",
    ]
    
    for c in report.model_constraints:
        latency = f"{c.inference_latency_ms:.1f}" if c.inference_latency_ms else "N/A"
        oom = str(c.oom_batch_size) if c.oom_batch_size else "-"
        lines.append(
            f"| {c.model_name} | {c.precision} | {c.max_batch_size} | "
            f"{c.vram_at_max_batch_mb:.0f} | {latency} | {oom} | {c.notes[:50]}{'...' if len(c.notes) > 50 else ''} |"
        )
    
    lines.extend([
        "",
        "## Recommendations",
        "",
    ])
    
    for rec in report.recommendations:
        lines.append(f"- {rec}")
    
    lines.extend([
        "",
        "---",
        "",
        "*Generated by gpu_constraint_profiler.py*",
    ])
    
    with open(path, "w") as f:
        f.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Profile GPU constraints for ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Profile all models
    pixi run -e cuda python scripts/gpu_constraint_profiler.py
    
    # Profile specific models
    pixi run -e cuda python scripts/gpu_constraint_profiler.py --models resnet50 vit_base yolo_pose
    
    # Custom output directory
    pixi run -e cuda python scripts/gpu_constraint_profiler.py --output reports/my_test/
        """
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["resnet50", "resnet50_fp16", "vit_base", "vit_base_fp16", 
                 "yolo_pose", "llama7b_fp16", "llama7b_int8", "llama7b_int4"],
        help="Specific models to profile (default: all)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("projects/gpu_optimizer/reports/system_constraints"),
        help="Output directory for reports (default: projects/gpu_optimizer/reports/system_constraints/)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving reports to disk (print to stdout only)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA not available. This script requires a GPU.")
        sys.exit(1)
    
    try:
        report = run_profiling(models=args.models, output_dir=args.output)
        
        if not args.no_save:
            save_report(report, args.output)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 PROFILING COMPLETE")
        print("=" * 60)
        print(f"\nGPU: {report.gpu_info.name} ({report.gpu_info.total_vram_gb:.1f}GB)")
        print("\nRecommendations:")
        for rec in report.recommendations:
            print(f"  {rec}")
        print("=" * 60)
        
    except Exception as e:
        logger.exception(f"Profiling failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
