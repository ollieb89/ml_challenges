#!/usr/bin/env python3
"""
Comprehensive profiling script for YOLO pose detection pipeline.
Uses torch.profiler to analyze bottlenecks and generate detailed HTML reports.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity, schedule
from ultralytics import YOLO

from pose_analyzer.pose_detector import YOLOPosev11Detector


class PoseProfiler:
    """Comprehensive profiler for YOLO pose detection models."""
    
    def __init__(
        self,
        model_variant: str = "n",
        device: Optional[str] = None,
        imgsz: int = 1080,
        warmup_iterations: int = 10,
        profile_iterations: int = 50,
    ):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.imgsz = imgsz
        self.warmup_iterations = warmup_iterations
        self.profile_iterations = profile_iterations
        
        # Initialize YOLO model
        model_name = f"yolo11{model_variant}-pose.pt"
        data_dir = Path(__file__).resolve().parents[3] / "data" / "models"
        model_path = data_dir / model_name
        
        if not model_path.exists():
            model_path = Path(model_name)
            
        self.model = YOLO(str(model_path), task="pose")
        self.model.to(self.device)
        
        # Test frame for profiling (1080p)
        self.test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
    def warmup_model(self) -> None:
        """Warm up the model to ensure stable performance measurements."""
        print(f"Warming up model with {self.warmup_iterations} iterations...")
        for i in range(self.warmup_iterations):
            _ = self.model(
                self.test_frame,
                imgsz=self.imgsz,
                conf=0.5,
                iou=0.45,
                device=self.device,
                verbose=False,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        print("Warmup complete.")
    
    def profile_single_stream(self) -> profile:
        """Profile single-stream inference performance."""
        print(f"Profiling single-stream inference with {self.profile_iterations} iterations...")
        
        # Define profiling schedule
        wait_steps = 5
        warmup_steps = 5
        active_steps = self.profile_iterations
        
        profiler_schedule = schedule(
            wait=wait_steps,
            warmup=warmup_steps,
            active=active_steps,
            repeat=1,
        )
        
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        
        with profile(
            activities=activities,
            schedule=profiler_schedule,
            on_trace_ready=self._trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
        ) as prof:
            for step in range(wait_steps + warmup_steps + active_steps):
                # Run inference
                _ = self.model(
                    self.test_frame,
                    imgsz=self.imgsz,
                    conf=0.5,
                    iou=0.45,
                    device=self.device,
                    verbose=False,
                )
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Tell profiler which step we're on
                prof.step()
        
        return prof
    
    def profile_memory_usage(self) -> Dict[str, float]:
        """Profile detailed memory usage patterns."""
        print("Profiling memory usage patterns...")
        
        if not torch.cuda.is_available():
            return {"error": "CUDA not available for memory profiling"}
        
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        initial_memory = torch.cuda.memory_allocated()
        peak_memory = 0
        
        # Run multiple iterations to capture memory patterns
        for i in range(20):
            _ = self.model(
                self.test_frame,
                imgsz=self.imgsz,
                conf=0.5,
                iou=0.45,
                device=self.device,
                verbose=False,
            )
            torch.cuda.synchronize()
            
            current_peak = torch.cuda.max_memory_allocated()
            peak_memory = max(peak_memory, current_peak)
        
        final_memory = torch.cuda.memory_allocated()
        
        return {
            "initial_memory_mb": initial_memory / (1024**2),
            "final_memory_mb": final_memory / (1024**2),
            "peak_memory_mb": peak_memory / (1024**2),
            "memory_increase_mb": (final_memory - initial_memory) / (1024**2),
        }
    
    def analyze_layer_performance(self, prof: profile) -> Dict[str, Dict]:
        """Analyze performance by layer to identify bottlenecks."""
        print("Analyzing layer-wise performance...")
        
        # Get key performance metrics
        key_averages = prof.key_averages()
        
        layer_stats = {}
        
        for stat in key_averages:
            op_name = stat.key if hasattr(stat, 'key') else str(stat)
            
            # Skip if this is not a compute operation
            if not any(keyword in op_name.lower() for keyword in ['conv', 'linear', 'matmul', 'attention']):
                continue
            
            layer_stats[op_name] = {
                "cpu_time_total_ms": stat.cpu_time_total / 1000,  # Convert to ms
                "cuda_time_total_ms": getattr(stat, 'cuda_time_total', 0) / 1000 if getattr(stat, 'cuda_time_total', 0) > 0 else 0,
                "cpu_time_avg_ms": stat.cpu_time_total / (stat.count * 1000),
                "cuda_time_avg_ms": getattr(stat, 'cuda_time_total', 0) / (stat.count * 1000) if getattr(stat, 'cuda_time_total', 0) > 0 else 0,
                "self_cpu_memory_usage_mb": stat.self_cpu_memory_usage / (1024**2),
                "self_cuda_memory_usage_mb": getattr(stat, 'self_cuda_memory_usage', 0) / (1024**2) if getattr(stat, 'self_cuda_memory_usage', 0) > 0 else 0,
                "count": stat.count,
                "flops": getattr(stat, 'flops', 0),
            }
        
        # Sort by total CUDA time (or CPU time if CUDA not available)
        sorted_layers = sorted(
            layer_stats.items(),
            key=lambda x: x[1]['cuda_time_total_ms'] if x[1]['cuda_time_total_ms'] > 0 else x[1]['cpu_time_total_ms'],
            reverse=True
        )
        
        return {
            "layer_statistics": dict(sorted_layers),
            "dominant_layer": sorted_layers[0] if sorted_layers else None,
            "top_5_layers": sorted_layers[:5],
        }
    
    def measure_data_transfer_time(self) -> Dict[str, float]:
        """Measure data transfer overheads."""
        print("Measuring data transfer times...")
        
        if not torch.cuda.is_available():
            return {"error": "CUDA not available for transfer profiling"}
        
        # Test CPU to GPU transfer
        cpu_tensor = torch.randn(1, 3, 1080, 1920)
        
        # Warm up
        for _ in range(5):
            gpu_tensor = cpu_tensor.to(self.device)
            torch.cuda.synchronize()
        
        # Measure transfer time
        transfer_times = []
        for _ in range(20):
            start = time.perf_counter()
            gpu_tensor = cpu_tensor.to(self.device)
            torch.cuda.synchronize()
            end = time.perf_counter()
            transfer_times.append((end - start) * 1000)  # Convert to ms
        
        return {
            "avg_cpu_to_gpu_transfer_ms": np.mean(transfer_times),
            "min_cpu_to_gpu_transfer_ms": np.min(transfer_times),
            "max_cpu_to_gpu_transfer_ms": np.max(transfer_times),
            "data_size_mb": cpu_tensor.numel() * cpu_tensor.element_size() / (1024**2),
        }
    
    def _trace_handler(self, prof: profile) -> None:
        """Handle profiler trace output."""
        # Don't export traces here to avoid "already saved" error
        pass
    
    def generate_comprehensive_report(
        self,
        output_dir: Path,
        model_variant: str
    ) -> Dict:
        """Generate comprehensive profiling report."""
        print(f"Generating comprehensive profiling report for YOLOv11-{model_variant}...")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Warm up model
        self.warmup_model()
        
        # Run main profiling
        prof = self.profile_single_stream()
        
        # Generate HTML report
        html_report_path = output_dir / f"single_stream_profile_{model_variant}.html"
        prof.export_chrome_trace(str(output_dir / f"trace_{model_variant}.json"))
        
        # Create custom HTML report
        html_content = self._create_html_report(prof, model_variant)
        with open(html_report_path, 'w') as f:
            f.write(html_content)
        
        # Analyze layer performance
        layer_analysis = self.analyze_layer_performance(prof)
        
        # Profile memory usage
        memory_analysis = self.profile_memory_usage()
        
        # Measure data transfer
        transfer_analysis = self.measure_data_transfer_time()
        
        # Get key statistics for report
        key_stats = prof.key_averages()
        
        # Compile comprehensive report
        report = {
            "model": f"YOLOv11-{model_variant}",
            "device": self.device,
            "image_size": self.imgsz,
            "profile_iterations": self.profile_iterations,
            "layer_analysis": layer_analysis,
            "memory_analysis": memory_analysis,
            "transfer_analysis": transfer_analysis,
            "top_cpu_operations": [
                {
                    "name": stat.key if hasattr(stat, 'key') else str(stat),
                    "cpu_time_ms": stat.cpu_time_total / 1000,
                    "count": stat.count
                }
                for stat in sorted(key_stats, key=lambda x: x.cpu_time_total, reverse=True)[:10]
            ],
            "top_cuda_operations": [
                {
                    "name": stat.key if hasattr(stat, 'key') else str(stat),
                    "cuda_time_ms": getattr(stat, 'cuda_time_total', 0) / 1000 if getattr(stat, 'cuda_time_total', 0) > 0 else 0,
                    "count": stat.count
                }
                for stat in sorted(key_stats, key=lambda x: getattr(x, 'cuda_time_total', 0), reverse=True)[:10]
            ] if torch.cuda.is_available() else [],
        }
        
        # Save JSON report
        with open(output_dir / f"profile_report_{model_variant}.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report generated: {html_report_path}")
        return report
    
    def _create_html_report(self, prof: profile, model_variant: str) -> str:
        """Create comprehensive HTML report."""
        
        # Get key statistics
        key_stats = prof.key_averages()
        total_cpu_time = sum(stat.cpu_time_total for stat in key_stats)
        total_cuda_time = sum(getattr(stat, 'cuda_time_total', 0) for stat in key_stats)
        
        # Memory statistics - filter operations with memory usage
        memory_stats = [stat for stat in key_stats if stat.self_cpu_memory_usage > 0]
        
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>YOLOv11-{model_variant} Profiling Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4fd; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .dominant {{ background-color: #ffebee; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>YOLOv11-{model_variant} Profiling Report</h1>
        <p>Device: {self.device} | Image Size: {self.imgsz}x{self.imgsz} | Profile Iterations: {self.profile_iterations}</p>
    </div>
    
    <div class="section">
        <h2>Performance Summary</h2>
        <div class="metric">
            <strong>Total CPU Time:</strong> {total_cpu_time / 1000:.2f} ms
        </div>
        <div class="metric">
            <strong>Total CUDA Time:</strong> {total_cuda_time / 1000:.2f} ms
        </div>
        <div class="metric">
            <strong>Avg Inference Time:</strong> {(total_cpu_time + total_cuda_time) / (self.profile_iterations * 1000):.2f} ms
        </div>
    </div>
    
    <div class="section">
        <h2>Top CPU Operations</h2>
        <table>
            <tr><th>Operation</th><th>Total Time (ms)</th><th>Calls</th><th>Avg Time (ms)</th></tr>
            {self._generate_operation_table(key_stats, 'cpu')}
        </table>
    </div>
    
    <div class="section">
        <h2>Top CUDA Operations</h2>
        <table>
            <tr><th>Operation</th><th>Total Time (ms)</th><th>Calls</th><th>Avg Time (ms)</th></tr>
            {self._generate_operation_table(key_stats, 'cuda')}
        </table>
    </div>
    
    <div class="section">
        <h2>Memory Usage</h2>
        <table>
            <tr><th>Operation</th><th>Memory Usage (MB)</th><th>Calls</th></tr>
            {self._generate_memory_table(memory_stats)}
        </table>
    </div>
    
    <div class="section">
        <h2>Performance Insights</h2>
        <ul>
            <li>Profile generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</li>
            <li>Use Chrome Trace Viewer (chrome://tracing) to view trace.json for detailed timeline analysis</li>
            <li>Focus on operations with highest total time for optimization opportunities</li>
        </ul>
    </div>
</body>
</html>
        """
        
        return html_template
    
    def _generate_operation_table(self, stats, metric_type: str) -> str:
        """Generate HTML table for operations."""
        if metric_type == 'cpu':
            sorted_stats = sorted(stats, key=lambda x: x.cpu_time_total, reverse=True)[:10]
        else:
            sorted_stats = sorted(stats, key=lambda x: getattr(x, 'cuda_time_total', 0), reverse=True)[:10]
        
        rows = []
        for stat in sorted_stats:
            if metric_type == 'cpu':
                time_ms = stat.cpu_time_total / 1000
                avg_time = stat.cpu_time_total / (stat.count * 1000)
            else:
                time_ms = getattr(stat, 'cuda_time_total', 0) / 1000
                avg_time = getattr(stat, 'cuda_time_total', 0) / (stat.count * 1000)
            
            if time_ms > 0.01:  # Only include operations with meaningful time
                operation_name = stat.key if hasattr(stat, 'key') else str(stat)
                rows.append(f"""
                <tr>
                    <td>{operation_name}</td>
                    <td>{time_ms:.2f}</td>
                    <td>{stat.count}</td>
                    <td>{avg_time:.2f}</td>
                </tr>
                """)
        
        return ''.join(rows)
    
    def _generate_memory_table(self, stats) -> str:
        """Generate HTML table for memory usage."""
        sorted_stats = sorted(stats, key=lambda x: x.self_cpu_memory_usage, reverse=True)[:10]
        
        rows = []
        for stat in sorted_stats:
            if stat.self_cpu_memory_usage > 1024:  # Only include operations with meaningful memory usage
                memory_mb = stat.self_cpu_memory_usage / (1024**2)
                operation_name = stat.key if hasattr(stat, 'key') else str(stat)
                rows.append(f"""
                <tr>
                    <td>{operation_name}</td>
                    <td>{memory_mb:.2f}</td>
                    <td>{stat.count}</td>
                </tr>
                """)
        
        return ''.join(rows)


def main():
    """Main entry point for profiling script."""
    parser = argparse.ArgumentParser(description="YOLO Pose Detection Profiler")
    parser.add_argument(
        "--model-variant",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="YOLOv11 model variant"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "data" / "reports",
        help="Output directory for profiling reports"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda:0, cpu, etc.)"
    )
    parser.add_argument(
        "--profile-iterations",
        type=int,
        default=50,
        help="Number of profiling iterations"
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=10,
        help="Number of warmup iterations"
    )
    
    args = parser.parse_args()
    
    # Initialize profiler
    profiler = PoseProfiler(
        model_variant=args.model_variant,
        device=args.device,
        warmup_iterations=args.warmup_iterations,
        profile_iterations=args.profile_iterations,
    )
    
    # Generate comprehensive report
    report = profiler.generate_comprehensive_report(
        output_dir=args.output_dir,
        model_variant=args.model_variant
    )
    
    # Print summary
    print("\n" + "="*50)
    print("PROFILING SUMMARY")
    print("="*50)
    print(f"Model: {report['model']}")
    print(f"Device: {report['device']}")
    print(f"Profile Iterations: {report['profile_iterations']}")
    
    if 'dominant_layer' in report['layer_analysis'] and report['layer_analysis']['dominant_layer']:
        layer_name, layer_stats = report['layer_analysis']['dominant_layer']
        print(f"\nDominant Layer: {layer_name}")
        print(f"  - Total Time: {layer_stats['cuda_time_total_ms']:.2f} ms" if layer_stats['cuda_time_total_ms'] > 0 else f"  - Total Time: {layer_stats['cpu_time_total_ms']:.2f} ms")
        print(f"  - Calls: {layer_stats['count']}")
    
    if 'peak_memory_mb' in report['memory_analysis']:
        print(f"\nPeak Memory Usage: {report['memory_analysis']['peak_memory_mb']:.2f} MB")
    
    if 'avg_cpu_to_gpu_transfer_ms' in report['transfer_analysis']:
        print(f"Avg Data Transfer Time: {report['transfer_analysis']['avg_cpu_to_gpu_transfer_ms']:.2f} ms")
    
    print(f"\nDetailed report saved to: {args.output_dir}/single_stream_profile_{args.model_variant}.html")


if __name__ == "__main__":
    main()
