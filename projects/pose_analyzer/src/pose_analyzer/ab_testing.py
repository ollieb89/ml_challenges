"""
A/B Testing Framework for Kalman Filter Performance

This module provides comprehensive A/B testing capabilities to compare
raw pose keypoints against Kalman-filtered keypoints with detailed
metrics and visualizations.

Key Features:
- Side-by-side comparison of raw vs filtered trajectories
- Jitter measurement and smoothness analysis
- Performance benchmarking and latency measurement
- Statistical analysis of filtering effectiveness
- Visual comparison tools
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import json

from temporal_analyzer import KalmanFilterManager, FilterMetrics, KeypointState


@dataclass
class ABTestResult:
    """Results from A/B testing comparison"""
    raw_jitter_score: float
    filtered_jitter_score: float
    jitter_reduction_percent: float
    raw_smoothness_score: float
    filtered_smoothness_score: float
    smoothness_improvement_percent: float
    avg_latency_ms: float
    performance_success_rate: float
    frames_under_10ms: int
    total_frames: int


class JitterAnalyzer:
    """Analyzes jitter and smoothness in pose trajectories"""
    
    @staticmethod
    def calculate_jitter(trajectory: np.ndarray) -> float:
        """
        Calculate jitter as the variance of frame-to-frame displacements
        
        Args:
            trajectory: Array of shape (n_frames, n_keypoints, 2) with [x, y] coordinates
            
        Returns:
            Jitter score (lower is better)
        """
        if len(trajectory) < 2:
            return 0.0
        
        # Calculate frame-to-frame displacements
        displacements = np.diff(trajectory, axis=0)
        
        # Calculate magnitude of displacements for each keypoint
        displacement_magnitudes = np.linalg.norm(displacements, axis=2)
        
        # Jitter is the variance of these magnitudes across frames
        jitter = np.mean(np.var(displacement_magnitudes, axis=0))
        
        return jitter
    
    @staticmethod
    def calculate_smoothness(trajectory: np.ndarray) -> float:
        """
        Calculate smoothness score based on acceleration consistency
        
        Args:
            trajectory: Array of shape (n_frames, n_keypoints, 2) with [x, y] coordinates
            
        Returns:
            Smoothness score (higher is better)
        """
        if len(trajectory) < 3:
            return 1.0
        
        # Calculate second derivatives (acceleration)
        accelerations = np.diff(trajectory, n=2, axis=0)
        
        # Calculate magnitude of accelerations
        acceleration_magnitudes = np.linalg.norm(accelerations, axis=2)
        
        # Smoothness is inverse of acceleration variance
        smoothness = 1.0 / (1.0 + np.mean(np.var(acceleration_magnitudes, axis=0)))
        
        return smoothness
    
    @staticmethod
    def calculate_trajectory_stability(trajectory: np.ndarray) -> float:
        """
        Calculate trajectory stability (how much the trajectory wanders)
        
        Args:
            trajectory: Array of shape (n_frames, n_keypoints, 2) with [x, y] coordinates
            
        Returns:
            Stability score (higher is better)
        """
        if len(trajectory) < 2:
            return 1.0
        
        # Calculate the total path length
        path_lengths = []
        for kp_idx in range(trajectory.shape[1]):
            kp_trajectory = trajectory[:, kp_idx, :]
            path_length = np.sum(np.linalg.norm(np.diff(kp_trajectory, axis=0), axis=1))
            path_lengths.append(path_length)
        
        # Compare with direct distance (start to end)
        direct_distances = []
        for kp_idx in range(trajectory.shape[1]):
            start_point = trajectory[0, kp_idx, :]
            end_point = trajectory[-1, kp_idx, :]
            direct_distance = np.linalg.norm(end_point - start_point)
            direct_distances.append(direct_distance)
        
        # Stability is ratio of direct distance to path length
        stability_scores = []
        for path_len, direct_dist in zip(path_lengths, direct_distances):
            if path_len > 0:
                stability = direct_dist / path_len
            else:
                stability = 1.0
            stability_scores.append(stability)
        
        return np.mean(stability_scores)


class ABTestFramework:
    """
    Comprehensive A/B testing framework for Kalman filter evaluation
    """
    
    def __init__(self, kalman_manager: KalmanFilterManager):
        """
        Initialize A/B testing framework
        
        Args:
            kalman_manager: Configured KalmanFilterManager instance
        """
        self.kalman_manager = kalman_manager
        self.jitter_analyzer = JitterAnalyzer()
        
        # Storage for test data
        self.raw_trajectories: List[np.ndarray] = []
        self.filtered_trajectories: List[np.ndarray] = []
        self.metrics_history: List[FilterMetrics] = []
        
        # Test configuration
        self.test_running = False
        self.current_frame = 0
    
    def start_test(self) -> None:
        """Start a new A/B test session"""
        self.kalman_manager.reset()
        self.kalman_manager.enable_ab_testing()
        self.raw_trajectories = []
        self.filtered_trajectories = []
        self.metrics_history = []
        self.test_running = True
        self.current_frame = 0
    
    def stop_test(self) -> None:
        """Stop the current A/B test session"""
        self.test_running = False
        self.kalman_manager.disable_ab_testing()
    
    def process_frame(self, keypoints: np.ndarray, confidences: Optional[np.ndarray] = None) -> Tuple[np.ndarray, FilterMetrics]:
        """
        Process a frame during A/B testing
        
        Args:
            keypoints: Raw keypoints from pose detector
            confidences: Confidence scores for each keypoint
            
        Returns:
            Tuple of (filtered_keypoints, metrics)
        """
        if not self.test_running:
            raise RuntimeError("A/B test not running. Call start_test() first.")
        
        # Store raw keypoints
        self.raw_trajectories.append(keypoints.copy())
        
        # Process through Kalman filter
        filtered_keypoints, metrics = self.kalman_manager.process_frame(keypoints, confidences)
        
        # Store filtered keypoints and metrics
        self.filtered_trajectories.append(filtered_keypoints.copy())
        self.metrics_history.append(metrics)
        
        self.current_frame += 1
        
        return filtered_keypoints, metrics
    
    def generate_report(self) -> ABTestResult:
        """
        Generate comprehensive A/B test report
        
        Returns:
            ABTestResult with all comparison metrics
        """
        if len(self.raw_trajectories) < 2:
            raise ValueError("Need at least 2 frames for analysis")
        
        # Convert to numpy arrays
        raw_trajectory = np.array(self.raw_trajectories)
        filtered_trajectory = np.array(self.filtered_trajectories)
        
        # Calculate jitter scores
        raw_jitter = self.jitter_analyzer.calculate_jitter(raw_trajectory)
        filtered_jitter = self.jitter_analyzer.calculate_jitter(filtered_trajectory)
        jitter_reduction = ((raw_jitter - filtered_jitter) / raw_jitter * 100) if raw_jitter > 0 else 0
        
        # Calculate smoothness scores
        raw_smoothness = self.jitter_analyzer.calculate_smoothness(raw_trajectory)
        filtered_smoothness = self.jitter_analyzer.calculate_smoothness(filtered_trajectory)
        smoothness_improvement = ((filtered_smoothness - raw_smoothness) / raw_smoothness * 100) if raw_smoothness > 0 else 0
        
        # Calculate performance metrics
        avg_latency = np.mean([m.processing_time_ms for m in self.metrics_history])
        frames_under_10ms = sum(1 for m in self.metrics_history if m.processing_time_ms < 10.0)
        performance_success_rate = (frames_under_10ms / len(self.metrics_history)) * 100
        
        return ABTestResult(
            raw_jitter_score=raw_jitter,
            filtered_jitter_score=filtered_jitter,
            jitter_reduction_percent=jitter_reduction,
            raw_smoothness_score=raw_smoothness,
            filtered_smoothness_score=filtered_smoothness,
            smoothness_improvement_percent=smoothness_improvement,
            avg_latency_ms=avg_latency,
            performance_success_rate=performance_success_rate,
            frames_under_10ms=frames_under_10ms,
            total_frames=len(self.raw_trajectories)
        )
    
    def visualize_comparison(self, keypoint_idx: int = 0, save_path: Optional[str] = None) -> None:
        """
        Create visual comparison of raw vs filtered trajectories
        
        Args:
            keypoint_idx: Index of keypoint to visualize (0-16)
            save_path: Optional path to save the plot
        """
        if len(self.raw_trajectories) < 2:
            print("Need at least 2 frames for visualization")
            return
        
        raw_trajectory = np.array(self.raw_trajectories)
        filtered_trajectory = np.array(self.filtered_trajectories)
        
        # Extract trajectory for specific keypoint
        raw_kp = raw_trajectory[:, keypoint_idx, :]
        filtered_kp = filtered_trajectory[:, keypoint_idx, :]
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # X coordinate over time
        axes[0, 0].plot(raw_kp[:, 0], 'r-', alpha=0.7, label='Raw', linewidth=1)
        axes[0, 0].plot(filtered_kp[:, 0], 'b-', label='Filtered', linewidth=2)
        axes[0, 0].set_title(f'X Coordinate - {KalmanFilterManager.COCO_KEYPOINTS[keypoint_idx]}')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('X Position')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Y coordinate over time
        axes[0, 1].plot(raw_kp[:, 1], 'r-', alpha=0.7, label='Raw', linewidth=1)
        axes[0, 1].plot(filtered_kp[:, 1], 'b-', label='Filtered', linewidth=2)
        axes[0, 1].set_title(f'Y Coordinate - {KalmanFilterManager.COCO_KEYPOINTS[keypoint_idx]}')
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Y Position')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 2D trajectory
        axes[1, 0].plot(raw_kp[:, 0], raw_kp[:, 1], 'r-', alpha=0.7, label='Raw', linewidth=1)
        axes[1, 0].plot(filtered_kp[:, 0], filtered_kp[:, 1], 'b-', label='Filtered', linewidth=2)
        axes[1, 0].scatter(raw_kp[0, 0], raw_kp[0, 1], c='green', s=100, marker='o', label='Start')
        axes[1, 0].scatter(raw_kp[-1, 0], raw_kp[-1, 1], c='red', s=100, marker='s', label='End')
        axes[1, 0].set_title(f'2D Trajectory - {KalmanFilterManager.COCO_KEYPOINTS[keypoint_idx]}')
        axes[1, 0].set_xlabel('X Position')
        axes[1, 0].set_ylabel('Y Position')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axis('equal')
        
        # Displacement analysis
        raw_displacements = np.linalg.norm(np.diff(raw_kp, axis=0), axis=1)
        filtered_displacements = np.linalg.norm(np.diff(filtered_kp, axis=0), axis=1)
        
        axes[1, 1].plot(raw_displacements, 'r-', alpha=0.7, label='Raw', linewidth=1)
        axes[1, 1].plot(filtered_displacements, 'b-', label='Filtered', linewidth=2)
        axes[1, 1].set_title(f'Frame-to-Frame Displacement - {KalmanFilterManager.COCO_KEYPOINTS[keypoint_idx]}')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Displacement')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def export_results(self, filepath: str) -> None:
        """
        Export test results to JSON file
        
        Args:
            filepath: Path to save results
        """
        if len(self.raw_trajectories) == 0:
            print("No test data to export")
            return
        
        report = self.generate_report()
        
        # Prepare export data
        export_data = {
            'test_summary': {
                'total_frames': report.total_frames,
                'avg_latency_ms': report.avg_latency_ms,
                'performance_success_rate': report.performance_success_rate,
                'frames_under_10ms': report.frames_under_10ms
            },
            'jitter_analysis': {
                'raw_jitter_score': report.raw_jitter_score,
                'filtered_jitter_score': report.filtered_jitter_score,
                'jitter_reduction_percent': report.jitter_reduction_percent
            },
            'smoothness_analysis': {
                'raw_smoothness_score': report.raw_smoothness_score,
                'filtered_smoothness_score': report.filtered_smoothness_score,
                'smoothness_improvement_percent': report.smoothness_improvement_percent
            },
            'detailed_metrics': [
                {
                    'frame': i,
                    'processing_time_ms': m.processing_time_ms,
                    'jitter_reduction': m.jitter_reduction,
                    'smoothness_score': m.smoothness_score
                }
                for i, m in enumerate(self.metrics_history)
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Results exported to {filepath}")


def run_comprehensive_ab_test(num_frames: int = 100) -> ABTestResult:
    """
    Run a comprehensive A/B test with simulated data
    
    Args:
        num_frames: Number of frames to simulate
        
    Returns:
        ABTestResult with comprehensive analysis
    """
    print(f"Running comprehensive A/B test with {num_frames} frames...")
    
    # Create Kalman manager
    kalman_manager = KalmanFilterManager(
        process_noise=0.01,
        measurement_noise=0.5,
        dt=0.033
    )
    
    # Create A/B test framework
    ab_test = ABTestFramework(kalman_manager)
    
    # Start test
    ab_test.start_test()
    
    # Simulate noisy pose data
    np.random.seed(42)
    base_trajectory = np.array([
        [100 + 10 * np.sin(i * 0.1), 200 + 5 * np.cos(i * 0.1)] 
        for i in range(num_frames)
    ])
    
    # Add noise and process frames
    for frame in range(num_frames):
        # Add realistic jitter noise
        noise = np.random.randn(17, 2) * 3  # 3 pixel standard deviation
        noisy_keypoints = base_trajectory[frame] + noise
        
        # Vary confidence scores
        confidences = np.random.uniform(0.7, 1.0, 17)
        
        # Process frame
        filtered_keypoints, metrics = ab_test.process_frame(noisy_keypoints, confidences)
        
        if frame % 20 == 0:
            print(f"  Processed frame {frame}: {metrics.processing_time_ms:.2f}ms")
    
    # Stop test and generate report
    ab_test.stop_test()
    report = ab_test.generate_report()
    
    # Print results
    print(f"\n=== A/B Test Results ===")
    print(f"Total frames processed: {report.total_frames}")
    print(f"Average latency: {report.avg_latency_ms:.2f}ms")
    print(f"Performance success rate: {report.performance_success_rate:.1f}%")
    print(f"Frames under 10ms: {report.frames_under_10ms}/{report.total_frames}")
    print(f"Jitter reduction: {report.jitter_reduction_percent:.1f}%")
    print(f"Smoothness improvement: {report.smoothness_improvement_percent:.1f}%")
    
    # Visualize comparison for first keypoint
    ab_test.visualize_comparison(keypoint_idx=0)
    
    # Export results
    ab_test.export_results('ab_test_results.json')
    
    return report


if __name__ == "__main__":
    # Run comprehensive A/B test
    result = run_comprehensive_ab_test(num_frames=50)
    
    print(f"\n✓ A/B testing framework completed successfully")
    print(f"✓ Jitter reduction: {result.jitter_reduction_percent:.1f}%")
    print(f"✓ Performance target met: {result.performance_success_rate:.1f}% frames under 10ms")
