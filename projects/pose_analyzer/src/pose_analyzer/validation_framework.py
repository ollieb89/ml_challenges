"""
Biomechanics Validation Framework

This module provides comprehensive validation tools for joint angle calculations,
including error measurement, statistical analysis, and systematic testing
infrastructure.

Key Features:
- Error measurement utilities for joint angles
- Statistical analysis and reporting
- Systematic validation test harness
- Performance benchmarking
- Comprehensive accuracy metrics

Author: AI/ML Pipeline Team
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json
import time

from .biomechanics import JointAngles, JointAngleCalculator
from .synthetic_poses import SyntheticPoseGenerator, SyntheticPose, SquatType


@dataclass
class ValidationError:
    """Error measurement for a single joint angle"""
    joint_name: str
    ground_truth: float
    measured: float
    absolute_error: float
    relative_error: float
    is_within_tolerance: bool


@dataclass
class ValidationResults:
    """Comprehensive validation results for a test sequence"""
    total_frames: int
    joint_errors: Dict[str, List[ValidationError]] = field(default_factory=dict)
    summary_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    performance_stats: Dict[str, float] = field(default_factory=dict)
    tolerance_degrees: float = 5.0


@dataclass
class TestConfiguration:
    """Configuration for validation tests"""
    tolerance_degrees: float = 5.0
    noise_levels: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5])
    num_squats: int = 50
    frames_per_squat: int = 60
    squat_types: List[SquatType] = field(default_factory=lambda: list(SquatType))
    include_temporal_smoothing: bool = True


class ErrorAnalyzer:
    """Analyzes and reports on joint angle calculation errors"""
    
    def __init__(self, tolerance_degrees: float = 5.0):
        """
        Initialize error analyzer
        
        Args:
            tolerance_degrees: Acceptable error threshold in degrees
        """
        self.tolerance_degrees = tolerance_degrees
    
    def calculate_angle_error(
        self, 
        ground_truth: Optional[float], 
        measured: Optional[float],
        joint_name: str
    ) -> Optional[ValidationError]:
        """
        Calculate error between ground truth and measured angles
        
        Args:
            ground_truth: Ground truth angle in degrees
            measured: Measured angle in degrees
            joint_name: Name of the joint
            
        Returns:
            ValidationError object or None if either angle is None
        """
        if ground_truth is None or measured is None:
            return None
        
        absolute_error = abs(measured - ground_truth)
        relative_error = (absolute_error / abs(ground_truth)) * 100 if ground_truth != 0 else 0
        is_within_tolerance = absolute_error <= self.tolerance_degrees
        
        return ValidationError(
            joint_name=joint_name,
            ground_truth=ground_truth,
            measured=measured,
            absolute_error=absolute_error,
            relative_error=relative_error,
            is_within_tolerance=is_within_tolerance
        )
    
    def analyze_sequence_errors(
        self, 
        ground_truth_sequence: List[JointAngles],
        measured_sequence: List[JointAngles]
    ) -> Dict[str, List[ValidationError]]:
        """
        Analyze errors for a complete sequence
        
        Args:
            ground_truth_sequence: List of ground truth joint angles
            measured_sequence: List of measured joint angles
            
        Returns:
            Dictionary mapping joint names to lists of validation errors
        """
        if len(ground_truth_sequence) != len(measured_sequence):
            raise ValueError("Sequence lengths must match")
        
        joint_errors = defaultdict(list)
        joint_names = [
            'shoulder_left', 'shoulder_right', 'elbow_left', 'elbow_right',
            'hip_left', 'hip_right', 'knee_left', 'knee_right',
            'ankle_left', 'ankle_right'
        ]
        
        for gt_angles, measured_angles in zip(ground_truth_sequence, measured_sequence):
            for joint_name in joint_names:
                gt_value = getattr(gt_angles, joint_name)
                measured_value = getattr(measured_angles, joint_name)
                
                error = self.calculate_angle_error(gt_value, measured_value, joint_name)
                if error:
                    joint_errors[joint_name].append(error)
        
        return dict(joint_errors)
    
    def calculate_summary_statistics(self, joint_errors: Dict[str, List[ValidationError]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate summary statistics for each joint
        
        Args:
            joint_errors: Dictionary of joint errors
            
        Returns:
            Dictionary of summary statistics per joint
        """
        summary_stats = {}
        
        for joint_name, errors in joint_errors.items():
            if not errors:
                continue
            
            absolute_errors = [e.absolute_error for e in errors]
            relative_errors = [e.relative_error for e in errors]
            within_tolerance = [e.is_within_tolerance for e in errors]
            
            stats = {
                'mean_absolute_error': np.mean(absolute_errors),
                'std_absolute_error': np.std(absolute_errors),
                'max_absolute_error': np.max(absolute_errors),
                'min_absolute_error': np.min(absolute_errors),
                'mean_relative_error': np.mean(relative_errors),
                'std_relative_error': np.std(relative_errors),
                'tolerance_success_rate': np.mean(within_tolerance) * 100,
                'total_measurements': len(errors)
            }
            
            summary_stats[joint_name] = stats
        
        return summary_stats


class ValidationHarness:
    """
    Systematic validation test harness for biomechanics analysis
    """
    
    def __init__(self, config: Optional[TestConfiguration] = None):
        """
        Initialize validation harness
        
        Args:
            config: Test configuration, defaults to standard settings
        """
        self.config = config or TestConfiguration()
        self.pose_generator = SyntheticPoseGenerator()
        self.angle_calculator = JointAngleCalculator()
        self.error_analyzer = ErrorAnalyzer(self.config.tolerance_degrees)
    
    def run_single_squat_validation(
        self, 
        squat_type: SquatType = SquatType.PARALLEL,
        noise_level: float = 1.0
    ) -> ValidationResults:
        """
        Run validation on a single squat sequence
        
        Args:
            squat_type: Type of squat to test
            noise_level: Noise level for pose generation
            
        Returns:
            ValidationResults with comprehensive error analysis
        """
        # Generate test sequence
        sequence = self.pose_generator.generate_squat_sequence(
            num_frames=self.config.frames_per_squat,
            squat_type=squat_type,
            noise_level=noise_level
        )
        
        # Extract ground truth and calculate measured angles
        ground_truth_sequence = [pose.ground_truth_angles for pose in sequence]
        measured_sequence = []
        
        processing_times = []
        
        for pose in sequence:
            start_time = time.perf_counter()
            
            # Calculate angles using the biomechanics module
            measured_angles = self.angle_calculator.calculate_angles(
                pose.keypoints, pose.confidences
            )
            
            processing_time = (time.perf_counter() - start_time) * 1000
            processing_times.append(processing_time)
            
            measured_sequence.append(measured_angles)
        
        # Analyze errors
        joint_errors = self.error_analyzer.analyze_sequence_errors(
            ground_truth_sequence, measured_sequence
        )
        
        # Calculate summary statistics
        summary_stats = self.error_analyzer.calculate_summary_statistics(joint_errors)
        
        # Performance statistics
        performance_stats = {
            'avg_processing_time_ms': np.mean(processing_times),
            'max_processing_time_ms': np.max(processing_times),
            'total_processing_time_ms': np.sum(processing_times),
            'frames_per_second': 1000 / np.mean(processing_times)
        }
        
        return ValidationResults(
            total_frames=len(sequence),
            joint_errors=joint_errors,
            summary_stats=summary_stats,
            performance_stats=performance_stats,
            tolerance_degrees=self.config.tolerance_degrees
        )
    
    def run_comprehensive_validation(self) -> Dict[str, ValidationResults]:
        """
        Run comprehensive validation across all configurations
        
        Returns:
            Dictionary mapping test names to validation results
        """
        results = {}
        
        for noise_level in self.config.noise_levels:
            for squat_type in self.config.squat_types:
                test_name = f"{squat_type.value}_noise_{noise_level}"
                
                print(f"Running validation: {test_name}...")
                
                result = self.run_single_squat_validation(squat_type, noise_level)
                results[test_name] = result
                
                # Print quick summary
                knee_stats = result.summary_stats.get('knee_left', {})
                success_rate = knee_stats.get('tolerance_success_rate', 0)
                avg_error = knee_stats.get('mean_absolute_error', 0)
                
                print(f"  Knee success rate: {success_rate:.1f}%, Avg error: {avg_error:.2f}°")
        
        return results
    
    def run_50_squat_validation(self) -> ValidationResults:
        """
        Run validation on 50 squat repetitions as specified in the challenge
        
        Returns:
            Aggregated validation results across all squats
        """
        print("Running 50 squat validation...")
        
        all_joint_errors = defaultdict(list)
        all_processing_times = []
        total_frames = 0
        
        for squat_idx in range(self.config.num_squats):
            if squat_idx % 10 == 0:
                print(f"  Processing squat {squat_idx + 1}/{self.config.num_squats}")
            
            # Vary squat type for diversity
            squat_types = list(SquatType)
            squat_type = squat_types[squat_idx % len(squat_types)]
            
            # Use moderate noise level
            noise_level = 1.0
            
            # Generate and process single squat
            sequence = self.pose_generator.generate_squat_sequence(
                num_frames=self.config.frames_per_squat,
                squat_type=squat_type,
                noise_level=noise_level
            )
            
            # Process each frame
            for pose in sequence:
                start_time = time.perf_counter()
                
                measured_angles = self.angle_calculator.calculate_angles(
                    pose.keypoints, pose.confidences
                )
                
                processing_time = (time.perf_counter() - start_time) * 1000
                all_processing_times.append(processing_time)
                
                # Calculate error
                error = self.error_analyzer.calculate_angle_error(
                    pose.ground_truth_angles.knee_left,
                    measured_angles.knee_left,
                    'knee_left'
                )
                if error:
                    all_joint_errors['knee_left'].append(error)
            
            total_frames += len(sequence)
        
        # Calculate aggregated statistics
        summary_stats = self.error_analyzer.calculate_summary_statistics(dict(all_joint_errors))
        
        performance_stats = {
            'avg_processing_time_ms': np.mean(all_processing_times),
            'max_processing_time_ms': np.max(all_processing_times),
            'total_processing_time_ms': np.sum(all_processing_times),
            'frames_per_second': 1000 / np.mean(all_processing_times),
            'total_frames_processed': total_frames
        }
        
        return ValidationResults(
            total_frames=total_frames,
            joint_errors=dict(all_joint_errors),
            summary_stats=summary_stats,
            performance_stats=performance_stats,
            tolerance_degrees=self.config.tolerance_degrees
        )


class ValidationReporter:
    """Generates comprehensive validation reports"""
    
    def __init__(self, tolerance_degrees: float = 5.0):
        """
        Initialize validation reporter
        
        Args:
            tolerance_degrees: Error tolerance threshold
        """
        self.tolerance_degrees = tolerance_degrees
    
    def generate_error_distribution_plot(
        self, 
        joint_errors: Dict[str, List[ValidationError]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Generate error distribution plots
        
        Args:
            joint_errors: Dictionary of joint errors
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot error distributions for major joints
        major_joints = ['knee_left', 'hip_left', 'shoulder_left', 'elbow_left']
        
        for i, joint_name in enumerate(major_joints):
            if joint_name not in joint_errors:
                continue
            
            errors = [e.absolute_error for e in joint_errors[joint_name]]
            
            axes[i].hist(errors, bins=20, alpha=0.7, edgecolor='black')
            axes[i].axvline(x=self.tolerance_degrees, color='red', linestyle='--', 
                          label=f'Tolerance ({self.tolerance_degrees}°)')
            axes[i].axvline(x=np.mean(errors), color='green', linestyle='-', 
                          label=f'Mean ({np.mean(errors):.2f}°)')
            
            axes[i].set_title(f'{joint_name.replace("_", " ").title()} Error Distribution')
            axes[i].set_xlabel('Absolute Error (degrees)')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Error distribution plot saved to {save_path}")
        
        plt.close()  # Close figure to free memory
    
    def generate_summary_report(self, results: ValidationResults) -> str:
        """
        Generate text summary of validation results
        
        Args:
            results: Validation results to summarize
            
        Returns:
            Formatted summary string
        """
        report = []
        report.append("# Biomechanics Validation Report")
        report.append("")
        report.append(f"Tolerance: ±{results.tolerance_degrees}°")
        report.append(f"Total Frames: {results.total_frames}")
        report.append("")
        
        # Performance summary
        report.append("## Performance Summary")
        for key, value in results.performance_stats.items():
            if 'time' in key:
                report.append(f"- {key.replace('_', ' ').title()}: {value:.2f}ms")
            else:
                report.append(f"- {key.replace('_', ' ').title()}: {value:.2f}")
        report.append("")
        
        # Accuracy summary
        report.append("## Accuracy Summary")
        for joint_name, stats in results.summary_stats.items():
            report.append(f"### {joint_name.replace('_', ' ').title()}")
            report.append(f"- Mean Absolute Error: {stats['mean_absolute_error']:.2f}°")
            report.append(f"- Standard Deviation: {stats['std_absolute_error']:.2f}°")
            report.append(f"- Maximum Error: {stats['max_absolute_error']:.2f}°")
            report.append(f"- Tolerance Success Rate: {stats['tolerance_success_rate']:.1f}%")
            report.append("")
        
        # Overall assessment
        knee_success = results.summary_stats.get('knee_left', {}).get('tolerance_success_rate', 0)
        overall_assessment = "PASS" if knee_success >= 90 else "NEEDS IMPROVEMENT"
        
        report.append("## Overall Assessment")
        report.append(f"Knee Angle Accuracy: {overall_assessment}")
        report.append(f"Success Rate: {knee_success:.1f}%")
        report.append("")
        
        return "\n".join(report)


def create_validation_harness() -> ValidationHarness:
    """Factory function to create optimized validation harness"""
    config = TestConfiguration(
        tolerance_degrees=5.0,
        noise_levels=[0.5, 1.0, 1.5],
        num_squats=50,
        frames_per_squat=60
    )
    return ValidationHarness(config)


if __name__ == "__main__":
    # Quick test
    print("Testing ValidationHarness...")
    
    harness = create_validation_harness()
    
    # Run single test
    result = harness.run_single_squat_validation(
        squat_type=SquatType.PARALLEL,
        noise_level=1.0
    )
    
    print(f"Processed {result.total_frames} frames")
    print(f"Average processing time: {result.performance_stats['avg_processing_time_ms']:.2f}ms")
    
    knee_stats = result.summary_stats.get('knee_left', {})
    print(f"Knee mean error: {knee_stats.get('mean_absolute_error', 0):.2f}°")
    print(f"Knee success rate: {knee_stats.get('tolerance_success_rate', 0):.1f}%")
    
    # Test 50 squat validation
    print("\nRunning 50 squat validation...")
    fifty_squat_result = harness.run_50_squat_validation()
    
    knee_stats_50 = fifty_squat_result.summary_stats.get('knee_left', {})
    print(f"50 squat knee success rate: {knee_stats_50.get('tolerance_success_rate', 0):.1f}%")
    
    print("\n✓ ValidationHarness test completed successfully")
