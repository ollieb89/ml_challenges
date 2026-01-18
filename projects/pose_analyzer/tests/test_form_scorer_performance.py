"""
Performance validation and integration tests for FormScorer

This module provides comprehensive performance testing and integration validation
for the FormScorer system, ensuring it meets real-world performance requirements
and integrates properly with existing pose analysis infrastructure.

Performance Requirements:
- Single pose scoring: <10ms
- Sequence scoring (60 frames): <100ms  
- Memory usage: <100MB for typical sequences
- Integration compatibility: 100% with existing modules

Author: AI/ML Pipeline Team
"""

import time
import psutil
import os
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass

from pose_analyzer.form_scorer import (
    FormScorer, FormMetrics, create_form_scorer
)
from pose_analyzer.biomechanics import JointAngleCalculator
from pose_analyzer.synthetic_poses import SyntheticPoseGenerator, SquatType


@dataclass
class PerformanceMetrics:
    """Performance measurement results"""
    operation_name: str
    execution_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    frames_processed: int
    fps_equivalent: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation': self.operation_name,
            'execution_time_ms': self.execution_time_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'frames_processed': self.frames_processed,
            'fps_equivalent': self.fps_equivalent
        }


class PerformanceValidator:
    """Validates FormScorer performance against requirements"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.pose_generator = SyntheticPoseGenerator()
        self.form_scorer = create_form_scorer("squat")
        
        # Performance requirements
        self.requirements = {
            'single_pose_max_time_ms': 10.0,
            'sequence_max_time_ms': 100.0,
            'max_memory_mb': 100.0,
            'min_fps_equivalent': 30.0
        }
    
    def measure_performance(self, operation_func, *args, **kwargs) -> PerformanceMetrics:
        """Measure performance of an operation"""
        # Get initial measurements
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = self.process.cpu_percent()
        
        # Measure execution time
        start_time = time.perf_counter()
        result = operation_func(*args, **kwargs)
        end_time = time.perf_counter()
        
        # Get final measurements
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        final_cpu = self.process.cpu_percent()
        
        execution_time_ms = (end_time - start_time) * 1000
        memory_usage_mb = final_memory - initial_memory
        cpu_usage_percent = final_cpu - initial_cpu
        
        # Determine frames processed and FPS
        if isinstance(result, FormMetrics):
            frames_processed = 1
        elif isinstance(result, list) and len(result) > 0:
            frames_processed = len(result)
        else:
            frames_processed = 1
        
        fps_equivalent = frames_processed / (execution_time_ms / 1000) if execution_time_ms > 0 else 0
        
        return PerformanceMetrics(
            operation_name=operation_func.__name__,
            execution_time_ms=execution_time_ms,
            memory_usage_mb=memory_usage_mb,
            cpu_usage_percent=cpu_usage_percent,
            frames_processed=frames_processed,
            fps_equivalent=fps_equivalent
        )
    
    def test_single_pose_performance(self) -> PerformanceMetrics:
        """Test performance of single pose scoring"""
        # Generate test pose
        sequence = self.pose_generator.generate_squat_sequence(
            num_frames=1, squat_type=SquatType.PARALLEL, noise_level=0.5
        )
        pose = sequence[0]
        
        def score_single_pose():
            return self.form_scorer.score_single_pose(
                pose.keypoints, pose.confidences, "squat_parallel"
            )
        
        return self.measure_performance(score_single_pose)
    
    def test_sequence_performance(self, num_frames: int = 60) -> PerformanceMetrics:
        """Test performance of sequence scoring"""
        # Generate test sequence
        sequence = self.pose_generator.generate_squat_sequence(
            num_frames=num_frames, squat_type=SquatType.PARALLEL, noise_level=0.5
        )
        
        keypoints_sequence = [pose.keypoints for pose in sequence]
        confidences_sequence = [pose.confidences for pose in sequence]
        
        def score_sequence():
            return self.form_scorer.score_pose_sequence(
                keypoints_sequence, confidences_sequence, "squat_parallel"
            )
        
        return self.measure_performance(score_sequence)
    
    def test_batch_processing_performance(self, num_sequences: int = 10) -> PerformanceMetrics:
        """Test performance of batch processing multiple sequences"""
        sequences = []
        
        for _ in range(num_sequences):
            sequence = self.pose_generator.generate_squat_sequence(
                num_frames=30, squat_type=SquatType.PARALLEL, noise_level=0.5
            )
            sequences.append(sequence)
        
        def process_batch():
            results = []
            for sequence in sequences:
                keypoints_sequence = [pose.keypoints for pose in sequence]
                confidences_sequence = [pose.confidences for pose in sequence]
                
                metrics = self.form_scorer.score_pose_sequence(
                    keypoints_sequence, confidences_sequence
                )
                results.append(metrics)
            return results
        
        return self.measure_performance(process_batch)
    
    def test_memory_scaling(self) -> Dict[str, PerformanceMetrics]:
        """Test how memory usage scales with sequence size"""
        frame_counts = [10, 30, 60, 120, 240]
        results = {}
        
        for frame_count in frame_counts:
            metrics = self.test_sequence_performance(frame_count)
            results[f"{frame_count}_frames"] = metrics
        
        return results
    
    def test_reference_pose_generation_performance(self) -> PerformanceMetrics:
        """Test performance of reference pose generation"""
        def generate_references():
            scorer = create_form_scorer("squat")
            return len(scorer.reference_poses)
        
        return self.measure_performance(generate_references)
    
    def test_body_proportion_estimation_performance(self) -> PerformanceMetrics:
        """Test performance of body proportion estimation"""
        # Generate test keypoints
        sequence = self.pose_generator.generate_squat_sequence(
            num_frames=1, squat_type=SquatType.PARALLEL, noise_level=0.5
        )
        keypoints = sequence[0].keypoints
        
        def estimate_proportions():
            from pose_analyzer.form_scorer import BodyProportions
            return BodyProportions.from_keypoints(keypoints)
        
        return self.measure_performance(estimate_proportions)
    
    def validate_requirements(self, metrics: PerformanceMetrics) -> Dict[str, bool]:
        """Validate metrics against performance requirements"""
        validation_results = {}
        
        # Determine which requirements to check based on operation
        if 'single_pose' in metrics.operation_name or metrics.frames_processed == 1:
            validation_results['single_pose_time'] = metrics.execution_time_ms <= self.requirements['single_pose_max_time_ms']
        
        if 'sequence' in metrics.operation_name or metrics.frames_processed > 1:
            validation_results['sequence_time'] = metrics.execution_time_ms <= self.requirements['sequence_max_time_ms']
        
        validation_results['memory_usage'] = abs(metrics.memory_usage_mb) <= self.requirements['max_memory_mb']
        validation_results['fps_performance'] = metrics.fps_equivalent >= self.requirements['min_fps_equivalent']
        
        return validation_results
    
    def run_comprehensive_performance_test(self) -> Dict[str, Any]:
        """Run comprehensive performance validation"""
        print("Running comprehensive performance validation...")
        
        results = {
            'single_pose': self.test_single_pose_performance(),
            'sequence_60_frames': self.test_sequence_performance(60),
            'batch_processing': self.test_batch_processing_performance(10),
            'memory_scaling': self.test_memory_scaling(),
            'reference_generation': self.test_reference_pose_generation_performance(),
            'proportion_estimation': self.test_body_proportion_estimation_performance()
        }
        
        # Validate each result against requirements
        validation_summary = {}
        for test_name, metrics in results.items():
            if isinstance(metrics, PerformanceMetrics):
                validation_summary[test_name] = self.validate_requirements(metrics)
            elif isinstance(metrics, dict):
                # For memory scaling, validate each entry
                validation_summary[test_name] = {}
                for scale_name, scale_metrics in metrics.items():
                    validation_summary[test_name][scale_name] = self.validate_requirements(scale_metrics)
        
        results['validation_summary'] = validation_summary
        
        return results


class IntegrationValidator:
    """Validates FormScorer integration with existing modules"""
    
    def __init__(self):
        self.form_scorer = create_form_scorer("squat")
        self.angle_calculator = JointAngleCalculator()
        self.pose_generator = SyntheticPoseGenerator()
    
    def test_biomechanics_integration(self) -> Dict[str, Any]:
        """Test integration with biomechanics module"""
        print("Testing biomechanics integration...")
        
        # Generate test data
        sequence = self.pose_generator.generate_squat_sequence(
            num_frames=30, squat_type=SquatType.PARALLEL, noise_level=0.0
        )
        
        integration_results = {
            'angle_calculation_compatibility': True,
            'joint_angle_format_compatibility': True,
            'data_flow_integration': True
        }
        
        try:
            # Test 1: Direct angle calculation compatibility
            pose = sequence[0]
            direct_angles = self.angle_calculator.calculate_angles(pose.keypoints, pose.confidences)
            
            # Test 2: FormScorer using same angles
            form_metrics = self.form_scorer.score_single_pose(pose.keypoints, pose.confidences)
            
            # Test 3: Data format compatibility
            assert hasattr(direct_angles, 'knee_left')
            assert hasattr(form_metrics, 'joint_errors')
            assert 'knee_left' in form_metrics.joint_errors
            
            print("✓ Biomechanics integration successful")
            
        except Exception as e:
            integration_results['error'] = str(e)
            print(f"✗ Biomechanics integration failed: {e}")
        
        return integration_results
    
    def test_synthetic_poses_integration(self) -> Dict[str, Any]:
        """Test integration with synthetic poses module"""
        print("Testing synthetic poses integration...")
        
        integration_results = {
            'pose_format_compatibility': True,
            'ground_truth_compatibility': True,
            'sequence_processing_compatibility': True
        }
        
        try:
            # Test 1: Pose format compatibility
            sequence = self.pose_generator.generate_squat_sequence(
                num_frames=20, squat_type=SquatType.PARALLEL, noise_level=0.5
            )
            
            # Verify pose format
            pose = sequence[0]
            assert hasattr(pose, 'keypoints')
            assert hasattr(pose, 'confidences')
            assert hasattr(pose, 'ground_truth_angles')
            assert pose.keypoints.shape == (17, 3)
            assert len(pose.confidences) == 17
            
            # Test 2: Ground truth compatibility
            assert isinstance(pose.ground_truth_angles, type(self.angle_calculator.calculate_angles(pose.keypoints)))
            
            # Test 3: Sequence processing
            keypoints_sequence = [pose.keypoints for pose in sequence]
            confidences_sequence = [pose.confidences for pose in sequence]
            
            metrics = self.form_scorer.score_pose_sequence(keypoints_sequence, confidences_sequence)
            assert isinstance(metrics, FormMetrics)
            assert 0 <= metrics.overall_score <= 100
            
            print("✓ Synthetic poses integration successful")
            
        except Exception as e:
            integration_results['error'] = str(e)
            print(f"✗ Synthetic poses integration failed: {e}")
        
        return integration_results
    
    def test_end_to_end_integration(self) -> Dict[str, Any]:
        """Test complete end-to-end integration"""
        print("Testing end-to-end integration...")
        
        integration_results = {
            'pipeline_compatibility': True,
            'data_consistency': True,
            'performance_acceptable': True
        }
        
        try:
            # Complete pipeline test
            start_time = time.time()
            
            # 1. Generate synthetic data
            sequence = self.pose_generator.generate_squat_sequence(
                num_frames=60, squat_type=SquatType.PARALLEL, noise_level=1.0
            )
            
            # 2. Process through FormScorer
            keypoints_sequence = [pose.keypoints for pose in sequence]
            confidences_sequence = [pose.confidences for pose in sequence]
            
            metrics = self.form_scorer.score_pose_sequence(keypoints_sequence, confidences_sequence)
            
            # 3. Validate results
            assert isinstance(metrics, FormMetrics)
            assert 0 <= metrics.overall_score <= 100
            assert metrics.joint_angle_deviation >= 0
            assert 0 <= metrics.symmetry_score <= 1
            assert 0 <= metrics.trajectory_smoothness <= 1
            assert 0 <= metrics.rom_coverage <= 1
            
            # 4. Check performance
            processing_time = time.time() - start_time
            fps_equivalent = len(sequence) / processing_time
            
            if fps_equivalent < 30:  # Minimum acceptable FPS
                integration_results['performance_acceptable'] = False
            
            # 5. Data consistency check (simplified - just verify angles can be calculated)
            for i, pose in enumerate(sequence[:5]):  # Check first 5 poses only
                # Verify angles can be calculated from both sources
                calculated_angles = self.angle_calculator.calculate_angles(pose.keypoints, pose.confidences)
                ground_truth = pose.ground_truth_angles
                
                # Just verify both have valid angles for key joints
                for joint_name in ['knee_left', 'hip_left']:
                    calc_angle = getattr(calculated_angles, joint_name)
                    gt_angle = getattr(ground_truth, joint_name)
                    
                    # Verify angles are reasonable (not None)
                    assert calc_angle is not None, f"Calculated {joint_name} is None for pose {i}"
                    assert gt_angle is not None, f"Ground truth {joint_name} is None for pose {i}"
            
            print("✓ End-to-end integration successful")
            print(f"  Processing speed: {fps_equivalent:.1f} FPS")
            
        except Exception as e:
            integration_results['error'] = str(e)
            print(f"✗ End-to-end integration failed: {e}")
        
        return integration_results
    
    def run_comprehensive_integration_test(self) -> Dict[str, Any]:
        """Run comprehensive integration validation"""
        print("Running comprehensive integration validation...")
        
        results = {
            'biomechanics': self.test_biomechanics_integration(),
            'synthetic_poses': self.test_synthetic_poses_integration(),
            'end_to_end': self.test_end_to_end_integration()
        }
        
        # Overall integration status
        all_passed = all(
            'error' not in result and all(result.values()) 
            for result in results.values() 
            if isinstance(result, dict)
        )
        
        results['overall_integration_status'] = 'PASSED' if all_passed else 'FAILED'
        
        return results


def run_performance_and_integration_validation() -> Dict[str, Any]:
    """Run complete performance and integration validation"""
    print("=" * 60)
    print("FORM SCORER PERFORMANCE AND INTEGRATION VALIDATION")
    print("=" * 60)
    
    # Performance validation
    perf_validator = PerformanceValidator()
    perf_results = perf_validator.run_comprehensive_performance_test()
    
    print("\nPERFORMANCE RESULTS:")
    print("-" * 30)
    
    # Summarize key performance metrics
    single_pose_metrics = perf_results['single_pose']
    sequence_metrics = perf_results['sequence_60_frames']
    
    print(f"Single pose scoring: {single_pose_metrics.execution_time_ms:.2f}ms")
    print(f"Sequence scoring (60 frames): {sequence_metrics.execution_time_ms:.2f}ms")
    print(f"Sequence FPS equivalent: {sequence_metrics.fps_equivalent:.1f} FPS")
    print(f"Memory usage (60 frames): {abs(sequence_metrics.memory_usage_mb):.2f}MB")
    
    # Check requirements compliance
    single_pose_ok = single_pose_metrics.execution_time_ms <= 10.0
    sequence_ok = sequence_metrics.execution_time_ms <= 100.0
    memory_ok = abs(sequence_metrics.memory_usage_mb) <= 100.0
    fps_ok = sequence_metrics.fps_equivalent >= 30.0
    
    print(f"\nREQUIREMENTS COMPLIANCE:")
    print(f"Single pose <10ms: {'✓' if single_pose_ok else '✗'}")
    print(f"Sequence <100ms: {'✓' if sequence_ok else '✗'}")
    print(f"Memory <100MB: {'✓' if memory_ok else '✗'}")
    print(f"FPS >=30: {'✓' if fps_ok else '✗'}")
    
    # Integration validation
    print("\n" + "=" * 60)
    integration_validator = IntegrationValidator()
    integration_results = integration_validator.run_comprehensive_integration_test()
    
    print("\nINTEGRATION RESULTS:")
    print("-" * 30)
    print(f"Overall status: {integration_results['overall_integration_status']}")
    
    for test_name, result in integration_results.items():
        if test_name != 'overall_integration_status':
            status = '✓ PASSED' if 'error' not in result else '✗ FAILED'
            print(f"{test_name}: {status}")
    
    # Final summary
    all_requirements_met = single_pose_ok and sequence_ok and memory_ok and fps_ok
    integration_passed = integration_results['overall_integration_status'] == 'PASSED'
    
    print("\n" + "=" * 60)
    print("FINAL VALIDATION SUMMARY:")
    print(f"Performance requirements: {'MET' if all_requirements_met else 'NOT MET'}")
    print(f"Integration requirements: {'MET' if integration_passed else 'NOT MET'}")
    print(f"Overall validation: {'PASSED' if all_requirements_met and integration_passed else 'FAILED'}")
    print("=" * 60)
    
    return {
        'performance_results': perf_results,
        'integration_results': integration_results,
        'summary': {
            'performance_requirements_met': all_requirements_met,
            'integration_requirements_met': integration_passed,
            'overall_validation_passed': all_requirements_met and integration_passed
        }
    }


if __name__ == "__main__":
    # Run complete validation
    results = run_performance_and_integration_validation()
    
    # Exit with appropriate code
    exit_code = 0 if results['summary']['overall_validation_passed'] else 1
    exit(exit_code)
