"""
Comprehensive Performance Tests for Kalman Filter Implementation

This module provides extensive testing capabilities to validate the Kalman filter
implementation meets all requirements including performance, accuracy, and reliability.

Test Categories:
- Basic functionality tests
- Performance benchmarking (<10ms latency constraint)
- Accuracy validation with synthetic data
- Edge case handling
- Integration tests with existing pose pipeline
- Stress testing with high-frequency data
"""

import numpy as np
import time
import unittest
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import json

from temporal_analyzer import KalmanFilterManager, FilterMetrics, KeypointState, create_kalman_manager
from ab_testing import ABTestFramework, JitterAnalyzer, run_comprehensive_ab_test


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@dataclass
class TestResult:
    """Result of a performance test"""
    test_name: str
    passed: bool
    execution_time_ms: float
    details: Dict[str, Any]
    error_message: Optional[str] = None


class KalmanFilterPerformanceTests:
    """
    Comprehensive test suite for Kalman filter implementation
    """
    
    def __init__(self):
        self.test_results: List[TestResult] = []
        self.kalman_manager = create_kalman_manager()
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all test categories"""
        print("🧪 Running Comprehensive Kalman Filter Performance Tests...")
        
        # Basic functionality tests
        self.test_basic_functionality()
        
        # Performance tests
        self.test_latency_constraint()
        self.test_high_frequency_processing()
        
        # Accuracy tests
        self.test_jitter_reduction()
        
        # Edge cases
        self.test_missing_keypoints()
        
        # Integration tests
        self.test_ab_testing_integration()
        
        return self.test_results
    
    def test_basic_functionality(self) -> None:
        """Test basic Kalman filter operations"""
        print("\n📋 Testing Basic Functionality...")
        
        try:
            start_time = time.perf_counter()
            
            # Test initialization
            manager = KalmanFilterManager()
            assert len(manager.filters) == 17, "Should have 17 filters"
            assert len(manager.keypoint_states) == 17, "Should have 17 keypoint states"
            
            # Test single keypoint update
            state = manager.update_keypoint(0, 100.0, 200.0, 0.8)
            assert isinstance(state, KeypointState), "Should return KeypointState"
            assert state.x == 100.0, "X position should match input"
            assert state.y == 200.0, "Y position should match input"
            assert state.confidence == 0.8, "Confidence should match input"
            
            # Test frame processing
            keypoints = np.random.randn(17, 2) * 10 + np.array([100, 200])
            confidences = np.ones(17) * 0.9
            
            smoothed_keypoints, metrics = manager.process_frame(keypoints, confidences)
            assert smoothed_keypoints.shape == (17, 2), "Output shape should match input"
            assert isinstance(metrics, FilterMetrics), "Should return FilterMetrics"
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            self.test_results.append(TestResult(
                test_name="Basic Functionality",
                passed=True,
                execution_time_ms=execution_time,
                details={"filters_created": 17, "frames_processed": 1}
            ))
            
            print("  ✅ Basic functionality tests passed")
            
        except Exception as e:
            self.test_results.append(TestResult(
                test_name="Basic Functionality",
                passed=False,
                execution_time_ms=0,
                details={},
                error_message=str(e)
            ))
            print(f"  ❌ Basic functionality test failed: {e}")
    
    def test_latency_constraint(self) -> None:
        """Test <10ms latency constraint"""
        print("\n⚡ Testing Latency Constraint...")
        
        try:
            manager = create_kalman_manager()
            processing_times = []
            
            # Process 100 frames to get reliable statistics
            for frame in range(100):
                keypoints = np.random.randn(17, 2) * 5 + np.array([100, 200])
                confidences = np.random.uniform(0.7, 1.0, 17)
                
                start_time = time.perf_counter()
                smoothed_keypoints, metrics = manager.process_frame(keypoints, confidences)
                processing_time = (time.perf_counter() - start_time) * 1000
                
                processing_times.append(processing_time)
            
            avg_time = np.mean(processing_times)
            max_time = np.max(processing_times)
            success_rate = np.mean(np.array(processing_times) < 10.0) * 100
            
            passed = bool(avg_time < 10.0 and success_rate > 95.0)
            
            self.test_results.append(TestResult(
                test_name="Latency Constraint",
                passed=passed,
                execution_time_ms=float(avg_time),
                details={
                    "avg_processing_time_ms": avg_time,
                    "max_processing_time_ms": max_time,
                    "success_rate_percent": success_rate,
                    "frames_under_10ms": sum(1 for t in processing_times if t < 10.0)
                }
            ))
            
            if passed:
                print(f"  ✅ Latency constraint met: {avg_time:.2f}ms avg, {success_rate:.1f}% under 10ms")
            else:
                print(f"  ❌ Latency constraint failed: {avg_time:.2f}ms avg, {success_rate:.1f}% under 10ms")
                
        except Exception as e:
            self.test_results.append(TestResult(
                test_name="Latency Constraint",
                passed=False,
                execution_time_ms=0,
                details={},
                error_message=str(e)
            ))
            print(f"  ❌ Latency test failed: {e}")
    
    def test_jitter_reduction(self) -> None:
        """Test jitter reduction effectiveness"""
        print("\n🎯 Testing Jitter Reduction...")
        
        try:
            # Create test data with known jitter pattern
            np.random.seed(42)
            base_position = np.array([100, 200])
            jitter_amplitude = 5.0
            
            manager = create_kalman_manager()
            ab_test = ABTestFramework(manager)
            ab_test.start_test()
            
            raw_trajectory = []
            filtered_trajectory = []
            
            # Generate 50 frames with controlled jitter
            for frame in range(50):
                # Add sinusoidal jitter
                jitter_x = jitter_amplitude * np.sin(frame * 0.5) + np.random.randn() * 2
                jitter_y = jitter_amplitude * np.cos(frame * 0.5) + np.random.randn() * 2
                
                noisy_keypoints = np.random.randn(17, 2) * 0.5 + base_position + np.array([jitter_x, jitter_y])
                confidences = np.ones(17) * 0.9
                
                raw_trajectory.append(noisy_keypoints.copy())
                
                smoothed_keypoints, metrics = ab_test.process_frame(noisy_keypoints, confidences)
                filtered_trajectory.append(smoothed_keypoints.copy())
            
            ab_test.stop_test()
            
            # Calculate jitter metrics
            raw_trajectory = np.array(raw_trajectory)
            filtered_trajectory = np.array(filtered_trajectory)
            
            jitter_analyzer = JitterAnalyzer()
            raw_jitter = jitter_analyzer.calculate_jitter(raw_trajectory)
            filtered_jitter = jitter_analyzer.calculate_jitter(filtered_trajectory)
            jitter_reduction = ((raw_jitter - filtered_jitter) / raw_jitter * 100) if raw_jitter > 0 else 0
            
            passed = bool(jitter_reduction > 20.0)  # Expect at least 20% jitter reduction
            
            self.test_results.append(TestResult(
                test_name="Jitter Reduction",
                passed=passed,
                execution_time_ms=0.0,
                details={
                    "raw_jitter_score": float(raw_jitter),
                    "filtered_jitter_score": float(filtered_jitter),
                    "jitter_reduction_percent": float(jitter_reduction)
                }
            ))
            
            if passed:
                print(f"  ✅ Jitter reduction effective: {jitter_reduction:.1f}% reduction")
            else:
                print(f"  ❌ Jitter reduction insufficient: {jitter_reduction:.1f}% reduction")
                
        except Exception as e:
            self.test_results.append(TestResult(
                test_name="Jitter Reduction",
                passed=False,
                execution_time_ms=0,
                details={},
                error_message=str(e)
            ))
            print(f"  ❌ Jitter reduction test failed: {e}")
    
    def test_high_frequency_processing(self) -> None:
        """Test processing at high frame rates"""
        print("\n🚀 Testing High Frequency Processing...")
        
        try:
            manager = create_kalman_manager()
            
            # Simulate 60 FPS processing
            target_fps = 60
            frame_time_ms = 1000.0 / target_fps
            processing_times = []
            
            for frame in range(120):  # 2 seconds at 60 FPS
                keypoints = np.random.randn(17, 2) * 3 + np.array([100, 200])
                confidences = np.ones(17) * 0.95
                
                start_time = time.perf_counter()
                smoothed_keypoints, metrics = manager.process_frame(keypoints, confidences)
                processing_time = (time.perf_counter() - start_time) * 1000
                
                processing_times.append(processing_time)
            
            avg_time = np.mean(processing_times)
            max_time = np.max(processing_times)
            success_rate = np.mean(np.array(processing_times) < frame_time_ms) * 100
            
            passed = bool(success_rate > 90.0)  # 90% of frames should process within frame time
            
            self.test_results.append(TestResult(
                test_name="High Frequency Processing",
                passed=passed,
                execution_time_ms=float(avg_time),
                details={
                    "target_fps": target_fps,
                    "frame_time_ms": frame_time_ms,
                    "avg_processing_time_ms": avg_time,
                    "max_processing_time_ms": max_time,
                    "success_rate_percent": success_rate
                }
            ))
            
            if passed:
                print(f"  ✅ High frequency processing successful: {success_rate:.1f}% frames within {frame_time_ms:.1f}ms")
            else:
                print(f"  ❌ High frequency processing failed: {success_rate:.1f}% frames within {frame_time_ms:.1f}ms")
                
        except Exception as e:
            self.test_results.append(TestResult(
                test_name="High Frequency Processing",
                passed=False,
                execution_time_ms=0,
                details={},
                error_message=str(e)
            ))
            print(f"  ❌ High frequency test failed: {e}")
    
    def test_missing_keypoints(self) -> None:
        """Test handling of missing/low-confidence keypoints"""
        print("\n🔍 Testing Missing Keypoints...")
        
        try:
            manager = create_kalman_manager()
            
            # Create keypoints with some missing (low confidence)
            keypoints = np.random.randn(17, 2) * 10 + np.array([100, 200])
            confidences = np.ones(17)
            confidences[5] = 0.05  # Left shoulder missing
            confidences[12] = 0.03  # Right hip missing
            
            smoothed_keypoints, metrics = manager.process_frame(keypoints, confidences)
            
            # Check that output shape is maintained
            assert smoothed_keypoints.shape == (17, 2), "Should maintain output shape"
            
            # Check that low confidence keypoints are handled gracefully
            # (they should use last known state or reasonable defaults)
            assert not np.any(np.isnan(smoothed_keypoints)), "No NaN values in output"
            assert not np.any(np.isinf(smoothed_keypoints)), "No infinite values in output"
            
            passed = True
            
            self.test_results.append(TestResult(
                test_name="Missing Keypoints",
                passed=passed,
                execution_time_ms=metrics.processing_time_ms,
                details={
                    "low_confidence_keypoints": [5, 12],
                    "output_shape": smoothed_keypoints.shape,
                    "has_nan": np.any(np.isnan(smoothed_keypoints)),
                    "has_inf": np.any(np.isinf(smoothed_keypoints))
                }
            ))
            
            print("  ✅ Missing keypoints handled correctly")
            
        except Exception as e:
            self.test_results.append(TestResult(
                test_name="Missing Keypoints",
                passed=False,
                execution_time_ms=0,
                details={},
                error_message=str(e)
            ))
            print(f"  ❌ Missing keypoints test failed: {e}")
    
    def test_ab_testing_integration(self) -> None:
        """Test A/B testing framework integration"""
        print("\n🧪 Testing A/B Testing Integration...")
        
        try:
            start_time = time.perf_counter()
            
            # Run comprehensive A/B test
            result = run_comprehensive_ab_test(num_frames=30)
            
            execution_time = (time.perf_counter() - start_time) * 1000
            
            # Validate results
            passed = bool(
                result.total_frames == 30 and
                result.performance_success_rate > 80.0 and
                result.jitter_reduction_percent > 0.0
            )
            
            self.test_results.append(TestResult(
                test_name="A/B Testing Integration",
                passed=passed,
                execution_time_ms=execution_time,
                details={
                    "total_frames": result.total_frames,
                    "jitter_reduction_percent": result.jitter_reduction_percent,
                    "performance_success_rate": result.performance_success_rate,
                    "avg_latency_ms": result.avg_latency_ms
                }
            ))
            
            if passed:
                print(f"  ✅ A/B testing integration successful: {result.jitter_reduction_percent:.1f}% jitter reduction")
            else:
                print(f"  ❌ A/B testing integration issues detected")
                
        except Exception as e:
            self.test_results.append(TestResult(
                test_name="A/B Testing Integration",
                passed=False,
                execution_time_ms=0,
                details={},
                error_message=str(e)
            ))
            print(f"  ❌ A/B testing integration failed: {e}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.passed)
        failed_tests = total_tests - passed_tests
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": float((passed_tests / total_tests * 100) if total_tests > 0 else 0)
            },
            "performance_summary": {
                "avg_execution_time_ms": float(np.mean([r.execution_time_ms for r in self.test_results if r.execution_time_ms > 0])),
                "max_execution_time_ms": float(np.max([r.execution_time_ms for r in self.test_results if r.execution_time_ms > 0]))
            },
            "detailed_results": [
                {
                    "test_name": result.test_name,
                    "passed": result.passed,
                    "execution_time_ms": result.execution_time_ms,
                    "details": result.details,
                    "error_message": result.error_message
                }
                for result in self.test_results
            ]
        }
        
        return report
    
    def save_report(self, filepath: str) -> None:
        """Save test report to JSON file"""
        report = self.generate_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        
        print(f"📄 Test report saved to {filepath}")


def run_performance_tests() -> Dict[str, Any]:
    """Run all performance tests and return results"""
    test_suite = KalmanFilterPerformanceTests()
    test_results = test_suite.run_all_tests()
    
    # Generate and save report
    report = test_suite.generate_report()
    
    # Save to data/reports
    from pathlib import Path
    report_dir = Path(__file__).resolve().parents[4] / "data" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / 'kalman_filter_performance_report.json'
    
    test_suite.save_report(str(report_path))
    
    # Print summary
    print(f"\n🎯 Performance Test Summary:")
    print(f"   Total Tests: {report['test_summary']['total_tests']}")
    print(f"   Passed: {report['test_summary']['passed_tests']}")
    print(f"   Failed: {report['test_summary']['failed_tests']}")
    print(f"   Success Rate: {report['test_summary']['success_rate']:.1f}%")
    
    if report['test_summary']['success_rate'] >= 90:
        print("🎉 All critical performance tests passed!")
    else:
        print("⚠️  Some performance tests failed - review detailed report")
    
    return report


if __name__ == "__main__":
    # Run comprehensive performance tests
    results = run_performance_tests()
    
    print(f"\n✅ Kalman filter performance testing completed")
    print(f"✅ Success rate: {results['test_summary']['success_rate']:.1f}%")
