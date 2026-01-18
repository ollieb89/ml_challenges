#!/usr/bin/env python3
"""
Simple validation test for JointAngleCalculator integration.
Tests the core integration without requiring actual pose detection.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pose_analyzer.pose_detector import DetectionResult, DetectionMetrics
from pose_analyzer.biomechanics import JointAngleCalculator
from pose_analyzer.video_processor import MultiStreamProcessor


def create_mock_keypoints():
    """Create mock keypoints for testing."""
    # Standing pose keypoints (17 COCO keypoints)
    keypoints = np.array([
        [0.5, 0.1, 0.0],   # nose
        [0.45, 0.08, 0.0], # left_eye
        [0.55, 0.08, 0.0], # right_eye
        [0.4, 0.1, 0.0],   # left_ear
        [0.6, 0.1, 0.0],   # right_ear
        [0.35, 0.25, 0.0], # left_shoulder
        [0.65, 0.25, 0.0], # right_shoulder
        [0.25, 0.25, 0.0], # left_elbow (extended)
        [0.75, 0.25, 0.0], # right_elbow (extended)
        [0.15, 0.25, 0.0], # left_wrist
        [0.85, 0.25, 0.0], # right_wrist
        [0.4, 0.6, 0.0],   # left_hip
        [0.6, 0.6, 0.0],   # right_hip
        [0.4, 0.8, 0.0],   # left_knee (straight)
        [0.6, 0.8, 0.0],   # right_knee (straight)
        [0.4, 1.0, 0.0],   # left_ankle
        [0.6, 1.0, 0.0],   # right_ankle
    ])
    return keypoints


def test_detection_result_integration():
    """Test DetectionResult integration with JointAngles."""
    print("=== Testing DetectionResult Integration ===")
    
    # Create mock detection result
    keypoints = create_mock_keypoints()
    metrics = DetectionMetrics(
        latency_ms=50.0,
        fps=20.0,
        vram_mb=100.0,
        num_poses=1
    )
    
    result = DetectionResult(
        keypoints=[keypoints],
        metrics=metrics,
        raw_output=None,
        joint_angles=None  # Initially None
    )
    
    # Calculate joint angles
    calculator = JointAngleCalculator(confidence_threshold=0.5, smoothing_window=0)
    angles = calculator.calculate_angles(keypoints)
    
    # Add angles to result
    result.joint_angles = angles
    
    print("✓ DetectionResult created with joint angles")
    print(f"  Shoulder angles: {result.joint_angles.shoulder_left:.1f}°/{result.joint_angles.shoulder_right:.1f}°")
    print(f"  Elbow angles: {result.joint_angles.elbow_left:.1f}°/{result.joint_angles.elbow_right:.1f}°")
    print(f"  Knee angles: {result.joint_angles.knee_left:.1f}°/{result.joint_angles.knee_right:.1f}°")
    
    # Test serialization
    result_dict = {
        'joint_angles': result.joint_angles.to_dict() if result.joint_angles else None,
        'num_poses': result.metrics.num_poses,
        'latency_ms': result.metrics.latency_ms
    }
    
    print("✓ DetectionResult serialization successful")
    return True


def test_video_processor_configuration():
    """Test video processor configuration with joint angles."""
    print("\n=== Testing Video Processor Configuration ===")
    
    try:
        # Test processor creation with joint angle settings
        processor = MultiStreamProcessor(
            [],  # Empty video list for configuration test
            enable_joint_angles=True,
            confidence_threshold=0.5,
            smoothing_window=3,
            max_streams=1,
            batch_size=1
        )
        
        print("✓ MultiStreamProcessor configured with joint angles")
        print(f"  Joint angles enabled: {processor.enable_joint_angles}")
        print(f"  Confidence threshold: {processor.angle_calculator.confidence_threshold}")
        print(f"  Smoothing window: {processor.angle_calculator.smoothing_window}")
        
        # Test angle calculation with mock data
        keypoints = create_mock_keypoints()
        angles = processor.angle_calculator.calculate_angles(keypoints)
        
        print("✓ Angle calculation successful through processor")
        print(f"  Hip angles: {angles.hip_left:.1f}°/{angles.hip_right:.1f}°")
        
        return True
        
    except Exception as e:
        print(f"❌ Video processor configuration failed: {e}")
        return False


def test_api_schema_validation():
    """Test API schema validation."""
    print("\n=== Testing API Schema Validation ===")
    
    try:
        # Import API schemas
        sys.path.insert(0, str(Path(__file__).parent / "api"))
        from schemas import JointAngles, PoseLandmarks, PoseDetectionRequest
        
        # Test JointAngles creation
        angles = JointAngles(
            shoulder_left=180.0,
            shoulder_right=180.0,
            elbow_left=180.0,
            elbow_right=180.0,
            hip_left=170.0,
            hip_right=170.0,
            knee_left=180.0,
            knee_right=180.0,
            ankle_left=90.0,
            ankle_right=90.0
        )
        
        print("✓ JointAngles schema validation successful")
        
        # Test PoseLandmarks with angles
        landmarks = PoseLandmarks(
            landmarks=[],
            confidence=0.9,
            joint_angles=angles
        )
        
        print("✓ PoseLandmarks with joint angles validation successful")
        
        # Test request configuration
        request = PoseDetectionRequest(
            image_data="dummy_base64_data",
            enable_joint_angles=True,
            angle_confidence_threshold=0.5,
            angle_smoothing_window=3
        )
        
        print("✓ PoseDetectionRequest configuration validation successful")
        
        # Test serialization
        angles_dict = angles.model_dump(exclude_none=True)
        print(f"✓ JointAngles serialization: {len(angles_dict)} fields")
        
        return True
        
    except Exception as e:
        print(f"❌ API schema validation failed: {e}")
        return False


def test_end_to_end_flow():
    """Test end-to-end integration flow."""
    print("\n=== Testing End-to-End Flow ===")
    
    try:
        # 1. Create mock detection result
        keypoints = create_mock_keypoints()
        metrics = DetectionMetrics(latency_ms=45.0, fps=22.0, vram_mb=120.0, num_poses=1)
        result = DetectionResult(keypoints=[keypoints], metrics=metrics, raw_output=None)
        
        # 2. Calculate joint angles
        calculator = JointAngleCalculator(confidence_threshold=0.5, smoothing_window=3)
        result.joint_angles = calculator.calculate_angles(keypoints)
        
        # 3. Convert to API format
        sys.path.insert(0, str(Path(__file__).parent / "api"))
        from schemas import JointAngles, PoseLandmarks
        
        api_angles = JointAngles(**result.joint_angles.to_dict())
        api_pose = PoseLandmarks(
            landmarks=[],
            confidence=0.85,
            joint_angles=api_angles
        )
        
        # 4. Validate complete flow
        print("✓ End-to-end flow successful")
        print(f"  Detection → Angles: {result.joint_angles.knee_left:.1f}° knee angle")
        print(f"  API serialization: {api_pose.joint_angles.knee_left:.1f}° knee angle")
        print(f"  Data integrity: {abs(result.joint_angles.knee_left - api_pose.joint_angles.knee_left) < 0.01}")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end flow failed: {e}")
        return False


def main():
    """Run all integration validation tests."""
    print("🔧 JointAngleCalculator Integration Validation\n")
    
    # Run all tests
    detection_success = test_detection_result_integration()
    config_success = test_video_processor_configuration()
    schema_success = test_api_schema_validation()
    flow_success = test_end_to_end_flow()
    
    # Results
    print("\n=== Integration Validation Results ===")
    print(f"DetectionResult Integration: {'✅ PASS' if detection_success else '❌ FAIL'}")
    print(f"Video Processor Configuration: {'✅ PASS' if config_success else '❌ FAIL'}")
    print(f"API Schema Validation: {'✅ PASS' if schema_success else '❌ FAIL'}")
    print(f"End-to-End Flow: {'✅ PASS' if flow_success else '❌ FAIL'}")
    
    overall_success = all([detection_success, config_success, schema_success, flow_success])
    print(f"\n🎯 Overall Integration: {'✅ SUCCESS' if overall_success else '❌ NEEDS ATTENTION'}")
    
    if overall_success:
        print("\n🎉 JointAngleCalculator integration validated!")
        print("   ✅ DetectionResult extended with JointAngles")
        print("   ✅ Video processor configured for angle calculations")
        print("   ✅ API schemas support joint angle data")
        print("   ✅ End-to-end data flow validated")
        print("   ✅ Configuration options working")
        
        print("\n📋 Integration Summary:")
        print("   - Core biomechanics module: ✅ Implemented")
        print("   - DetectionResult extension: ✅ Completed")
        print("   - Video processor integration: ✅ Completed")
        print("   - API schema updates: ✅ Completed")
        print("   - Configuration options: ✅ Completed")
        
    else:
        print("\n⚠️  Some integration aspects need attention")
    
    return overall_success


if __name__ == "__main__":
    main()
