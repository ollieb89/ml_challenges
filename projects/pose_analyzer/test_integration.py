#!/usr/bin/env python3
"""
Integration test for JointAngleCalculator with pose detection pipeline.
Tests the complete integration with both video processor and API schemas.
"""

import numpy as np
import cv2
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pose_analyzer.pose_detector import YOLOPosev11Detector, DetectionResult
from pose_analyzer.biomechanics import JointAngleCalculator
from pose_analyzer.video_processor import MultiStreamProcessor


def create_test_frame():
    """Create a simple test frame with a person silhouette."""
    # Create a simple test image (640x480)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add a simple person silhouette (standing pose)
    # Head
    cv2.circle(frame, (320, 80), 30, (255, 255, 255), -1)
    # Body
    cv2.rectangle(frame, (300, 110), (340, 300), (255, 255, 255), -1)
    # Arms
    cv2.rectangle(frame, (250, 120), (300, 140), (255, 255, 255), -1)  # Left arm
    cv2.rectangle(frame, (340, 120), (390, 140), (255, 255, 255), -1)  # Right arm
    # Legs
    cv2.rectangle(frame, (305, 300), (325, 420), (255, 255, 255), -1)  # Left leg
    cv2.rectangle(frame, (335, 300), (355, 420), (255, 255, 255), -1)  # Right leg
    
    return frame


def test_detector_integration():
    """Test JointAngleCalculator integration with pose detector."""
    print("=== Testing Detector Integration ===")
    
    # Create detector and calculator
    detector = YOLOPosev11Detector(model_variant="n")
    calculator = JointAngleCalculator(confidence_threshold=0.5, smoothing_window=0)
    
    # Create test frame
    frame = create_test_frame()
    
    # Detect pose
    result = detector.detect(frame)
    
    print(f"✓ Detected {len(result.keypoints)} poses")
    print(f"✓ Processing time: {result.metrics.latency_ms:.2f}ms")
    
    if result.keypoints:
        # Calculate angles
        keypoints = result.keypoints[0]
        angles = calculator.calculate_angles(keypoints)
        
        print(f"✓ Joint angles calculated:")
        print(f"  Shoulder: {angles.shoulder_left:.1f}°/{angles.shoulder_right:.1f}°")
        print(f"  Elbow: {angles.elbow_left:.1f}°/{angles.elbow_right:.1f}°")
        print(f"  Knee: {angles.knee_left:.1f}°/{angles.knee_right:.1f}°")
        
        # Test integration with DetectionResult
        result.joint_angles = angles
        assert result.joint_angles is not None
        print("✓ DetectionResult integration successful")
        
        return True
    else:
        print("⚠ No poses detected in test frame")
        return False


def test_video_processor_integration():
    """Test JointAngleCalculator integration with video processor."""
    print("\n=== Testing Video Processor Integration ===")
    
    # Create a simple test video file path (dummy)
    test_video_path = Path("/tmp/test_pose_video.mp4")
    
    # Create a simple test video using OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(test_video_path), fourcc, 1.0, (640, 480))
    
    # Write 10 test frames
    for i in range(10):
        frame = create_test_frame()
        out.write(frame)
    
    out.release()
    
    try:
        # Test video processor with joint angles enabled
        processor = MultiStreamProcessor(
            [test_video_path],
            detector="yolo:n",
            enable_joint_angles=True,
            confidence_threshold=0.5,
            smoothing_window=3,
            max_streams=1,
            batch_size=1
        )
        
        print("✓ MultiStreamProcessor created with joint angles enabled")
        
        # Process a single frame to test integration
        frame = create_test_frame()
        result = processor._process_frame(0, frame)
        
        if result.joint_angles:
            print("✓ Joint angles calculated in video processor")
            print(f"  Knee angles: {result.joint_angles.knee_left:.1f}°/{result.joint_angles.knee_right:.1f}°")
            return True
        else:
            print("⚠ No joint angles calculated (may be due to no pose detection)")
            return False
            
    except Exception as e:
        print(f"❌ Video processor integration failed: {e}")
        return False
    finally:
        # Clean up test video
        if test_video_path.exists():
            test_video_path.unlink()


def test_api_schema_integration():
    """Test API schema integration."""
    print("\n=== Testing API Schema Integration ===")
    
    try:
        # Import API schemas
        sys.path.insert(0, str(Path(__file__).parent / "api"))
        from schemas import JointAngles, PoseLandmarks, PoseDetectionRequest
        
        # Test JointAngles schema
        angles = JointAngles(
            shoulder_left=180.0,
            elbow_right=90.0,
            knee_left=175.0
        )
        
        print("✓ JointAngles schema created")
        print(f"  Shoulder left: {angles.shoulder_left}°")
        print(f"  Elbow right: {angles.elbow_right}°")
        
        # Test PoseLandmarks with angles
        pose_landmarks = PoseLandmarks(
            landmarks=[],  # Would be populated with actual landmarks
            confidence=0.8,
            joint_angles=angles
        )
        
        print("✓ PoseLandmarks schema with joint angles created")
        
        # Test request with angle configuration
        request = PoseDetectionRequest(
            image_data="dummy_base64_data",
            enable_joint_angles=True,
            angle_confidence_threshold=0.5,
            angle_smoothing_window=3
        )
        
        print("✓ PoseDetectionRequest with angle configuration created")
        print(f"  Enable joint angles: {request.enable_joint_angles}")
        print(f"  Confidence threshold: {request.angle_confidence_threshold}")
        
        return True
        
    except Exception as e:
        print(f"❌ API schema integration failed: {e}")
        return False


def main():
    """Run all integration tests."""
    print("🚀 Starting JointAngleCalculator Integration Tests\n")
    
    # Test individual components
    detector_success = test_detector_integration()
    video_success = test_video_processor_integration()
    api_success = test_api_schema_integration()
    
    # Overall result
    print("\n=== Integration Test Results ===")
    print(f"Detector Integration: {'✅ PASS' if detector_success else '❌ FAIL'}")
    print(f"Video Processor Integration: {'✅ PASS' if video_success else '❌ FAIL'}")
    print(f"API Schema Integration: {'✅ PASS' if api_success else '❌ FAIL'}")
    
    overall_success = detector_success and video_success and api_success
    print(f"\n🎯 Overall Integration: {'✅ SUCCESS' if overall_success else '❌ NEEDS ATTENTION'}")
    
    if overall_success:
        print("\n🎉 JointAngleCalculator successfully integrated!")
        print("   - ✅ DetectionResult extended with joint angles")
        print("   - ✅ Video processor supports angle calculations")
        print("   - ✅ API schemas include angle data")
        print("   - ✅ Configuration options available")
    else:
        print("\n⚠️  Some integration aspects need attention")
    
    return overall_success


if __name__ == "__main__":
    main()
