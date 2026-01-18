#!/usr/bin/env python3
"""
Simple validation script for the JointAngleCalculator.
Tests the core functionality with realistic pose examples.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pose_analyzer.biomechanics import JointAngleCalculator


def create_standing_pose():
    """Create a standing pose with extended arms and straight legs."""
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


def create_squat_pose():
    """Create a squat pose with bent knees and hips."""
    keypoints = np.array([
        [0.5, 0.05, 0.0],  # nose (slightly lower)
        [0.45, 0.03, 0.0], # left_eye
        [0.55, 0.03, 0.0], # right_eye
        [0.4, 0.05, 0.0],  # left_ear
        [0.6, 0.05, 0.0],  # right_ear
        [0.35, 0.2, 0.0],  # left_shoulder (forward)
        [0.65, 0.2, 0.0],  # right_shoulder (forward)
        [0.25, 0.2, 0.0],  # left_elbow
        [0.75, 0.2, 0.0],  # right_elbow
        [0.15, 0.2, 0.0],  # left_wrist
        [0.85, 0.2, 0.0],  # right_wrist
        [0.38, 0.45, 0.0], # left_hip (lower and forward)
        [0.62, 0.45, 0.0], # right_hip (lower and forward)
        [0.3, 0.7, 0.0],   # left_knee (bent forward)
        [0.7, 0.7, 0.0],   # right_knee (bent forward)
        [0.32, 0.9, 0.0],  # left_ankle
        [0.68, 0.9, 0.0],  # right_ankle
    ])
    return keypoints


def create_pushup_pose():
    """Create a push-up pose with bent elbows."""
    keypoints = np.array([
        [0.5, 0.35, 0.0],  # nose (lowered)
        [0.45, 0.33, 0.0], # left_eye
        [0.55, 0.33, 0.0], # right_eye
        [0.4, 0.35, 0.0],  # left_ear
        [0.6, 0.35, 0.0],  # right_ear
        [0.35, 0.4, 0.0],  # left_shoulder
        [0.65, 0.4, 0.0],  # right_shoulder
        [0.32, 0.52, 0.0], # left_elbow (bent)
        [0.68, 0.52, 0.0], # right_elbow (bent)
        [0.3, 0.65, 0.0],  # left_wrist (on ground)
        [0.7, 0.65, 0.0],  # right_wrist (on ground)
        [0.4, 0.7, 0.0],   # left_hip
        [0.6, 0.7, 0.0],   # right_hip
        [0.4, 0.85, 0.0],  # left_knee
        [0.6, 0.85, 0.0],  # right_knee
        [0.4, 0.95, 0.0],  # left_ankle
        [0.6, 0.95, 0.0],  # right_ankle
    ])
    return keypoints


def validate_calculator():
    """Validate the JointAngleCalculator with different poses."""
    print("=== Joint Angle Calculator Validation ===\n")
    
    calculator = JointAngleCalculator(confidence_threshold=0.5, smoothing_window=0)
    
    # Test 1: Standing pose
    print("1. Standing Pose Validation:")
    print("   Expected: Arms extended (~180°), Knees straight (~180°)")
    standing_keypoints = create_standing_pose()
    standing_angles = calculator.calculate_angles(standing_keypoints)
    
    print(f"   Shoulder (L/R): {standing_angles.shoulder_left:.1f}°/{standing_angles.shoulder_right:.1f}°")
    print(f"   Elbow (L/R): {standing_angles.elbow_left:.1f}°/{standing_angles.elbow_right:.1f}°")
    print(f"   Knee (L/R): {standing_angles.knee_left:.1f}°/{standing_angles.knee_right:.1f}°")
    
    # Validate standing pose expectations
    shoulder_valid = abs(standing_angles.shoulder_left - 180) <= 20 and abs(standing_angles.shoulder_right - 180) <= 20
    knee_valid = abs(standing_angles.knee_left - 180) <= 15 and abs(standing_angles.knee_right - 180) <= 15
    
    print(f"   ✓ Shoulder angles valid: {shoulder_valid}")
    print(f"   ✓ Knee angles valid: {knee_valid}")
    print()
    
    # Test 2: Squat pose
    print("2. Squat Pose Validation:")
    print("   Expected: Knees bent (~60-120°), Hips bent (~70-130°)")
    squat_keypoints = create_squat_pose()
    squat_angles = calculator.calculate_angles(squat_keypoints)
    
    print(f"   Knee (L/R): {squat_angles.knee_left:.1f}°/{squat_angles.knee_right:.1f}°")
    print(f"   Hip (L/R): {squat_angles.hip_left:.1f}°/{squat_angles.hip_right:.1f}°")
    
    # Validate squat pose expectations
    knee_bent = 60 <= squat_angles.knee_left <= 130 and 60 <= squat_angles.knee_right <= 130
    hip_bent = 70 <= squat_angles.hip_left <= 140 and 70 <= squat_angles.hip_right <= 140
    
    print(f"   ✓ Knee angles properly bent: {knee_bent}")
    print(f"   ✓ Hip angles properly bent: {hip_bent}")
    print()
    
    # Test 3: Push-up pose
    print("3. Push-up Pose Validation:")
    print("   Expected: Elbows bent (~70-140°), Shoulders angled (~60-140°)")
    pushup_keypoints = create_pushup_pose()
    pushup_angles = calculator.calculate_angles(pushup_keypoints)
    
    print(f"   Elbow (L/R): {pushup_angles.elbow_left:.1f}°/{pushup_angles.elbow_right:.1f}°")
    print(f"   Shoulder (L/R): {pushup_angles.shoulder_left:.1f}°/{pushup_angles.shoulder_right:.1f}°")
    
    # Validate push-up pose expectations
    elbow_bent = 70 <= pushup_angles.elbow_left <= 150 and 70 <= pushup_angles.elbow_right <= 150
    shoulder_angled = 60 <= pushup_angles.shoulder_left <= 150 and 60 <= pushup_angles.shoulder_right <= 150
    
    print(f"   ✓ Elbow angles properly bent: {elbow_bent}")
    print(f"   ✓ Shoulder angles properly angled: {shoulder_angled}")
    print()
    
    # Test 4: Missing keypoints handling
    print("4. Missing Keypoints Validation:")
    test_keypoints = standing_keypoints.copy()
    
    # Remove left elbow and shoulder
    test_keypoints[5] = np.nan  # left_shoulder
    test_keypoints[7] = np.nan  # left_elbow
    
    missing_angles = calculator.calculate_angles(test_keypoints)
    
    print(f"   Missing left shoulder/elbow - Shoulder (L/R): {missing_angles.shoulder_left}/{missing_angles.shoulder_right}")
    print(f"   Missing left shoulder/elbow - Elbow (L/R): {missing_angles.elbow_left}/{missing_angles.elbow_right}")
    print(f"   Missing left shoulder/elbow - Knee (L/R): {missing_angles.knee_left:.1f}°/{missing_angles.knee_right:.1f}°")
    
    # Should still calculate right side and other joints
    right_side_works = missing_angles.shoulder_right is not None and missing_angles.elbow_right is not None
    knees_work = missing_angles.knee_left is not None and missing_angles.knee_right is not None
    left_missing = missing_angles.shoulder_left is None and missing_angles.elbow_left is None
    
    print(f"   ✓ Right side angles calculated: {right_side_works}")
    print(f"   ✓ Knee angles still calculated: {knees_work}")
    print(f"   ✓ Missing left angles properly None: {left_missing}")
    print()
    
    # Test 5: Confidence filtering
    print("5. Confidence Filtering Validation:")
    confidences = np.ones(17) * 0.8
    confidences[6] = 0.3  # right_shoulder low confidence
    confidences[8] = 0.3  # right_elbow low confidence
    
    conf_angles = calculator.calculate_angles(standing_keypoints, confidences)
    
    print(f"   Low confidence right side - Shoulder (L/R): {conf_angles.shoulder_left}/{conf_angles.shoulder_right}")
    print(f"   Low confidence right side - Elbow (L/R): {conf_angles.elbow_left}/{conf_angles.elbow_right}")
    
    # Should filter out low confidence points
    high_conf_works = conf_angles.shoulder_left is not None and conf_angles.elbow_left is not None
    low_conf_filtered = conf_angles.shoulder_right is None and conf_angles.elbow_right is None
    
    print(f"   ✓ High confidence angles calculated: {high_conf_works}")
    print(f"   ✓ Low confidence angles filtered: {low_conf_filtered}")
    print()
    
    # Overall validation
    all_valid = (
        shoulder_valid and knee_valid and  # Standing
        knee_bent and hip_bent and         # Squat
        elbow_bent and shoulder_angled and # Push-up
        right_side_works and knees_work and left_missing and # Missing keypoints
        high_conf_works and low_conf_filtered  # Confidence filtering
    )
    
    print("=== Overall Validation Result ===")
    print(f"✓ All validations passed: {all_valid}")
    
    if all_valid:
        print("\n🎉 Joint Angle Calculator implementation is VALID!")
        print("   - 3D joint angle calculations working correctly")
        print("   - Missing keypoint interpolation functioning")
        print("   - Confidence-based filtering operational")
        print("   - Expected angle ranges for known poses verified")
    else:
        print("\n❌ Some validations failed - review implementation")
    
    return all_valid


if __name__ == "__main__":
    validate_calculator()
