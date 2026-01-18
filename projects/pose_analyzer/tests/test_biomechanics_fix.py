import numpy as np
import pytest
from pose_analyzer.biomechanics import JointAngleCalculator, JointAngles

class TestJointAngleCalculatorFix:
    def setup_method(self):
        # Window 0 to avoid smoothing effects and use clear historical state
        self.calculator = JointAngleCalculator(confidence_threshold=0.5, smoothing_window=0)

    def create_test_keypoints(self, pose_config: dict) -> np.ndarray:
        keypoints = np.zeros((17, 3))
        keypoint_map = {
            "nose": 0, "left_shoulder": 5, "right_shoulder": 6,
            "left_elbow": 7, "right_elbow": 8, "left_wrist": 9, "right_wrist": 10,
            "left_hip": 11, "right_hip": 12, "left_knee": 13, "right_knee": 14,
            "left_ankle": 15, "right_ankle": 16
        }
        # Default standing position
        defaults = {
            "left_shoulder": [0.4, 0.2, 0.0], "right_shoulder": [0.6, 0.2, 0.0],
            "left_elbow": [0.4, 0.4, 0.0], "right_elbow": [0.6, 0.4, 0.0],
            "left_wrist": [0.4, 0.6, 0.0], "right_wrist": [0.6, 0.6, 0.0],
            "left_hip": [0.4, 0.6, 0.0], "right_hip": [0.6, 0.6, 0.0],
            "left_knee": [0.4, 0.8, 0.0], "right_knee": [0.6, 0.8, 0.0],
            "left_ankle": [0.4, 1.0, 0.0], "right_ankle": [0.6, 1.0, 0.0],
        }
        for name, pos in defaults.items():
            keypoints[keypoint_map[name]] = pos
        for name, pos in pose_config.items():
            keypoints[keypoint_map[name]] = pos
        return keypoints

    def test_standing_pose(self):
        keypoints = self.create_test_keypoints({})
        angles = self.calculator.calculate_angles(keypoints)
        assert angles.knee_left is not None
        assert abs(angles.knee_left - 180) <= 5

    def test_squat_pose(self):
        # Knee angle calculation: (hip, knee, ankle)
        # Hip=(0.4, 0.6), Knee=(0.4, 0.8), Ankle=(0.6, 0.8) -> 90 degrees
        squat_config = {
            "left_hip": [0.4, 0.6, 0.0],
            "left_knee": [0.4, 0.8, 0.0],
            "left_ankle": [0.6, 0.8, 0.0],
        }
        keypoints = self.create_test_keypoints(squat_config)
        angles = self.calculator.calculate_angles(keypoints)
        assert angles.knee_left is not None
        assert abs(angles.knee_left - 90) <= 5

    def test_temporal_smoothing(self):
        calculator = JointAngleCalculator(smoothing_window=3)
        kp1 = self.create_test_keypoints({})
        angles1 = calculator.calculate_angles(kp1)
        
        # Change knee angle in next frame
        kp2 = self.create_test_keypoints({"left_knee": [0.35, 0.8, 0.0]})
        angles2 = calculator.calculate_angles(kp2)
        
        raw_calculator = JointAngleCalculator(smoothing_window=0)
        raw_angles1 = raw_calculator.calculate_angles(kp1)
        raw_angles2 = raw_calculator.calculate_angles(kp2)
        
        smooth_diff = abs(angles2.knee_left - angles1.knee_left)
        raw_diff = abs(raw_angles2.knee_left - raw_angles1.knee_left)
        
        assert smooth_diff < raw_diff

    def test_missing_keypoints(self):
        keypoints = self.create_test_keypoints({})
        # The calculator uses interpolation. 
        # To truly test "None", we need to disable interpolation or clear everything.
        # Let's just clear the relevant ones and their mirrors.
        for i in range(17):
            keypoints[i] = np.nan
            
        angles = self.calculator.calculate_angles(keypoints)
        assert angles.elbow_left is None
