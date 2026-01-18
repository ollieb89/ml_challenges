"""
Reference-Based Form Scorer for Pose Analysis

This module implements a comprehensive form scoring system that compares user poses
to reference poses and provides detailed feedback on exercise form quality.

Key Features:
- Joint angle deviation analysis (RMS error)
- Left/right symmetry assessment
- Trajectory smoothness evaluation
- Range of motion (ROM) coverage analysis
- Body proportion normalization for different user sizes
- 0-100 scoring system with detailed metrics

Author: AI/ML Pipeline Team
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json

from .biomechanics import JointAngles, JointAngleCalculator
from .synthetic_poses import SyntheticPoseGenerator, SyntheticPose, SquatType


@dataclass
class BodyProportions:
    """Normalized body proportions for user-specific scaling"""
    torso_length: float = 0.4      # Torso length relative to height
    thigh_length: float = 0.25     # Upper leg length
    shin_length: float = 0.25      # Lower leg length
    upper_arm_length: float = 0.15 # Upper arm length
    forearm_length: float = 0.15    # Lower arm length
    shoulder_width: float = 0.15    # Shoulder width relative to height
    
    @classmethod
    def from_keypoints(cls, keypoints: np.ndarray) -> 'BodyProportions':
        """
        Estimate body proportions from detected keypoints
        
        Args:
            keypoints: Array of (17, 3) COCO keypoints
            
        Returns:
            BodyProportions object with estimated measurements
        """
        # Extract key joint positions
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_knee = keypoints[13]
        right_knee = keypoints[14]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        left_elbow = keypoints[7]
        right_elbow = keypoints[8]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        
        # Calculate body segment lengths (using average of left/right where applicable)
        shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
        
        torso_length = np.linalg.norm(
            (left_shoulder + right_shoulder) / 2 - (left_hip + right_hip) / 2
        )
        
        thigh_length_left = np.linalg.norm(left_knee - left_hip)
        thigh_length_right = np.linalg.norm(right_knee - right_hip)
        thigh_length = (thigh_length_left + thigh_length_right) / 2
        
        shin_length_left = np.linalg.norm(left_ankle - left_knee)
        shin_length_right = np.linalg.norm(right_ankle - right_knee)
        shin_length = (shin_length_left + shin_length_right) / 2
        
        upper_arm_length_left = np.linalg.norm(left_elbow - left_shoulder)
        upper_arm_length_right = np.linalg.norm(right_elbow - right_shoulder)
        upper_arm_length = (upper_arm_length_left + upper_arm_length_right) / 2
        
        forearm_length_left = np.linalg.norm(left_wrist - left_elbow)
        forearm_length_right = np.linalg.norm(right_wrist - right_elbow)
        forearm_length = (forearm_length_left + forearm_length_right) / 2
        
        # Normalize by total height (estimate from head to feet)
        head_y = min(left_shoulder[1], right_shoulder[1]) - 0.1  # Estimate head position
        feet_y = max(left_ankle[1], right_ankle[1])
        total_height = abs(feet_y - head_y)
        
        # Check if we have valid keypoints (more than just noise/zeros)
        if total_height < 0.2 or np.all(keypoints == 0):
            return cls()
        
        if total_height > 0:
            return cls(
                torso_length=torso_length / total_height,
                thigh_length=thigh_length / total_height,
                shin_length=shin_length / total_height,
                upper_arm_length=upper_arm_length / total_height,
                forearm_length=forearm_length / total_height,
                shoulder_width=shoulder_width / total_height
            )
        # Fallback to defaults
        return cls()


@dataclass
class FormMetrics:
    """Detailed form quality metrics"""
    joint_angle_deviation: float = 0.0      # RMS error in degrees
    symmetry_score: float = 0.0             # 0-1 scale (1 = perfect symmetry)
    trajectory_smoothness: float = 0.0       # 0-1 scale (1 = perfectly smooth)
    rom_coverage: float = 0.0                # 0-1 scale (1 = full expected ROM)
    overall_score: float = 0.0              # 0-100 scale
    
    # Detailed joint-specific metrics
    joint_errors: Dict[str, float] = field(default_factory=dict)
    symmetry_details: Dict[str, float] = field(default_factory=dict)
    rom_details: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'overall_score': self.overall_score,
            'joint_angle_deviation': self.joint_angle_deviation,
            'symmetry_score': self.symmetry_score,
            'trajectory_smoothness': self.trajectory_smoothness,
            'rom_coverage': self.rom_coverage,
            'joint_errors': self.joint_errors,
            'symmetry_details': self.symmetry_details,
            'rom_details': self.rom_details
        }


@dataclass
class ReferencePose:
    """Reference pose for form comparison"""
    name: str
    exercise_type: str
    keypoints: np.ndarray          # (17, 3) ideal keypoints
    angles: JointAngles            # Ideal joint angles
    proportions: BodyProportions   # Expected body proportions
    rom_targets: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # Joint: (min, max) angles


class FormScorer:
    """
    Reference-based form scoring system for pose analysis
    
    Compares user poses to reference poses and provides comprehensive
    form quality assessment with body proportion normalization.
    """
    
    def __init__(self, exercise_type: str = "squat"):
        """
        Initialize form scorer for specific exercise type
        
        Args:
            exercise_type: Type of exercise (squat, pushup, etc.)
        """
        self.exercise_type = exercise_type
        self.angle_calculator = JointAngleCalculator()
        self.pose_generator = SyntheticPoseGenerator()
        
        # Initialize reference poses
        self.reference_poses: Dict[str, ReferencePose] = {}
        self._load_reference_poses()
        
        # Scoring weights (tunable for different exercises)
        self.weights = {
            'joint_angle_deviation': 0.4,
            'symmetry_score': 0.2,
            'trajectory_smoothness': 0.2,
            'rom_coverage': 0.2
        }
    
    def _load_reference_poses(self) -> None:
        """Load or generate reference poses for the exercise type"""
        if self.exercise_type == "squat":
            self._generate_squat_references()
        # Add other exercise types as needed
    
    def _generate_squat_references(self) -> None:
        """Generate ideal squat reference poses"""
        # Generate perfect parallel squat
        perfect_sequence = self.pose_generator.generate_squat_sequence(
            num_frames=60,
            squat_type=SquatType.PARALLEL,
            noise_level=0.0  # No noise for reference
        )
        
        # Extract key poses (bottom, middle, top)
        bottom_frame = perfect_sequence[15]  # Bottom position
        middle_frame = perfect_sequence[30]  # Middle position  
        top_frame = perfect_sequence[45]     # Top position
        
        # Create reference poses
        proportions = BodyProportions()  # Use default proportions
        
        # Define ROM targets for squats (in degrees)
        rom_targets = {
            'knee_left': (90, 180),    # 90° at bottom, 180° at top
            'knee_right': (90, 180),
            'hip_left': (70, 180),     # 70° hip flexion at bottom
            'hip_right': (70, 180),
            'ankle_left': (70, 90),    # Ankle range during squat
            'ankle_right': (70, 90)
        }
        
        self.reference_poses['squat_bottom'] = ReferencePose(
            name="Squat Bottom Position",
            exercise_type="squat",
            keypoints=bottom_frame.keypoints,
            angles=bottom_frame.ground_truth_angles,
            proportions=proportions,
            rom_targets=rom_targets
        )
        
        self.reference_poses['squat_parallel'] = ReferencePose(
            name="Parallel Squat",
            exercise_type="squat", 
            keypoints=middle_frame.keypoints,
            angles=middle_frame.ground_truth_angles,
            proportions=proportions,
            rom_targets=rom_targets
        )
        
        self.reference_poses['squat_top'] = ReferencePose(
            name="Squat Top Position",
            exercise_type="squat",
            keypoints=top_frame.keypoints,
            angles=top_frame.ground_truth_angles,
            proportions=proportions,
            rom_targets=rom_targets
        )
    
    def normalize_pose_by_proportions(
        self, 
        keypoints: np.ndarray, 
        user_proportions: BodyProportions,
        reference_proportions: BodyProportions
    ) -> np.ndarray:
        """
        Normalize pose keypoints to account for different body proportions
        
        Args:
            keypoints: User pose keypoints (17, 3)
            user_proportions: Estimated user body proportions
            reference_proportions: Reference body proportions
            
        Returns:
            Normalized keypoints matching reference proportions
        """
        normalized_keypoints = keypoints.copy()
        
        # Calculate scaling factors for each body segment
        scale_factors = {
            'torso': reference_proportions.torso_length / user_proportions.torso_length,
            'thigh': reference_proportions.thigh_length / user_proportions.thigh_length,
            'shin': reference_proportions.shin_length / user_proportions.shin_length,
            'upper_arm': reference_proportions.upper_arm_length / user_proportions.upper_arm_length,
            'forearm': reference_proportions.forearm_length / user_proportions.forearm_length,
            'shoulder_width': reference_proportions.shoulder_width / user_proportions.shoulder_width
        }
        
        # Apply scaling to appropriate keypoints
        # This is a simplified approach - in practice, more sophisticated
        # anatomical modeling would be used
        
        # Scale shoulder width
        left_shoulder = normalized_keypoints[5]
        right_shoulder = normalized_keypoints[6]
        shoulder_center = (left_shoulder + right_shoulder) / 2
        
        normalized_keypoints[5] = shoulder_center + (left_shoulder - shoulder_center) * scale_factors['shoulder_width']
        normalized_keypoints[6] = shoulder_center + (right_shoulder - shoulder_center) * scale_factors['shoulder_width']
        
        # Scale arm segments
        if scale_factors['upper_arm'] > 0:
            # Left arm
            left_shoulder = normalized_keypoints[5]
            left_elbow = normalized_keypoints[7]
            left_wrist = normalized_keypoints[9]
            
            normalized_keypoints[7] = left_shoulder + (left_elbow - left_shoulder) * scale_factors['upper_arm']
            normalized_keypoints[9] = left_elbow + (left_wrist - left_elbow) * scale_factors['forearm']
            
            # Right arm
            right_shoulder = normalized_keypoints[6]
            right_elbow = normalized_keypoints[8]
            right_wrist = normalized_keypoints[10]
            
            normalized_keypoints[8] = right_shoulder + (right_elbow - right_shoulder) * scale_factors['upper_arm']
            normalized_keypoints[10] = right_elbow + (right_wrist - right_elbow) * scale_factors['forearm']
        
        return normalized_keypoints
    
    def calculate_joint_angle_deviation(
        self, 
        user_angles: JointAngles, 
        reference_angles: JointAngles
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate RMS error between user and reference joint angles
        
        Args:
            user_angles: User's joint angles
            reference_angles: Reference joint angles
            
        Returns:
            Tuple of (RMS error, joint-specific errors)
        """
        joint_names = [
            'shoulder_left', 'shoulder_right', 'elbow_left', 'elbow_right',
            'hip_left', 'hip_right', 'knee_left', 'knee_right',
            'ankle_left', 'ankle_right'
        ]
        
        errors = {}
        squared_errors = []
        
        for joint_name in joint_names:
            user_angle = getattr(user_angles, joint_name)
            reference_angle = getattr(reference_angles, joint_name)
            
            if user_angle is not None and reference_angle is not None:
                error = abs(user_angle - reference_angle)
                errors[joint_name] = error
                squared_errors.append(error ** 2)
            else:
                errors[joint_name] = 0.0  # Default to no error for missing data
        
        # Calculate RMS error
        rms_error = math.sqrt(np.mean(squared_errors)) if squared_errors else 0.0
        
        return rms_error, errors
    
    def calculate_symmetry_score(self, angles: JointAngles) -> Tuple[float, Dict[str, float]]:
        """
        Calculate left/right symmetry score
        
        Args:
            angles: Joint angles to analyze
            
        Returns:
            Tuple of (symmetry_score, symmetry_details)
        """
        symmetry_pairs = [
            ('shoulder_left', 'shoulder_right'),
            ('elbow_left', 'elbow_right'),
            ('hip_left', 'hip_right'),
            ('knee_left', 'knee_right'),
            ('ankle_left', 'ankle_right')
        ]
        
        symmetry_details = {}
        symmetry_scores = []
        
        for left_joint, right_joint in symmetry_pairs:
            left_angle = getattr(angles, left_joint)
            right_angle = getattr(angles, right_joint)
            
            if left_angle is not None and right_angle is not None:
                # Calculate symmetry as 1 - normalized difference
                max_angle = max(abs(left_angle), abs(right_angle))
                if max_angle > 0:
                    difference = abs(left_angle - right_angle) / max_angle
                    symmetry = 1.0 - min(difference, 1.0)
                else:
                    symmetry = 1.0  # Perfect symmetry if both are zero
                
                symmetry_details[f"{left_joint.replace('_left', '')}_symmetry"] = symmetry
                symmetry_scores.append(symmetry)
            else:
                symmetry_details[f"{left_joint.replace('_left', '')}_symmetry"] = 0.5  # Neutral for missing data
        
        overall_symmetry = np.mean(symmetry_scores) if symmetry_scores else 0.5
        
        return overall_symmetry, symmetry_details
    
    def calculate_trajectory_smoothness(
        self, 
        angle_sequence: List[JointAngles]
    ) -> float:
        """
        Calculate trajectory smoothness from angle sequence
        
        Args:
            angle_sequence: List of joint angles over time
            
        Returns:
            Smoothness score (0-1, higher is smoother)
        """
        if len(angle_sequence) < 3:
            return 1.0  # Perfect smoothness for very short sequences
        
        joint_names = [
            'knee_left', 'knee_right', 'hip_left', 'hip_right',
            'shoulder_left', 'shoulder_right'
        ]
        
        smoothness_scores = []
        
        for joint_name in joint_names:
            # Extract angle sequence for this joint
            angles = []
            for angles_obj in angle_sequence:
                angle = getattr(angles_obj, joint_name)
                if angle is not None:
                    angles.append(angle)
            
            if len(angles) >= 3:
                # Calculate jerk (third derivative) as smoothness metric
                angles_array = np.array(angles)
                
                # First derivative (velocity)
                velocity = np.gradient(angles_array)
                
                # Second derivative (acceleration)  
                acceleration = np.gradient(velocity)
                
                # Third derivative (jerk)
                jerk = np.gradient(acceleration)
                
                # Smoothness = 1 / (1 + normalized jerk magnitude)
                jerk_magnitude = np.mean(np.abs(jerk))
                smoothness = 1.0 / (1.0 + jerk_magnitude)
                
                smoothness_scores.append(smoothness)
        
        return np.mean(smoothness_scores) if smoothness_scores else 1.0
    
    def calculate_rom_coverage(
        self, 
        angle_sequence: List[JointAngles],
        rom_targets: Dict[str, Tuple[float, float]]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate range of motion coverage
        
        Args:
            angle_sequence: List of joint angles over time
            rom_targets: Expected (min, max) angles for each joint
            
        Returns:
            Tuple of (rom_coverage_score, rom_details)
        """
        rom_details = {}
        rom_scores = []
        
        for joint_name, (target_min, target_max) in rom_targets.items():
            # Extract angle sequence for this joint
            angles = []
            for angles_obj in angle_sequence:
                angle = getattr(angles_obj, joint_name)
                if angle is not None:
                    angles.append(angle)
            
            if angles:
                actual_min = min(angles)
                actual_max = max(angles)
                actual_range = actual_max - actual_min
                target_range = target_max - target_min
                
                # Calculate coverage as percentage of expected range achieved
                if target_range > 0:
                    coverage = min(actual_range / target_range, 1.0)
                else:
                    coverage = 1.0  # Perfect coverage if no range expected
                
                # Also check if the range overlaps with expected range
                range_overlap = 1.0
                if actual_max < target_min or actual_min > target_max:
                    # No overlap with expected range
                    range_overlap = 0.0
                else:
                    # Partial overlap
                    overlap_start = max(actual_min, target_min)
                    overlap_end = min(actual_max, target_max)
                    overlap_range = max(0.0, float(overlap_end - overlap_start))
                    if target_range > 0:
                        range_overlap = overlap_range / target_range
                    else:
                        range_overlap = 1.0
                
                rom_details[joint_name] = coverage * range_overlap
                rom_scores.append(coverage * range_overlap)
            else:
                rom_details[joint_name] = 0.0
        
        overall_rom = np.mean(rom_scores) if rom_scores else 0.0
        
        return overall_rom, rom_details
    
    def calculate_overall_score(self, metrics: FormMetrics) -> float:
        """
        Calculate overall form score (0-100) from individual metrics
        
        Args:
            metrics: Form metrics object
            
        Returns:
            Overall score from 0-100
        """
        # Convert individual metrics to 0-1 scale if needed
        joint_score = max(0, 1.0 - (metrics.joint_angle_deviation / 30.0))  # 30° = 0 score
        symmetry_score = metrics.symmetry_score
        smoothness_score = metrics.trajectory_smoothness
        rom_score = metrics.rom_coverage
        
        # Weighted combination
        weighted_score = (
            joint_score * self.weights['joint_angle_deviation'] +
            symmetry_score * self.weights['symmetry_score'] +
            smoothness_score * self.weights['trajectory_smoothness'] +
            rom_score * self.weights['rom_coverage']
        )
        
        # Convert to 0-100 scale
        overall_score = weighted_score * 100
        
        return max(0.0, min(100.0, overall_score))
    
    def score_pose_sequence(
        self, 
        keypoints_sequence: List[np.ndarray],
        confidences_sequence: Optional[List[np.ndarray]] = None,
        reference_name: str = "squat_parallel"
    ) -> FormMetrics:
        """
        Score a complete pose sequence against reference
        
        Args:
            keypoints_sequence: List of pose keypoints over time
            confidences_sequence: Optional confidence scores for each frame
            reference_name: Name of reference pose to compare against
            
        Returns:
            Comprehensive form metrics
        """
        if reference_name not in self.reference_poses:
            raise ValueError(f"Reference pose '{reference_name}' not found")
        
        reference = self.reference_poses[reference_name]
        
        # Calculate angles for each frame
        angle_sequence = []
        user_proportions_list = []
        
        for i, keypoints in enumerate(keypoints_sequence):
            confidences = confidences_sequence[i] if confidences_sequence else None
            
            # Estimate user proportions from first valid frame
            if i == 0:
                user_proportions = BodyProportions.from_keypoints(keypoints)
                user_proportions_list.append(user_proportions)
            
            # Normalize keypoints to reference proportions
            normalized_keypoints = self.normalize_pose_by_proportions(
                keypoints, user_proportions, reference.proportions
            )
            
            # Calculate angles
            angles = self.angle_calculator.calculate_angles(normalized_keypoints, confidences)
            angle_sequence.append(angles)
        
        # Calculate metrics
        # Joint angle deviation (use middle frame for comparison)
        middle_idx = len(angle_sequence) // 2
        joint_deviation, joint_errors = self.calculate_joint_angle_deviation(
            angle_sequence[middle_idx], reference.angles
        )
        
        # Symmetry score (use middle frame)
        symmetry_score, symmetry_details = self.calculate_symmetry_score(angle_sequence[middle_idx])
        
        # Trajectory smoothness (entire sequence)
        smoothness_score = self.calculate_trajectory_smoothness(angle_sequence)
        
        # ROM coverage (entire sequence)
        rom_score, rom_details = self.calculate_rom_coverage(angle_sequence, reference.rom_targets)
        
        # Create metrics object
        metrics = FormMetrics(
            joint_angle_deviation=joint_deviation,
            symmetry_score=symmetry_score,
            trajectory_smoothness=smoothness_score,
            rom_coverage=rom_score,
            joint_errors=joint_errors,
            symmetry_details=symmetry_details,
            rom_details=rom_details
        )
        
        # Calculate overall score
        metrics.overall_score = self.calculate_overall_score(metrics)
        
        return metrics
    
    def score_single_pose(
        self, 
        keypoints: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        reference_name: str = "squat_parallel"
    ) -> FormMetrics:
        """
        Score a single pose against reference
        
        Args:
            keypoints: Single pose keypoints
            confidences: Optional confidence scores
            reference_name: Name of reference pose to compare against
            
        Returns:
            Form metrics for single pose
        """
        # For single pose, create sequence of length 1
        return self.score_pose_sequence(
            [keypoints], [confidences] if confidences is not None else None,
            reference_name
        )


def create_form_scorer(exercise_type: str = "squat") -> FormScorer:
    """Factory function to create optimized form scorer"""
    return FormScorer(exercise_type)


if __name__ == "__main__":
    # Quick test
    print("Testing FormScorer...")
    
    scorer = create_form_scorer("squat")
    
    # Generate test sequence
    generator = SyntheticPoseGenerator()
    test_sequence = generator.generate_squat_sequence(
        num_frames=30,
        squat_type=SquatType.PARALLEL,
        noise_level=1.0
    )
    
    # Extract keypoints
    keypoints_sequence = [pose.keypoints for pose in test_sequence]
    confidences_sequence = [pose.confidences for pose in test_sequence]
    
    # Score the sequence
    metrics = scorer.score_pose_sequence(
        keypoints_sequence, confidences_sequence, "squat_parallel"
    )
    
    print(f"Overall score: {metrics.overall_score:.1f}/100")
    print(f"Joint angle deviation: {metrics.joint_angle_deviation:.2f}°")
    print(f"Symmetry score: {metrics.symmetry_score:.3f}")
    print(f"Trajectory smoothness: {metrics.trajectory_smoothness:.3f}")
    print(f"ROM coverage: {metrics.rom_coverage:.3f}")
    
    print("\n✓ FormScorer test completed successfully")