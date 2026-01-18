"""Biomechanical analysis module for joint angle calculations.

This module implements geometric joint angle calculations for COCO 17-keypoint pose format,
with robust handling of missing keypoints and confidence-based smoothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class JointAngles:
    """Container for calculated joint angles in degrees."""
    
    shoulder_left: Optional[float] = None
    shoulder_right: Optional[float] = None
    elbow_left: Optional[float] = None
    elbow_right: Optional[float] = None
    hip_left: Optional[float] = None
    hip_right: Optional[float] = None
    knee_left: Optional[float] = None
    knee_right: Optional[float] = None
    ankle_left: Optional[float] = None
    ankle_right: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Optional[float]]:
        """Convert to dictionary format."""
        return {
            "shoulder_left": self.shoulder_left,
            "shoulder_right": self.shoulder_right,
            "elbow_left": self.elbow_left,
            "elbow_right": self.elbow_right,
            "hip_left": self.hip_left,
            "hip_right": self.hip_right,
            "knee_left": self.knee_left,
            "knee_right": self.knee_right,
            "ankle_left": self.ankle_left,
            "ankle_right": self.ankle_right,
        }


class JointAngleCalculator:
    """Calculate joint angles from COCO 17 keypoints using vectorized operations.
    
    Supports:
    - 3D joint angle calculations using vector geometry
    - Missing keypoint interpolation and smoothing
    - Confidence-based filtering
    - Vectorized batch processing
    
    COCO Keypoint Mapping (17 points):
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    """
    
    # COCO keypoint indices for joint calculations
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    # Joint angle calculation triplets (proximal, joint, distal)
    JOINT_TRIPLETS = {
        "shoulder_left": (5, 7, 9),      # left_shoulder - left_elbow - left_wrist
        "shoulder_right": (6, 8, 10),     # right_shoulder - right_elbow - right_wrist  
        "elbow_left": (5, 7, 9),         # left_shoulder - left_elbow - left_wrist
        "elbow_right": (6, 8, 10),       # right_shoulder - right_elbow - right_wrist
        "hip_left": (5, 11, 13),         # left_shoulder - left_hip - left_knee
        "hip_right": (6, 12, 14),       # right_shoulder - right_hip - right_knee
        "knee_left": (11, 13, 15),       # left_hip - left_knee - left_ankle
        "knee_right": (12, 14, 16),      # right_hip - right_knee - right_ankle
        "ankle_left": (13, 15, 15),      # left_knee - left_ankle - left_ankle (foot angle)
        "ankle_right": (14, 16, 16),     # right_knee - right_ankle - right_ankle (foot angle)
    }
    
    def __init__(self, confidence_threshold: float = 0.5, smoothing_window: int = 3):
        """Initialize calculator with confidence and smoothing parameters.
        
        Args:
            confidence_threshold: Minimum keypoint confidence for calculations
            smoothing_window: Window size for temporal smoothing (0 = no smoothing)
        """
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self._angle_history: List[JointAngles] = []
    
    def calculate_angles(
        self, 
        keypoints: np.ndarray, 
        confidences: Optional[np.ndarray] = None
    ) -> JointAngles:
        """Calculate all joint angles from keypoints.
        
        Args:
            keypoints: Array of shape (17, 3) with (x, y, z) coordinates
            confidences: Optional array of shape (17,) with confidence scores
            
        Returns:
            JointAngles object with calculated angles in degrees
        """
        if keypoints.shape != (17, 3):
            raise ValueError(f"Expected keypoints shape (17, 3), got {keypoints.shape}")
        
        # Apply confidence filtering if provided
        valid_keypoints = self._filter_by_confidence(keypoints, confidences)
        
        # Interpolate missing keypoints if needed
        interpolated_keypoints = self._interpolate_missing_keypoints(valid_keypoints)
        
        # Calculate individual joint angles
        angles = JointAngles()
        
        # Calculate angles for each joint
        angles.shoulder_left = self._calculate_joint_angle(
            interpolated_keypoints, self.JOINT_TRIPLETS["shoulder_left"]
        )
        angles.shoulder_right = self._calculate_joint_angle(
            interpolated_keypoints, self.JOINT_TRIPLETS["shoulder_right"]
        )
        
        angles.elbow_left = self._calculate_joint_angle(
            interpolated_keypoints, self.JOINT_TRIPLETS["elbow_left"]
        )
        angles.elbow_right = self._calculate_joint_angle(
            interpolated_keypoints, self.JOINT_TRIPLETS["elbow_right"]
        )
        
        angles.hip_left = self._calculate_joint_angle(
            interpolated_keypoints, self.JOINT_TRIPLETS["hip_left"]
        )
        angles.hip_right = self._calculate_joint_angle(
            interpolated_keypoints, self.JOINT_TRIPLETS["hip_right"]
        )
        
        angles.knee_left = self._calculate_joint_angle(
            interpolated_keypoints, self.JOINT_TRIPLETS["knee_left"]
        )
        angles.knee_right = self._calculate_joint_angle(
            interpolated_keypoints, self.JOINT_TRIPLETS["knee_right"]
        )
        
        # For ankle angles, calculate foot orientation relative to shin
        angles.ankle_left = self._calculate_ankle_angle(interpolated_keypoints, left=True)
        angles.ankle_right = self._calculate_ankle_angle(interpolated_keypoints, left=False)
        
        # Apply temporal smoothing if enabled
        if self.smoothing_window > 0:
            angles = self._apply_smoothing(angles)
        
        return angles
    
    def _filter_by_confidence(
        self, 
        keypoints: np.ndarray, 
        confidences: Optional[np.ndarray]
    ) -> np.ndarray:
        """Filter keypoints by confidence threshold."""
        if confidences is None:
            return keypoints
        
        # Create mask for valid keypoints
        valid_mask = confidences >= self.confidence_threshold
        
        # Set invalid keypoints to NaN
        filtered_keypoints = keypoints.copy().astype(float)
        filtered_keypoints[~valid_mask] = np.nan
        
        return filtered_keypoints
    
    def _interpolate_missing_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """Interpolate missing keypoints using linear interpolation."""
        interpolated = keypoints.copy()
        
        # Find missing keypoints (NaN values)
        missing_mask = np.isnan(keypoints[:, 0])  # Check x coordinate for NaN
        
        if not np.any(missing_mask):
            return interpolated
        
        # Simple interpolation strategy: use symmetric keypoints
        # Left/right pairs for interpolation
        left_right_pairs = [
            (5, 6),  # shoulders
            (7, 8),  # elbows  
            (9, 10), # wrists
            (11, 12), # hips
            (13, 14), # knees
            (15, 16), # ankles
        ]
        
        for left_idx, right_idx in left_right_pairs:
            if missing_mask[left_idx] and not missing_mask[right_idx]:
                # Interpolate left from right (mirror across center)
                interpolated[left_idx] = self._mirror_keypoint(
                    interpolated[right_idx], left=True
                )
            elif missing_mask[right_idx] and not missing_mask[left_idx]:
                # Interpolate right from left
                interpolated[right_idx] = self._mirror_keypoint(
                    interpolated[left_idx], left=False
                )
        
        # For still missing keypoints, use neighboring interpolation
        for i in range(17):
            if missing_mask[i] and np.isnan(interpolated[i, 0]):
                interpolated[i] = self._interpolate_from_neighbors(interpolated, i)
        
        return interpolated
    
    def _mirror_keypoint(self, keypoint: np.ndarray, left: bool) -> np.ndarray:
        """Mirror a keypoint across the sagittal plane."""
        mirrored = keypoint.copy()
        # Flip x coordinate (assuming normalized coordinates)
        mirrored[0] = 1.0 - keypoint[0]
        return mirrored
    
    def _interpolate_from_neighbors(self, keypoints: np.ndarray, missing_idx: int) -> np.ndarray:
        """Interpolate a missing keypoint from its neighbors."""
        # Define neighbor relationships for interpolation
        neighbor_map = {
            0: [1, 2],      # nose from eyes
            1: [0, 3],      # left_eye from nose, left_ear
            2: [0, 4],      # right_eye from nose, right_ear
            3: [1, 5],      # left_ear from left_eye, left_shoulder
            4: [2, 6],      # right_ear from right_eye, right_shoulder
            5: [3, 7, 11],  # left_shoulder from left_ear, left_elbow, left_hip
            6: [4, 8, 12],  # right_shoulder from right_ear, right_elbow, right_hip
            7: [5, 9],      # left_elbow from left_shoulder, left_wrist
            8: [6, 10],     # right_elbow from right_shoulder, right_wrist
            9: [7],         # left_wrist from left_elbow
            10: [8],        # right_wrist from right_elbow
            11: [5, 12, 13], # left_hip from left_shoulder, right_hip, left_knee
            12: [6, 11, 14], # right_hip from right_shoulder, left_hip, right_knee
            13: [11, 15],    # left_knee from left_hip, left_ankle
            14: [12, 16],    # right_knee from right_hip, right_ankle
            15: [13],        # left_ankle from left_knee
            16: [14],        # right_ankle from right_knee
        }
        
        neighbors = neighbor_map.get(missing_idx, [])
        valid_neighbors = [kp for kp in neighbors if not np.isnan(keypoints[kp, 0])]
        
        if valid_neighbors:
            # Average valid neighbors
            return np.nanmean(keypoints[valid_neighbors], axis=0)
        else:
            # Fallback: return NaNs to indicate interpolation failed
            return np.full(3, np.nan)
    
    def _calculate_joint_angle(
        self, 
        keypoints: np.ndarray, 
        triplet: Tuple[int, int, int]
    ) -> Optional[float]:
        """Calculate joint angle from three keypoints.
        
        Args:
            keypoints: Array of keypoints
            triplet: (proximal_idx, joint_idx, distal_idx)
            
        Returns:
            Angle in degrees or None if calculation fails
        """
        proximal_idx, joint_idx, distal_idx = triplet
        
        # Check if all keypoints are valid
        if any(np.isnan(keypoints[idx]).any() for idx in triplet):
            return None
        
        # Extract 3D coordinates
        proximal = keypoints[proximal_idx]
        joint = keypoints[joint_idx]
        distal = keypoints[distal_idx]
        
        # Calculate vectors
        vector1 = proximal - joint  # Vector from joint to proximal
        vector2 = distal - joint   # Vector from joint to distal
        
        # Calculate angle using dot product
        angle = self._vector_angle_3d(vector1, vector2)
        
        return angle
    
    def _calculate_ankle_angle(self, keypoints: np.ndarray, left: bool) -> Optional[float]:
        """Calculate ankle angle (foot orientation relative to shin)."""
        if left:
            knee_idx, ankle_idx = 13, 15
        else:
            knee_idx, ankle_idx = 14, 16
        
        if any(np.isnan(keypoints[idx]).any() for idx in [knee_idx, ankle_idx]):
            return None
        
        # Calculate shin vector (knee to ankle)
        knee = keypoints[knee_idx]
        ankle = keypoints[ankle_idx]
        shin_vector = ankle - knee
        
        # Estimate foot direction using ankle position and assuming forward pointing
        # This is a simplified approach - in practice, foot orientation would need
        # additional keypoints or assumptions
        foot_vector = np.array([shin_vector[0], shin_vector[1], shin_vector[2] * 0.5])
        
        # Calculate angle between shin and estimated foot direction
        angle = self._vector_angle_3d(shin_vector, foot_vector)
        
        return angle
    
    def _vector_angle_3d(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Calculate angle between two 3D vectors in degrees."""
        # Normalize vectors
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0 or np.isnan(norm1) or np.isnan(norm2):
            return None
        
        v1_normalized = vector1 / norm1
        v2_normalized = vector2 / norm2
        
        # Calculate angle using dot product
        cos_angle = np.clip(np.dot(v1_normalized, v2_normalized), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg
    
    def _apply_smoothing(self, current_angles: JointAngles) -> JointAngles:
        """Apply temporal smoothing to angle measurements."""
        # Add current angles to history
        self._angle_history.append(current_angles)
        
        # Maintain window size
        if len(self._angle_history) > self.smoothing_window:
            self._angle_history.pop(0)
        
        # Calculate smoothed angles
        smoothed = JointAngles()
        
        # Get all angle values as lists
        angle_fields = [
            "shoulder_left", "shoulder_right", "elbow_left", "elbow_right",
            "hip_left", "hip_right", "knee_left", "knee_right",
            "ankle_left", "ankle_right"
        ]
        
        for field in angle_fields:
            values = []
            for angles in self._angle_history:
                value = getattr(angles, field)
                if value is not None:
                    values.append(value)
            
            if values:
                setattr(smoothed, field, np.mean(values))
            else:
                setattr(smoothed, field, None)
        
        return smoothed
    
    def calculate_batch_angles(
        self, 
        keypoints_batch: List[np.ndarray], 
        confidences_batch: Optional[List[np.ndarray]] = None
    ) -> List[JointAngles]:
        """Calculate angles for a batch of keypoint sequences.
        
        Args:
            keypoints_batch: List of keypoint arrays
            confidences_batch: Optional list of confidence arrays
            
        Returns:
            List of JointAngles objects
        """
        results = []
        
        for i, keypoints in enumerate(keypoints_batch):
            confidences = confidences_batch[i] if confidences_batch else None
            angles = self.calculate_angles(keypoints, confidences)
            results.append(angles)
        
        return results
    
    def reset_smoothing_history(self) -> None:
        """Reset the temporal smoothing history."""
        self._angle_history.clear()


__all__ = [
    "JointAngles",
    "JointAngleCalculator",
]