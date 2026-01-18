"""
Synthetic Pose Generator for Biomechanics Validation

This module creates anatomically realistic synthetic pose data for testing
joint angle calculations, with focus on squat movements and validation
scenarios.

Key Features:
- Anatomically correct joint angle ranges
- Realistic squat movement patterns
- Configurable noise and confidence levels
- Ground truth angle generation
- Multiple squat variations (depth, speed, form)

Author: AI/ML Pipeline Team
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

from .biomechanics import JointAngles, JointAngleCalculator


class SquatType(Enum):
    """Types of squat movements for validation"""
    SHALLOW = "shallow"      # ~45° knee flexion
    PARALLEL = "parallel"    # ~90° knee flexion  
    DEEP = "deep"           # ~120° knee flexion
    VARIATION = "variation"  # Mixed depths


@dataclass
class PoseParameters:
    """Parameters for synthetic pose generation"""
    # Body dimensions (normalized units)
    torso_length: float = 0.4
    thigh_length: float = 0.25
    shin_length: float = 0.25
    upper_arm_length: float = 0.15
    forearm_length: float = 0.15
    
    # Position and orientation
    hip_center_x: float = 0.5
    hip_center_y: float = 0.6
    shoulder_width: float = 0.15
    
    # Movement parameters
    squat_depth: float = 90.0  # degrees of knee flexion
    squat_speed: float = 1.0    # relative speed
    forward_lean: float = 5.0   # degrees of torso forward lean


@dataclass 
class SyntheticPose:
    """Synthetic pose with keypoints and ground truth angles"""
    keypoints: np.ndarray  # (17, 2) COCO format
    confidences: np.ndarray  # (17,) confidence scores
    ground_truth_angles: JointAngles
    frame_number: int
    squat_phase: str  # 'descending', 'bottom', 'ascending', 'standing'


class SyntheticPoseGenerator:
    """
    Generate anatomically realistic synthetic poses for validation testing
    """
    
    # COCO 17 keypoint names in order
    COCO_KEYPOINTS = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    def __init__(self, params: Optional[PoseParameters] = None):
        """
        Initialize pose generator
        
        Args:
            params: Optional pose parameters, defaults to standard proportions
        """
        self.params = params or PoseParameters()
        self.angle_calculator = JointAngleCalculator()
        
        # Noise parameters
        self.position_noise_std = 0.02  # 2% of image dimensions
        self.confidence_base = 0.9
        self.confidence_variation = 0.1
    
    def generate_squat_sequence(
        self, 
        num_frames: int = 60,
        squat_type: SquatType = SquatType.PARALLEL,
        noise_level: float = 1.0
    ) -> List[SyntheticPose]:
        """
        Generate a complete squat movement sequence
        
        Args:
            num_frames: Number of frames in the sequence
            squat_type: Type of squat to generate
            noise_level: Multiplier for noise intensity
            
        Returns:
            List of synthetic poses with ground truth angles
        """
        poses = []
        
        for frame in range(num_frames):
            # Calculate squat progress (0.0 to 1.0 to 0.0)
            progress = self._calculate_squat_progress(frame, num_frames)
            
            # Determine squat depth based on type
            target_depth = self._get_squat_depth(squat_type, progress)
            
            # Generate pose for this frame
            pose = self._generate_frame_pose(
                frame, target_depth, progress, noise_level
            )
            
            poses.append(pose)
        
        return poses
    
    def _calculate_squat_progress(self, frame: int, num_frames: int) -> float:
        """Calculate squat progress (0.0 = standing, 1.0 = deepest point)"""
        # Use sine wave for smooth movement
        phase = (frame / num_frames) * 2 * math.pi
        # Map to 0-1-0 pattern (stand -> squat -> stand)
        progress = (1.0 - math.cos(phase)) / 2.0
        return progress
    
    def _get_squat_depth(self, squat_type: SquatType, progress: float) -> float:
        """Get target knee flexion angle based on squat type and progress"""
        base_depths = {
            SquatType.SHALLOW: 45.0,
            SquatType.PARALLEL: 90.0,
            SquatType.DEEP: 120.0
        }
        
        if squat_type == SquatType.VARIATION:
            # Vary depth between 60° and 100°
            base_depth = 60.0 + 40.0 * math.sin(progress * math.pi)
        else:
            base_depth = base_depths[squat_type]
        
        # Apply progress (standing = 180°, full squat = base_depth)
        knee_angle = 180.0 - (180.0 - base_depth) * progress
        return knee_angle
    
    def _generate_frame_pose(
        self, 
        frame: int, 
        knee_angle: float, 
        progress: float,
        noise_level: float
    ) -> SyntheticPose:
        """Generate a single frame pose with specified knee angle"""
        
        # Calculate joint positions based on anatomical constraints
        keypoints = self._calculate_keypoints(knee_angle, progress)
        
        # Add realistic noise
        noisy_keypoints = self._add_position_noise(keypoints, noise_level)
        
        # Generate confidences
        confidences = self._generate_confidences(noise_level)
        
        # Calculate ground truth angles
        ground_truth_angles = self._calculate_ground_truth_angles(keypoints)
        
        # Determine squat phase
        squat_phase = self._determine_squat_phase(progress)
        
        return SyntheticPose(
            keypoints=noisy_keypoints,
            confidences=confidences,
            ground_truth_angles=ground_truth_angles,
            frame_number=frame,
            squat_phase=squat_phase
        )
    
    def _calculate_keypoints(self, knee_angle: float, progress: float) -> np.ndarray:
        """
        Calculate keypoint positions based on anatomical constraints
        
        Args:
            knee_angle: Target knee flexion angle in degrees
            progress: Squat progress (0.0 to 1.0)
            
        Returns:
            Array of (17, 3) keypoint coordinates with z=0 for 2D poses
        """
        keypoints = np.zeros((17, 3))
        
        # Convert angles to radians
        knee_rad = math.radians(knee_angle)
        hip_flexion = math.radians(5.0 + 10.0 * progress)  # Slight hip flexion
        torso_lean = math.radians(self.params.forward_lean * progress)
        
        # Calculate positions
        # Hip center (reference point)
        hip_x = self.params.hip_center_x
        hip_y = self.params.hip_center_y
        
        # Shoulders
        shoulder_y = hip_y - self.params.torso_length * math.cos(torso_lean)
        shoulder_x_offset = self.params.torso_length * math.sin(torso_lean)
        
        # Left shoulder
        keypoints[5] = [hip_x - self.params.shoulder_width/2 + shoulder_x_offset, shoulder_y, 0.0]
        # Right shoulder  
        keypoints[6] = [hip_x + self.params.shoulder_width/2 + shoulder_x_offset, shoulder_y, 0.0]
        
        # Hips
        keypoints[11] = [hip_x - self.params.shoulder_width/4, hip_y, 0.0]  # Left hip
        keypoints[12] = [hip_x + self.params.shoulder_width/4, hip_y, 0.0]  # Right hip
        
        # Knees (based on knee angle)
        knee_height = hip_y + self.params.thigh_length * math.sin(knee_rad)
        knee_forward = self.params.thigh_length * math.cos(knee_rad)
        
        keypoints[13] = [hip_x - self.params.shoulder_width/4 - knee_forward, knee_height, 0.0]  # Left knee
        keypoints[14] = [hip_x + self.params.shoulder_width/4 - knee_forward, knee_height, 0.0]  # Right knee
        
        # Ankles (assuming feet flat on ground)
        ankle_y = knee_height + self.params.shin_length * math.cos(knee_rad)
        ankle_forward = knee_forward + self.params.shin_length * math.sin(knee_rad)
        
        keypoints[15] = [hip_x - self.params.shoulder_width/4 - ankle_forward, ankle_y, 0.0]  # Left ankle
        keypoints[16] = [hip_x + self.params.shoulder_width/4 - ankle_forward, ankle_y, 0.0]  # Right ankle
        
        # Arms (slight movement for balance)
        arm_swing = math.sin(progress * math.pi) * 0.05  # Small arm swing
        
        # Elbows
        keypoints[7] = [keypoints[5][0] - self.params.upper_arm_length + arm_swing, 
                        keypoints[5][1] + self.params.upper_arm_length * 0.3, 0.0]  # Left elbow
        keypoints[8] = [keypoints[6][0] + self.params.upper_arm_length - arm_swing,
                        keypoints[6][1] + self.params.upper_arm_length * 0.3, 0.0]  # Right elbow
        
        # Wrists
        keypoints[9] = [keypoints[7][0] - self.params.forearm_length + arm_swing * 0.5,
                        keypoints[7][1] + self.params.forearm_length * 0.2, 0.0]  # Left wrist
        keypoints[10] = [keypoints[8][0] + self.params.forearm_length - arm_swing * 0.5,
                         keypoints[8][1] + self.params.forearm_length * 0.2, 0.0]  # Right wrist
        
        # Head (simplified - centered above shoulders)
        head_center_x = (keypoints[5][0] + keypoints[6][0]) / 2
        head_center_y = keypoints[5][1] - 0.1
        
        keypoints[0] = [head_center_x, head_center_y, 0.0]  # Nose
        keypoints[1] = [head_center_x - 0.02, head_center_y - 0.02, 0.0]  # Left eye
        keypoints[2] = [head_center_x + 0.02, head_center_y - 0.02, 0.0]  # Right eye
        keypoints[3] = [head_center_x - 0.04, head_center_y, 0.0]  # Left ear
        keypoints[4] = [head_center_x + 0.04, head_center_y, 0.0]  # Right ear
        
        return keypoints
    
    def _add_position_noise(self, keypoints: np.ndarray, noise_level: float) -> np.ndarray:
        """Add realistic position noise to keypoints"""
        noise = np.zeros((17, 3))
        noise[:, :2] = np.random.randn(17, 2) * self.position_noise_std * noise_level
        return keypoints + noise
    
    def _generate_confidences(self, noise_level: float) -> np.ndarray:
        """Generate realistic confidence scores"""
        base_conf = self.confidence_base - noise_level * 0.2
        variation = self.confidence_variation * noise_level
        
        confidences = base_conf + np.random.randn(17) * variation
        confidences = np.clip(confidences, 0.3, 1.0)  # Keep within reasonable range
        
        # Lower confidence for extremities
        confidences[9:11] *= 0.9  # Wrists
        confidences[15:17] *= 0.95  # Ankles
        
        return confidences
    
    def _calculate_ground_truth_angles(self, keypoints: np.ndarray) -> JointAngles:
        """Calculate ground truth joint angles from clean keypoints"""
        return self.angle_calculator.calculate_angles(keypoints)
    
    def _determine_squat_phase(self, progress: float) -> str:
        """Determine squat phase based on progress"""
        if progress < 0.4:
            return 'descending'
        elif progress < 0.6:
            return 'bottom'
        elif progress < 1.0:
            return 'ascending'
        else:
            return 'standing'
    
    def generate_validation_dataset(
        self, 
        num_squats: int = 50,
        frames_per_squat: int = 60,
        noise_levels: List[float] = [0.5, 1.0, 1.5]
    ) -> Dict[str, List[SyntheticPose]]:
        """
        Generate comprehensive validation dataset
        
        Args:
            num_squats: Number of squat repetitions to generate
            frames_per_squat: Frames per squat repetition
            noise_levels: Different noise levels to test
            
        Returns:
            Dictionary mapping noise levels to pose sequences
        """
        dataset = {}
        
        for noise_level in noise_levels:
            squat_sequences = []
            
            for squat_idx in range(num_squats):
                # Vary squat type for diversity
                squat_types = list(SquatType)
                squat_type = squat_types[squat_idx % len(squat_types)]
                
                # Generate sequence
                sequence = self.generate_squat_sequence(
                    frames_per_squat, squat_type, noise_level
                )
                
                squat_sequences.append(sequence)
            
            dataset[f"noise_{noise_level}"] = squat_sequences
        
        return dataset


def create_pose_generator() -> SyntheticPoseGenerator:
    """Factory function to create optimized pose generator"""
    return SyntheticPoseGenerator()


if __name__ == "__main__":
    # Quick test
    print("Testing SyntheticPoseGenerator...")
    
    generator = create_pose_generator()
    
    # Generate a test squat sequence
    sequence = generator.generate_squat_sequence(
        num_frames=30, 
        squat_type=SquatType.PARALLEL,
        noise_level=1.0
    )
    
    print(f"Generated {len(sequence)} frames")
    
    # Check first and last frames
    first_frame = sequence[0]
    last_frame = sequence[-1]
    
    print(f"First frame knee angle: {first_frame.ground_truth_angles.knee_left:.1f}°")
    print(f"Last frame knee angle: {last_frame.ground_truth_angles.knee_left:.1f}°")
    print(f"Average confidence: {np.mean(first_frame.confidences):.3f}")
    
    # Test validation dataset generation
    dataset = generator.generate_validation_dataset(
        num_squats=5, 
        frames_per_squat=20,
        noise_levels=[0.5, 1.0]
    )
    
    print(f"Generated validation dataset with {len(dataset)} noise levels")
    for noise_level, sequences in dataset.items():
        print(f"  {noise_level}: {len(sequences)} sequences")
    
    print("\n✓ SyntheticPoseGenerator test completed successfully")
