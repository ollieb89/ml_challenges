"""
Comprehensive tests for FormScorer module

This test suite validates all aspects of the form scoring system including:
- Body proportion estimation and normalization
- Joint angle deviation calculation
- Symmetry analysis
- Trajectory smoothness evaluation
- Range of motion coverage
- Overall scoring algorithm
- Integration with existing modules

Author: AI/ML Pipeline Team
"""

import pytest
import numpy as np
import math
from typing import List, Dict, Any

from pose_analyzer.form_scorer import (
    FormScorer, BodyProportions, FormMetrics, ReferencePose,
    create_form_scorer
)
from pose_analyzer.biomechanics import JointAngles, JointAngleCalculator
from pose_analyzer.synthetic_poses import SyntheticPoseGenerator, SquatType


class TestBodyProportions:
    """Test body proportion estimation and normalization"""
    
    def test_default_proportions(self):
        """Test default body proportions"""
        proportions = BodyProportions()
        
        assert proportions.torso_length == 0.4
        assert proportions.thigh_length == 0.25
        assert proportions.shin_length == 0.25
        assert proportions.upper_arm_length == 0.15
        assert proportions.forearm_length == 0.15
        assert proportions.shoulder_width == 0.15
    
    def test_proportions_from_keypoints(self):
        """Test proportion estimation from keypoints"""
        # Create mock keypoints with known proportions
        keypoints = np.zeros((17, 3))
        
        # Set up a simple humanoid pose
        keypoints[5] = [0.4, 0.6, 0]  # left_shoulder
        keypoints[6] = [0.6, 0.6, 0]  # right_shoulder
        keypoints[11] = [0.45, 0.8, 0]  # left_hip
        keypoints[12] = [0.55, 0.8, 0]  # right_hip
        keypoints[13] = [0.45, 1.0, 0]  # left_knee
        keypoints[14] = [0.55, 1.0, 0]  # right_knee
        keypoints[15] = [0.45, 1.2, 0]  # left_ankle
        keypoints[16] = [0.55, 1.2, 0]  # right_ankle
        keypoints[7] = [0.35, 0.7, 0]  # left_elbow
        keypoints[8] = [0.65, 0.7, 0]  # right_elbow
        keypoints[9] = [0.3, 0.8, 0]   # left_wrist
        keypoints[10] = [0.7, 0.8, 0]  # right_wrist
        
        proportions = BodyProportions.from_keypoints(keypoints)
        
        # Verify proportions are reasonable
        assert 0 < proportions.torso_length < 1.0
        assert 0 < proportions.thigh_length < 1.0
        assert 0 < proportions.shin_length < 1.0
        assert 0 < proportions.upper_arm_length < 1.0
        assert 0 < proportions.forearm_length < 1.0
        assert 0 < proportions.shoulder_width < 1.0
    
    def test_proportions_fallback(self):
        """Test fallback to default proportions with invalid keypoints"""
        # Use zero keypoints (invalid)
        keypoints = np.zeros((17, 3))
        
        proportions = BodyProportions.from_keypoints(keypoints)
        
        # Should fall back to defaults
        assert proportions.torso_length == 0.4
        assert proportions.thigh_length == 0.25


class TestFormScorer:
    """Test FormScorer functionality"""
    
    @pytest.fixture
    def form_scorer(self):
        """Create FormScorer instance for testing"""
        return create_form_scorer("squat")
    
    @pytest.fixture
    def test_sequence(self):
        """Generate test pose sequence"""
        generator = SyntheticPoseGenerator()
        return generator.generate_squat_sequence(
            num_frames=30,
            squat_type=SquatType.PARALLEL,
            noise_level=0.5
        )
    
    def test_form_scorer_initialization(self, form_scorer):
        """Test FormScorer initialization"""
        assert form_scorer.exercise_type == "squat"
        assert isinstance(form_scorer.angle_calculator, JointAngleCalculator)
        assert isinstance(form_scorer.pose_generator, SyntheticPoseGenerator)
        assert len(form_scorer.reference_poses) > 0
        assert "squat_parallel" in form_scorer.reference_poses
        assert "squat_bottom" in form_scorer.reference_poses
        assert "squat_top" in form_scorer.reference_poses
    
    def test_reference_poses_generation(self, form_scorer):
        """Test reference pose generation"""
        for name, reference in form_scorer.reference_poses.items():
            assert isinstance(reference, ReferencePose)
            assert reference.exercise_type == "squat"
            assert reference.keypoints.shape == (17, 3)
            assert isinstance(reference.angles, JointAngles)
            assert isinstance(reference.proportions, BodyProportions)
            assert len(reference.rom_targets) > 0
    
    def test_body_proportion_normalization(self, form_scorer):
        """Test pose normalization by body proportions"""
        # Create test keypoints
        keypoints = np.random.rand(17, 3)
        
        # Create different proportions
        user_proportions = BodyProportions(
            torso_length=0.45, thigh_length=0.3, shin_length=0.2,
            upper_arm_length=0.18, forearm_length=0.12, shoulder_width=0.18
        )
        
        reference_proportions = BodyProportions()  # Default proportions
        
        # Normalize keypoints
        normalized = form_scorer.normalize_pose_by_proportions(
            keypoints, user_proportions, reference_proportions
        )
        
        # Verify shape is preserved
        assert normalized.shape == keypoints.shape
        
        # Verify some transformation occurred
        assert not np.array_equal(normalized, keypoints)
    
    def test_joint_angle_deviation_calculation(self, form_scorer):
        """Test joint angle deviation calculation"""
        # Create test angles
        user_angles = JointAngles(
            knee_left=90.0, knee_right=95.0,
            hip_left=70.0, hip_right=75.0
        )
        
        reference_angles = JointAngles(
            knee_left=85.0, knee_right=85.0,
            hip_left=70.0, hip_right=70.0
        )
        
        deviation, errors = form_scorer.calculate_joint_angle_deviation(
            user_angles, reference_angles
        )
        
        # Verify RMS error is calculated
        assert deviation > 0
        assert isinstance(deviation, float)
        
        # Verify joint-specific errors
        assert 'knee_left' in errors
        assert 'knee_right' in errors
        assert 'hip_left' in errors
        assert 'hip_right' in errors
        
        # Check specific error values
        assert errors['knee_left'] == 5.0  # |90 - 85|
        assert errors['knee_right'] == 10.0  # |95 - 85|
        assert errors['hip_left'] == 0.0   # |70 - 70|
        assert errors['hip_right'] == 5.0  # |75 - 70|
    
    def test_symmetry_score_calculation(self, form_scorer):
        """Test symmetry score calculation"""
        # Perfect symmetry
        perfect_angles = JointAngles(
            knee_left=90.0, knee_right=90.0,
            hip_left=70.0, hip_right=70.0
        )
        
        symmetry_score, details = form_scorer.calculate_symmetry_score(perfect_angles)
        
        assert symmetry_score > 0.5  # Should be mostly symmetric
        assert details['knee_symmetry'] == 1.0
        assert details['hip_symmetry'] == 1.0
        
        # Imperfect symmetry
        imperfect_angles = JointAngles(
            knee_left=90.0, knee_right=80.0,
            hip_left=70.0, hip_right=60.0
        )
        
        symmetry_score, details = form_scorer.calculate_symmetry_score(imperfect_angles)
        
        assert 0.0 < symmetry_score < 1.0
        assert 'knee_symmetry' in details
        assert 'hip_symmetry' in details
        assert details['knee_symmetry'] < 1.0
        assert details['hip_symmetry'] < 1.0
    
    def test_trajectory_smoothness_calculation(self, form_scorer, test_sequence):
        """Test trajectory smoothness calculation"""
        # Extract angle sequence
        angle_sequence = [pose.ground_truth_angles for pose in test_sequence]
        
        smoothness = form_scorer.calculate_trajectory_smoothness(angle_sequence)
        
        assert 0.0 <= smoothness <= 1.0
        assert isinstance(smoothness, float)
        
        # Test with very short sequence
        short_sequence = angle_sequence[:2]
        smoothness_short = form_scorer.calculate_trajectory_smoothness(short_sequence)
        assert smoothness_short == 1.0  # Perfect smoothness for short sequences
    
    def test_rom_coverage_calculation(self, form_scorer, test_sequence):
        """Test range of motion coverage calculation"""
        angle_sequence = [pose.ground_truth_angles for pose in test_sequence]
        
        # Define ROM targets
        rom_targets = {
            'knee_left': (90, 180),
            'knee_right': (90, 180),
            'hip_left': (70, 180)
        }
        
        rom_score, details = form_scorer.calculate_rom_coverage(
            angle_sequence, rom_targets
        )
        
        assert 0.0 <= rom_score <= 1.0
        assert 'knee_left' in details
        assert 'knee_right' in details
        assert 'hip_left' in details
        
        # All values should be between 0 and 1
        assert all(0.0 <= score <= 1.0 for score in details.values())
    
    def test_overall_score_calculation(self, form_scorer):
        """Test overall score calculation"""
        # Create test metrics
        metrics = FormMetrics(
            joint_angle_deviation=5.0,  # Low error
            symmetry_score=0.9,         # High symmetry
            trajectory_smoothness=0.8,   # Good smoothness
            rom_coverage=0.85           # Good ROM coverage
        )
        
        overall_score = form_scorer.calculate_overall_score(metrics)
        
        assert 0.0 <= overall_score <= 100.0
        assert overall_score > 50.0  # Should be high for good metrics
        
        # Test with poor metrics
        poor_metrics = FormMetrics(
            joint_angle_deviation=25.0,  # High error
            symmetry_score=0.3,          # Low symmetry
            trajectory_smoothness=0.2,    # Poor smoothness
            rom_coverage=0.4             # Poor ROM coverage
        )
        
        poor_score = form_scorer.calculate_overall_score(poor_metrics)
        
        assert poor_score < overall_score
        assert poor_score < 50.0  # Should be low for poor metrics
    
    def test_score_single_pose(self, form_scorer, test_sequence):
        """Test single pose scoring"""
        # Use middle frame
        pose = test_sequence[len(test_sequence) // 2]
        
        metrics = form_scorer.score_single_pose(
            pose.keypoints, pose.confidences, "squat_parallel"
        )
        
        assert isinstance(metrics, FormMetrics)
        assert 0.0 <= metrics.overall_score <= 100.0
        assert metrics.joint_angle_deviation >= 0.0
        assert 0.0 <= metrics.symmetry_score <= 1.0
        assert 0.0 <= metrics.trajectory_smoothness <= 1.0
        assert 0.0 <= metrics.rom_coverage <= 1.0
        
        # Check detailed metrics are populated
        assert len(metrics.joint_errors) > 0
        assert len(metrics.symmetry_details) > 0
        assert len(metrics.rom_details) > 0
    
    def test_score_pose_sequence(self, form_scorer, test_sequence):
        """Test pose sequence scoring"""
        # Extract keypoints and confidences
        keypoints_sequence = [pose.keypoints for pose in test_sequence]
        confidences_sequence = [pose.confidences for pose in test_sequence]
        
        metrics = form_scorer.score_pose_sequence(
            keypoints_sequence, confidences_sequence, "squat_parallel"
        )
        
        assert isinstance(metrics, FormMetrics)
        assert 0.0 <= metrics.overall_score <= 100.0
        assert metrics.joint_angle_deviation >= 0.0
        assert 0.0 <= metrics.symmetry_score <= 1.0
        assert 0.0 <= metrics.trajectory_smoothness <= 1.0
        assert 0.0 <= metrics.rom_coverage <= 1.0
    
    def test_invalid_reference_pose(self, form_scorer, test_sequence):
        """Test error handling for invalid reference pose"""
        pose = test_sequence[0]
        
        with pytest.raises(ValueError, match="Reference pose 'invalid' not found"):
            form_scorer.score_single_pose(
                pose.keypoints, pose.confidences, "invalid"
            )
    
    def test_different_reference_poses(self, form_scorer, test_sequence):
        """Test scoring against different reference poses"""
        pose = test_sequence[len(test_sequence) // 2]
        
        # Score against different references
        bottom_metrics = form_scorer.score_single_pose(
            pose.keypoints, pose.confidences, "squat_bottom"
        )
        
        parallel_metrics = form_scorer.score_single_pose(
            pose.keypoints, pose.confidences, "squat_parallel"
        )
        
        top_metrics = form_scorer.score_single_pose(
            pose.keypoints, pose.confidences, "squat_top"
        )
        
        # Scores should be different for different references
        # (though they might be similar for some poses)
        assert isinstance(bottom_metrics.overall_score, float)
        assert isinstance(parallel_metrics.overall_score, float)
        assert isinstance(top_metrics.overall_score, float)


class TestFormScorerIntegration:
    """Integration tests for FormScorer with existing modules"""
    
    def test_integration_with_biomechanics(self):
        """Test integration with biomechanics module"""
        scorer = create_form_scorer("squat")
        generator = SyntheticPoseGenerator()
        
        # Generate test pose
        sequence = generator.generate_squat_sequence(
            num_frames=10, squat_type=SquatType.PARALLEL, noise_level=0.0
        )
        
        pose = sequence[0]
        
        # Score using FormScorer
        metrics = scorer.score_single_pose(pose.keypoints, pose.confidences)
        
        # Verify integration worked
        assert isinstance(metrics, FormMetrics)
        assert hasattr(metrics, 'joint_angle_deviation')
        assert hasattr(metrics, 'overall_score')
    
    def test_performance_with_large_sequences(self):
        """Test performance with larger sequences"""
        scorer = create_form_scorer("squat")
        generator = SyntheticPoseGenerator()
        
        # Generate larger sequence
        large_sequence = generator.generate_squat_sequence(
            num_frames=120,  # 2 seconds at 60 FPS
            squat_type=SquatType.PARALLEL,
            noise_level=1.0
        )
        
        keypoints_sequence = [pose.keypoints for pose in large_sequence]
        confidences_sequence = [pose.confidences for pose in large_sequence]
        
        # Time the scoring process
        import time
        start_time = time.time()
        
        metrics = scorer.score_pose_sequence(
            keypoints_sequence, confidences_sequence
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify results
        assert isinstance(metrics, FormMetrics)
        
        # Performance should be reasonable (less than 1 second for 120 frames)
        assert processing_time < 1.0
        print(f"Processed {len(large_sequence)} frames in {processing_time:.3f}s")
    
    def test_form_metrics_serialization(self):
        """Test FormMetrics serialization"""
        metrics = FormMetrics(
            overall_score=85.5,
            joint_angle_deviation=3.2,
            symmetry_score=0.9,
            trajectory_smoothness=0.85,
            rom_coverage=0.8
        )
        
        # Test to_dict conversion
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert 'overall_score' in metrics_dict
        assert 'joint_angle_deviation' in metrics_dict
        assert 'symmetry_score' in metrics_dict
        assert 'trajectory_smoothness' in metrics_dict
        assert 'rom_coverage' in metrics_dict
        
        assert metrics_dict['overall_score'] == 85.5
        assert metrics_dict['joint_angle_deviation'] == 3.2


class TestFormScorerEdgeCases:
    """Test edge cases and error handling"""
    
    def test_missing_joint_angles(self):
        """Test handling of missing joint angles"""
        scorer = create_form_scorer("squat")
        
        # Create angles with some missing values
        user_angles = JointAngles(
            knee_left=90.0, knee_right=None,  # Missing right knee
            hip_left=70.0, hip_right=75.0
        )
        
        reference_angles = JointAngles(
            knee_left=85.0, knee_right=85.0,
            hip_left=70.0, hip_right=70.0
        )
        
        deviation, errors = scorer.calculate_joint_angle_deviation(
            user_angles, reference_angles
        )
        
        # Should handle missing values gracefully
        assert isinstance(deviation, float)
        assert errors['knee_right'] == 0.0  # Default to no error for missing data
        assert errors['knee_left'] == 5.0   # Still calculate for present data
    
    def test_empty_angle_sequence(self):
        """Test handling of empty angle sequences"""
        scorer = create_form_scorer("squat")
        
        # Test smoothness with empty sequence
        smoothness = scorer.calculate_trajectory_smoothness([])
        assert smoothness == 1.0  # Perfect smoothness for empty sequence
        
        # Test ROM coverage with empty sequence
        rom_score, rom_details = scorer.calculate_rom_coverage([], {})
        assert rom_score == 0.0
        assert rom_details == {}
    
    def test_zero_rom_targets(self):
        """Test ROM coverage with zero target range"""
        scorer = create_form_scorer("squat")
        
        # Create test angles
        angle_sequence = [
            JointAngles(knee_left=90.0),
            JointAngles(knee_left=95.0),
            JointAngles(knee_left=85.0)
        ]
        
        # Zero target range
        rom_targets = {'knee_left': (90.0, 90.0)}  # No expected range
        
        rom_score, rom_details = scorer.calculate_rom_coverage(
            angle_sequence, rom_targets
        )
        
        assert rom_score == 1.0  # Perfect coverage for zero range
        assert rom_details['knee_left'] == 1.0


if __name__ == "__main__":
    # Run quick test
    print("Running FormScorer tests...")
    
    # Test basic functionality
    scorer = create_form_scorer("squat")
    generator = SyntheticPoseGenerator()
    
    # Generate test data
    test_sequence = generator.generate_squat_sequence(
        num_frames=20, squat_type=SquatType.PARALLEL, noise_level=0.5
    )
    
    # Test single pose scoring
    pose = test_sequence[10]
    metrics = scorer.score_single_pose(pose.keypoints, pose.confidences)
    
    print(f"Single pose score: {metrics.overall_score:.1f}/100")
    
    # Test sequence scoring
    keypoints_sequence = [pose.keypoints for pose in test_sequence]
    confidences_sequence = [pose.confidences for pose in test_sequence]
    
    sequence_metrics = scorer.score_pose_sequence(
        keypoints_sequence, confidences_sequence
    )
    
    print(f"Sequence score: {sequence_metrics.overall_score:.1f}/100")
    print(f"Joint deviation: {sequence_metrics.joint_angle_deviation:.2f}°")
    print(f"Symmetry: {sequence_metrics.symmetry_score:.3f}")
    print(f"Smoothness: {sequence_metrics.trajectory_smoothness:.3f}")
    print(f"ROM coverage: {sequence_metrics.rom_coverage:.3f}")
    
    print("\n✓ FormScorer tests completed successfully")
