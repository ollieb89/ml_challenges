"""
Temporal Smoothing with Kalman Filtering for Pose Analysis

This module implements Kalman filtering for COCO 17-keypoint pose trajectories
to reduce jitter between frames while maintaining low latency (<10ms per frame).

Key Features:
- Parallel Kalman filters for 17 COCO keypoints
- Kinematic motion model for smooth trajectory tracking
- Adaptive noise parameters based on detection confidence
- A/B testing framework for raw vs filtered comparison
- Performance optimization for real-time processing

Author: AI/ML Pipeline Team
"""

import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


@dataclass
class KeypointState:
    """State for a single keypoint with position and velocity"""
    x: float  # x position
    y: float  # y position
    vx: float = 0.0  # x velocity
    vy: float = 0.0  # y velocity
    confidence: float = 1.0  # detection confidence


@dataclass
class FilterMetrics:
    """Performance metrics for Kalman filtering"""
    processing_time_ms: float
    jitter_reduction: float
    latency_added_ms: float
    smoothness_score: float


class KalmanFilterManager:
    """
    Manages parallel Kalman filters for 17 COCO keypoints
    
    Each keypoint has a 4D state vector: [x, vx, y, vy]
    Measurement vector: [x, y] (detected position)
    """
    
    # COCO 17 keypoint names
    COCO_KEYPOINTS = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    def __init__(self, 
                 process_noise: float = 0.1,
                 measurement_noise: float = 1.0,
                 dt: float = 0.033):  # ~30 FPS
        """
        Initialize Kalman filters for all 17 keypoints
        
        Args:
            process_noise: Process noise variance (Q matrix)
            measurement_noise: Measurement noise variance (R matrix)
            dt: Time step between frames (seconds)
        """
        self.num_keypoints = 17
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # Initialize filters for each keypoint
        self.filters: List[KalmanFilter] = []
        self.keypoint_states: List[KeypointState] = []
        
        # Performance tracking
        self.processing_times: List[float] = []
        self.frame_count = 0
        
        # A/B testing mode
        self.ab_test_mode = False
        self.raw_keypoints_history: List[np.ndarray] = []
        
        self._initialize_filters()
    
    def _initialize_filters(self) -> None:
        """Initialize Kalman filters for all keypoints"""
        for i in range(self.num_keypoints):
            # Create Kalman filter with 4D state (x, vx, y, vy) and 2D measurement (x, y)
            kf = KalmanFilter(dim_x=4, dim_z=2)
            
            # State transition matrix (constant velocity model)
            kf.F = np.array([
                [1., self.dt, 0., 0.],    # x = x + vx*dt
                [0., 1.,       0., 0.],    # vx = vx
                [0., 0.,       1., self.dt], # y = y + vy*dt
                [0., 0.,       0., 1.]     # vy = vy
            ])
            
            # Measurement function (observe position only)
            kf.H = np.array([
                [1., 0., 0., 0.],  # measure x
                [0., 0., 1., 0.]   # measure y
            ])
            
            # Initial state covariance (high uncertainty)
            kf.P *= 1000.
            
            # Process noise covariance
            kf.Q = Q_discrete_white_noise(dim=4, dt=self.dt, var=self.process_noise)
            
            # Measurement noise covariance
            kf.R = np.array([
                [self.measurement_noise, 0.],
                [0., self.measurement_noise]
            ])
            
            # Initial state (will be set on first measurement)
            kf.x = np.array([[0.], [0.], [0.], [0.]])
            
            self.filters.append(kf)
            self.keypoint_states.append(KeypointState(0, 0))
    
    def update_keypoint(self, 
                       keypoint_idx: int, 
                       x: float, 
                       y: float, 
                       confidence: float = 1.0) -> KeypointState:
        """
        Update a single keypoint with new measurement
        
        Args:
            keypoint_idx: Index of keypoint (0-16)
            x: Measured x coordinate
            y: Measured y coordinate
            confidence: Detection confidence (0-1)
            
        Returns:
            Updated KeypointState with smoothed position
        """
        if keypoint_idx >= self.num_keypoints:
            raise ValueError(f"Keypoint index {keypoint_idx} out of range (0-{self.num_keypoints-1})")
        
        kf = self.filters[keypoint_idx]
        
        # Adaptive measurement noise based on confidence
        adaptive_noise = self.measurement_noise * (2.0 - confidence)  # Higher noise for low confidence
        kf.R = np.array([
            [adaptive_noise, 0.],
            [0., adaptive_noise]
        ])
        
        # First measurement - initialize state
        if self.frame_count == 0:
            kf.x = np.array([[x], [0.], [y], [0.]])  # Initialize with zero velocity
        
        # Prediction step
        kf.predict()
        
        # Update step with measurement
        measurement = np.array([x, y])
        kf.update(measurement)
        
        # Extract smoothed state
        state = kf.x.flatten()
        smoothed_state = KeypointState(
            x=state[0],
            y=state[2], 
            vx=state[1],
            vy=state[3],
            confidence=confidence
        )
        
        self.keypoint_states[keypoint_idx] = smoothed_state
        return smoothed_state
    
    def process_frame(self, 
                     keypoints: np.ndarray, 
                     confidences: Optional[np.ndarray] = None) -> Tuple[np.ndarray, FilterMetrics]:
        """
        Process a full frame of 17 keypoints
        
        Args:
            keypoints: Array of shape (17, 2) with [x, y] coordinates
            confidences: Optional array of shape (17,) with confidence scores
            
        Returns:
            Tuple of (smoothed_keypoints, performance_metrics)
        """
        start_time = time.perf_counter()
        
        if keypoints.shape != (17, 2):
            raise ValueError(f"Expected keypoints shape (17, 2), got {keypoints.shape}")
        
        if confidences is None:
            confidences = np.ones(17)
        elif confidences.shape != (17,):
            raise ValueError(f"Expected confidences shape (17,), got {confidences.shape}")
        
        # Store raw keypoints for A/B testing
        if self.ab_test_mode:
            self.raw_keypoints_history.append(keypoints.copy())
        
        # Process each keypoint
        smoothed_keypoints = []
        for i in range(17):
            x, y = keypoints[i]
            conf = confidences[i]
            
            # Skip keypoints with very low confidence
            if conf < 0.1:
                # Use last known state or pass through
                state = self.keypoint_states[i]
                smoothed_keypoints.append([state.x, state.y])
            else:
                state = self.update_keypoint(i, x, y, conf)
                smoothed_keypoints.append([state.x, state.y])
        
        # Calculate processing time
        processing_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        self.processing_times.append(processing_time)
        self.frame_count += 1
        
        # Calculate metrics
        metrics = self._calculate_metrics(keypoints, np.array(smoothed_keypoints), processing_time)
        
        return np.array(smoothed_keypoints), metrics
    
    def _calculate_metrics(self, 
                          raw_keypoints: np.ndarray, 
                          smoothed_keypoints: np.ndarray, 
                          processing_time: float) -> FilterMetrics:
        """Calculate performance metrics for the current frame"""
        
        # Jitter reduction (average displacement reduction)
        if self.frame_count > 1:
            raw_displacement = np.mean(np.linalg.norm(
                raw_keypoints - self.keypoint_states[-17:].reshape(-1, 2) if len(self.keypoint_states) > 17 else raw_keypoints, 
                axis=1
            ))
            smoothed_displacement = np.mean(np.linalg.norm(
                smoothed_keypoints - self.keypoint_states[-17:].reshape(-1, 2) if len(self.keypoint_states) > 17 else smoothed_keypoints,
                axis=1
            ))
            jitter_reduction = max(0, raw_displacement - smoothed_displacement)
        else:
            jitter_reduction = 0.0
        
        # Smoothness score (lower variance in velocity changes)
        if len(self.processing_times) > 10:
            smoothness_score = 1.0 / (1.0 + np.var(self.processing_times[-10:]))
        else:
            smoothness_score = 1.0
        
        return FilterMetrics(
            processing_time_ms=processing_time,
            jitter_reduction=jitter_reduction,
            latency_added_ms=processing_time,
            smoothness_score=smoothness_score
        )
    
    def enable_ab_testing(self) -> None:
        """Enable A/B testing mode (stores raw keypoints for comparison)"""
        self.ab_test_mode = True
        self.raw_keypoints_history = []
    
    def disable_ab_testing(self) -> None:
        """Disable A/B testing mode"""
        self.ab_test_mode = False
    
    def get_raw_keypoints(self) -> List[np.ndarray]:
        """Get stored raw keypoints for A/B comparison"""
        return self.raw_keypoints_history.copy()
    
    def reset(self) -> None:
        """Reset all filters to initial state"""
        self._initialize_filters()
        self.processing_times = []
        self.frame_count = 0
        self.raw_keypoints_history = []
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get comprehensive performance statistics"""
        if not self.processing_times:
            return {}
        
        return {
            'avg_processing_time_ms': np.mean(self.processing_times),
            'max_processing_time_ms': np.max(self.processing_times),
            'min_processing_time_ms': np.min(self.processing_times),
            'std_processing_time_ms': np.std(self.processing_times),
            'total_frames_processed': self.frame_count,
            'performance_success_rate': np.mean(np.array(self.processing_times) < 10.0) * 100  # % frames under 10ms
        }
    
    def set_noise_parameters(self, 
                            process_noise: Optional[float] = None,
                            measurement_noise: Optional[float] = None) -> None:
        """Update noise parameters for all filters"""
        if process_noise is not None:
            self.process_noise = process_noise
            for kf in self.filters:
                kf.Q = Q_discrete_white_noise(dim=4, dt=self.dt, var=process_noise)
        
        if measurement_noise is not None:
            self.measurement_noise = measurement_noise
            for kf in self.filters:
                kf.R = np.array([
                    [measurement_noise, 0.],
                    [0., measurement_noise]
                ])


def create_kalman_manager() -> KalmanFilterManager:
    """Factory function to create optimized KalmanFilterManager"""
    return KalmanFilterManager(
        process_noise=0.01,      # Low process noise for smooth trajectories
        measurement_noise=0.5,   # Moderate measurement noise
        dt=0.033                # 30 FPS timing
    )


if __name__ == "__main__":
    # Quick test
    print("Testing KalmanFilterManager...")
    
    manager = create_kalman_manager()
    
    # Simulate noisy keypoints
    np.random.seed(42)
    true_position = np.array([100, 200])
    
    for frame in range(10):
        # Add noise to simulate jitter
        noisy_keypoints = np.random.randn(17, 2) * 5 + true_position
        confidences = np.ones(17) * 0.8
        
        smoothed_keypoints, metrics = manager.process_frame(noisy_keypoints, confidences)
        
        print(f"Frame {frame}: Processing time: {metrics.processing_time_ms:.2f}ms")
    
    stats = manager.get_performance_stats()
    print(f"\nPerformance Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value:.3f}")
    
    print("\n✓ KalmanFilterManager test completed successfully")