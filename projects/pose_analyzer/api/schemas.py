"""Pydantic schemas for pose analysis API requests and responses."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class JointAngle(BaseModel):
    """Individual joint angle measurement."""
    
    angle_degrees: Optional[float] = Field(None, description="Joint angle in degrees")
    confidence: Optional[float] = Field(None, description="Confidence score for angle measurement")


class JointAngles(BaseModel):
    """Complete set of joint angles for a person."""
    
    shoulder_left: Optional[float] = Field(None, description="Left shoulder angle in degrees")
    shoulder_right: Optional[float] = Field(None, description="Right shoulder angle in degrees")
    elbow_left: Optional[float] = Field(None, description="Left elbow angle in degrees")
    elbow_right: Optional[float] = Field(None, description="Right elbow angle in degrees")
    hip_left: Optional[float] = Field(None, description="Left hip angle in degrees")
    hip_right: Optional[float] = Field(None, description="Right hip angle in degrees")
    knee_left: Optional[float] = Field(None, description="Left knee angle in degrees")
    knee_right: Optional[float] = Field(None, description="Right knee angle in degrees")
    ankle_left: Optional[float] = Field(None, description="Left ankle angle in degrees")
    ankle_right: Optional[float] = Field(None, description="Right ankle angle in degrees")


class PoseLandmark(BaseModel):
    """Individual pose landmark point."""
    
    x: float = Field(..., description="Normalized x coordinate [0.0, 1.0]")
    y: float = Field(..., description="Normalized y coordinate [0.0, 1.0]")
    z: Optional[float] = Field(None, description="Normalized z coordinate (depth)")
    visibility: Optional[float] = Field(None, description="Landmark visibility score [0.0, 1.0]")


class PoseLandmarks(BaseModel):
    """Complete set of pose landmarks for a single person."""
    
    landmarks: List[PoseLandmark] = Field(..., description="List of pose landmarks")
    confidence: float = Field(..., description="Overall pose detection confidence")
    pose_id: Optional[int] = Field(None, description="Unique identifier for this pose")
    joint_angles: Optional[JointAngles] = Field(None, description="Calculated joint angles for this pose")


class PoseDetectionRequest(BaseModel):
    """Request for pose detection on image data."""
    
    image_data: str = Field(..., description="Base64 encoded image data")
    min_pose_detection_confidence: float = Field(default=0.5, description="Minimum pose detection confidence")
    min_pose_presence_confidence: float = Field(default=0.5, description="Minimum pose presence confidence")
    min_tracking_confidence: float = Field(default=0.5, description="Minimum tracking confidence")
    max_poses: int = Field(default=2, description="Maximum number of poses to detect")
    model_complexity: int = Field(default=1, description="Model complexity: 0, 1, or 2")
    enable_joint_angles: bool = Field(default=False, description="Enable joint angle calculations")
    angle_confidence_threshold: float = Field(default=0.5, description="Confidence threshold for joint angles")
    angle_smoothing_window: int = Field(default=3, description="Temporal smoothing window for joint angles")


class PoseDetectionResponse(BaseModel):
    """Response containing detected poses."""
    
    poses: List[PoseLandmarks] = Field(..., description="List of detected poses")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    image_width: int = Field(..., description="Original image width")
    image_height: int = Field(..., description="Original image height")
    model_used: str = Field(..., description="Model used for detection")


class FormAnalysisRequest(BaseModel):
    """Request for fitness form analysis."""
    
    poses: List[PoseLandmarks] = Field(..., description="Detected poses to analyze")
    exercise_type: str = Field(..., description="Type of exercise (squat, deadlift, pushup, etc.)")
    analysis_level: str = Field(default="basic", description="Analysis level: basic, detailed, expert")
    user_preferences: Optional[Dict[str, Any]] = Field(default=None, description="User analysis preferences")


class FormIssue(BaseModel):
    """Individual form issue detected."""
    
    issue_type: str = Field(..., description="Type of form issue")
    severity: str = Field(..., description="Severity level: low, medium, high")
    description: str = Field(..., description="Human-readable description of the issue")
    affected_landmarks: List[int] = Field(..., description="Indices of affected landmarks")
    suggestion: str = Field(..., description="Suggestion for correction")


class FormScore(BaseModel):
    """Form scoring components."""
    
    overall_score: float = Field(..., description="Overall form score [0.0, 100.0]")
    stability_score: float = Field(..., description="Stability score [0.0, 100.0]")
    alignment_score: float = Field(..., description="Alignment score [0.0, 100.0]")
    range_of_motion_score: float = Field(..., description="Range of motion score [0.0, 100.0]")


class FormAnalysisResponse(BaseModel):
    """Response containing form analysis results."""
    
    exercise_type: str = Field(..., description="Type of exercise analyzed")
    form_score: FormScore = Field(..., description="Detailed form scoring")
    issues: List[FormIssue] = Field(..., description="List of detected form issues")
    feedback: str = Field(..., description="Overall feedback message")
    analysis_time_ms: float = Field(..., description="Analysis time in milliseconds")
    recommendations: List[str] = Field(..., description="Improvement recommendations")


class VideoAnalysisRequest(BaseModel):
    """Request for video-based pose analysis."""
    
    video_data: str = Field(..., description="Base64 encoded video data or file path")
    analysis_type: str = Field(..., description="Type of analysis: pose_detection, form_analysis, both")
    exercise_type: Optional[str] = Field(None, description="Exercise type for form analysis")
    frame_sampling_rate: int = Field(default=1, description="Sample every Nth frame")
    start_time_seconds: float = Field(default=0.0, description="Start time in seconds")
    end_time_seconds: Optional[float] = Field(None, description="End time in seconds")
    detection_confidence: float = Field(default=0.5, description="Detection confidence threshold")


class FrameResult(BaseModel):
    """Results for a single video frame."""
    
    frame_number: int = Field(..., description="Frame number in video")
    timestamp_seconds: float = Field(..., description="Timestamp in seconds")
    poses: List[PoseLandmarks] = Field(..., description="Detected poses in this frame")
    form_analysis: Optional[FormAnalysisResponse] = Field(None, description="Form analysis if requested")


class VideoAnalysisResponse(BaseModel):
    """Response containing video analysis results."""
    
    video_info: Dict[str, Any] = Field(..., description="Video metadata")
    total_frames_analyzed: int = Field(..., description="Total number of frames analyzed")
    frame_results: List[FrameResult] = Field(..., description="Results for each analyzed frame")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    average_confidence: float = Field(..., description="Average detection confidence")


class RealTimeConfig(BaseModel):
    """Configuration for real-time pose analysis."""
    
    detection_confidence: float = Field(default=0.5, description="Detection confidence threshold")
    tracking_confidence: float = Field(default=0.5, description="Tracking confidence threshold")
    max_poses: int = Field(default=1, description="Maximum poses to track")
    enable_form_analysis: bool = Field(default=False, description="Enable real-time form analysis")
    exercise_type: Optional[str] = Field(None, description="Exercise type for form analysis")
    smoothing_enabled: bool = Field(default=True, description="Enable pose smoothing")
    buffer_size: int = Field(default=10, description="Frame buffer size for smoothing")
    enable_joint_angles: bool = Field(default=False, description="Enable joint angle calculations")
    angle_confidence_threshold: float = Field(default=0.5, description="Confidence threshold for joint angles")
    angle_smoothing_window: int = Field(default=3, description="Temporal smoothing window for joint angles")


class RealTimePoseUpdate(BaseModel):
    """Real-time pose update message."""
    
    timestamp: float = Field(..., description="Timestamp of update")
    frame_id: int = Field(..., description="Unique frame identifier")
    poses: List[PoseLandmarks] = Field(..., description="Current pose detections")
    form_feedback: Optional[str] = Field(None, description="Real-time form feedback")
    confidence: float = Field(..., description="Average confidence for this frame")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: float = Field(..., description="Error timestamp")


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Service status")
    mediapipe_available: bool = Field(..., description="MediaPipe library availability")
    yolo_available: bool = Field(..., description="YOLO models availability")
    gpu_available: bool = Field(..., description="GPU acceleration availability")
    active_sessions: int = Field(..., description="Number of active analysis sessions")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")