"""FastAPI routes for pose analysis with MediaPipe and YOLO integration."""

import asyncio
import base64
import io
import json
import time
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from pose_analyzer.pose_detector import MediaPipePoseDetector
from pose_analyzer.yolo_detector import YOLOPoseDetector
from pose_analyzer.form_scorer import FormScorer
from pose_analyzer.video_processor import VideoProcessor
from pose_analyzer.temporal_analyzer import TemporalAnalyzer

from .schemas import (
    PoseDetectionRequest,
    PoseDetectionResponse,
    FormAnalysisRequest,
    FormAnalysisResponse,
    VideoAnalysisRequest,
    VideoAnalysisResponse,
    RealTimeConfig,
    RealTimePoseUpdate,
    HealthResponse,
    ErrorResponse,
    PoseLandmark,
    PoseLandmarks,
    FormIssue,
    FormScore,
    FrameResult
)


# Global state for pose detectors and sessions
mediapipe_detector = None
yolo_detector = None
form_scorer = None
video_processor = None
temporal_analyzer = None
active_sessions: Dict[str, Dict] = {}


def initialize_detectors() -> bool:
    """Initialize pose detection models."""
    global mediapipe_detector, yolo_detector, form_scorer, video_processor, temporal_analyzer
    
    try:
        # Initialize MediaPipe detector
        mediapipe_detector = MediaPipePoseDetector()
        
        # Initialize YOLO detector
        yolo_detector = YOLOPoseDetector()
        
        # Initialize form scorer
        form_scorer = FormScorer()
        
        # Initialize video processor
        video_processor = VideoProcessor()
        
        # Initialize temporal analyzer
        temporal_analyzer = TemporalAnalyzer()
        
        return True
    except Exception as e:
        print(f"Failed to initialize detectors: {e}")
        return False


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 image string to numpy array."""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")


def convert_mediapipe_landmarks(landmarks) -> List[PoseLandmark]:
    """Convert MediaPipe landmarks to our schema format."""
    pose_landmarks = []
    
    for landmark in landmarks:
        pose_landmark = PoseLandmark(
            x=landmark.x,
            y=landmark.y,
            z=landmark.z if hasattr(landmark, 'z') else None,
            visibility=landmark.visibility if hasattr(landmark, 'visibility') else None
        )
        pose_landmarks.append(pose_landmark)
    
    return pose_landmarks


def convert_yolo_landmarks(landmarks) -> List[PoseLandmark]:
    """Convert YOLO landmarks to our schema format."""
    pose_landmarks = []
    
    for landmark in landmarks:
        pose_landmark = PoseLandmark(
            x=landmark[0] if len(landmark) > 0 else 0.0,
            y=landmark[1] if len(landmark) > 1 else 0.0,
            z=landmark[2] if len(landmark) > 2 else None,
            visibility=None  # YOLO doesn't provide visibility
        )
        pose_landmarks.append(pose_landmark)
    
    return pose_landmarks


async def detect_poses_in_image(
    image: np.ndarray,
    request: PoseDetectionRequest,
    use_mediapipe: bool = True
) -> List[PoseLandmarks]:
    """Detect poses in image using specified detector."""
    poses = []
    
    try:
        if use_mediapipe and mediapipe_detector:
            # Update detector configuration
            mediapipe_detector.landmarker.options.min_pose_detection_confidence = request.min_pose_detection_confidence
            mediapipe_detector.landmarker.options.min_pose_presence_confidence = request.min_pose_presence_confidence
            mediapipe_detector.landmarker.options.min_tracking_confidence = request.min_tracking_confidence
            mediapipe_detector.landmarker.options.num_poses = request.max_poses
            
            # Detect poses
            result = mediapipe_detector.detect(image)
            
            if result.pose_landmarks:
                for i, landmarks in enumerate(result.pose_landmarks):
                    pose_landmarks = convert_mediapipe_landmarks(landmarks)
                    pose = PoseLandmarks(
                        landmarks=pose_landmarks,
                        confidence=0.8,  # MediaPipe doesn't provide per-pose confidence
                        pose_id=i
                    )
                    poses.append(pose)
        
        elif yolo_detector:
            # Use YOLO detector
            results = yolo_detector.detect(image)
            
            for i, landmarks in enumerate(results):
                pose_landmarks = convert_yolo_landmarks(landmarks)
                pose = PoseLandmarks(
                    landmarks=pose_landmarks,
                    confidence=0.8,  # YOLO provides confidence but we'll use a default
                    pose_id=i
                )
                poses.append(pose)
        
        return poses
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pose detection failed: {str(e)}")


# FastAPI app will be created in the main module
def create_pose_routes(app: FastAPI) -> None:
    """Create and register pose analysis routes."""
    
    @app.post("/api/pose/detect", response_model=PoseDetectionResponse)
    async def detect_pose(request: PoseDetectionRequest):
        """Detect poses in an image."""
        if not mediapipe_detector and not yolo_detector:
            raise HTTPException(status_code=503, detail="Pose detectors not available")
        
        start_time = time.time()
        
        try:
            # Decode image
            image = decode_base64_image(request.image_data)
            image_height, image_width = image.shape[:2]
            
            # Detect poses
            poses = await detect_poses_in_image(image, request)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            return PoseDetectionResponse(
                poses=poses,
                processing_time_ms=processing_time_ms,
                image_width=image_width,
                image_height=image_height,
                model_used="MediaPipe" if mediapipe_detector else "YOLO"
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    
    
    @app.post("/api/pose/analyze-form", response_model=FormAnalysisResponse)
    async def analyze_form(request: FormAnalysisRequest):
        """Analyze fitness form for detected poses."""
        if not form_scorer:
            raise HTTPException(status_code=503, detail="Form analyzer not available")
        
        start_time = time.time()
        
        try:
            # Analyze form using the form scorer
            form_result = form_scorer.analyze_form(
                poses=request.poses,
                exercise_type=request.exercise_type,
                analysis_level=request.analysis_level
            )
            
            # Convert to response format
            issues = []
            for issue in form_result.get('issues', []):
                form_issue = FormIssue(
                    issue_type=issue.get('type', 'unknown'),
                    severity=issue.get('severity', 'medium'),
                    description=issue.get('description', ''),
                    affected_landmarks=issue.get('affected_landmarks', []),
                    suggestion=issue.get('suggestion', '')
                )
                issues.append(form_issue)
            
            scores = form_result.get('scores', {})
            form_score = FormScore(
                overall_score=scores.get('overall', 0.0),
                stability_score=scores.get('stability', 0.0),
                alignment_score=scores.get('alignment', 0.0),
                range_of_motion_score=scores.get('range_of_motion', 0.0)
            )
            
            analysis_time_ms = (time.time() - start_time) * 1000
            
            return FormAnalysisResponse(
                exercise_type=request.exercise_type,
                form_score=form_score,
                issues=issues,
                feedback=form_result.get('feedback', 'Form analysis completed'),
                analysis_time_ms=analysis_time_ms,
                recommendations=form_result.get('recommendations', [])
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Form analysis failed: {str(e)}")
    
    
    @app.post("/api/pose/analyze-video", response_model=VideoAnalysisResponse)
    async def analyze_video(request: VideoAnalysisRequest):
        """Analyze poses in video frames."""
        if not video_processor:
            raise HTTPException(status_code=503, detail="Video processor not available")
        
        start_time = time.time()
        
        try:
            # Process video
            video_result = video_processor.process_video(
                video_data=request.video_data,
                analysis_type=request.analysis_type,
                exercise_type=request.exercise_type,
                frame_sampling_rate=request.frame_sampling_rate,
                start_time=request.start_time_seconds,
                end_time=request.end_time_seconds,
                detection_confidence=request.detection_confidence
            )
            
            # Convert frame results
            frame_results = []
            for frame_data in video_result.get('frames', []):
                # Convert poses
                poses = []
                for pose_data in frame_data.get('poses', []):
                    landmarks = [PoseLandmark(**lm) for lm in pose_data.get('landmarks', [])]
                    pose = PoseLandmarks(
                        landmarks=landmarks,
                        confidence=pose_data.get('confidence', 0.0),
                        pose_id=pose_data.get('pose_id')
                    )
                    poses.append(pose)
                
                # Convert form analysis if present
                form_analysis = None
                if frame_data.get('form_analysis'):
                    form_data = frame_data['form_analysis']
                    issues = [FormIssue(**issue) for issue in form_data.get('issues', [])]
                    scores = FormScore(**form_data.get('form_score', {}))
                    
                    form_analysis = FormAnalysisResponse(
                        exercise_type=form_data.get('exercise_type', ''),
                        form_score=scores,
                        issues=issues,
                        feedback=form_data.get('feedback', ''),
                        analysis_time_ms=form_data.get('analysis_time_ms', 0.0),
                        recommendations=form_data.get('recommendations', [])
                    )
                
                frame_result = FrameResult(
                    frame_number=frame_data.get('frame_number', 0),
                    timestamp_seconds=frame_data.get('timestamp_seconds', 0.0),
                    poses=poses,
                    form_analysis=form_analysis
                )
                frame_results.append(frame_result)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            return VideoAnalysisResponse(
                video_info=video_result.get('video_info', {}),
                total_frames_analyzed=video_result.get('total_frames_analyzed', 0),
                frame_results=frame_results,
                processing_time_ms=processing_time_ms,
                average_confidence=video_result.get('average_confidence', 0.0)
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")
    
    
    @app.websocket("/api/pose/realtime")
    async def realtime_pose_analysis(websocket: WebSocket):
        """WebSocket endpoint for real-time pose analysis."""
        await websocket.accept()
        
        session_id = f"session_{int(time.time() * 1000)}"
        active_sessions[session_id] = {
            'websocket': websocket,
            'config': None,
            'frame_count': 0
        }
        
        try:
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get('type') == 'config':
                    # Configure real-time analysis
                    config_data = message.get('config', {})
                    try:
                        config = RealTimeConfig(**config_data)
                        active_sessions[session_id]['config'] = config
                        await websocket.send_text(json.dumps({
                            'type': 'config_ack',
                            'session_id': session_id
                        }))
                    except ValidationError as e:
                        await websocket.send_text(json.dumps({
                            'type': 'error',
                            'message': f'Invalid configuration: {str(e)}'
                        }))
                
                elif message.get('type') == 'frame':
                    # Process frame
                    if not active_sessions[session_id]['config']:
                        await websocket.send_text(json.dumps({
                            'type': 'error',
                            'message': 'Configuration not set'
                        }))
                        continue
                    
                    try:
                        frame_data = message.get('image_data', '')
                        config = active_sessions[session_id]['config']
                        
                        # Decode and process frame
                        image = decode_base64_image(frame_data)
                        
                        # Create detection request
                        detection_request = PoseDetectionRequest(
                            image_data=frame_data,
                            min_pose_detection_confidence=config.detection_confidence,
                            min_pose_presence_confidence=config.tracking_confidence,
                            min_tracking_confidence=config.tracking_confidence,
                            max_poses=config.max_poses
                        )
                        
                        # Detect poses
                        poses = await detect_poses_in_image(image, detection_request)
                        
                        # Form analysis if enabled
                        form_feedback = None
                        if config.enable_form_analysis and config.exercise_type and poses:
                            form_request = FormAnalysisRequest(
                                poses=poses,
                                exercise_type=config.exercise_type,
                                analysis_level="basic"
                            )
                            form_result = await analyze_form(form_request)
                            form_feedback = form_result.feedback
                        
                        # Send update
                        update = RealTimePoseUpdate(
                            timestamp=time.time(),
                            frame_id=active_sessions[session_id]['frame_count'],
                            poses=poses,
                            form_feedback=form_feedback,
                            confidence=sum(p.confidence for p in poses) / len(poses) if poses else 0.0
                        )
                        
                        await websocket.send_text(json.dumps({
                            'type': 'pose_update',
                            'data': update.dict()
                        }))
                        
                        active_sessions[session_id]['frame_count'] += 1
                        
                    except Exception as e:
                        await websocket.send_text(json.dumps({
                            'type': 'error',
                            'message': f'Frame processing failed: {str(e)}'
                        }))
                
                elif message.get('type') == 'close':
                    break
                    
        except WebSocketDisconnect:
            pass
        finally:
            # Clean up session
            if session_id in active_sessions:
                del active_sessions[session_id]
    
    
    @app.get("/api/pose/health", response_model=HealthResponse)
    async def health_check():
        """Health check for pose analysis service."""
        return HealthResponse(
            status="healthy",
            mediapipe_available=mediapipe_detector is not None,
            yolo_available=yolo_detector is not None,
            gpu_available=cv2.cuda.getCudaEnabledDeviceCount() > 0,
            active_sessions=len(active_sessions),
            uptime_seconds=time.time()
        )
    
    
    @app.get("/api/pose/sessions")
    async def list_active_sessions():
        """List active real-time analysis sessions."""
        return {
            "active_sessions": len(active_sessions),
            "session_ids": list(active_sessions.keys())
        }
    
    
    @app.delete("/api/pose/sessions/{session_id}")
    async def close_session(session_id: str):
        """Close a specific real-time analysis session."""
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        try:
            websocket = active_sessions[session_id]['websocket']
            await websocket.close()
        except Exception:
            pass
        
        del active_sessions[session_id]
        return {"message": "Session closed successfully"}


# Initialize detectors when module is imported
initialize_detectors()