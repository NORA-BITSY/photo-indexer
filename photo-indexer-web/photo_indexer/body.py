"""Body pose detection and measurement extraction using MediaPipe Pose (tasks API)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, PoseLandmarkerResult
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

from .models import BodyLandmarkSet, BodyPose, BodyMeasurement, MeasurementType

# MediaPipe Pose landmark indices (33 landmarks)
MP_POSE_LANDMARKS = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


# Measurement landmark pairs (indices) for 2D landmark-based measurements
MEASUREMENT_LANDMARKS = {
    MeasurementType.HEIGHT: (0, 27),  # nose to left_ankle (approximate)
    MeasurementType.HEAD_HEIGHT: (0, 9),  # nose to mouth_left
    MeasurementType.NECK_LENGTH: (0, 11),  # nose to left_shoulder
    MeasurementType.SHOULDER_WIDTH: (11, 12),  # left_shoulder to right_shoulder
    MeasurementType.CHEST_WIDTH: (11, 12),  # shoulders as proxy
    MeasurementType.WAIST_WIDTH: (23, 24),  # hips as proxy
    MeasurementType.HIP_WIDTH: (23, 24),  # left_hip to right_hip
    MeasurementType.UPPER_ARM_LENGTH: (11, 13),  # left_shoulder to left_elbow
    MeasurementType.FOREARM_LENGTH: (13, 15),  # left_elbow to left_wrist
    MeasurementType.HAND_LENGTH: (15, 19),  # left_wrist to left_index
    MeasurementType.THIGH_LENGTH: (23, 25),  # left_hip to left_knee
    MeasurementType.SHIN_LENGTH: (25, 27),  # left_knee to left_ankle
    MeasurementType.FOOT_LENGTH: (27, 31),  # left_ankle to left_foot_index
    MeasurementType.ARM_SPAN: (15, 16),  # left_wrist to right_wrist
    MeasurementType.TORSO_LENGTH: (11, 23),  # left_shoulder to left_hip
    MeasurementType.LEG_LENGTH: (23, 27),  # left_hip to left_ankle
    MeasurementType.INSEAM: (24, 28),  # right_hip to right_ankle
}


@dataclass
class BodyDetectorConfig:
    """Configuration for body pose detection."""
    model_complexity: int = 1
    min_confidence: float = 0.5
    model_asset_path: str = "pose_landmarker_heavy.task"


class BodyDetector:
    """Wrapper around MediaPipe PoseLandmarker for body pose detection and landmark extraction."""

    def __init__(self, config: BodyDetectorConfig = None):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe not available. Install with: pip install mediapipe")
        
        self.config = config or BodyDetectorConfig()
        self._detector: Optional[PoseLandmarker] = None

    def __enter__(self):
        self._init_detector()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._detector:
            self._detector.close()

    def _init_detector(self):
        base_options = mp.tasks.BaseOptions(
            model_asset_path=self.config.model_asset_path
        )
        options = PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=self.config.min_confidence,
            min_pose_presence_confidence=self.config.min_confidence,
            min_tracking_confidence=self.config.min_confidence,
            output_segmentation_masks=False,
        )
        self._detector = PoseLandmarker.create_from_options(options)

    def detect_pose(self, image_path: Union[str, Path]) -> Optional[BodyPose]:
        """Detect body pose in an image."""
        if not self._detector:
            self._init_detector()

        image = mp.Image.create_from_file(str(image_path))
        result: PoseLandmarkerResult = self._detector.detect(image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        landmarks_2d = []
        landmarks_3d = []
        for landmark in result.pose_landmarks[0]:
            landmarks_2d.append([landmark.x, landmark.y, landmark.visibility])
            if landmark.HasField("z"):
                landmarks_3d.append([landmark.x, landmark.y, landmark.z, landmark.visibility])

        # Compute bounding box from landmarks
        xs = [lm[0] for lm in landmarks_2d]
        ys = [lm[1] for lm in landmarks_2d]
        bbox = [min(xs), min(ys), max(xs), max(ys)]

        return BodyPose(
            landmark_set=BodyLandmarkSet.MEDIAPIPE,
            landmarks_2d=landmarks_2d,
            landmarks_3d=landmarks_3d if landmarks_3d else None,
            bbox=bbox,
            det_score=float(np.mean([lm[2] for lm in landmarks_2d])) if landmarks_2d else 0.0,
            model_name="mediapipe_pose",
            raw_response=None,
        )


def detect_pose_in_image(
    image_path: Union[str, Path],
    config: Optional[BodyDetectorConfig] = None,
) -> Optional[BodyPose]:
    """Convenience function to detect pose in a single image."""
    detector = BodyDetector(config)
    return detector.detect_pose(image_path)


def compute_measurements_from_pose(
    pose: BodyPose,
    reference_height_cm: Optional[float] = None,
) -> List[BodyMeasurement]:
    """Compute body measurements from detected pose landmarks."""
    measurements = []
    
    if not pose.landmarks_2d:
        return measurements

    landmarks = pose.landmarks_2d
    
    # Calculate pixel distances for each measurement type
    for meas_type, (idx1, idx2) in MEASUREMENT_LANDMARKS.items():
        if idx1 < len(landmarks) and idx2 < len(landmarks):
            p1 = np.array(landmarks[idx1][:2])
            p2 = np.array(landmarks[idx2][:2])
            
            # Euclidean distance in normalized coordinates (0-1)
            pixel_dist = float(np.linalg.norm(p1 - p2))
            
            # Convert to cm if reference height provided
            value_cm = None
            if reference_height_cm:
                # Use height measurement (nose to ankle) as reference
                height_landmarks = MEASUREMENT_LANDMARKS.get(MeasurementType.HEIGHT, (0, 27))
                if height_landmarks[0] < len(landmarks) and height_landmarks[1] < len(landmarks):
                    h1 = np.array(landmarks[height_landmarks[0]][:2])
                    h2 = np.array(landmarks[height_landmarks[1]][:2])
                    height_pixels = float(np.linalg.norm(h1 - h2))
                    if height_pixels > 0:
                        scale = reference_height_cm / height_pixels
                        value_cm = pixel_dist * scale
            
            confidence = float(min(landmarks[idx1][2], landmarks[idx2][2])) if len(landmarks[idx1]) > 2 and len(landmarks[idx2]) > 2 else 0.5
            
            measurements.append(BodyMeasurement(
                measurement_type=meas_type,
                value=value_cm if value_cm is not None else pixel_dist,
                unit="cm" if value_cm is not None else "normalized",
                confidence=confidence,
                method="landmark_2d",
                reference_scale=reference_height_cm,
                landmarks_used=[idx1, idx2],
            ))
    
    return measurements