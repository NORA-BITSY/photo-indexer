"""Performer profile aggregation from multi-image data."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np

from .models import (
    BodyMeasurement, BodyPose, FaceEmbedding, FacialBiometrics,
    MeasurementType, PerformerProfile
)
from .database import PhotoIndexDB


@dataclass
class AggregationConfig:
    """Configuration for performer profile aggregation."""
    min_images_for_profile: int = 3
    embedding_aggregation: str = "mean"  # mean, median, weighted_mean
    outlier_std_threshold: float = 2.0
    require_face_match: bool = True
    face_match_threshold: float = 0.6


class PerformerAggregator:
    """Aggregate multi-image data into comprehensive performer profiles."""

    def __init__(self, db: PhotoIndexDB, config: AggregationConfig = None):
        self.db = db
        self.config = config or AggregationConfig()

    def build_performer_profile(
        self,
        performer_id: str,
        asset_ids: List[str],
        aliases: List[str] = None,
        reference_height: Optional[float] = None,
        store: bool = True,
    ) -> PerformerProfile:
        """Build a performer profile from a list of asset IDs."""
        if len(asset_ids) < self.config.min_images_for_profile:
            raise ValueError(f"Need at least {self.config.min_images_for_profile} images")

        embeddings = []
        all_measurements = defaultdict(list)
        measurement_confidences = defaultdict(list)
        first_seen = None
        last_seen = None
        source_tags = set()

        for asset_id in asset_ids:
            asset = self.db.get_asset(asset_id)
            if not asset:
                continue

            # Collect face embeddings
            if asset.face_embeddings:
                for face in asset.face_embeddings:
                    if face.embedding:
                        embeddings.append(np.array(face.embedding))

            # Collect measurements
            measurements = self.db.get_body_measurements(asset_id=asset_id)
            for m in measurements:
                mtype = m["measurement_type"]
                all_measurements[mtype].append(m["value"])
                measurement_confidences[mtype].append(m.get("confidence", 0.0))

            # Track dates
            if asset.enriched_at:
                if first_seen is None or asset.enriched_at < first_seen:
                    first_seen = asset.enriched_at
                if last_seen is None or asset.enriched_at > last_seen:
                    last_seen = asset.enriched_at

            # Collect tags
            for tag in asset.ner_entities:
                if hasattr(tag, 'source') and tag.source == 'file':
                    source_tags.add(tag.text)

        # Aggregate embeddings
        canonical_embedding = self._aggregate_embeddings(embeddings) if embeddings else None
        embedding_quality = 1.0 if canonical_embedding is not None else 0.0

        # Aggregate measurements
        aggregated_measurements = {}
        aggregated_confidences = {}
        for mtype, values in all_measurements.items():
            confs = measurement_confidences[mtype]
            clean_values, clean_confs = self._remove_outliers(values, confs)
            if clean_values:
                aggregated_measurements[mtype] = float(np.mean(clean_values))
                aggregated_confidences[mtype] = float(np.mean(clean_confs))

        # Classify body type
        body_type = self._classify_body_type(aggregated_measurements)

        # Estimate height from measurements
        height_cm = None
        height_conf = 0.0
        if MeasurementType.HEIGHT.value in aggregated_measurements:
            height_cm = aggregated_measurements[MeasurementType.HEIGHT.value]
            height_conf = aggregated_confidences[MeasurementType.HEIGHT.value]
        elif reference_height:
            height_cm = reference_height
            height_conf = 0.8

        # Weight estimation (rough)
        weight_kg = None
        if height_cm and body_type:
            # Very rough BMI-based estimation
            bmi_map = {"ectomorph": 19, "mesomorph": 23, "endomorph": 27}
            bmi = bmi_map.get(body_type, 22)
            weight_kg = bmi * (height_cm / 100) ** 2

        profile = PerformerProfile(
            id=UUID(performer_id) if performer_id else uuid4(),
            name="",  # Will be set by caller
            aliases=aliases or [],
            face_embedding=canonical_embedding.tolist() if canonical_embedding is not None else None,
            face_embedding_quality=embedding_quality,
            measurements=aggregated_measurements,
            measurement_confidence=aggregated_confidences,
            body_type=body_type,
            height_cm=height_cm,
            height_confidence=height_conf,
            weight_kg_estimate=weight_kg,
            asset_count=len(asset_ids),
            first_seen=first_seen,
            last_seen=last_seen,
            source_tags=list(source_tags),
            notes=None,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        if store:
            # Update in database
            self.db.update_performer(str(profile.id), {
                "face_embedding_json": json.dumps(canonical_embedding.tolist()) if canonical_embedding is not None else None,
                "face_embedding_quality": embedding_quality,
                "measurements_json": json.dumps(aggregated_measurements),
                "measurement_confidence_json": json.dumps(aggregated_confidences),
                "body_type": body_type,
                "height_cm": height_cm,
                "height_confidence": height_conf,
                "weight_kg_estimate": weight_kg,
                "asset_count": len(asset_ids),
                "first_seen": first_seen.isoformat() if first_seen else None,
                "last_seen": last_seen.isoformat() if last_seen else None,
                "source_tags_json": json.dumps(list(source_tags)),
                "updated_at": datetime.utcnow().isoformat(),
            })

        return profile

    def _aggregate_embeddings(self, embeddings: List[np.ndarray]) -> Optional[np.ndarray]:
        """Aggregate multiple face embeddings into a canonical one."""
        if not embeddings:
            return None
        
        if self.config.embedding_aggregation == "median":
            agg = np.median(np.array(embeddings), axis=0)
        elif self.config.embedding_aggregation == "weighted_mean":
            # Weight by norms
            norms = np.linalg.norm(embeddings, axis=1)
            weights = norms / np.sum(norms) if np.sum(norms) > 0 else None
            agg = np.average(np.array(embeddings), axis=0, weights=weights)
        else:  # mean
            agg = np.mean(np.array(embeddings), axis=0)
        
        # Normalize
        norm = np.linalg.norm(agg)
        if norm > 0:
            agg = agg / norm
        return agg

    def _remove_outliers(
        self, values: List[float], confidences: List[float]
    ) -> Tuple[List[float], List[float]]:
        """Remove outliers using z-score method."""
        if len(values) < 3:
            return values, confidences
        
        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr)
        
        if std == 0:
            return values, confidences
        
        z_scores = np.abs((arr - mean) / std)
        mask = z_scores <= self.config.outlier_std_threshold
        
        return (
            arr[mask].tolist(),
            np.array(confidences)[mask].tolist()
        )

    def _classify_body_type(self, measurements: Dict) -> Optional[str]:
        """Classify body type from aggregated measurements."""
        shoulder = measurements.get(MeasurementType.SHOULDER_WIDTH.value, 0)
        waist = measurements.get(MeasurementType.WAIST_WIDTH.value, 0)
        hip = measurements.get(MeasurementType.HIP_WIDTH.value, 0)
        
        if shoulder == 0 or waist == 0 or hip == 0:
            return None
        
        shoulder_hip_ratio = shoulder / hip
        waist_hip_ratio = waist / hip
        
        # Simple heuristic
        if shoulder_hip_ratio > 1.05 and waist_hip_ratio < 0.85:
            return "mesomorph"  # Athletic
        elif shoulder_hip_ratio < 0.95 and waist_hip_ratio > 0.9:
            return "endomorph"  # Rounder
        else:
            return "ectomorph"  # Lean, narrow

    def build_facial_biometrics(
        self,
        performer_id: str,
        asset_ids: List[str],
        store: bool = True,
    ) -> FacialBiometrics:
        """Build comprehensive facial biometrics from multiple face images."""
        all_embeddings = []
        all_landmarks_68 = []
        all_landmarks_468 = []
        all_blendshapes = []
        all_expressions = []
        pose_coverage = defaultdict(int)
        expression_coverage = defaultdict(int)
        
        image_count = 0
        
        for asset_id in asset_ids:
            asset = self.db.get_asset(asset_id)
            if not asset:
                continue
            
            image_count += 1
            
            for face in asset.face_embeddings:
                if face.embedding:
                    all_embeddings.append(np.array(face.embedding))
                
                if face.landmarks:
                    # Extract 68 landmarks from available landmarks
                    lm = face.landmarks
                    if len(lm) >= 68:
                        all_landmarks_68.append(np.array(lm[:68]))
                    elif len(lm) >= 468:
                        # Map 468 to 68
                        all_landmarks_68.append(self._get_68_from_468(lm))
                
                if face.blendshapes:
                    all_blendshapes.append(np.array(face.blendshapes))
                
                if face.expression_vector:
                    all_expressions.append(np.array(face.expression_vector))
                
                # Track pose coverage
                if face.head_pose:
                    pose_coverage[str(face.head_pose)] += 1
                
                if face.expression_vector:
                    expression_coverage[str(face.expression_vector[:3])] += 1

        # Aggregate identity embedding
        identity_embedding = self._aggregate_embeddings(all_embeddings) if all_embeddings else None
        identity_quality = float(np.mean([np.linalg.norm(e) for e in all_embeddings])) if all_embeddings else 0.0
        
        # Aggregate landmarks
        canonical_68 = None
        if all_landmarks_68:
            canonical_68 = np.mean(np.array(all_landmarks_68), axis=0).tolist()
        
        # Compute facial measurements from canonical landmarks
        facial_measurements = {}
        if canonical_68:
            facial_measurements = self._compute_facial_measurements_from_68(canonical_68)
        
        biometrics = FacialBiometrics(
            id=uuid4(),
            performer_id=UUID(performer_id) if isinstance(performer_id, str) else performer_id,
            identity_embedding=identity_embedding.tolist() if identity_embedding is not None else None,
            identity_embedding_std=None,  # Could compute std across embeddings
            identity_quality_mean=identity_quality,
            shape_coefficients=None,  # Would need PCA
            expression_coefficients_mean=None,
            expression_coefficients_std=None,
            canonical_landmarks_68=canonical_68,
            canonical_landmarks_68_cov=None,
            canonical_landmarks_468=None,
            facial_measurements=facial_measurements,
            skin_tone_lab=None,
            skin_texture_descriptor=None,
            source_image_count=image_count,
            pose_coverage=dict(pose_coverage),
            expression_coverage=dict(expression_coverage),
            updated_at=datetime.utcnow(),
            model_versions={},
        )

        if store and identity_embedding is not None:
            self.db.upsert_facial_biometrics({
                "performer_id": performer_id,
                "identity_embedding": identity_embedding.tolist(),
                "identity_quality_mean": identity_quality,
                "canonical_landmarks_68": canonical_68,
                "facial_measurements": facial_measurements,
                "source_image_count": image_count,
                "pose_coverage": dict(pose_coverage),
                "expression_coverage": dict(expression_coverage),
                "updated_at": datetime.utcnow().isoformat(),
            })

        return biometrics

    def _get_68_from_468(self, landmarks_468: List) -> List:
        """Map 468 MediaPipe landmarks to 68 standard landmarks."""
        # Standard 68-point mapping from 468 MediaPipe landmarks
        mapping = [
            162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 361, 323,  # jaw
            70, 63, 105, 66, 107,  # right eyebrow
            336, 296, 334, 293, 300,  # left eyebrow
            168, 6, 197, 195, 5,  # nose bridge
            48, 115, 220, 45, 4, 275, 440, 344, 278,  # nose
            33, 160, 158, 133, 153, 144,  # right eye
            362, 385, 387, 263, 373, 380,  # left eye
            61, 39, 37, 0, 267, 269, 270, 409, 291,  # outer mouth
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,  # inner mouth
        ]
        return [landmarks_468[i] for i in mapping if i < len(landmarks_468)]

    def _compute_facial_measurements_from_68(self, landmarks_68: List) -> Dict:
        """Compute facial measurements from 68 landmarks."""
        lm = np.array(landmarks_68)
        
        # Eye width (left: 36-39, right: 42-45)
        left_eye_width = np.linalg.norm(lm[39] - lm[36])
        right_eye_width = np.linalg.norm(lm[45] - lm[42])
        
        # Eye height
        left_eye_height = np.linalg.norm(lm[37] - lm[41])
        right_eye_height = np.linalg.norm(lm[43] - lm[47])
        
        # Mouth width (48-54)
        mouth_width = np.linalg.norm(lm[54] - lm[48])
        
        # Nose width (31-35)
        nose_width = np.linalg.norm(lm[35] - lm[31])
        
        # Face width (0-16)
        face_width = np.linalg.norm(lm[16] - lm[0])
        
        # Face height (8-27)
        face_height = np.linalg.norm(lm[8] - lm[27])
        
        return {
            "left_eye_width": float(left_eye_width),
            "right_eye_width": float(right_eye_width),
            "left_eye_height": float(left_eye_height),
            "right_eye_height": float(right_eye_height),
            "mouth_width": float(mouth_width),
            "nose_width": float(nose_width),
            "face_width": float(face_width),
            "face_height": float(face_height),
            "eye_aspect_ratio_left": float(left_eye_height / left_eye_width) if left_eye_width > 0 else 0,
            "eye_aspect_ratio_right": float(right_eye_height / right_eye_width) if right_eye_width > 0 else 0,
            "mouth_aspect_ratio": float((lm[62][1] - lm[66][1]) / mouth_width) if mouth_width > 0 else 0,
        }


import json
from uuid import uuid4