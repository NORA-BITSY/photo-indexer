"""Face detection and embedding extraction using InsightFace."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
from uuid import uuid4

try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

import cv2

from .models import FaceEmbedding
from .metadata import find_images


@dataclass
class FaceDetectionConfig:
    """Configuration for face detection."""
    det_size: Tuple[int, int] = (640, 640)
    det_thresh: float = 0.5
    ctx_id: int = 0  # GPU context ID, -1 for CPU
    providers: List[str] = None

    def __post_init__(self):
        if self.providers is None:
            self.providers = ['CPUExecutionProvider']


class FaceDetector:
    """Wrapper around InsightFace FaceAnalysis for face detection and embedding extraction."""

    def __init__(self, config: FaceDetectionConfig = None):
        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError("insightface not installed. Install with: pip install insightface onnxruntime")
        self.config = config or FaceDetectionConfig()
        self._app = None

    def _ensure_loaded(self):
        if self._app is None:
            self._app = insightface.app.FaceAnalysis(providers=self.config.providers)
            self._app.prepare(ctx_id=self.config.ctx_id, det_size=self.config.det_size)

    def detect_faces(self, image_path: Union[str, Path]) -> List[FaceEmbedding]:
        """Detect faces in an image and extract embeddings.

        Returns:
            List of FaceEmbedding objects with bounding boxes, landmarks, and 512-dim embeddings.
        """
        self._ensure_loaded()

        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            return []

        # Detect faces
        faces = self._app.get(img)

        embeddings = []
        for face in faces:
            # Bounding box: [x1, y1, x2, y2]
            bbox = face.bbox.astype(float).tolist() if face.bbox is not None else None

            # Landmarks: 5 or 68 points
            landmarks = face.landmark_2d_106.astype(float).tolist() if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None else None
            if landmarks is None and hasattr(face, 'landmark_3d_68') and face.landmark_3d_68 is not None:
                landmarks = face.landmark_3d_68[:, :2].astype(float).tolist()  # x,y only

            # Embedding: 512-dim normalized vector
            embedding = face.embedding.astype(float).tolist() if face.embedding is not None else None

            # Detection score
            det_score = float(face.det_score) if face.det_score is not None else None

            # Age/gender if available
            age = int(face.age) if hasattr(face, 'age') and face.age is not None else None
            gender = int(face.gender) if hasattr(face, 'gender') and face.gender is not None else None  # 0=female, 1=male

            from .models import FaceEmbedding
            embeddings.append(FaceEmbedding(
                id=uuid4(),
                asset_id=None,  # Set by caller
                bbox=bbox,
                landmarks=landmarks,
                embedding=embedding,
                det_score=det_score,
                age=age,
                gender=gender,
                model_name="buffalo_l",
            ))

        return embeddings


def detect_faces_in_image(
    image_path: Union[str, Path],
    config: FaceDetectionConfig = None,
) -> List[FaceEmbedding]:
    """Convenience function to detect faces in a single image."""
    detector = FaceDetector(config)
    return detector.detect_faces(image_path)


def detect_faces_batch(
    image_paths: List[Union[str, Path]],
    config: FaceDetectionConfig = None,
) -> List[Tuple[Path, List[FaceEmbedding]]]:
    """Detect faces in multiple images."""
    detector = FaceDetector(config)
    results = []
    for path in image_paths:
        faces = detector.detect_faces(path)
        results.append((Path(path), faces))
    return results


def index_faces_in_staging(
    staging_path: Union[str, Path],
    db_path: Optional[str] = None,
    config: FaceDetectionConfig = None,
    batch_size: int = 10,
) -> int:
    """Index faces for all images in a staging directory.

    Args:
        staging_path: Path to staging directory
        db_path: SQLite database path (uses default if None)
        config: Face detection config
        batch_size: Number of images to process per batch (for progress reporting)

    Returns:
        Total number of faces detected and indexed.
    """
    from .database import PhotoIndexDB
    from .models import Asset, AssetStatus, deterministic_asset_id

    if db_path is None:
        import os
        db_path = os.environ.get("PHOTO_INDEX_DB", "./photo_index.db")

    db = PhotoIndexDB(db_path)
    staging = Path(staging_path)
    images = find_images(staging, recursive=True)

    total_faces = 0
    detector = FaceDetector(config)

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        for img_path in batch:
            # Find or create asset
            rel = str(img_path)
            # Try to find existing asset by path
            existing = None
            for asset in db.list_all(limit=10000):
                if asset.path == rel:
                    existing = asset
                    break

            if existing is None:
                # Create minimal asset
                import hashlib
                sha256 = hashlib.sha256()
                with open(img_path, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        sha256.update(chunk)
                existing = Asset(
                    id=deterministic_asset_id(rel, sha256.hexdigest()),
                    path=rel,
                    filename=img_path.name,
                    content_sha256=sha256.hexdigest(),
                    file_size=img_path.stat().st_size,
                    status=AssetStatus.INDEXED,
                )

            # Detect faces
            faces = detector.detect_faces(img_path)
            if not faces:
                continue

            # Update asset with face embeddings
            face_embeddings = []
            for face in faces:
                face.asset_id = str(existing.id)
                face_embeddings.append(face)

            # Store in asset (extend model if needed)
            existing.face_embeddings = face_embeddings
            db.upsert(existing)
            total_faces += len(faces)

        print(f"Processed {min(i + batch_size, len(images))}/{len(images)} images, {total_faces} faces total")

    return total_faces