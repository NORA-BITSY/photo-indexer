"""Pydantic data models for photo indexing, VLM results, taxonomies, and NER entities."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4, uuid5, NAMESPACE_URL

from pydantic import BaseModel, Field


class AssetStatus(str, Enum):
    PENDING = "pending"
    INDEXED = "indexed"
    ANALYZED = "analyzed"
    ENRICHED = "enriched"
    ERROR = "error"
    SKIPPED = "skipped"


class EntityType(str, Enum):
    PERSON = "person"
    ANIMAL = "animal"
    PET = "pet"
    PLACE = "place"
    LOCATION = "location"
    OBJECT = "object"
    FACE = "face"


class VLMProvider(str, Enum):
    OLLAMA = "ollama"
    API = "api"  # OpenAI-compatible / generic HTTP VLM API


class SidecarStatus(str, Enum):
    WRITTEN = "written"
    SKIPPED_UNSAFE = "skipped_unsafe"
    ERROR = "error"


class ExifMetadata(BaseModel):
    """Flattened ExifTool output fields."""

    file_name: Optional[str] = None
    directory: Optional[str] = None
    file_size: Optional[int] = None
    file_modify_date: Optional[str] = None
    file_access_date: Optional[str] = None
    file_inode_change_date: Optional[str] = None
    file_permissions: Optional[str] = None
    file_type: Optional[str] = None
    mime_type: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    encoding_process: Optional[str] = None
    bits_per_sample: Optional[int] = None
    color_components: Optional[int] = None
    ycbcr_sub_sampling: Optional[str] = None
    image_type: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    orientation: Optional[int] = None
    x_resolution: Optional[float] = None
    y_resolution: Optional[float] = None
    resolution_unit: Optional[str] = None
    modify_date: Optional[str] = None
    exif_offset: Optional[int] = None
    gps_position: Optional[str] = None
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    gps_altitude: Optional[float] = None
    gps_timestamp: Optional[str] = None
    date_time_original: Optional[str] = None
    create_date: Optional[str] = None
    software: Optional[str] = None
    artist: Optional[str] = None
    copyright: Optional[str] = None
    image_description: Optional[str] = None
    keywords: Optional[List[str]] = None
    rating: Optional[int] = None
    lens_model: Optional[str] = None
    focal_length: Optional[float] = None
    aperture: Optional[float] = None
    shutter_speed: Optional[float] = None
    iso: Optional[int] = None
    flash: Optional[str] = None
    white_balance: Optional[str] = None
    color_space: Optional[str] = None
    compression: Optional[str] = None
    thumbnail_offset: Optional[int] = None
    thumbnail_length: Optional[int] = None
    raw: Dict[str, Any] = Field(default_factory=dict)


def deterministic_asset_id(path: str, content_sha256: Optional[str] = None) -> UUID:
    """Derive a stable UUIDv5 for an asset from its path (+ content hash if available)."""
    name = content_sha256 or path
    return uuid5(NAMESPACE_URL, name)


class Asset(BaseModel):
    """A photo asset in the index."""

    id: Optional[UUID] = None
    path: str
    filename: str
    content_sha256: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    status: AssetStatus = AssetStatus.PENDING
    indexed_at: Optional[datetime] = None
    analyzed_at: Optional[datetime] = None
    enriched_at: Optional[datetime] = None
    exif: Optional[ExifMetadata] = None
    vlm: Optional["VLMResult"] = None
    ner_entities: List["NEREntity"] = Field(default_factory=list)
    face_embeddings: List["FaceEmbedding"] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    sidecar_status: Optional[SidecarStatus] = None
    sidecar_path: Optional[str] = None
    error_message: Optional[str] = None
    class Config:
        use_enum_values = True


class VLMResult(BaseModel):
    """Result from VLM image analysis."""

    provider: VLMProvider = VLMProvider.OLLAMA
    model: str
    caption: Optional[str] = None
    detailed_description: Optional[str] = None
    confidence: Optional[float] = None
    processing_time_ms: Optional[int] = None
    raw_response: Optional[str] = None
    analyzed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    class Config:
        use_enum_values = True


class NEREntity(BaseModel):
    """Named entity extracted from VLM text output."""

    id: UUID = Field(default_factory=uuid4)
    text: str
    entity_type: EntityType
    label: Optional[str] = None  # canonical taxonomy label
    confidence: Optional[float] = None
    bbox: Optional[List[float]] = None  # [x1, y1, x2, y2] normalized 0-1
    source: str = "vlm"  # vlm, exif, xmp
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    class Config:
        use_enum_values = True


class FaceEmbedding(BaseModel):
    """Face embedding extracted from an image using InsightFace."""

    id: UUID = Field(default_factory=uuid4)
    asset_id: Optional[UUID] = None  # Reference to parent asset
    bbox: Optional[List[float]] = None  # [x1, y1, x2, y2] in image coordinates
    landmarks: Optional[List[List[float]]] = None  # [[x, y], ...] landmark points
    embedding: Optional[List[float]] = None  # 512-dim face embedding vector
    det_score: Optional[float] = None  # Detection confidence score
    age: Optional[int] = None  # Estimated age
    gender: Optional[int] = None  # 0=female, 1=male
    model_name: str = "buffalo_l"  # InsightFace model used
    detected_at: Optional[datetime] = None

    class Config:
        use_enum_values = True



class BodyLandmarkSet(str, Enum):
    """Body landmark model variants."""
    MEDIAPIPE = "mediapipe"  # 33 keypoints
    OPENPOSE = "openpose"  # 25/135 keypoints


class BodyPose(BaseModel):
    """Body pose detection result with landmarks and optional 3D world coordinates."""
    landmark_set: BodyLandmarkSet
    landmarks_2d: List[List[float]]  # [[x, y, visibility], ...] normalized 0-1
    landmarks_3d: Optional[List[List[float]]] = None  # [[x, y, z, visibility], ...] world coordinates
    bbox: Optional[List[float]] = None  # [x1, y1, x2, y2] normalized 0-1
    det_score: float = 0.0
    model_name: str = "mediapipe_pose"
    raw_response: Optional[Dict] = None

    class Config:
        use_enum_values = True


class MeasurementType(str, Enum):
    """Anthropometric measurement types."""
    HEIGHT = "height"
    HEAD_HEIGHT = "head_height"
    NECK_LENGTH = "neck_length"
    SHOULDER_WIDTH = "shoulder_width"
    CHEST_WIDTH = "chest_width"
    WAIST_WIDTH = "waist_width"
    HIP_WIDTH = "hip_width"
    UPPER_ARM_LENGTH = "upper_arm_length"
    FOREARM_LENGTH = "forearm_length"
    HAND_LENGTH = "hand_length"
    THIGH_LENGTH = "thigh_length"
    SHIN_LENGTH = "shin_length"
    FOOT_LENGTH = "foot_length"
    ARM_SPAN = "arm_span"
    TORSO_LENGTH = "torso_length"
    LEG_LENGTH = "leg_length"
    INSEAM = "inseam"
    BODY_TYPE = "body_type"  # ectomorph/mesomorph/endomorph classification


class BodyMeasurement(BaseModel):
    """Single body measurement with value, unit, and confidence."""
    id: UUID = Field(default_factory=uuid4)
    asset_id: Optional[UUID] = None
    performer_id: Optional[UUID] = None
    measurement_type: MeasurementType
    value: float
    unit: str = "cm"
    confidence: float = 0.0
    method: str = "landmark_2d"
    reference_scale: Optional[float] = None
    landmarks_used: Optional[List[int]] = None
    computed_at: Optional[datetime] = None
    notes: Optional[str] = None

    class Config:
        use_enum_values = True


class FacialBiometrics(BaseModel):
    """Comprehensive facial biometric profile for a performer (aggregated multi-image)."""
    id: UUID = Field(default_factory=uuid4)
    performer_id: UUID
    identity_embedding: Optional[List[float]] = None  # Canonical 512-dim embedding
    identity_embedding_std: Optional[List[float]] = None
    identity_quality_mean: float = 0.0
    shape_coefficients: Optional[List[float]] = None  # PCA shape coefficients
    expression_coefficients_mean: Optional[List[float]] = None
    expression_coefficients_std: Optional[List[float]] = None
    canonical_landmarks_68: Optional[List[List[float]]] = None
    canonical_landmarks_68_cov: Optional[List[List[float]]] = None
    canonical_landmarks_468: Optional[List[List[float]]] = None
    facial_measurements: Dict[str, float] = Field(default_factory=dict)
    skin_tone_lab: Optional[List[float]] = None
    skin_texture_descriptor: Optional[List[float]] = None
    source_image_count: int = 0
    pose_coverage: Dict[str, int] = Field(default_factory=dict)
    expression_coverage: Dict[str, int] = Field(default_factory=dict)
    updated_at: Optional[datetime] = None
    model_versions: Dict[str, str] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class PerformerProfile(BaseModel):
    """Aggregated performer profile from multiple assets."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    aliases: List[str] = Field(default_factory=list)
    face_embedding: Optional[List[float]] = None
    face_embedding_quality: float = 0.0
    measurements: Dict[str, float] = Field(default_factory=dict)
    measurement_confidence: Dict[str, float] = Field(default_factory=dict)
    body_type: Optional[str] = None
    height_cm: Optional[float] = None
    height_confidence: float = 0.0
    weight_kg_estimate: Optional[float] = None
    facial_biometrics_id: Optional[UUID] = None
    asset_count: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    source_tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True

class TaxonomyEntry(BaseModel):
    """A single entry within a taxonomy."""

    label: str
    synonyms: List[str] = Field(default_factory=list)
    parent: Optional[str] = None
    confidence_threshold: float = 0.7


class Taxonomy(BaseModel):
    """A domain taxonomy for classifying image content."""

    name: str
    version: str = "1.0.0"
    description: str
    entries: List[TaxonomyEntry] = Field(default_factory=list)

    def get_entry(self, label: str) -> Optional[TaxonomyEntry]:
        for entry in self.entries:
            if entry.label.lower() == label.lower():
                return entry
            if label.lower() in [s.lower() for s in entry.synonyms]:
                return entry
        return None

    def match(self, text: str, threshold: float = 0.7) -> List[TaxonomyEntry]:
        """Fuzzy-match text against taxonomy entries."""
        text_lower = text.lower()
        matched: List[TaxonomyEntry] = []
        for entry in self.entries:
            if entry.label.lower() in text_lower or text_lower in entry.label.lower():
                matched.append(entry)
                continue
            for syn in entry.synonyms:
                if syn.lower() in text_lower:
                    matched.append(entry)
                    break
        return matched


class EnrichmentRecord(BaseModel):
    """Database enrichment record keyed by asset."""

    asset_id: UUID
    content_sha256: Optional[str] = None
    exif_raw: Dict[str, Any] = Field(default_factory=dict)
    vlm_caption: Optional[str] = None
    vlm_description: Optional[str] = None
    vlm_model: Optional[str] = None
    ner_entities: List[NEREntity] = Field(default_factory=list)
    sidecar_status: Optional[SidecarStatus] = None
    enriched_at: Optional[datetime] = None



# Fix forward references
Asset.model_rebuild()
VLMResult.model_rebuild()
NEREntity.model_rebuild()
Taxonomy.model_rebuild()
EnrichmentRecord.model_rebuild()
FaceEmbedding.model_rebuild()
BodyPose.model_rebuild()
BodyMeasurement.model_rebuild()
FacialBiometrics.model_rebuild()
PerformerProfile.model_rebuild()
