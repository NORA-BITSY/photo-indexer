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
    keywords: Optional[str] = None
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
