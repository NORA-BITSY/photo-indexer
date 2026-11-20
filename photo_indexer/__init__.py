"""Photo Indexer - Self-hosted, local-first photo indexing CLI."""

__version__ = "0.1.0"

# Export main models
from .models import (
    Asset,
    AssetStatus,
    BodyLandmarkSet,
    BodyMeasurement,
    BodyPose,
    EntityType,
    ExifMetadata,
    FaceEmbedding,
    FacialBiometrics,
    MeasurementType,
    NEREntity,
    PerformerProfile,
    SidecarStatus,
    Taxonomy,
    TaxonomyEntry,
    VLMProvider,
    VLMResult,
    deterministic_asset_id,
)

# Export database
from .database import PhotoIndexDB

# Export CLI
from .cli import main

__all__ = [
    "Asset",
    "AssetStatus",
    "BodyLandmarkSet",
    "BodyMeasurement",
    "BodyPose",
    "EntityType",
    "ExifMetadata",
    "FaceEmbedding",
    "FacialBiometrics",
    "MeasurementType",
    "NEREntity",
    "PerformerProfile",
    "SidecarStatus",
    "Taxonomy",
    "TaxonomyEntry",
    "VLMProvider",
    "VLMResult",
    "deterministic_asset_id",
    "PhotoIndexDB",
    "main",
]