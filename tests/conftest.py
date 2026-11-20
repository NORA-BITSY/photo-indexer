"""Test configuration and fixtures."""

import tempfile
import uuid
from collections.abc import Generator
from pathlib import Path

import pytest

from photo_indexer.database import PhotoIndexDB
from photo_indexer.models import Asset, AssetStatus, ExifMetadata, VLMResult


@pytest.fixture
def temp_db() -> Generator[PhotoIndexDB, None, None]:
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    db = PhotoIndexDB(db_path)
    yield db
    db_path.unlink(missing_ok=True)


@pytest.fixture
def sample_asset() -> Asset:
    """Create a sample asset for testing."""
    return Asset(
        id=str(uuid.uuid4()),
        path="/test/photo.jpg",
        filename="photo.jpg",
        content_sha256="abc123",
        file_size=1024,
        exif=ExifMetadata(
            mime_type="image/jpeg",
            width=1920,
            height=1080,
            datetime_original="2024-01-01T12:00:00",
            gps_latitude=37.7749,
            gps_longitude=-122.4194,
        ),
        status=AssetStatus.INDEXED,
        vlm=VLMResult(
            caption="A beautiful sunset over the ocean",
            detailed_description="The sun sets over the Pacific Ocean with vibrant orange and pink colors",
            model="llava",
        ),
        tags=["sunset", "ocean", "nature"],
    )


@pytest.fixture
def sample_embedding() -> list[float]:
    """Create a sample face embedding."""
    # 512-dimensional normalized embedding (typical for face recognition)
    import random
    random.seed(42)
    vec = [random.gauss(0, 1) for _ in range(512)]
    # Normalize
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]