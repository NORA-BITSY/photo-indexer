"""Tests for PhotoIndexDB face recognition and tagging."""

import json
import uuid

from photo_indexer.models import FaceEmbedding


def test_create_person(temp_db, sample_embedding):
    """Test creating a known person."""
    person_id = temp_db.create_person("John Doe", sample_embedding)
    assert person_id is not None
    assert len(person_id) == 36  # UUID format

    person = temp_db.get_person(person_id)
    assert person is not None
    assert person["name"] == "John Doe"
    assert person["id"] == person_id
    assert json.loads(person["embedding_json"]) == sample_embedding


def test_create_duplicate_person_fails(temp_db, sample_embedding):
    """Test that creating a person with duplicate name fails."""
    temp_db.create_person("John Doe", sample_embedding)
    # Should raise an exception due to UNIQUE constraint on name
    import sqlite3
    try:
        temp_db.create_person("John Doe", sample_embedding)
        assert False, "Expected IntegrityError"
    except sqlite3.IntegrityError:
        pass


def test_list_persons(temp_db, sample_embedding):
    """Test listing all known persons."""
    temp_db.create_person("Alice", sample_embedding)
    temp_db.create_person("Bob", sample_embedding)

    persons = temp_db.list_persons()
    assert len(persons) == 2
    names = {p["name"] for p in persons}
    assert names == {"Alice", "Bob"}


def test_get_person_by_name(temp_db, sample_embedding):
    """Test getting a person by name."""
    person_id = temp_db.create_person("Charlie", sample_embedding)
    person = temp_db.get_person_by_name("Charlie")
    assert person is not None
    assert person["id"] == person_id
    assert person["name"] == "Charlie"

    # Non-existent person
    assert temp_db.get_person_by_name("NonExistent") is None


def test_match_face_to_person(temp_db, sample_embedding):
    """Test matching a face embedding to a known person."""
    person_id = temp_db.create_person("Test Person", sample_embedding)

    # Match with same embedding (should be ~1.0 similarity)
    match = temp_db.match_face_to_person(sample_embedding, threshold=0.5)
    assert match is not None
    matched_id, similarity = match
    assert matched_id == person_id
    assert similarity > 0.99

    # Match with different embedding (should fail threshold)
    import random
    random.seed(123)
    diff_embedding = [random.gauss(0, 1) for _ in range(512)]
    norm = sum(x * x for x in diff_embedding) ** 0.5
    diff_embedding = [x / norm for x in diff_embedding]

    match = temp_db.match_face_to_person(diff_embedding, threshold=0.6)
    assert match is None


def test_add_and_remove_tags(temp_db, sample_asset):
    """Test adding and removing tags from an asset."""
    # First upsert the asset (sample_asset has tags, but we want to test add_tag)
    # Create a fresh asset with no tags for this test
    from photo_indexer.models import Asset, AssetStatus, ExifMetadata, VLMResult
    import uuid
    fresh_asset = Asset(
        id=uuid.uuid4(),
        path="/test/photo2.jpg",
        filename="photo2.jpg",
        content_sha256="def456",
        file_size=2048,
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
            caption="A mountain landscape",
            detailed_description="Beautiful mountains",
            model="llava",
        ),
        tags=[],  # Empty tags for testing add_tag
    )
    temp_db.upsert(fresh_asset)

    # Add tags - asset_id is a UUID object, need string
    asset_id_str = str(fresh_asset.id)
    assert temp_db.add_tag(asset_id_str, "sunset") is True
    assert temp_db.add_tag(asset_id_str, "ocean") is True
    assert temp_db.add_tag(asset_id_str, "nature") is True

    # Duplicate tag should return False
    assert temp_db.add_tag(asset_id_str, "sunset") is False

    # Get tags
    tags = temp_db.get_tags(asset_id_str)
    assert set(tags) == {"sunset", "ocean", "nature"}

    # Remove tag
    assert temp_db.remove_tag(asset_id_str, "ocean") is True
    tags = temp_db.get_tags(asset_id_str)
    assert set(tags) == {"sunset", "nature"}

    # Remove non-existent tag
    assert temp_db.remove_tag(asset_id_str, "nonexistent") is False


def test_find_assets_by_tag(temp_db, sample_asset):
    """Test finding assets by tag."""
    temp_db.upsert(sample_asset)
    asset_id_str = str(sample_asset.id)
    temp_db.add_tag(asset_id_str, "sunset")
    temp_db.add_tag(asset_id_str, "ocean")

    # Find by tag
    assets = temp_db.find_assets_by_tag("sunset")
    assert len(assets) == 1
    assert assets[0].id == sample_asset.id

    assets = temp_db.find_assets_by_tag("ocean")
    assert len(assets) == 1

    # Non-existent tag
    assets = temp_db.find_assets_by_tag("nonexistent")
    assert len(assets) == 0


def test_import_tags_from_file(temp_db, sample_asset, tmp_path):
    """Test bulk importing tags from a file."""
    # Create a fresh asset with no tags for this test
    from photo_indexer.models import Asset, AssetStatus, ExifMetadata, VLMResult
    import uuid
    fresh_asset = Asset(
        id=uuid.uuid4(),
        path="/test/photo3.jpg",
        filename="photo3.jpg",
        content_sha256="ghi789",
        file_size=3072,
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
            caption="A cityscape",
            detailed_description="Urban skyline",
            model="llava",
        ),
        tags=[],
    )
    temp_db.upsert(fresh_asset)
    asset_id_str = str(fresh_asset.id)

    # Create a test file (TSV format: asset_id\ttag1,tag2)
    test_file = tmp_path / "tags.tsv"
    test_file.write_text(f"{asset_id_str}\tsunset,ocean,nature\n")

    count = temp_db.import_tags_from_file(str(test_file))
    assert count == 3

    tags = temp_db.get_tags(asset_id_str)
    assert set(tags) == {"sunset", "ocean", "nature"}

def test_face_match_and_link(temp_db, sample_asset, sample_embedding):
    """Test matching faces in an asset and linking to persons."""
    # Add a face to the asset with proper UUID
    face = FaceEmbedding(
        id=uuid.uuid4(),
        asset_id=sample_asset.id,
        embedding=sample_embedding,
        bbox=[0.1, 0.1, 0.5, 0.5],
        confidence=0.99,
    )
    sample_asset.face_embeddings = [face]
    temp_db.upsert(sample_asset)

    # Create a person
    person_id = temp_db.create_person("Test Person", sample_embedding)

    # Match faces in asset
    asset_id_str = str(sample_asset.id)
    matches = temp_db.match_faces_in_asset(asset_id_str, threshold=0.5)
    assert len(matches) == 1
    assert matches[0]["person_id"] == person_id
    assert matches[0]["similarity"] > 0.99

    # Link face to person
    temp_db.link_face_to_person(matches[0]["face_id"], person_id, matches[0]["similarity"])

    # Verify link exists
    faces = temp_db.get_faces_by_person(person_id)
    assert len(faces) == 1
    assert faces[0]["id"] == matches[0]["face_id"]
    assert faces[0]["person_name"] == "Test Person"