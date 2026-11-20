"""Metadata extraction: ExifTool integration, XMP sidecar writing, and database operations."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .models import Asset, ExifMetadata, NEREntity, SidecarStatus, deterministic_asset_id


# Image extensions we consider indexable
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".heic", ".heif",
    ".raw", ".cr2", ".nef", ".arw", ".dng", ".webp", ".bmp", ".gif",
    # Video formats (Live Photo videos, screen recordings, etc.)
    ".mov", ".mp4", ".m4v", ".avi", ".mkv",
}


def is_image_file(path: Path) -> bool:
    """Return True if the file has an image extension."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def find_images(root: Union[str, Path], recursive: bool = True) -> List[Path]:
    """Recursively find all image files under root."""
    root_path = Path(root)
    images: List[Path] = []
    if root_path.is_file():
        if is_image_file(root_path):
            images.append(root_path)
        return images
    pattern = "**/*" if recursive else "*"
    for p in root_path.glob(pattern):
        if p.is_file() and is_image_file(p):
            images.append(p)
    return sorted(images)


def run_exiftool(image_path: Union[str, Path]) -> Dict[str, Any]:
    """Run ExifTool on a single image and return the parsed JSON output.

    Uses `exiftool -j -a -G1 -s` for full, structured metadata.
    Returns an empty dict if exiftool is not available or fails.
    """
    path = str(image_path)
    cmd = ["exiftool", "-j", "-a", "-G1", "-s", path]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {"_extraction_error": f"exiftool exit={result.returncode}: {result.stderr[:200]}"}
        data = json.loads(result.stdout)
        return data[0] if isinstance(data, list) and data else {}
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as e:
        return {"_extraction_error": str(e)}


def _strip_group_prefix(tag: str) -> str:
    """Remove ExifTool group prefix (e.g., 'ExifIFD:' -> '') if present."""
    if ":" in tag:
        parts = tag.split(":", 1)
        # Known groups: ExifIFD, GPS, XMP, IPTC, MakerNotes, etc.
        # Keep the tag name after the first colon
        return parts[1]
    return tag


def _to_snake(name: str) -> str:
    """Convert a tag name like 'DateTimeOriginal' to 'date_time_original'."""
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    s1 = re.sub(r"[^a-zA-Z0-9]", "_", s1)
    return s1.lower()


# Map ExifTool tag names (with or without group prefix) to model field names
EXIF_TO_FIELD: Dict[str, str] = {
    # File
    "FileName": "file_name",
    "Directory": "directory",
    "FileSize": "file_size",
    "FileModifyDate": "file_modify_date",
    "FileAccessDate": "file_access_date",
    "FileInodeChangeDate": "file_inode_change_date",
    "FilePermissions": "file_permissions",
    "FileType": "file_type",
    "MIMEType": "mime_type",
    # Image
    "ImageWidth": "image_width",
    "ImageHeight": "image_height",
    "EncodingProcess": "encoding_process",
    "BitsPerSample": "bits_per_sample",
    "ColorComponents": "color_components",
    "YCbCrSubSampling": "ycbcr_sub_sampling",
    "ImageType": "image_type",
    # Camera
    "Make": "make",
    "Model": "model",
    "Orientation": "orientation",
    "XResolution": "x_resolution",
    "YResolution": "y_resolution",
    "ResolutionUnit": "resolution_unit",
    "ModifyDate": "modify_date",
    "ExifOffset": "exif_offset",
    # GPS
    "GPSPosition": "gps_position",
    "GPSLatitude": "gps_latitude",
    "GPSLongitude": "gps_longitude",
    "GPSAltitude": "gps_altitude",
    "GPSTimeStamp": "gps_timestamp",
    # Timestamps
    "DateTimeOriginal": "date_time_original",
    "CreateDate": "create_date",
    # Descriptive
    "Software": "software",
    "Artist": "artist",
    "Copyright": "copyright",
    "ImageDescription": "image_description",
    "Keywords": "keywords",
    "Rating": "rating",
    # Lens
    "LensModel": "lens_model",
    "FocalLength": "focal_length",
    "Aperture": "aperture",
    "ShutterSpeed": "shutter_speed",
    "ISO": "iso",
    # Flash/WhiteBalance/ColorSpace
    "Flash": "flash",
    "WhiteBalance": "white_balance",
    "ColorSpace": "color_space",
    "Compression": "compression",
    # Thumbnail
    "ThumbnailOffset": "thumbnail_offset",
    "ThumbnailLength": "thumbnail_length",
}


def _convert_value(field_name: str, value: Any) -> Any:
    """Convert ExifTool string values to appropriate Python types."""
    if value is None:
        return None
    # Float fields
    if field_name in {"gps_latitude", "gps_longitude", "gps_altitude",
                       "x_resolution", "y_resolution", "focal_length",
                       "aperture"}:
        try:
            if isinstance(value, str):
                value = value.split()[0]
            return float(value)
        except (ValueError, AttributeError):
            return None
    # Shutter speed: handle fractions like "1/120" or "0.008"
    if field_name == "shutter_speed":
        try:
            if isinstance(value, str):
                if "/" in value:
                    num, den = value.split("/")
                    return float(num) / float(den)
                return float(value.split()[0])
            return float(value)
        except (ValueError, AttributeError, ZeroDivisionError):
            return None
    # Int fields
    if field_name in {"image_width", "image_height", "orientation",
                       "bits_per_sample", "color_components", "rating",
                       "iso", "thumbnail_offset", "thumbnail_length",
                       "file_size", "exif_offset"}:
        try:
            if isinstance(value, str):
                value = value.split()[0]
            return int(float(value))
        except (ValueError, AttributeError):
            return None
    # Keywords: comma-separated string -> list
    if field_name == "keywords":
        if isinstance(value, str):
            return [k.strip() for k in value.split(",") if k.strip()]
        return value
    # GPS coordinates sometimes come as "40 deg 26' 46.00\" N"
    if field_name in {"gps_latitude", "gps_longitude"} and isinstance(value, str):
        # Already handled above for float, but could add DMS parsing here if needed
        pass
    return value


def parse_exif(raw: Dict[str, Any]) -> ExifMetadata:
    """Parse raw ExifTool output into an ExifMetadata model.

    Handles ExifTool group prefixes (ExifIFD:DateTimeOriginal -> DateTimeOriginal).
    Maps tags to model fields via EXIF_TO_FIELD.
    Converts values to appropriate types.
    Preserves ALL tags in `raw` including extraction errors.
    """
    if not raw:
        return ExifMetadata(raw={"_empty": True})

    fields: Dict[str, Any] = {"raw": {}}
    recognized: set = set()

    for tag, value in raw.items():
        # Strip group prefix for matching
        clean_tag = _strip_group_prefix(tag)

        # Try exact match in mapping
        field_name = EXIF_TO_FIELD.get(clean_tag)
        if field_name is None:
            # Try snake_case as fallback
            field_name = _to_snake(clean_tag)

        # Convert value
        converted = _convert_value(field_name, value)

        if field_name in EXIF_TO_FIELD.values() or field_name in {"gps_position", "gps_timestamp"}:
            # Known model field
            fields[field_name] = converted
            recognized.add(tag)
        else:
            # Unknown -> raw
            fields["raw"][tag] = value

    # Add extraction error if present
    if "_extraction_error" in raw:
        fields["raw"]["_extraction_error"] = raw["_extraction_error"]

    return ExifMetadata(**fields)


def write_xmp_sidecar(
    image_path: Union[str, Path],
    description: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    caption: Optional[str] = None,
    subjects: Optional[List[str]] = None,
    force: bool = False,
    dry_run: bool = False,
    sidecar_path: Optional[Union[str, Path]] = None,
) -> SidecarStatus:
    """Write a non-destructive XMP sidecar for an image.

    Creates a `.xmp` sidecar file next to the image (e.g. `photo.jpg.xmp`)
    with Dublin Core / XMP metadata. Skips RAW/video formats unless --force.

    Returns the sidecar write status.
    """
    path = Path(image_path)
    if sidecar_path:
        sidecar = Path(sidecar_path)
    else:
        sidecar = path.with_suffix(path.suffix + ".xmp")

    # Safety check for RAW/video unless forced
    if not force and path.suffix.lower() in {".raw", ".cr2", ".nef", ".arw", ".dng", ".mp4", ".mov"}:
        return SidecarStatus.SKIPPED_UNSAFE

    if sidecar.exists() and not force:
        return SidecarStatus.SKIPPED_UNSAFE

    if dry_run:
        return SidecarStatus.WRITTEN

    # Build XMP RDF
    xmp = _build_xmp(
        description=description or caption,
        keywords=keywords or subjects or [],
    )

    try:
        sidecar.write_text(xmp, encoding="utf-8")
        return SidecarStatus.WRITTEN
    except OSError:
        return SidecarStatus.ERROR


def _build_xmp(description: Optional[str], keywords: List[str]) -> str:
    """Build an XMP RDF XML string."""
    from xml.sax.saxutils import escape

    desc = escape(description or "")
    kw_xml = "\n".join(f"        <rdf:li>{escape(k)}</rdf:li>" for k in keywords)

    return f"""<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:dc="http://purl.org/dc/elements/1.1/">
      {f'<dc:description><rdf:Alt><rdf:li xml:lang="x-default">{desc}</rdf:li></rdf:Alt></dc:description>' if desc else ''}
      {f'<dc:subject><rdf:Bag>{kw_xml}</rdf:Bag></dc:subject>' if keywords else ''}
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"""


@dataclass
class DBRecord:
    """A flat enrichment record for database upsert."""

    asset_id: str
    content_sha256: Optional[str]
    file_path: str
    file_name: str
    file_size: Optional[int]
    mime_type: Optional[str]
    exif_raw: Dict[str, Any]
    exif_structured: Dict[str, Any]
    vlm_caption: Optional[str]
    vlm_description: Optional[str]
    vlm_model: Optional[str]
    ner_entities_json: str
    face_embeddings_json: str
    tags_json: str
    sidecar_status: Optional[str]
    enriched_at: str


def asset_to_record(asset: Asset) -> DBRecord:
    """Convert an Asset model to a flat DB record."""
    from datetime import datetime

    # Extract structured exif fields for persistence
    exif_structured = {}
    if asset.exif:
        # Get all model fields that are not 'raw'
        exif_data = asset.exif.model_dump(exclude={"raw"})
        exif_structured = {k: v for k, v in exif_data.items() if v is not None}
        exif_raw = asset.exif.raw
    else:
        exif_raw = {}

    return DBRecord(
        asset_id=str(asset.id),
        content_sha256=asset.content_sha256,
        file_path=asset.path,
        file_name=asset.filename,
        file_size=asset.file_size,
        mime_type=asset.mime_type,
        exif_raw=exif_raw,
        exif_structured=exif_structured,
        vlm_caption=asset.vlm.caption if asset.vlm else None,
        vlm_description=asset.vlm.detailed_description if asset.vlm else None,
        vlm_model=asset.vlm.model if asset.vlm else None,
        ner_entities_json=json.dumps([e.model_dump() for e in asset.ner_entities], default=str),
        face_embeddings_json=json.dumps([e.model_dump() for e in asset.face_embeddings], default=str),
        tags_json=json.dumps(asset.tags),
        sidecar_status=asset.sidecar_status.value if hasattr(asset.sidecar_status, "value") else asset.sidecar_status,
        enriched_at=datetime.utcnow().isoformat(),
    )


def record_to_asset(record: DBRecord) -> Asset:
    """Convert a DB record back to an Asset model."""
    from .models import ExifMetadata, SidecarStatus, VLMResult  # noqa: F811

    asset = Asset(
        id=deterministic_asset_id(record.file_path, record.content_sha256),
        path=record.file_path,
        filename=record.file_name,
        content_sha256=record.content_sha256,
        file_size=record.file_size,
        mime_type=record.mime_type,
        sidecar_status=SidecarStatus(record.sidecar_status) if record.sidecar_status else None,
    )
    # Restore exif: merge structured fields with raw
    if record.exif_structured:
        asset.exif = ExifMetadata(**record.exif_structured, raw=record.exif_raw)
    elif record.exif_raw:
        asset.exif = ExifMetadata(raw=record.exif_raw)
    if record.vlm_caption or record.vlm_description:
        asset.vlm = VLMResult(
            model=record.vlm_model or "unknown",
            caption=record.vlm_caption,
            detailed_description=record.vlm_description,
        )
    if record.ner_entities_json:
        try:
            entities = json.loads(record.ner_entities_json)
            asset.ner_entities = [NEREntity(**e) for e in entities]
        except (json.JSONDecodeError, TypeError):
            asset.ner_entities = []
    return asset


def sha256_file(path: Union[str, Path]) -> Optional[str]:
    """Compute SHA-256 hash of a file."""
    import hashlib

    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None
