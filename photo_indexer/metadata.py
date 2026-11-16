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
            return {}
        data = json.loads(result.stdout)
        return data[0] if isinstance(data, list) and data else {}
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return {}


def parse_exif(raw: Dict[str, Any]) -> ExifMetadata:
    """Parse raw ExifTool output into an ExifMetadata model.

    Unknown keys are stored in `raw`.
    """
    known = {
        "FileName", "Directory", "FileSize", "FileModifyDate", "FileAccessDate",
        "FileInodeChangeDate", "FilePermissions", "FileType", "MIMEType",
        "ImageWidth", "ImageHeight", "EncodingProcess", "BitsPerSample",
        "ColorComponents", "YCbCrSubSampling", "ImageType", "Make", "Model",
        "Orientation", "XResolution", "YResolution", "ResolutionUnit",
        "ModifyDate", "ExifOffset", "GPSPosition", "GPSLatitude", "GPSLongitude",
        "GPSAltitude", "GPSTimeStamp", "DateTimeOriginal", "CreateDate",
        "Software", "Artist", "Copyright", "ImageDescription", "Keywords",
        "Rating", "LensModel", "FocalLength", "Aperture", "ShutterSpeed",
        "ISO", "Flash", "WhiteBalance", "ColorSpace", "Compression",
        "ThumbnailOffset", "ThumbnailLength",
    }
    fields: Dict[str, Any] = {"raw": {}}
    for key, value in raw.items():
        # Normalize key to snake_case
        snake = _to_snake(key)
        if key in known:
            fields[snake] = value
        else:
            fields["raw"][key] = value
    return ExifMetadata(**fields)


def _to_snake(name: str) -> str:
    """Convert a tag name like 'DateTimeOriginal' to 'date_time_original'."""
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    s1 = re.sub(r"[^a-zA-Z0-9]", "_", s1)
    return s1.lower()


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
    vlm_caption: Optional[str]
    vlm_description: Optional[str]
    vlm_model: Optional[str]
    ner_entities_json: str
    sidecar_status: Optional[str]
    enriched_at: str


def asset_to_record(asset: Asset) -> DBRecord:
    """Convert an Asset model to a flat DB record."""
    from datetime import datetime

    return DBRecord(
        asset_id=str(asset.id),
        content_sha256=asset.content_sha256,
        file_path=asset.path,
        file_name=asset.filename,
        file_size=asset.file_size,
        mime_type=asset.mime_type,
        exif_raw=asset.exif.raw if asset.exif else {},
        vlm_caption=asset.vlm.caption if asset.vlm else None,
        vlm_description=asset.vlm.detailed_description if asset.vlm else None,
        vlm_model=asset.vlm.model if asset.vlm else None,
        ner_entities_json=json.dumps([e.model_dump() for e in asset.ner_entities], default=str),
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
