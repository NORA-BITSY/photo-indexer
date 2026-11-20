from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse, Response

from web.auth import get_current_user
from web.db import AsyncPhotoIndexDB, get_db
from web.jobs.manager import get_job_manager
from web.immich import get_immich_client
from web.media import serve_media_file
from web.config import get_settings
from pathlib import Path


router = APIRouter(prefix="/api/v1/assets", tags=["assets"])


class AnalyzeRequest(BaseModel):
    model: Optional[str] = None
    prompt: Optional[str] = None


class EnrichRequest(BaseModel):
    pass


class SidecarRequest(BaseModel):
    description: Optional[str] = None
    keywords: List[str] = []


class BodyPoseRequest(BaseModel):
    model_complexity: int = 1
    min_confidence: float = 0.5
    reference_height: Optional[float] = None


class BatchRequest(BaseModel):
    asset_ids: List[str]
    operations: List[str]  # analyze, enrich, faces, body-pose, sidecar


@router.get("")
async def list_assets(
    status: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    has_faces: Optional[bool] = Query(None),
    has_measurements: Optional[bool] = Query(None),
    performer_id: Optional[str] = Query(None),
    tags: Optional[List[str]] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """List assets with filters and pagination."""
    filters = {}
    if status:
        filters["status"] = status
    if date_from:
        filters["date_from"] = date_from
    if date_to:
        filters["date_to"] = date_to
    if has_faces:
        filters["has_faces"] = True
    if has_measurements:
        filters["has_measurements"] = True
    if performer_id:
        filters["performer_id"] = performer_id

    assets = await db.list_assets(limit, offset, filters)
    total = await db.count_assets(filters)

    # Convert to dict for JSON serialization
    asset_list = []
    for asset in assets:
        asset_list.append({
            "id": str(asset.id) if asset.id else None,
            "path": asset.path,
            "filename": asset.filename,
            "content_sha256": asset.content_sha256,
            "file_size": asset.file_size,
            "mime_type": asset.mime_type,
            "status": asset.status.value if hasattr(asset.status, "value") else asset.status,
            "indexed_at": asset.indexed_at.isoformat() if asset.indexed_at else None,
            "analyzed_at": asset.analyzed_at.isoformat() if asset.analyzed_at else None,
            "enriched_at": asset.enriched_at.isoformat() if asset.enriched_at else None,
            "error_message": asset.error_message,
            "has_vlm": asset.vlm is not None,
            "has_faces": len(asset.face_embeddings) > 0,
            "face_count": len(asset.face_embeddings),
            "has_measurements": len(asset.ner_entities) > 0,  # approximate
        })

    return {
        "assets": asset_list,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/stats")
async def get_asset_stats(
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Get dashboard statistics for assets."""
    stats = await db.get_stats()
    return stats


@router.get("/{asset_id}")
async def get_asset(
    asset_id: str,
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Get full asset detail with all enrichment data."""
    asset = await db.get_asset(asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")

    # Get linked performers
    from web.config import get_settings
    settings = get_settings()
    sync_db = __import__("photo_indexer.database", fromlist=["PhotoIndexDB"]).PhotoIndexDB(settings.database_path)
    performers = sync_db.get_asset_performers(asset_id)

    # Get semantic tags
    semantic_tags = await db.get_semantic_tags_for_asset(asset_id)

    # Get measurements
    measurements = await db.get_measurements(asset_id=asset_id)

    # Get poses
    poses = sync_db.get_body_poses(asset_id)

    return {
        "asset": {
            "id": str(asset.id) if asset.id else None,
            "path": asset.path,
            "filename": asset.filename,
            "content_sha256": asset.content_sha256,
            "file_size": asset.file_size,
            "mime_type": asset.mime_type,
            "status": asset.status.value if hasattr(asset.status, "value") else asset.status,
            "indexed_at": asset.indexed_at.isoformat() if asset.indexed_at else None,
            "analyzed_at": asset.analyzed_at.isoformat() if asset.analyzed_at else None,
            "enriched_at": asset.enriched_at.isoformat() if asset.enriched_at else None,
            "error_message": asset.error_message,
        },
        "exif": asset.exif.model_dump() if asset.exif else None,
        "vlm": asset.vlm.model_dump() if asset.vlm else None,
        "entities": [
            {
                "text": e.text,
                "entity_type": e.entity_type.value if hasattr(e.entity_type, "value") else e.entity_type,
                "label": e.label,
                "confidence": e.confidence,
                "source": e.source,
            }
            for e in asset.ner_entities
        ],
        "faces": [
            {
                "id": str(f.id) if f.id else None,
                "bbox": f.bbox,
                "landmarks": f.landmarks,
                "embedding": f.embedding[:10] if f.embedding else None,  # truncated for preview
                "det_score": f.det_score,
                "age": f.age,
                "gender": f.gender,
                "model_name": f.model_name,
                "detected_at": f.detected_at.isoformat() if f.detected_at else None,
            }
            for f in asset.face_embeddings
        ],
        "poses": poses,
        "measurements": measurements,
        "performers": performers,
        "semantic_tags": semantic_tags,
        "sidecar_status": asset.sidecar_status.value if asset.sidecar_status else None,
    }


@router.post("/{asset_id}/analyze")
async def analyze_asset(
    asset_id: str,
    request: AnalyzeRequest,
    job_manager=Depends(get_job_manager),
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Trigger VLM analysis for an asset (async job)."""
    asset = await db.get_asset(asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")

    job_id = await job_manager.create_job("vlm_analyze", {
        "asset_id": asset_id,
        "model": request.model,
        "prompt": request.prompt,
    })

    return {"job_id": job_id, "message": "VLM analysis started"}


@router.post("/{asset_id}/enrich")
async def enrich_asset(
    asset_id: str,
    job_manager=Depends(get_job_manager),
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Trigger NER enrichment for an asset (async job)."""
    asset = await db.get_asset(asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")

    job_id = await job_manager.create_job("ner_enrich", {"asset_id": asset_id})

    return {"job_id": job_id, "message": "NER enrichment started"}


@router.post("/{asset_id}/sidecar")
async def write_sidecar(
    asset_id: str,
    request: SidecarRequest,
    job_manager=Depends(get_job_manager),
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Write XMP sidecar for an asset (async job)."""
    asset = await db.get_asset(asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")

    job_id = await job_manager.create_job("sidecar_write", {
        "asset_id": asset_id,
        "description": request.description,
        "keywords": request.keywords,
    })

    return {"job_id": job_id, "message": "Sidecar write started"}


@router.post("/{asset_id}/body-pose")
async def detect_body_pose(
    asset_id: str,
    request: BodyPoseRequest,
    job_manager=Depends(get_job_manager),
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Detect body pose and compute measurements (async job)."""
    asset = await db.get_asset(asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")

    job_id = await job_manager.create_job("body_pose", {
        "asset_ids": [asset_id],
        "model_complexity": request.model_complexity,
        "min_confidence": request.min_confidence,
        "reference_height": request.reference_height,
    })

    return {"job_id": job_id, "message": "Body pose detection started"}


@router.post("/batch")
async def batch_operations(
    request: BatchRequest,
    job_manager=Depends(get_job_manager),
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Run batch operations on multiple assets (async job)."""
    # Validate asset IDs
    valid_assets = []
    for aid in request.asset_ids:
        asset = await db.get_asset(aid)
        if asset:
            valid_assets.append(aid)

    if not valid_assets:
        raise HTTPException(status_code=400, detail="No valid assets found")

    valid_ops = ["analyze", "enrich", "faces", "body-pose", "sidecar"]
    invalid_ops = [op for op in request.operations if op not in valid_ops]
    if invalid_ops:
        raise HTTPException(status_code=400, detail=f"Invalid operations: {invalid_ops}")

    job_id = await job_manager.create_job("batch_operation", {
        "asset_ids": valid_assets,
        "operations": request.operations,
    })

    return {"job_id": job_id, "message": "Batch operation started", "asset_count": len(valid_assets)}


@router.get("/{asset_id}/file")
async def get_asset_file(
    asset_id: str,
    request: Request,
    user=Depends(get_current_user),
):
    """Get full-resolution asset file (proxies to Immich or serves directly)."""
    asset = await get_db().get_asset(asset_id)  # Need proper dependency
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")

    # Try Immich first
    immich = get_immich_client()
    content = await immich.get_asset_file(asset_id)
    if content:
        return Response(content=content, media_type=asset.mime_type or "application/octet-stream")

    # Fallback to direct media serve
    for root_name, root_path in get_settings().media_root_dict.items():
        try:
            rel_path = Path(asset.path).relative_to(root_path)
            return await serve_media_file(root_name, str(rel_path), request)
        except ValueError:
            continue

    raise HTTPException(status_code=404, detail="File not found in media roots")


@router.get("/{asset_id}/thumbnail")
async def get_asset_thumbnail(
    asset_id: str,
    size: str = Query("preview"),
    user=Depends(get_current_user),
):
    """Get asset thumbnail (proxies to Immich)."""
    immich = get_immich_client()
    content = await immich.get_asset_thumbnail(asset_id, size)
    if content:
        return Response(content=content, media_type="image/jpeg")

    raise HTTPException(status_code=404, detail="Thumbnail not found")


@router.get("/export")
async def export_assets(
    status: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    performer_id: Optional[str] = Query(None),
    format: str = Query("csv", pattern="^(csv|json)$"),
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Export assets as CSV or JSON."""
    filters = {}
    if status:
        filters["status"] = status
    if date_from:
        filters["date_from"] = date_from
    if date_to:
        filters["date_to"] = date_to
    if performer_id:
        filters["performer_id"] = performer_id

    assets = await db.list_assets(limit=10000, offset=0, filters=filters)

    if format == "json":
        return {"assets": [a.model_dump() for a in assets]}

    import io
    import csv

    output = io.StringIO()
    if assets:
        fieldnames = ["id", "path", "filename", "content_sha256", "file_size", "mime_type",
                      "status", "indexed_at", "analyzed_at", "enriched_at"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for a in assets:
            writer.writerow({
                "id": str(a.id) if a.id else "",
                "path": a.path,
                "filename": a.filename,
                "content_sha256": a.content_sha256 or "",
                "file_size": a.file_size or "",
                "mime_type": a.mime_type or "",
                "status": a.status.value if hasattr(a.status, "value") else a.status,
                "indexed_at": a.indexed_at.isoformat() if a.indexed_at else "",
                "analyzed_at": a.analyzed_at.isoformat() if a.analyzed_at else "",
                "enriched_at": a.enriched_at.isoformat() if a.enriched_at else "",
            })

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=assets.csv"},
    )