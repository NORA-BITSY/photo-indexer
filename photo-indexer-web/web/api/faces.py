from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, Query

from web.auth import get_current_user
from web.db import AsyncPhotoIndexDB, get_db
from web.jobs.manager import get_job_manager


router = APIRouter(prefix="/api/v1/faces", tags=["faces"])


class FaceMatchRequest(BaseModel):
    performer_id: str
    similarity: float
    method: str = "manual"


class MatchAllRequest(BaseModel):
    asset_ids: Optional[List[str]] = None
    threshold: float = 0.6


@router.get("/unmatched")
async def get_unmatched_faces(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Get paginated list of faces without person match."""
    faces = await db.get_unmatched_faces(limit, offset)
    return {"faces": faces, "limit": limit, "offset": offset}


@router.get("/asset/{asset_id}")
async def get_faces_for_asset(
    asset_id: str,
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Get all faces for an asset with match status."""
    faces = await db.get_faces_for_asset(asset_id)
    return {"faces": faces}


@router.post("/{face_id}/match")
async def match_face(
    face_id: str,
    request: FaceMatchRequest,
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Link a face to a performer."""
    # Verify performer exists
    performer = await db.get_performer(request.performer_id)
    if not performer:
        raise HTTPException(status_code=404, detail="Performer not found")

    await db.match_face_to_person(face_id, request.performer_id, request.similarity, request.method)
    return {"message": "Face matched to performer"}


@router.post("/{face_id}/unmatch")
async def unmatch_face(
    face_id: str,
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Remove face match."""
    success = await db.unmatch_face(face_id)
    if not success:
        raise HTTPException(status_code=404, detail="Face match not found")
    return {"message": "Face unmatched"}


@router.post("/match-all")
async def match_all_faces(
    request: MatchAllRequest,
    job_manager=Depends(get_job_manager),
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Run face detection and matching on assets (async job)."""
    job_id = await job_manager.create_job("face_detect", {
        "asset_ids": request.asset_ids,
        "threshold": request.threshold,
    })
    return {"job_id": job_id, "message": "Face detection and matching started"}


@router.get("/{face_id}/embedding")
async def get_face_embedding(
    face_id: str,
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Get face embedding vector."""
    # Need to query the sync DB for embedding
    from photo_indexer.database import PhotoIndexDB
    from web.config import get_settings
    settings = get_settings()
    sync_db = PhotoIndexDB(settings.database_path)

    with sync_db._connect() as conn:
        row = conn.execute(
            "SELECT embedding_json FROM enrich_faces WHERE id = ?", (face_id,)
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Face not found")

    import json
    return {"face_id": face_id, "embedding": json.loads(row["embedding_json"])}