from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

from web.auth import get_current_user
from web.db import AsyncPhotoIndexDB, get_db
from web.jobs.manager import get_job_manager


router = APIRouter(prefix="/api/v1/performers", tags=["performers"])


# Request/Response models
class PerformerCreate(BaseModel):
    name: str
    aliases: List[str] = []
    face_embedding: Optional[List[float]] = None
    face_embedding_quality: float = 0.0
    notes: Optional[str] = None


class PerformerUpdate(BaseModel):
    aliases: Optional[List[str]] = None
    body_type: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg_estimate: Optional[float] = None
    notes: Optional[str] = None


class BuildProfileRequest(BaseModel):
    asset_ids: List[str]
    aliases: Optional[List[str]] = None
    reference_height: Optional[float] = None
    store: bool = True


class FacialBiometricsRequest(BaseModel):
    asset_ids: List[str]
    store: bool = True


class SearchByFaceRequest(BaseModel):
    embedding: List[float]
    threshold: float = 0.6
    limit: int = 10


@router.get("")
async def list_performers(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    search: Optional[str] = Query(None),
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """List performers with pagination and search."""
    performers = await db.list_performers(limit, offset, search)
    return {"performers": performers, "limit": limit, "offset": offset}


@router.get("/{performer_id}")
async def get_performer(
    performer_id: str,
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Get full performer profile with biometrics, measurements, assets, faces."""
    performer = await db.get_performer(performer_id)
    if not performer:
        raise HTTPException(status_code=404, detail="Performer not found")

    # Get linked assets
    assets = await db.get_performer_assets(performer_id)

    # Get facial biometrics
    from photo_indexer.database import PhotoIndexDB
    from web.config import get_settings
    settings = get_settings()
    sync_db = PhotoIndexDB(settings.database_path)
    biometrics = sync_db.get_facial_biometrics(performer_id)

    # Get measurements
    measurements = await db.get_measurements(performer_id=performer_id)

    # Get matched faces
    matched_faces = []
    for asset in assets:
        faces = await db.get_faces_for_asset(asset["asset_id"])
        for face in faces:
            if face.get("person_id") == performer_id:
                matched_faces.append(face)

    return {
        "performer": performer,
        "assets": assets,
        "biometrics": biometrics,
        "measurements": measurements,
        "matched_faces": matched_faces,
    }


@router.post("", status_code=201)
async def create_performer(
    request: PerformerCreate,
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Create a new performer."""
    # Check if name already exists
    existing = await db.get_performer_by_name(request.name)
    if existing:
        raise HTTPException(status_code=400, detail="Performer with this name already exists")

    performer_id = await db.create_performer(
        name=request.name,
        aliases=request.aliases,
        face_embedding=request.face_embedding,
        face_embedding_quality=request.face_embedding_quality,
        notes=request.notes,
    )

    return {"id": performer_id, "name": request.name}


@router.patch("/{performer_id}")
async def update_performer(
    performer_id: str,
    request: PerformerUpdate,
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Update performer details."""
    performer = await db.get_performer(performer_id)
    if not performer:
        raise HTTPException(status_code=404, detail="Performer not found")

    updates = request.model_dump(exclude_unset=True)
    if "aliases" in updates:
        updates["aliases_json"] = updates.pop("aliases")

    success = await db.update_performer(performer_id, updates)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to update performer")

    return {"message": "Performer updated"}


@router.delete("/{performer_id}")
async def delete_performer(
    performer_id: str,
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Delete a performer."""
    performer = await db.get_performer(performer_id)
    if not performer:
        raise HTTPException(status_code=404, detail="Performer not found")

    success = await db.delete_performer(performer_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to delete performer")

    return {"message": "Performer deleted"}


@router.post("/{performer_id}/build-profile")
async def build_profile(
    performer_id: str,
    request: BuildProfileRequest,
    job_manager=Depends(get_job_manager),
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Build performer profile from multiple assets (async job)."""
    performer = await db.get_performer(performer_id)
    if not performer:
        raise HTTPException(status_code=404, detail="Performer not found")

    job_id = await job_manager.create_job("build_profile", {
        "performer_id": performer_id,
        "asset_ids": request.asset_ids,
        "aliases": request.aliases,
        "reference_height": request.reference_height,
        "store": request.store,
    })

    return {"job_id": job_id, "message": "Profile building started"}


@router.post("/{performer_id}/facial-biometrics")
async def build_facial_biometrics(
    performer_id: str,
    request: FacialBiometricsRequest,
    job_manager=Depends(get_job_manager),
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Build facial biometrics for a performer (async job)."""
    performer = await db.get_performer(performer_id)
    if not performer:
        raise HTTPException(status_code=404, detail="Performer not found")

    job_id = await job_manager.create_job("facial_biometrics", {
        "performer_id": performer_id,
        "asset_ids": request.asset_ids,
        "store": request.store,
    })

    return {"job_id": job_id, "message": "Facial biometrics building started"}


@router.post("/search-by-face")
async def search_by_face(
    request: SearchByFaceRequest,
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Search performers by face embedding similarity."""
    # This would need a vector search implementation
    # For now, use the sync DB's cosine similarity
    from photo_indexer.database import PhotoIndexDB
    from web.config import get_settings
    settings = get_settings()
    sync_db = PhotoIndexDB(settings.database_path)

    match = sync_db.match_face_to_person(request.embedding, request.threshold)
    if match:
        person_id, similarity = match
        person = sync_db.get_person(person_id)
        return {
            "matches": [{
                "performer_id": person_id,
                "name": person["name"] if person else None,
                "similarity": similarity,
            }]
        }
    return {"matches": []}


@router.get("/{performer_id}/assets")
async def get_performer_assets(
    performer_id: str,
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Get all assets linked to a performer."""
    performer = await db.get_performer(performer_id)
    if not performer:
        raise HTTPException(status_code=404, detail="Performer not found")

    assets = await db.get_performer_assets(performer_id)
    return {"assets": assets}


@router.get("/{performer_id}/measurements")
async def get_performer_measurements(
    performer_id: str,
    measurement_type: Optional[str] = Query(None),
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Get measurements for a performer."""
    performer = await db.get_performer(performer_id)
    if not performer:
        raise HTTPException(status_code=404, detail="Performer not found")

    measurements = await db.get_measurements(performer_id=performer_id, measurement_type=measurement_type)
    return {"measurements": measurements}