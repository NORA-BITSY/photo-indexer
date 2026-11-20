from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from web.auth import get_current_user
from web.db import AsyncPhotoIndexDB, get_db
from web.jobs.manager import get_job_manager
from photo_indexer.models import MeasurementType


router = APIRouter(prefix="/api/v1/measurements", tags=["measurements"])


class RecomputeRequest(BaseModel):
    asset_ids: List[str]
    model_complexity: int = 1
    min_confidence: float = 0.5
    reference_height: Optional[float] = None


@router.get("")
async def list_measurements(
    asset_id: Optional[str] = Query(None),
    performer_id: Optional[str] = Query(None),
    measurement_type: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """List measurements with optional filters."""
    measurements = await db.get_measurements(
        asset_id=asset_id,
        performer_id=performer_id,
        measurement_type=measurement_type,
        limit=limit,
        offset=offset,
    )
    return {"measurements": measurements, "limit": limit, "offset": offset}


@router.get("/summary")
async def get_measurement_summary(
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Get aggregated statistics per measurement type."""
    summary = await db.get_measurement_summary()
    return {"summary": summary}


@router.get("/performer/{performer_id}")
async def get_performer_measurements(
    performer_id: str,
    measurement_type: Optional[str] = Query(None),
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Get all measurements for a performer."""
    measurements = await db.get_measurements(performer_id=performer_id, measurement_type=measurement_type)
    return {"measurements": measurements}


@router.post("/recompute")
async def recompute_measurements(
    request: RecomputeRequest,
    job_manager=Depends(get_job_manager),
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Recompute body measurements for assets (async job)."""
    job_id = await job_manager.create_job("body_pose", {
        "asset_ids": request.asset_ids,
        "model_complexity": request.model_complexity,
        "min_confidence": request.min_confidence,
        "reference_height": request.reference_height,
    })
    return {"job_id": job_id, "message": "Body pose detection and measurement started"}


@router.get("/types")
async def get_measurement_types(
    user=Depends(get_current_user),
):
    """List all measurement types with descriptions."""
    types = [
        {
            "value": m.value,
            "label": m.value.replace("_", " ").title(),
            "description": _get_measurement_description(m),
        }
        for m in MeasurementType
    ]
    return {"types": types}


def _get_measurement_description(m) -> str:
    descriptions = {
        "height": "Total body height from head to feet",
        "head_height": "Height of the head",
        "neck_length": "Length of the neck",
        "shoulder_width": "Distance between shoulder joints",
        "chest_width": "Width of the chest at widest point",
        "waist_width": "Width of the waist at narrowest point",
        "hip_width": "Width of the hips at widest point",
        "upper_arm_length": "Length from shoulder to elbow",
        "forearm_length": "Length from elbow to wrist",
        "hand_length": "Length from wrist to middle fingertip",
        "thigh_length": "Length from hip to knee",
        "shin_length": "Length from knee to ankle",
        "foot_length": "Length of the foot",
        "arm_span": "Distance between fingertips with arms outstretched",
        "torso_length": "Length from shoulder to hip",
        "leg_length": "Length from hip to ankle",
        "inseam": "Inner leg length from crotch to ankle",
        "body_type": "Somatotype classification (ectomorph/mesomorph/endomorph)",
    }
    return descriptions.get(m.value, "")


@router.get("/export")
async def export_measurements(
    asset_id: Optional[str] = Query(None),
    performer_id: Optional[str] = Query(None),
    measurement_type: Optional[str] = Query(None),
    format: str = Query("csv", pattern="^(csv|json)$"),
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Export measurements as CSV or JSON."""
    measurements = await db.get_measurements(
        asset_id=asset_id,
        performer_id=performer_id,
        measurement_type=measurement_type,
        limit=10000,
        offset=0,
    )

    if format == "json":
        return {"measurements": measurements}

    import io
    import csv
    from fastapi.responses import StreamingResponse

    output = io.StringIO()
    if measurements:
        writer = csv.DictWriter(output, fieldnames=measurements[0].keys())
        writer.writeheader()
        writer.writerows(measurements)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=measurements.csv"},
    )