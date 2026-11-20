from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, Query
from sse_starlette.sse import EventSourceResponse

from web.auth import get_current_user
from web.jobs.manager import JobManager, JobStatus, get_job_manager


router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


class JobCreateRequest(BaseModel):
    job_type: str
    payload: Dict[str, Any]
    priority: int = 0


class JobResponse(BaseModel):
    id: str
    type: str
    payload: Dict[str, Any]
    status: str
    progress: float
    message: str
    result: Any
    error: Optional[str]
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]


@router.post("", response_model=JobResponse, status_code=201)
async def create_job(
    request: JobCreateRequest,
    job_manager: JobManager = Depends(get_job_manager),
    user=Depends(get_current_user),
):
    """Create a new job."""
    # Validate job type
    valid_types = [
        "vlm_analyze", "ner_enrich", "face_detect", "body_pose",
        "build_profile", "facial_biometrics", "sidecar_write",
        "batch_operation", "tag_import", "backup", "semantic_auto_tag"
    ]
    if request.job_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid job type. Valid types: {valid_types}")

    job_id = await job_manager.create_job(request.job_type, request.payload, request.priority)
    job = await job_manager.get_job(job_id)
    return job


@router.get("", response_model=List[JobResponse])
async def list_jobs(
    job_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    job_manager: JobManager = Depends(get_job_manager),
    user=Depends(get_current_user),
):
    """List jobs with optional filters."""
    return await job_manager.list_jobs(job_type, status, limit, offset)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
    user=Depends(get_current_user),
):
    """Get job status and result."""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/{job_id}/stream")
async def stream_job_progress(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
    user=Depends(get_current_user),
):
    """Stream job progress via Server-Sent Events."""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        async for update in job_manager.subscribe_progress(job_id):
            yield {"data": json.dumps(update)}

    return EventSourceResponse(event_generator())


@router.delete("/{job_id}")
async def cancel_job(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager),
    user=Depends(get_current_user),
):
    """Cancel a job."""
    success = await job_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot cancel job (not found or already completed)")
    return {"message": "Job cancelled"}


import json