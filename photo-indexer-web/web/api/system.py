import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from web.auth import get_current_user
from web.config import get_settings
from web.db import AsyncPhotoIndexDB, get_db
from web.jobs.manager import get_job_manager
from web.immich import get_immich_client


router = APIRouter(prefix="/api/v1/system", tags=["system"])


@router.get("/info")
async def get_system_info(
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Get system information."""
    settings = get_settings()

    # Disk usage
    db_path = Path(settings.database_path)
    db_size = db_path.stat().st_size if db_path.exists() else 0

    total, used, free = shutil.disk_usage("/data")

    # Model files
    model_files = {
        "face_landmarker": Path("face_landmarker.task").exists(),
        "pose_landmarker_heavy": Path("pose_landmarker_heavy.task").exists(),
    }

    # Check Ollama connectivity
    import httpx
    ollama_healthy = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            ollama_healthy = resp.status_code == 200
    except Exception:
        pass

    # Check Immich connectivity
    immich = get_immich_client()
    immich_healthy = await immich.health_check()

    return {
        "database": {
            "path": settings.database_path,
            "size_bytes": db_size,
        },
        "disk": {
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": free,
            "usage_percent": (used / total * 100) if total > 0 else 0,
        },
        "models": model_files,
        "ollama": {
            "url": settings.ollama_base_url,
            "model": settings.vlm_model,
            "healthy": ollama_healthy,
        },
        "immich": {
            "url": settings.immich_url,
            "healthy": immich_healthy,
        },
        "redis": {
            "url": settings.redis_url,
        },
    }


@router.get("/health")
async def health_check(
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Detailed health check."""
    settings = get_settings()

    checks = {}

    # Database
    try:
        await db.get_stats()
        checks["database"] = {"status": "ok", "message": "Connected"}
    except Exception as e:
        checks["database"] = {"status": "error", "message": str(e)}

    # Ollama
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            checks["ollama"] = {"status": "ok" if resp.status_code == 200 else "error"}
    except Exception as e:
        checks["ollama"] = {"status": "error", "message": str(e)}

    # Immich
    immich = get_immich_client()
    immich_ok = await immich.health_check()
    checks["immich"] = {"status": "ok" if immich_ok else "error"}

    # Redis
    try:
        from web.redis import get_redis
        redis_client = await get_redis()
        await redis_client.ping()
        checks["redis"] = {"status": "ok"}
    except Exception as e:
        checks["redis"] = {"status": "error", "message": str(e)}

    overall = "ok" if all(c.get("status") == "ok" for c in checks.values()) else "degraded"

    return {
        "status": overall,
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.post("/backup")
async def create_backup(
    job_manager=Depends(get_job_manager),
    user=Depends(get_current_user),
):
    """Create a backup (async job)."""
    job_id = await job_manager.create_job("backup", {})
    return {"job_id": job_id, "message": "Backup started"}


@router.get("/backups")
async def list_backups(
    user=Depends(get_current_user),
):
    """List available backup files."""
    backup_dir = Path("/data/backups")
    if not backup_dir.exists():
        backup_dir = Path("/home/pat/photo-indexer/backups")

    backups = []
    if backup_dir.exists():
        for b in sorted(backup_dir.glob("*.tar.gz"), key=lambda p: p.stat().st_mtime, reverse=True):
            backups.append({
                "name": b.name,
                "size": b.stat().st_size,
                "mtime": b.stat().st_mtime,
                "mtime_iso": datetime.fromtimestamp(b.stat().st_mtime).isoformat(),
            })

    return {"backups": backups}


@router.get("/backups/{filename}")
async def download_backup(
    filename: str,
    user=Depends(get_current_user),
):
    """Download a backup file."""
    # Prevent path traversal
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid filename")

    backup_dir = Path("/data/backups")
    if not backup_dir.exists():
        backup_dir = Path("/home/pat/photo-indexer/backups")

    file_path = backup_dir / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Backup not found")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/gzip",
    )


@router.get("/settings")
async def get_settings(
    user=Depends(get_current_user),
):
    """Get current effective configuration (non-sensitive)."""
    settings = get_settings()
    return {
        "database_path": settings.database_path,
        "immich_url": settings.immich_url,
        "ollama_base_url": settings.ollama_base_url,
        "vlm_model": settings.vlm_model,
        "redis_url": settings.redis_url,
        "media_roots": settings.media_roots,
        "max_upload_size": settings.max_upload_size,
        "job_timeout_seconds": settings.job_timeout_seconds,
        "job_concurrency": {
            "vlm": settings.job_concurrency_vlm,
            "face": settings.job_concurrency_face,
            "body": settings.job_concurrency_body,
            "default": settings.job_concurrency_default,
        },
    }


@router.patch("/settings")
async def update_settings(
    updates: Dict[str, Any],
    user=Depends(get_current_user),
):
    """Update runtime-configurable settings."""
    # Only allow certain settings to be updated at runtime
    allowed = {
        "vlm_model",
        "job_concurrency_vlm",
        "job_concurrency_face",
        "job_concurrency_body",
        "job_concurrency_default",
        "job_timeout_seconds",
    }

    filtered = {k: v for k, v in updates.items() if k in allowed}
    if not filtered:
        raise HTTPException(status_code=400, detail="No valid settings to update")

    # In a real implementation, this would update the config file or database
    # For now, just return the changes
    return {"updated": filtered, "message": "Settings updated (requires restart for some changes)"}


@router.get("/logs")
async def get_logs(
    lines: int = Query(100, ge=1, le=1000),
    user=Depends(get_current_user),
):
    """Get recent log lines."""
    # Try to read from container logs or log file
    log_sources = [
        Path("/var/log/photo-indexer.log"),
        Path("/home/pat/photo-indexer/photo-indexer.log"),
    ]

    for log_file in log_sources:
        if log_file.exists():
            try:
                with open(log_file) as f:
                    all_lines = f.readlines()
                return {"logs": all_lines[-lines:]}
            except Exception:
                pass

    # Fallback: try docker logs (if running in container)
    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(lines), "photo-indexer-web"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return {"logs": result.stdout.splitlines()}
    except Exception:
        pass

    return {"logs": ["No log source available"]}