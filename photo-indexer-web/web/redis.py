import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

import redis.asyncio as redis
from redis.asyncio import Redis

from web.config import get_settings


class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobManager:
    """Redis-backed job manager with progress pub/sub."""

    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self._pubsub: Optional[redis.client.PubSub] = None
        self._worker_tasks: Dict[str, asyncio.Task] = {}
        self._concurrency_limits: Dict[str, int] = {}
        self._running_counts: Dict[str, int] = {}

    async def initialize(self, concurrency_limits: Dict[str, int]) -> None:
        self._concurrency_limits = concurrency_limits
        for job_type in concurrency_limits:
            self._running_counts[job_type] = 0

    def _job_key(self, job_id: str) -> str:
        return f"job:{job_id}"

    def _progress_channel(self, job_id: str) -> str:
        return f"job_progress:{job_id}"

    def _job_list_key(self) -> str:
        return "jobs:list"

    async def create_job(
        self,
        job_type: str,
        payload: Dict[str, Any],
        priority: int = 0,
    ) -> str:
        """Create a new job and add to queue."""
        job_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        job_data = {
            "id": job_id,
            "type": job_type,
            "payload": payload,
            "status": JobStatus.PENDING,
            "progress": 0.0,
            "message": "Queued",
            "result": None,
            "error": None,
            "priority": priority,
            "created_at": now,
            "started_at": None,
            "completed_at": None,
        }

        async with self.redis.pipeline() as pipe:
            pipe.set(self._job_key(job_id), json.dumps(job_data))
            pipe.zadd(self._job_list_key(), {job_id: priority})
            await pipe.execute()

        # Try to start worker if capacity available
        asyncio.create_task(self._maybe_start_worker(job_type))

        return job_id

    async def get_job(self, job_id: str) -> Optional[Dict]:
        """Get job status and result."""
        data = await self.redis.get(self._job_key(job_id))
        if not data:
            return None
        return json.loads(data)

    async def update_progress(
        self, job_id: str, progress: float, message: str = ""
    ) -> None:
        """Update job progress and publish to channel."""
        job = await self.get_job(job_id)
        if not job:
            return

        job["progress"] = max(0.0, min(1.0, progress))
        if message:
            job["message"] = message

        await self.redis.set(self._job_key(job_id), json.dumps(job))

        # Publish progress update
        await self.redis.publish(
            self._progress_channel(job_id),
            json.dumps({"job_id": job_id, "progress": job["progress"], "message": message}),
        )

    async def complete_job(self, job_id: str, result: Any) -> None:
        """Mark job as completed with result."""
        job = await self.get_job(job_id)
        if not job:
            return

        job["status"] = JobStatus.COMPLETED
        job["progress"] = 1.0
        job["result"] = result
        job["completed_at"] = datetime.utcnow().isoformat()
        job["message"] = "Completed"

        await self.redis.set(self._job_key(job_id), json.dumps(job))
        await self._decrement_running(job["type"])

        # Publish completion
        await self.redis.publish(
            self._progress_channel(job_id),
            json.dumps({"job_id": job_id, "status": JobStatus.COMPLETED, "result": result}),
        )

        # Try to start next worker
        asyncio.create_task(self._maybe_start_worker(job["type"]))

    async def fail_job(self, job_id: str, error: str) -> None:
        """Mark job as failed with error."""
        job = await self.get_job(job_id)
        if not job:
            return

        job["status"] = JobStatus.FAILED
        job["error"] = error
        job["completed_at"] = datetime.utcnow().isoformat()
        job["message"] = f"Failed: {error}"

        await self.redis.set(self._job_key(job_id), json.dumps(job))
        await self._decrement_running(job["type"])

        # Publish failure
        await self.redis.publish(
            self._progress_channel(job_id),
            json.dumps({"job_id": job_id, "status": JobStatus.FAILED, "error": error}),
        )

        asyncio.create_task(self._maybe_start_worker(job["type"]))

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        job = await self.get_job(job_id)
        if not job:
            return False

        if job["status"] in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return False

        job["status"] = JobStatus.CANCELLED
        job["completed_at"] = datetime.utcnow().isoformat()
        job["message"] = "Cancelled"

        await self.redis.set(self._job_key(job_id), json.dumps(job))

        if job["status"] == JobStatus.RUNNING:
            await self._decrement_running(job["type"])
            asyncio.create_task(self._maybe_start_worker(job["type"]))

        return True

    async def list_jobs(
        self,
        job_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict]:
        """List jobs with optional filters."""
        # Get job IDs from sorted set (newest first by using negative priority + timestamp)
        job_ids = await self.redis.zrevrange(
            self._job_list_key(), offset, offset + limit - 1
        )

        jobs = []
        for job_id in job_ids:
            job = await self.get_job(job_id)
            if not job:
                continue
            if job_type and job["type"] != job_type:
                continue
            if status and job["status"] != status:
                continue
            jobs.append(job)

        return jobs

    async def subscribe_progress(self, job_id: str) -> AsyncIterator[Dict]:
        """Subscribe to job progress updates via pub/sub."""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(self._progress_channel(job_id))

        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    yield json.loads(message["data"])
        finally:
            await pubsub.unsubscribe(self._progress_channel(job_id))
            await pubsub.close()

    async def _maybe_start_worker(self, job_type: str) -> None:
        """Start a worker for job_type if capacity available."""
        limit = self._concurrency_limits.get(job_type, self._concurrency_limits.get("default", 4))
        running = self._running_counts.get(job_type, 0)

        if running >= limit:
            return

        # Find next pending job of this type
        job_ids = await self.redis.zrange(self._job_list_key(), 0, -1)
        for job_id in job_ids:
            job = await self.get_job(job_id)
            if not job:
                continue
            if job["type"] != job_type:
                continue
            if job["status"] != JobStatus.PENDING:
                continue

            # Found a job to run
            self._running_counts[job_type] = running + 1
            await self._run_job(job_id)
            return

    async def _decrement_running(self, job_type: str) -> None:
        self._running_counts[job_type] = max(0, self._running_counts.get(job_type, 1) - 1)

    async def _run_job(self, job_id: str) -> None:
        """Execute a job. Override in subclass or register workers."""
        job = await self.get_job(job_id)
        if not job:
            return

        job["status"] = JobStatus.RUNNING
        job["started_at"] = datetime.utcnow().isoformat()
        job["message"] = "Starting..."
        await self.redis.set(self._job_key(job_id), json.dumps(job))

        # This should be overridden by specific workers
        await self.fail_job(job_id, "No worker registered for this job type")


# Global instances
_redis_client: Optional[Redis] = None
_job_manager: Optional[JobManager] = None


async def get_redis() -> Redis:
    global _redis_client
    if _redis_client is None:
        settings = get_settings()
        _redis_client = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
    return _redis_client


async def get_job_manager() -> JobManager:
    global _job_manager
    if _job_manager is None:
        redis_client = await get_redis()
        _job_manager = JobManager(redis_client)
        settings = get_settings()
        concurrency_limits = {
            "vlm_analyze": settings.job_concurrency_vlm,
            "face_detect": settings.job_concurrency_face,
            "body_pose": settings.job_concurrency_body,
            "ner_enrich": settings.job_concurrency_default,
            "build_profile": 1,
            "facial_biometrics": 1,
            "sidecar_write": settings.job_concurrency_default,
            "batch_operation": 1,
            "tag_import": 1,
            "backup": 1,
            "semantic_auto_tag": settings.job_concurrency_vlm,
            "default": settings.job_concurrency_default,
        }
        await _job_manager.initialize(concurrency_limits)
    return _job_manager


async def close_redis() -> None:
    global _redis_client, _job_manager
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
    _job_manager = None