import asyncio
import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from photo_indexer.body import BodyDetector, BodyDetectorConfig, compute_measurements_from_pose
from photo_indexer.database import PhotoIndexDB
from photo_indexer.faces import FaceDetector, FaceDetectionConfig, detect_faces_batch
from photo_indexer.metadata import write_xmp_sidecar
from photo_indexer.models import SidecarStatus
from photo_indexer.ner import extract_entities
from photo_indexer.performers import AggregationConfig, PerformerAggregator
from photo_indexer.vlm import VLMClient

from web.db import AsyncPhotoIndexDB, get_db
from web.redis import get_job_manager


class BaseWorker:
    """Base class for job workers."""

    def __init__(self, job_manager: JobManager, db: AsyncPhotoIndexDB):
        self.job_manager = job_manager
        self.db = db
        self._sync_db: Optional[PhotoIndexDB] = None

    @property
    def sync_db(self) -> PhotoIndexDB:
        """Lazy-initialized sync DB for CPU-heavy operations."""
        if self._sync_db is None:
            from web.config import get_settings
            settings = get_settings()
            self._sync_db = PhotoIndexDB(settings.database_path)
        return self._sync_db

    async def run(self, job_id: str) -> None:
        """Main entry point - override in subclasses."""
        raise NotImplementedError

    async def _run_in_thread(self, func, *args, **kwargs):
        """Run sync function in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)


class VLMAnalyzeWorker(BaseWorker):
    """VLM analysis worker using Ollama."""

    def __init__(self, job_manager: JobManager, db: AsyncPhotoIndexDB):
        super().__init__(job_manager, db)
        self.vlm_client: Optional[VLMClient] = None

    async def run(self, job_id: str) -> None:
        try:
            job = await self.job_manager.get_job(job_id)
            if not job:
                return

            payload = job["payload"]
            asset_id = payload["asset_id"]
            model = payload.get("model")
            prompt = payload.get("prompt")

            await self.job_manager.update_progress(job_id, 0.1, "Loading asset...")

            asset = await self.db.get_asset(asset_id)
            if not asset:
                await self.job_manager.fail_job(job_id, f"Asset not found: {asset_id}")
                return

            await self.job_manager.update_progress(job_id, 0.2, "Initializing VLM client...")

            if self.vlm_client is None:
                settings = self._get_settings()
                self.vlm_client = VLMClient(
                    provider="ollama",
                    model=model or settings.vlm_model,
                    base_url=settings.ollama_base_url,
                )

            await self.job_manager.update_progress(job_id, 0.4, "Analyzing image with VLM...")

            # Run VLM analysis in thread pool
            result = await self._run_in_thread(
                self.vlm_client.analyze,
                asset.path,
                prompt=prompt,
            )

            await self.job_manager.update_progress(job_id, 0.8, "Saving results...")

            # Update asset with VLM result
            asset.vlm = result
            asset.analyzed_at = datetime.now(timezone.utc)
            asset.status = "analyzed"
            self.sync_db.upsert(asset)

            await self.job_manager.complete_job(job_id, {
                "asset_id": asset_id,
                "caption": result.caption,
                "description": result.detailed_description,
                "model": result.model,
            })

        except Exception as e:
            await self.job_manager.fail_job(job_id, str(e))

    def _get_settings(self):
        from web.config import get_settings
        return get_settings()


class NEREnrichWorker(BaseWorker):
    """NER enrichment worker using taxonomies."""

    async def run(self, job_id: str) -> None:
        try:
            job = await self.job_manager.get_job(job_id)
            if not job:
                return

            asset_id = job["payload"]["asset_id"]

            await self.job_manager.update_progress(job_id, 0.2, "Loading asset...")

            asset = await self.db.get_asset(asset_id)
            if not asset:
                await self.job_manager.fail_job(job_id, f"Asset not found: {asset_id}")
                return

            if not asset.vlm or not asset.vlm.detailed_description:
                await self.job_manager.fail_job(job_id, "No VLM description to extract entities from")
                return

            await self.job_manager.update_progress(job_id, 0.5, "Extracting entities...")

            entities = extract_entities(asset.vlm.detailed_description)

            await self.job_manager.update_progress(job_id, 0.8, "Saving entities...")

            asset.ner_entities = entities
            asset.enriched_at = datetime.now(timezone.utc)
            asset.status = "enriched"
            self.sync_db.upsert(asset)

            await self.job_manager.complete_job(job_id, {
                "asset_id": asset_id,
                "entity_count": len(entities),
                "entities": [
                    {"text": e.text, "type": e.entity_type.value, "label": e.label, "confidence": e.confidence}
                    for e in entities
                ],
            })

        except Exception as e:
            await self.job_manager.fail_job(job_id, str(e))


class FaceDetectWorker(BaseWorker):
    """Face detection and embedding extraction worker."""

    def __init__(self, job_manager: JobManager, db: AsyncPhotoIndexDB):
        super().__init__(job_manager, db)
        self.face_detector: Optional[FaceDetector] = None

    async def run(self, job_id: str) -> None:
        try:
            job = await self.job_manager.get_job(job_id)
            if not job:
                return

            payload = job["payload"]
            asset_ids = payload.get("asset_ids")
            threshold = payload.get("threshold", 0.6)

            await self.job_manager.update_progress(job_id, 0.1, "Initializing face detector...")

            if self.face_detector is None:
                config = FaceDetectionConfig()
                self.face_detector = FaceDetector(config)

            if asset_ids:
                # Specific assets
                total = len(asset_ids)
                for i, asset_id in enumerate(asset_ids):
                    await self._process_asset(job_id, asset_id, threshold)
                    await self.job_manager.update_progress(
                        job_id,
                        (i + 1) / total,
                        f"Processed {i + 1}/{total} assets"
                    )
            else:
                # All assets without faces
                await self.job_manager.update_progress(job_id, 0.2, "Finding assets without faces...")
                assets = await self.db.list_assets(limit=10000, filters={"has_faces": False})
                total = len(assets)
                for i, asset in enumerate(assets):
                    await self._process_asset(job_id, asset.id, threshold)
                    await self.job_manager.update_progress(
                        job_id,
                        0.2 + 0.8 * (i + 1) / total if total > 0 else 1.0,
                        f"Processed {i + 1}/{total} assets"
                    )

            await self.job_manager.complete_job(job_id, {"processed": total})

        except Exception as e:
            await self.job_manager.fail_job(job_id, str(e))

    async def _process_asset(self, job_id: str, asset_id: str, threshold: float) -> None:
        asset = await self.db.get_asset(asset_id)
        if not asset:
            return

        try:
            # Detect faces
            faces = await self._run_in_thread(
                self.face_detector.detect_faces,
                asset.path,
            )

            if not faces:
                return

            # Convert to dict format for storage
            face_dicts = []
            for face in faces:
                face_dicts.append({
                    "id": str(face.id) if face.id else str(uuid.uuid4()),
                    "bbox": face.bbox,
                    "landmarks": face.landmarks,
                    "landmarks_468": face.landmarks_468,
                    "landmarks_3d": face.landmarks_3d,
                    "embedding": face.embedding,
                    "blendshapes": face.blendshapes,
                    "expression_vector": face.expression_vector,
                    "identity_vector": face.identity_vector,
                    "head_pose": face.head_pose,
                    "eye_gaze": face.eye_gaze,
                    "iris_landmarks": face.iris_landmarks,
                    "det_score": face.det_score,
                    "age": face.age,
                    "gender": face.gender,
                    "model_name": face.model_name,
                    "detected_at": face.detected_at.isoformat() if face.detected_at else datetime.now(timezone.utc).isoformat(),
                })

            # Update asset
            asset.face_embeddings = faces
            asset.status = "analyzed"
            self.sync_db.upsert(asset)

            # Try to match to known persons
            if face_dicts:
                matches = self.sync_db.match_faces_in_asset(asset_id, threshold)
                for match in matches:
                    self.sync_db.link_face_to_person(
                        match["face_id"],
                        match["person_id"],
                        match["similarity"],
                    )

        except Exception as e:
            # Log but continue with other assets
            print(f"Error processing faces for {asset_id}: {e}")


class BodyPoseWorker(BaseWorker):
    """Body pose detection and measurement worker."""

    def __init__(self, job_manager: JobManager, db: AsyncPhotoIndexDB):
        super().__init__(job_manager, db)
        self.body_detector: Optional[BodyDetector] = None

    async def run(self, job_id: str) -> None:
        try:
            job = await self.job_manager.get_job(job_id)
            if not job:
                return

            payload = job["payload"]
            asset_ids = payload.get("asset_ids")
            model_complexity = payload.get("model_complexity", 1)
            min_confidence = payload.get("min_confidence", 0.5)
            reference_height = payload.get("reference_height")

            await self.job_manager.update_progress(job_id, 0.1, "Initializing body detector...")

            if self.body_detector is None:
                config = BodyDetectorConfig(model_complexity=model_complexity)
                self.body_detector = BodyDetector(config)

            if asset_ids:
                total = len(asset_ids)
                for i, asset_id in enumerate(asset_ids):
                    await self._process_asset(job_id, asset_id, min_confidence, reference_height)
                    await self.job_manager.update_progress(
                        job_id,
                        (i + 1) / total,
                        f"Processed {i + 1}/{total} assets"
                    )
            else:
                await self.job_manager.update_progress(job_id, 0.2, "Finding assets without poses...")
                assets = await self.db.list_assets(limit=10000, filters={"has_measurements": False})
                total = len(assets)
                for i, asset in enumerate(assets):
                    await self._process_asset(job_id, asset.id, min_confidence, reference_height)
                    await self.job_manager.update_progress(
                        job_id,
                        0.2 + 0.8 * (i + 1) / total if total > 0 else 1.0,
                        f"Processed {i + 1}/{total} assets"
                    )

            await self.job_manager.complete_job(job_id, {"processed": total})

        except Exception as e:
            await self.job_manager.fail_job(job_id, str(e))

    async def _process_asset(self, job_id: str, asset_id: str, min_confidence: float, reference_height: Optional[float]) -> None:
        asset = await self.db.get_asset(asset_id)
        if not asset:
            return

        try:
            # Detect pose
            pose = await self._run_in_thread(
                self.body_detector.detect_pose,
                asset.path,
            )

            if not pose:
                return

            # Store pose
            pose_data = {
                "landmark_set": pose.landmark_set.value,
                "landmarks_2d": pose.landmarks_2d,
                "landmarks_3d": pose.landmarks_3d,
                "bbox": pose.bbox,
                "det_score": pose.det_score,
                "model_name": pose.model_name,
                "raw_response": pose.raw_response,
            }
            pose_id = self.sync_db.upsert_body_pose(asset_id, pose_data)

            # Compute measurements
            measurements = compute_measurements_from_pose(pose, reference_height)

            # Store measurements
            for meas in measurements:
                meas_dict = {
                    "asset_id": asset_id,
                    "measurement_type": meas.measurement_type.value,
                    "value": meas.value,
                    "unit": meas.unit,
                    "confidence": meas.confidence,
                    "method": meas.method,
                    "reference_scale": meas.reference_scale,
                    "landmarks_used": meas.landmarks_used,
                    "notes": meas.notes,
                }
                self.sync_db.upsert_body_measurement(meas_dict)

            # Update asset status
            asset.status = "analyzed"
            self.sync_db.upsert(asset)

        except Exception as e:
            print(f"Error processing body pose for {asset_id}: {e}")


class BuildProfileWorker(BaseWorker):
    """Build performer profile from multiple assets."""

    def __init__(self, job_manager: JobManager, db: AsyncPhotoIndexDB):
        super().__init__(job_manager, db)
        self.aggregator: Optional[PerformerAggregator] = None

    async def run(self, job_id: str) -> None:
        try:
            job = await self.job_manager.get_job(job_id)
            if not job:
                return

            payload = job["payload"]
            performer_id = payload["performer_id"]
            asset_ids = payload.get("asset_ids", [])
            reference_height = payload.get("reference_height")
            aliases = payload.get("aliases")
            store = payload.get("store", True)

            await self.job_manager.update_progress(job_id, 0.1, "Loading performer...")

            performer = await self.db.get_performer(performer_id)
            if not performer:
                await self.job_manager.fail_job(job_id, f"Performer not found: {performer_id}")
                return

            if not asset_ids:
                # Get all assets linked to this performer
                linked = await self.db.get_performer_assets(performer_id)
                asset_ids = [a["asset_id"] for a in linked]

            if not asset_ids:
                await self.job_manager.fail_job(job_id, "No assets to build profile from")
                return

            await self.job_manager.update_progress(job_id, 0.3, "Building profile...")

            if self.aggregator is None:
                self.aggregator = PerformerAggregator(self.sync_db)

            profile = await self._run_in_thread(
                self.aggregator.build_performer_profile,
                performer_id,
                asset_ids,
                aliases=aliases,
                reference_height=reference_height,
                store=store,
            )

            await self.job_manager.update_progress(job_id, 0.9, "Saving profile...")

            await self.job_manager.complete_job(job_id, {
                "performer_id": performer_id,
                "asset_count": len(asset_ids),
                "profile": profile.model_dump() if profile else None,
            })

        except Exception as e:
            await self.job_manager.fail_job(job_id, str(e))


class FacialBiometricsWorker(BaseWorker):
    """Build facial biometrics for a performer."""

    def __init__(self, job_manager: JobManager, db: AsyncPhotoIndexDB):
        super().__init__(job_manager, db)
        self.aggregator: Optional[PerformerAggregator] = None

    async def run(self, job_id: str) -> None:
        try:
            job = await self.job_manager.get_job(job_id)
            if not job:
                return

            payload = job["payload"]
            performer_id = payload["performer_id"]
            asset_ids = payload.get("asset_ids", [])
            store = payload.get("store", True)

            await self.job_manager.update_progress(job_id, 0.1, "Loading performer...")

            performer = await self.db.get_performer(performer_id)
            if not performer:
                await self.job_manager.fail_job(job_id, f"Performer not found: {performer_id}")
                return

            if not asset_ids:
                linked = await self.db.get_performer_assets(performer_id)
                asset_ids = [a["asset_id"] for a in linked]

            if not asset_ids:
                await self.job_manager.fail_job(job_id, "No assets for facial biometrics")
                return

            await self.job_manager.update_progress(job_id, 0.3, "Building facial biometrics...")

            if self.aggregator is None:
                self.aggregator = PerformerAggregator(self.sync_db)

            biometrics = await self._run_in_thread(
                self.aggregator.build_facial_biometrics,
                performer_id,
                asset_ids,
                store=store,
            )

            await self.job_manager.update_progress(job_id, 0.9, "Saving biometrics...")

            await self.job_manager.complete_job(job_id, {
                "performer_id": performer_id,
                "asset_count": len(asset_ids),
                "biometrics": biometrics.model_dump() if biometrics else None,
            })

        except Exception as e:
            await self.job_manager.fail_job(job_id, str(e))


class SidecarWriteWorker(BaseWorker):
    """Write XMP sidecar for an asset."""

    async def run(self, job_id: str) -> None:
        try:
            job = await self.job_manager.get_job(job_id)
            if not job:
                return

            payload = job["payload"]
            asset_id = payload["asset_id"]
            description = payload.get("description")
            keywords = payload.get("keywords", [])

            await self.job_manager.update_progress(job_id, 0.2, "Loading asset...")

            asset = await self.db.get_asset(asset_id)
            if not asset:
                await self.job_manager.fail_job(job_id, f"Asset not found: {asset_id}")
                return

            await self.job_manager.update_progress(job_id, 0.5, "Writing XMP sidecar...")

            status = await self._run_in_thread(
                write_xmp_sidecar,
                asset.path,
                description=description,
                keywords=keywords,
            )

            await self.job_manager.update_progress(job_id, 0.9, "Updating database...")

            asset.sidecar_status = status
            self.sync_db.upsert(asset)

            await self.job_manager.complete_job(job_id, {
                "asset_id": asset_id,
                "status": status.value,
            })

        except Exception as e:
            await self.job_manager.fail_job(job_id, str(e))


class BatchOperationWorker(BaseWorker):
    """Run batch operations on multiple assets."""

    def __init__(self, job_manager: JobManager, db: AsyncPhotoIndexDB):
        super().__init__(job_manager, db)
        self.vlm_worker = VLMAnalyzeWorker(job_manager, db)
        self.ner_worker = NEREnrichWorker(job_manager, db)
        self.face_worker = FaceDetectWorker(job_manager, db)
        self.body_worker = BodyPoseWorker(job_manager, db)
        self.sidecar_worker = SidecarWriteWorker(job_manager, db)

    async def run(self, job_id: str) -> None:
        try:
            job = await self.job_manager.get_job(job_id)
            if not job:
                return

            payload = job["payload"]
            asset_ids = payload["asset_ids"]
            operations = payload.get("operations", ["analyze"])

            total_steps = len(asset_ids) * len(operations)
            completed = 0

            results = {"vlm": 0, "ner": 0, "faces": 0, "body": 0, "sidecars": 0, "errors": []}

            for asset_id in asset_ids:
                for op in operations:
                    try:
                        if op == "analyze":
                            sub_job_id = await self.job_manager.create_job("vlm_analyze", {"asset_id": asset_id})
                            await self._wait_for_job(sub_job_id)
                            results["vlm"] += 1
                        elif op == "enrich":
                            sub_job_id = await self.job_manager.create_job("ner_enrich", {"asset_id": asset_id})
                            await self._wait_for_job(sub_job_id)
                            results["ner"] += 1
                        elif op == "faces":
                            sub_job_id = await self.job_manager.create_job("face_detect", {"asset_ids": [asset_id]})
                            await self._wait_for_job(sub_job_id)
                            results["faces"] += 1
                        elif op == "body-pose":
                            sub_job_id = await self.job_manager.create_job("body_pose", {"asset_ids": [asset_id]})
                            await self._wait_for_job(sub_job_id)
                            results["body"] += 1
                        elif op == "sidecar":
                            sub_job_id = await self.job_manager.create_job("sidecar_write", {"asset_id": asset_id})
                            await self._wait_for_job(sub_job_id)
                            results["sidecars"] += 1

                        completed += 1
                        await self.job_manager.update_progress(
                            job_id,
                            completed / total_steps,
                            f"Completed {completed}/{total_steps} operations"
                        )
                    except Exception as e:
                        results["errors"].append({"asset_id": asset_id, "operation": op, "error": str(e)})
                        completed += 1

            await self.job_manager.complete_job(job_id, results)

        except Exception as e:
            await self.job_manager.fail_job(job_id, str(e))

    async def _wait_for_job(self, job_id: str, timeout: int = 3600) -> None:
        """Wait for a sub-job to complete."""
        start = asyncio.get_event_loop().time()
        while True:
            job = await self.job_manager.get_job(job_id)
            if not job:
                raise Exception(f"Sub-job {job_id} not found")

            if job["status"] == JobStatus.COMPLETED:
                return
            elif job["status"] in (JobStatus.FAILED, JobStatus.CANCELLED):
                raise Exception(f"Sub-job failed: {job.get('error', 'Unknown error')}")

            if asyncio.get_event_loop().time() - start > timeout:
                raise Exception(f"Sub-job timeout: {job_id}")

            await asyncio.sleep(1)


class TagImportWorker(BaseWorker):
    """Import tags from TSV/CSV file."""

    async def run(self, job_id: str) -> None:
        try:
            job = await self.job_manager.get_job(job_id)
            if not job:
                return

            payload = job["payload"]
            file_path = payload["file_path"]

            await self.job_manager.update_progress(job_id, 0.2, "Reading file...")

            count = await self._run_in_thread(
                self.sync_db.import_tags_from_file,
                file_path,
            )

            await self.job_manager.complete_job(job_id, {
                "imported": count,
                "file": file_path,
            })

        except Exception as e:
            await self.job_manager.fail_job(job_id, str(e))


class BackupWorker(BaseWorker):
    """Run backup script."""

    async def run(self, job_id: str) -> None:
        try:
            job = await self.job_manager.get_job(job_id)
            if not job:
                return

            await self.job_manager.update_progress(job_id, 0.2, "Running backup...")

            # Run backup script
            script_path = Path("/app/scripts/backup.sh")
            if not script_path.exists():
                script_path = Path("/home/pat/photo-indexer/scripts/backup.sh")

            if not script_path.exists():
                await self.job_manager.fail_job(job_id, "Backup script not found")
                return

            result = await asyncio.create_subprocess_exec(
                str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                await self.job_manager.fail_job(job_id, f"Backup failed: {stderr.decode()}")
                return

            await self.job_manager.update_progress(job_id, 0.9, "Backup completed")

            # List backup files
            backup_dir = Path("/data/backups")
            if not backup_dir.exists():
                backup_dir = Path("/home/pat/photo-indexer/backups")

            backups = sorted(backup_dir.glob("*.tar.gz"), key=lambda p: p.stat().st_mtime, reverse=True)[:10]

            await self.job_manager.complete_job(job_id, {
                "output": stdout.decode(),
                "backups": [{"name": b.name, "size": b.stat().st_size, "mtime": b.stat().st_mtime} for b in backups],
            })

        except Exception as e:
            await self.job_manager.fail_job(job_id, str(e))


class SemanticAutoTagWorker(BaseWorker):
    """Auto-tag assets with semantic tags using VLM."""

    async def run(self, job_id: str) -> None:
        try:
            job = await self.job_manager.get_job(job_id)
            if not job:
                return

            payload = job["payload"]
            asset_ids = payload.get("asset_ids")
            namespaces = payload.get("namespaces")  # optional list of namespaces
            force = payload.get("force_recompute", False)

            await self.job_manager.update_progress(job_id, 0.1, "Initializing VLM client...")

            settings = self._get_settings()
            vlm_client = VLMClient(
                provider="ollama",
                model=settings.vlm_model,
                base_url=settings.ollama_base_url,
            )

            if asset_ids:
                assets_to_tag = [await self.db.get_asset(aid) for aid in asset_ids]
                assets_to_tag = [a for a in assets_to_tag if a is not None]
            else:
                assets_to_tag = await self.db.list_assets(limit=1000)

            total = len(assets_to_tag)
            tagged = 0

            for i, asset in enumerate(assets_to_tag):
                try:
                    await self._tag_asset(asset, vlm_client, namespaces, force)
                    tagged += 1
                except Exception as e:
                    print(f"Error tagging {asset.id}: {e}")

                await self.job_manager.update_progress(
                    job_id,
                    (i + 1) / total,
                    f"Tagged {tagged}/{total} assets"
                )

            await self.job_manager.complete_job(job_id, {
                "total": total,
                "tagged": tagged,
            })

        except Exception as e:
            await self.job_manager.fail_job(job_id, str(e))

    async def _tag_asset(self, asset: Asset, vlm_client: VLMClient, namespaces: Optional[List[str]], force: bool) -> None:
        """Tag a single asset with semantic tags."""
        # Check if already has semantic tags
        existing = await self.db.get_semantic_tags_for_asset(asset.id)
        if existing and not force:
            return

        # Get or generate VLM description
        if not asset.vlm or not asset.vlm.detailed_description:
            result = await asyncio.get_event_loop().run_in_executor(
                None, vlm_client.analyze, asset.path
            )
            asset.vlm = result
            asset.analyzed_at = datetime.now(timezone.utc)
            self.sync_db.upsert(asset)

        # Use VLM to generate structured semantic tags
        # This would use a specialized prompt for structured output
        # For now, extract from existing VLM + NER
        entities = extract_entities(asset.vlm.detailed_description)

        # Convert entities to semantic tags
        namespace_map = {
            "person": "content",
            "animal": "content",
            "object": "content",
            "place": "spatial",
            "location": "spatial",
        }

        for entity in entities:
            ns = namespace_map.get(entity.entity_type.value, "content")
            await self.db.add_semantic_tag(
                asset.id,
                namespace=ns,
                key=entity.text.lower().replace(" ", "_"),
                value=entity.label,
                confidence=entity.confidence or 0.8,
                source="vlm_ner",
                model_version=asset.vlm.model if asset.vlm else "unknown",
            )

        # Add temporal tags from EXIF
        if asset.exif and asset.exif.datetime_original:
            dt = asset.exif.datetime_original
            hour = dt.hour
            if 5 <= hour < 8:
                await self.db.add_semantic_tag(asset.id, "temporal", "golden_hour_morning", None, 0.9, "exif", "exif")
            elif 17 <= hour < 20:
                await self.db.add_semantic_tag(asset.id, "temporal", "golden_hour_evening", None, 0.9, "exif", "exif")
            elif 20 <= hour or hour < 5:
                await self.db.add_semantic_tag(asset.id, "temporal", "night", None, 0.8, "exif", "exif")

    def _get_settings(self):
        from web.config import get_settings
        return get_settings()


# Worker registry
WORKER_CLASSES = {
    "vlm_analyze": VLMAnalyzeWorker,
    "ner_enrich": NEREnrichWorker,
    "face_detect": FaceDetectWorker,
    "body_pose": BodyPoseWorker,
    "build_profile": BuildProfileWorker,
    "facial_biometrics": FacialBiometricsWorker,
    "sidecar_write": SidecarWriteWorker,
    "batch_operation": BatchOperationWorker,
    "tag_import": TagImportWorker,
    "backup": BackupWorker,
    "semantic_auto_tag": SemanticAutoTagWorker,
}


async def get_worker(job_type: str, job_manager: JobManager, db: AsyncPhotoIndexDB):
    """Get worker instance for job type."""
    worker_class = WORKER_CLASSES.get(job_type)
    if not worker_class:
        raise ValueError(f"No worker for job type: {job_type}")
    return worker_class(job_manager, db)