import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from uuid import UUID

import aiosqlite
from aiosqlite import Connection

from photo_indexer.database import PhotoIndexDB
from photo_indexer.models import (
    Asset,
    AssetStatus,
    BodyMeasurement,
    BodyPose,
    EntityType,
    FaceEmbedding,
    FacialBiometrics,
    NEREntity,
    PerformerProfile,
    SidecarStatus,
    VLMResult,
)


class AsyncPhotoIndexDB:
    """Async wrapper around SQLite for web concurrent access."""

    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path = Path(db_path)
        self._pool: List[Connection] = []
        self._pool_size = 10
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize connection pool and run migrations."""
        if self._initialized:
            return

        # Create initial connections
        for _ in range(self._pool_size):
            conn = await aiosqlite.connect(str(self.db_path))
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA foreign_keys=ON")
            await conn.execute("PRAGMA busy_timeout=5000")
            self._pool.append(conn)

        self._initialized = True

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[Connection]:
        """Get a connection from the pool."""
        if not self._initialized:
            await self.initialize()

        conn = self._pool.pop()
        try:
            yield conn
        finally:
            self._pool.append(conn)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[Connection]:
        """Transaction context manager."""
        async with self.connection() as conn:
            await conn.execute("BEGIN")
            try:
                yield conn
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

    async def close(self) -> None:
        """Close all connections."""
        for conn in self._pool:
            await conn.close()
        self._pool.clear()
        self._initialized = False

    # --- Async read methods ---

    async def get_asset(self, asset_id: str) -> Optional[Asset]:
        """Retrieve an asset by ID with all enrichment data."""
        async with self.connection() as conn:
            return await self._get_asset(conn, asset_id)

    async def _get_asset(self, conn: Connection, asset_id: str) -> Optional[Asset]:
        row = await conn.execute(
            "SELECT * FROM enrich_assets WHERE asset_id = ?", (asset_id,)
        )
        row = await row.fetchone()
        if not row:
            return None

        asset = Asset(
            id=UUID(row["asset_id"]) if row["asset_id"] else None,
            path=row["file_path"],
            filename=row["file_name"],
            content_sha256=row["content_sha256"],
            file_size=row["file_size"],
            mime_type=row["mime_type"],
            status=AssetStatus(row["status"]) if row["status"] else AssetStatus.PENDING,
            indexed_at=self._parse_dt(row["indexed_at"]),
            analyzed_at=self._parse_dt(row["analyzed_at"]),
            enriched_at=self._parse_dt(row["enriched_at"]),
            error_message=row["error_message"],
        )

        # Load EXIF
        exif_row = await conn.execute(
            "SELECT raw_json, structured_json FROM enrich_exif WHERE asset_id = ?", (asset_id,)
        )
        exif_row = await exif_row.fetchone()
        if exif_row:
            from photo_indexer.models import ExifMetadata

            raw = json.loads(exif_row["raw_json"]) if exif_row["raw_json"] else {}
            structured = json.loads(exif_row["structured_json"]) if exif_row["structured_json"] else {}
            if structured:
                asset.exif = ExifMetadata(**structured, raw=raw)
            else:
                asset.exif = ExifMetadata(raw=raw)

        # Load VLM
        vlm_row = await conn.execute(
            "SELECT model, caption, description FROM enrich_vlm WHERE asset_id = ?",
            (asset_id,),
        )
        vlm_row = await vlm_row.fetchone()
        if vlm_row:
            asset.vlm = VLMResult(
                model=vlm_row["model"],
                caption=vlm_row["caption"],
                detailed_description=vlm_row["description"],
            )

        # Load entities
        ent_rows = await conn.execute(
            "SELECT * FROM enrich_entities WHERE asset_id = ?", (asset_id,)
        )
        ent_rows = await ent_rows.fetchall()
        for er in ent_rows:
            asset.ner_entities.append(
                NEREntity(
                    text=er["text"],
                    entity_type=EntityType(er["entity_type"]),
                    label=er["label"],
                    confidence=er["confidence"],
                    source=er["source"],
                )
            )

        # Load faces
        face_rows = await conn.execute(
            "SELECT * FROM enrich_faces WHERE asset_id = ?", (asset_id,)
        )
        face_rows = await face_rows.fetchall()
        for fr in face_rows:
            asset.face_embeddings.append(
                FaceEmbedding(
                    id=UUID(fr["id"]) if fr["id"] else None,
                    asset_id=UUID(fr["asset_id"]) if fr["asset_id"] else None,
                    bbox=json.loads(fr["bbox_json"]) if fr["bbox_json"] else None,
                    landmarks=json.loads(fr["landmarks_json"]) if fr["landmarks_json"] else None,
                    embedding=json.loads(fr["embedding_json"]) if fr["embedding_json"] else None,
                    det_score=fr["det_score"],
                    age=fr["age"],
                    gender=fr["gender"],
                    model_name=fr["model_name"],
                    detected_at=self._parse_dt(fr["detected_at"]),
                )
            )

        # Load tags
        tag_rows = await conn.execute(
            "SELECT * FROM enrich_tags WHERE asset_id = ?", (asset_id,)
        )
        tag_rows = await tag_rows.fetchall()
        for tr in tag_rows:
            asset.ner_entities.append(
                NEREntity(
                    id=UUID(tr["id"]) if tr["id"] else None,
                    text=tr["tag_name"],
                    entity_type=EntityType.FACE,
                    label=tr["tag_name"],
                    confidence=1.0,
                    source=tr["source"],
                )
            )

        # Load sidecar
        sc_row = await conn.execute(
            "SELECT status FROM enrich_sidecars WHERE asset_id = ?", (asset_id,)
        )
        sc_row = await sc_row.fetchone()
        if sc_row and sc_row["status"]:
            asset.sidecar_status = SidecarStatus(sc_row["status"])

        return asset

    async def list_assets(
        self,
        limit: int = 50,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Asset]:
        """List assets with optional filters."""
        query = "SELECT * FROM enrich_assets WHERE 1=1"
        params = []

        if filters:
            if filters.get("status"):
                query += " AND status = ?"
                params.append(filters["status"])
            if filters.get("has_faces"):
                query += " AND asset_id IN (SELECT asset_id FROM enrich_faces)"
            if filters.get("has_measurements"):
                query += " AND asset_id IN (SELECT asset_id FROM enrich_body_measurements)"
            if filters.get("performer_id"):
                query += " AND asset_id IN (SELECT asset_id FROM enrich_performer_assets WHERE performer_id = ?)"
                params.append(filters["performer_id"])
            if filters.get("date_from"):
                query += " AND enriched_at >= ?"
                params.append(filters["date_from"])
            if filters.get("date_to"):
                query += " AND enriched_at <= ?"
                params.append(filters["date_to"])

        query += " ORDER BY enriched_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        async with self.connection() as conn:
            rows = await conn.execute(query, params)
            rows = await rows.fetchall()
            return [await self._get_asset(conn, row["asset_id"]) for row in rows]

    async def count_assets(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count assets matching filters."""
        query = "SELECT COUNT(*) FROM enrich_assets WHERE 1=1"
        params = []

        if filters:
            if filters.get("status"):
                query += " AND status = ?"
                params.append(filters["status"])
            if filters.get("has_faces"):
                query += " AND asset_id IN (SELECT asset_id FROM enrich_faces)"
            if filters.get("has_measurements"):
                query += " AND asset_id IN (SELECT asset_id FROM enrich_body_measurements)"
            if filters.get("performer_id"):
                query += " AND asset_id IN (SELECT asset_id FROM enrich_performer_assets WHERE performer_id = ?)"
                params.append(filters["performer_id"])

        async with self.connection() as conn:
            row = await conn.execute(query, params)
            row = await row.fetchone()
            return row[0] if row else 0

    # --- Performers ---

    async def list_performers(
        self, limit: int = 50, offset: int = 0, search: Optional[str] = None
    ) -> List[Dict]:
        query = "SELECT * FROM enrich_performers WHERE 1=1"
        params = []
        if search:
            query += " AND name LIKE ?"
            params.append(f"%{search}%")
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        async with self.connection() as conn:
            rows = await conn.execute(query, params)
            rows = await rows.fetchall()
            return [dict(row) for row in rows]

    async def get_performer(self, performer_id: str) -> Optional[Dict]:
        async with self.connection() as conn:
            row = await conn.execute(
                "SELECT * FROM enrich_performers WHERE id = ?", (performer_id,)
            )
            row = await row.fetchone()
            return dict(row) if row else None

    async def get_performer_by_name(self, name: str) -> Optional[Dict]:
        async with self.connection() as conn:
            row = await conn.execute(
                "SELECT * FROM enrich_performers WHERE name = ?", (name,)
            )
            row = await row.fetchone()
            return dict(row) if row else None

    async def create_performer(
        self,
        name: str,
        aliases: List[str] = None,
        face_embedding: List[float] = None,
        face_embedding_quality: float = 0.0,
        notes: str = None,
    ) -> str:
        performer_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        aliases_json = json.dumps(aliases or [])
        face_embedding_json = json.dumps(face_embedding) if face_embedding else None

        async with self.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO enrich_performers
                (id, name, aliases_json, face_embedding_json, face_embedding_quality,
                 created_at, updated_at, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    performer_id,
                    name,
                    aliases_json,
                    face_embedding_json,
                    face_embedding_quality,
                    now,
                    now,
                    notes,
                ),
            )
        return performer_id

    async def update_performer(self, performer_id: str, updates: Dict) -> bool:
        if not updates:
            return False
        updates["updated_at"] = datetime.utcnow().isoformat()
        set_clause = ", ".join(f"{k}=?" for k in updates.keys())
        values = list(updates.values()) + [performer_id]

        async with self.transaction() as conn:
            cursor = await conn.execute(
                f"UPDATE enrich_performers SET {set_clause} WHERE id = ?", values
            )
            return cursor.rowcount > 0

    async def delete_performer(self, performer_id: str) -> bool:
        async with self.transaction() as conn:
            cursor = await conn.execute(
                "DELETE FROM enrich_performers WHERE id = ?", (performer_id,)
            )
            return cursor.rowcount > 0

    # --- Faces ---

    async def get_unmatched_faces(
        self, limit: int = 50, offset: int = 0
    ) -> List[Dict]:
        async with self.connection() as conn:
            rows = await conn.execute(
                """
                SELECT f.*, a.file_path, a.filename
                FROM enrich_faces f
                JOIN enrich_assets a ON f.asset_id = a.asset_id
                WHERE f.id NOT IN (SELECT face_id FROM enrich_face_matches)
                ORDER BY f.detected_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
            rows = await rows.fetchall()
            return [dict(row) for row in rows]

    async def get_faces_for_asset(self, asset_id: str) -> List[Dict]:
        async with self.connection() as conn:
            rows = await conn.execute(
                """
                SELECT f.*, m.person_id, m.similarity, m.matched_at, p.name as person_name
                FROM enrich_faces f
                LEFT JOIN enrich_face_matches m ON f.id = m.face_id
                LEFT JOIN enrich_persons p ON m.person_id = p.id
                WHERE f.asset_id = ?
                ORDER BY f.det_score DESC
                """,
                (asset_id,),
            )
            rows = await rows.fetchall()
            return [dict(row) for row in rows]

    async def match_face_to_person(
        self, face_id: str, person_id: str, similarity: float, method: str = "manual"
    ) -> None:
        match_id = str(uuid.uuid4())
        matched_at = datetime.utcnow().isoformat()
        async with self.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO enrich_face_matches (id, face_id, person_id, similarity, matched_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (match_id, face_id, person_id, similarity, matched_at),
            )

    async def unmatch_face(self, face_id: str) -> bool:
        async with self.transaction() as conn:
            cursor = await conn.execute(
                "DELETE FROM enrich_face_matches WHERE face_id = ?", (face_id,)
            )
            return cursor.rowcount > 0

    # --- Measurements ---

    async def get_measurements(
        self,
        asset_id: Optional[str] = None,
        performer_id: Optional[str] = None,
        measurement_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict]:
        query = "SELECT * FROM enrich_body_measurements WHERE 1=1"
        params = []
        if asset_id:
            query += " AND asset_id = ?"
            params.append(asset_id)
        if performer_id:
            query += " AND performer_id = ?"
            params.append(performer_id)
        if measurement_type:
            query += " AND measurement_type = ?"
            params.append(measurement_type)
        query += " ORDER BY computed_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        async with self.connection() as conn:
            rows = await conn.execute(query, params)
            rows = await rows.fetchall()
            results = []
            for row in rows:
                r = dict(row)
                if r.get("landmarks_used_json"):
                    r["landmarks_used_json"] = json.loads(r["landmarks_used_json"])
                results.append(r)
            return results

    async def get_measurement_summary(self) -> Dict[str, Dict]:
        """Get aggregated stats per measurement type."""
        async with self.connection() as conn:
            rows = await conn.execute(
                """
                SELECT
                    measurement_type,
                    COUNT(*) as count,
                    AVG(value) as mean,
                    MIN(value) as min_val,
                    MAX(value) as max_val,
                    SQRT(AVG(value * value) - AVG(value) * AVG(value)) as std_dev
                FROM enrich_body_measurements
                GROUP BY measurement_type
                """
            )
            rows = await rows.fetchall()
            return {
                row["measurement_type"]: {
                    "count": row["count"],
                    "mean": row["mean"],
                    "min": row["min_val"],
                    "max": row["max_val"],
                    "std": row["std_dev"],
                }
                for row in rows
            }

    # --- Tags ---

    async def get_all_tags(self) -> List[Dict]:
        async with self.connection() as conn:
            rows = await conn.execute(
                """
                SELECT tag_name, COUNT(DISTINCT asset_id) as count
                FROM enrich_tags
                GROUP BY tag_name
                ORDER BY count DESC
                """
            )
            rows = await rows.fetchall()
            return [dict(row) for row in rows]

    async def get_asset_tags(self, asset_id: str) -> List[str]:
        async with self.connection() as conn:
            rows = await conn.execute(
                "SELECT tag_name FROM enrich_tags WHERE asset_id = ? ORDER BY tag_name",
                (asset_id,),
            )
            rows = await rows.fetchall()
            return [row["tag_name"] for row in rows]

    async def add_tag(self, asset_id: str, tag_name: str, source: str = "manual") -> bool:
        async with self.transaction() as conn:
            existing = await conn.execute(
                "SELECT 1 FROM enrich_tags WHERE asset_id = ? AND tag_name = ?",
                (asset_id, tag_name),
            )
            existing = await existing.fetchone()
            if existing:
                return False

            tag_id = str(uuid.uuid4())
            await conn.execute(
                """
                INSERT INTO enrich_tags (id, asset_id, tag_name, source)
                VALUES (?, ?, ?, ?)
                """,
                (tag_id, asset_id, tag_name, source),
            )
            return True

    async def remove_tag(self, asset_id: str, tag_name: str) -> bool:
        async with self.transaction() as conn:
            cursor = await conn.execute(
                "DELETE FROM enrich_tags WHERE asset_id = ? AND tag_name = ?",
                (asset_id, tag_name),
            )
            return cursor.rowcount > 0

    # --- Semantic Tags ---

    async def init_semantic_tags_table(self) -> None:
        """Create semantic tags tables if not exist."""
        async with self.transaction() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS enrich_semantic_tags (
                    id TEXT PRIMARY KEY,
                    asset_id TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT,
                    confidence REAL DEFAULT 1.0,
                    source TEXT NOT NULL,
                    model_version TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (asset_id) REFERENCES enrich_assets(asset_id) ON DELETE CASCADE
                )
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_semantic_tags_asset_ns
                ON enrich_semantic_tags(asset_id, namespace)
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_semantic_tags_ns_key
                ON enrich_semantic_tags(namespace, key)
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_semantic_tags_confidence
                ON enrich_semantic_tags(confidence)
                """
            )
            # Tag relations table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS enrich_tag_relations (
                    from_namespace TEXT,
                    from_key TEXT,
                    to_namespace TEXT,
                    to_key TEXT,
                    relation_type TEXT,
                    weight REAL DEFAULT 1.0,
                    PRIMARY KEY (from_namespace, from_key, to_namespace, to_key, relation_type)
                )
                """
            )

    async def get_semantic_tags_for_asset(self, asset_id: str) -> Dict[str, List[Dict]]:
        """Get all semantic tags for an asset, grouped by namespace."""
        async with self.connection() as conn:
            rows = await conn.execute(
                """
                SELECT * FROM enrich_semantic_tags
                WHERE asset_id = ?
                ORDER BY namespace, key, confidence DESC
                """,
                (asset_id,),
            )
            rows = await rows.fetchall()
            result = {}
            for row in rows:
                ns = row["namespace"]
                if ns not in result:
                    result[ns] = []
                result[ns].append(dict(row))
            return result

    async def get_namespace_keys(self, namespace: str) -> List[Dict]:
        """Get all keys in a namespace with counts."""
        async with self.connection() as conn:
            rows = await conn.execute(
                """
                SELECT key, COUNT(DISTINCT asset_id) as count, AVG(confidence) as avg_confidence
                FROM enrich_semantic_tags
                WHERE namespace = ?
                GROUP BY key
                ORDER BY count DESC
                """,
                (namespace,),
            )
            rows = await rows.fetchall()
            return [dict(row) for row in rows]

    async def search_assets_by_semantic_tags(
        self,
        filters: List[Dict],  # [{"namespace": "...", "key": "...", "op": "=", "value": "..."}]
        logic: str = "AND",
        limit: int = 50,
        offset: int = 0,
    ) -> List[str]:
        """Search assets by semantic tag criteria. Returns asset_ids."""
        if not filters:
            return []

        conditions = []
        params = []
        for f in filters:
            ns = f["namespace"]
            key = f["key"]
            op = f.get("op", "=")
            val = f.get("value")
            conf = f.get("confidence", 0.0)

            if op == "=" and val is not None:
                conditions.append(
                    f"EXISTS (SELECT 1 FROM enrich_semantic_tags st WHERE st.asset_id = enrich_assets.asset_id AND st.namespace = ? AND st.key = ? AND st.value = ? AND st.confidence >= ?)"
                )
                params.extend([ns, key, val, conf])
            elif op == "exists":
                conditions.append(
                    f"EXISTS (SELECT 1 FROM enrich_semantic_tags st WHERE st.asset_id = enrich_assets.asset_id AND st.namespace = ? AND st.key = ? AND st.confidence >= ?)"
                )
                params.extend([ns, key, conf])

        logic_op = " AND " if logic == "AND" else " OR "
        where_clause = logic_op.join(conditions)

        query = f"""
            SELECT DISTINCT asset_id FROM enrich_assets
            WHERE {where_clause}
            ORDER BY enriched_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        async with self.connection() as conn:
            rows = await conn.execute(query, params)
            rows = await rows.fetchall()
            return [row["asset_id"] for row in rows]

    # --- System / Stats ---

    async def get_stats(self) -> Dict:
        """Get dashboard statistics."""
        async with self.connection() as conn:
            # Asset counts by status
            status_rows = await conn.execute(
                "SELECT status, COUNT(*) as count FROM enrich_assets GROUP BY status"
            )
            status_rows = await status_rows.fetchall()
            by_status = {row["status"]: row["count"] for row in status_rows}

            # Total assets
            total_row = await conn.execute("SELECT COUNT(*) as count FROM enrich_assets")
            total_row = await total_row.fetchone()
            total = total_row["count"] if total_row else 0

            # Assets with faces
            faces_row = await conn.execute(
                "SELECT COUNT(DISTINCT asset_id) as count FROM enrich_faces"
            )
            faces_row = await faces_row.fetchone()
            with_faces = faces_row["count"] if faces_row else 0

            # Assets with measurements
            meas_row = await conn.execute(
                "SELECT COUNT(DISTINCT asset_id) as count FROM enrich_body_measurements"
            )
            meas_row = await meas_row.fetchone()
            with_measurements = meas_row["count"] if meas_row else 0

            # Performers
            perf_row = await conn.execute("SELECT COUNT(*) as count FROM enrich_performers")
            perf_row = await perf_row.fetchone()
            performers = perf_row["count"] if perf_row else 0

            # DB size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

            return {
                "total_assets": total,
                "by_status": by_status,
                "with_faces": with_faces,
                "with_measurements": with_measurements,
                "performers": performers,
                "db_size_bytes": db_size,
            }

    def _parse_dt(self, value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return None


# Global instance (initialized on startup)
_db_instance: Optional[AsyncPhotoIndexDB] = None


async def get_db() -> AsyncPhotoIndexDB:
    """FastAPI dependency for database access."""
    global _db_instance
    if _db_instance is None:
        from web.config import get_settings

        settings = get_settings()
        _db_instance = AsyncPhotoIndexDB(settings.database_path)
        await _db_instance.initialize()
        await _db_instance.init_semantic_tags_table()
    return _db_instance


async def close_db() -> None:
    """Close database connections."""
    global _db_instance
    if _db_instance:
        await _db_instance.close()
        _db_instance = None