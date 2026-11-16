"""Database operations for photo enrichment records.

Supports SQLite (default, local-first) and PostgreSQL (via Immich instance).
Schema mirrors sql/enrich_schema.sql.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional, Union
from uuid import UUID

from .metadata import DBRecord, asset_to_record
from .models import Asset, AssetStatus, EntityType, NEREntity, SidecarStatus, VLMResult




def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO datetime string from the DB, returning None on failure."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None

class PhotoIndexDB:
    """SQLite-backed store for enrichment records keyed by asset."""

    def __init__(self, db_path: Union[str, Path] = "./photo_index.db") -> None:
        self.db_path = Path(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS enrich_assets (
                    asset_id      TEXT PRIMARY KEY,
                    content_sha256 TEXT,
                    file_path     TEXT NOT NULL,
                    file_name     TEXT NOT NULL,
                    file_size     INTEGER,
                    mime_type     TEXT,
                    status        TEXT DEFAULT 'pending',
                    indexed_at    TEXT,
                    analyzed_at   TEXT,
                    enriched_at   TEXT,
                    error_message TEXT
                );

                CREATE TABLE IF NOT EXISTS enrich_exif (
                    asset_id        TEXT PRIMARY KEY,
                    raw_json        TEXT NOT NULL,
                    structured_json TEXT,
                    FOREIGN KEY (asset_id) REFERENCES enrich_assets(asset_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS enrich_faces (
                    id            TEXT PRIMARY KEY,
                    asset_id      TEXT NOT NULL,
                    bbox_json     TEXT,
                    landmarks_json TEXT,
                    embedding_json TEXT NOT NULL,
                    det_score     REAL,
                    age           INTEGER,
                    gender        INTEGER,
                    model_name    TEXT,
                    detected_at   TEXT,
                    FOREIGN KEY (asset_id) REFERENCES enrich_assets(asset_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_enrich_faces_asset
                    ON enrich_faces(asset_id);

                CREATE TABLE IF NOT EXISTS enrich_vlm (
                    asset_id        TEXT PRIMARY KEY,
                    model           TEXT,
                    caption         TEXT,
                    description     TEXT,
                    raw_response    TEXT,
                    processing_ms   INTEGER,
                    analyzed_at     TEXT,
                    FOREIGN KEY (asset_id) REFERENCES enrich_assets(asset_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS enrich_entities (
                    id            TEXT PRIMARY KEY,
                    asset_id      TEXT NOT NULL,
                    text          TEXT NOT NULL,
                    entity_type   TEXT NOT NULL,
                    label         TEXT,
                    confidence    REAL,
                    bbox_json     TEXT,
                    source        TEXT DEFAULT 'vlm',
                    start_char    INTEGER,
                    end_char      INTEGER,
                    FOREIGN KEY (asset_id) REFERENCES enrich_assets(asset_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS enrich_sidecars (
                    asset_id      TEXT PRIMARY KEY,
                    sidecar_path  TEXT,
                    status        TEXT,
                    FOREIGN KEY (asset_id) REFERENCES enrich_assets(asset_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_enrich_entities_asset
                    ON enrich_entities(asset_id);
                CREATE INDEX IF NOT EXISTS idx_enrich_entities_type
                    ON enrich_entities(entity_type);
                CREATE INDEX IF NOT EXISTS idx_enrich_assets_path
                    ON enrich_assets(file_path);
                CREATE INDEX IF NOT EXISTS idx_enrich_assets_sha
                    ON enrich_assets(content_sha256);
                """
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def upsert(self, asset: Asset) -> None:
        """Insert or update an asset and all related enrichment tables."""
        record = asset_to_record(asset)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO enrich_assets
                (asset_id, content_sha256, file_path, file_name, file_size, mime_type,
                 status, indexed_at, analyzed_at, enriched_at, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(asset_id) DO UPDATE SET
                  content_sha256=excluded.content_sha256,
                  file_path=excluded.file_path,
                  file_name=excluded.file_name,
                  file_size=excluded.file_size,
                  mime_type=excluded.mime_type,
                  status=excluded.status,
                  indexed_at=excluded.indexed_at,
                  analyzed_at=excluded.analyzed_at,
                  enriched_at=excluded.enriched_at,
                  error_message=excluded.error_message
                """,
                (
                    record.asset_id, record.content_sha256, record.file_path,
                    record.file_name, record.file_size, record.mime_type,
                    asset.status.value if hasattr(asset.status, "value") else asset.status,
                    asset.indexed_at.isoformat() if asset.indexed_at else None,
                    asset.analyzed_at.isoformat() if asset.analyzed_at else None,
                    record.enriched_at, asset.error_message,
                ),
            )
            self._upsert_exif(conn, record)
            self._upsert_vlm(conn, record)
            self._upsert_entities(conn, record)
            self._upsert_faces(conn, record)
            self._upsert_sidecar(conn, record)

    def _upsert_exif(self, conn: sqlite3.Connection, record: DBRecord) -> None:
        conn.execute(
            """
            INSERT INTO enrich_exif (asset_id, raw_json, structured_json)
            VALUES (?, ?, ?)
            ON CONFLICT(asset_id) DO UPDATE SET
                raw_json=excluded.raw_json,
                structured_json=excluded.structured_json
            """,
            (record.asset_id, json.dumps(record.exif_raw), json.dumps(record.exif_structured)),
        )

    def _upsert_vlm(self, conn: sqlite3.Connection, record: DBRecord) -> None:
        if not record.vlm_caption and not record.vlm_description:
            return
        conn.execute(
            """
            INSERT INTO enrich_vlm
            (asset_id, model, caption, description, analyzed_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(asset_id) DO UPDATE SET
              model=excluded.model,
              caption=excluded.caption,
              description=excluded.description,
              analyzed_at=excluded.analyzed_at
            """,
            (
                record.asset_id, record.vlm_model, record.vlm_caption,
                record.vlm_description, record.enriched_at,
            ),
        )

    def _upsert_entities(self, conn: sqlite3.Connection, record: DBRecord) -> None:
        conn.execute("DELETE FROM enrich_entities WHERE asset_id = ?", (record.asset_id,))
        try:
            entities = json.loads(record.ner_entities_json)
        except (json.JSONDecodeError, TypeError):
            entities = []
        for e in entities:
            conn.execute(
                """
                INSERT INTO enrich_entities
                (id, asset_id, text, entity_type, label, confidence, bbox_json, source, start_char, end_char)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(e.get("id", "")),
                    record.asset_id,
                    e.get("text", ""),
                    e.get("entity_type", ""),
                    e.get("label"),
                    e.get("confidence"),
                    json.dumps(e.get("bbox")) if e.get("bbox") else None,
                    e.get("source", "vlm"),
                    e.get("start_char"),
                    e.get("end_char"),
                ),
            )
    def _upsert_faces(self, conn: sqlite3.Connection, record: DBRecord) -> None:
        conn.execute("DELETE FROM enrich_faces WHERE asset_id = ?", (record.asset_id,))
        try:
            faces = json.loads(record.face_embeddings_json)
        except (json.JSONDecodeError, TypeError):
            faces = []
        for f in faces:
            conn.execute(
                """
                INSERT INTO enrich_faces
                (id, asset_id, bbox_json, landmarks_json, embedding_json, det_score, age, gender, model_name, detected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(f.get("id", "")),
                    record.asset_id,
                    json.dumps(f.get("bbox")) if f.get("bbox") else None,
                    json.dumps(f.get("landmarks")) if f.get("landmarks") else None,
                    json.dumps(f.get("embedding")) if f.get("embedding") else None,
                    f.get("det_score"),
                    f.get("age"),
                    f.get("gender"),
                    f.get("model_name", "buffalo_l"),
                    f.get("detected_at"),
                ),
            )


    def _upsert_sidecar(self, conn: sqlite3.Connection, record: DBRecord) -> None:
        conn.execute(
            """
            INSERT INTO enrich_sidecars (asset_id, sidecar_path, status)
            VALUES (?, ?, ?)
            ON CONFLICT(asset_id) DO UPDATE SET
              sidecar_path=excluded.sidecar_path,
              status=excluded.status
            """,
            (record.asset_id, None, record.sidecar_status),
        )

    def get(self, asset_id: str) -> Optional[Asset]:
        """Retrieve an asset by ID with all enrichment data."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM enrich_assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()
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
                indexed_at=_parse_dt(row["indexed_at"]),
                analyzed_at=_parse_dt(row["analyzed_at"]),
                enriched_at=_parse_dt(row["enriched_at"]),
                error_message=row["error_message"],
            )
            # Load exif
            exif_row = conn.execute(
                "SELECT raw_json, structured_json FROM enrich_exif WHERE asset_id = ?", (asset_id,)
            ).fetchone()
            if exif_row:
                raw = json.loads(exif_row["raw_json"]) if exif_row["raw_json"] else {}
                structured = json.loads(exif_row["structured_json"]) if exif_row["structured_json"] else {}
                from .models import ExifMetadata
                if structured:
                    asset.exif = ExifMetadata(**structured, raw=raw)
                else:
                    asset.exif = ExifMetadata(raw=raw)

            # Load vlm
            vlm_row = conn.execute(
                "SELECT model, caption, description FROM enrich_vlm WHERE asset_id = ?",
                (asset_id,),
            ).fetchone()
            if vlm_row:
                asset.vlm = VLMResult(
                    model=vlm_row["model"],
                    caption=vlm_row["caption"],
                    detailed_description=vlm_row["description"],
                )

            # Load entities
            ent_rows = conn.execute(
                "SELECT * FROM enrich_entities WHERE asset_id = ?", (asset_id,)
            ).fetchall()
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
            face_rows = conn.execute(
                "SELECT * FROM enrich_faces WHERE asset_id = ?", (asset_id,)
            ).fetchall()
            for fr in face_rows:
                from .models import FaceEmbedding
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
                        detected_at=_parse_dt(fr["detected_at"]),
                    )
                )

            # Load sidecar
            sc_row = conn.execute(
                "SELECT status FROM enrich_sidecars WHERE asset_id = ?", (asset_id,)
            ).fetchone()
            if sc_row and sc_row["status"]:
                asset.sidecar_status = SidecarStatus(sc_row["status"])

            return asset

    def list_all(self, limit: int = 100, offset: int = 0) -> List[Asset]:
        """List all indexed assets."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM enrich_assets LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
            return [self._row_to_asset(r, conn) for r in rows]

    def _row_to_asset(self, row: sqlite3.Row, conn: sqlite3.Connection) -> Asset:
        asset = Asset(
            id=UUID(row["asset_id"]) if row["asset_id"] else None,
            path=row["file_path"],
            filename=row["file_name"],
            content_sha256=row["content_sha256"],
            file_size=row["file_size"],
            mime_type=row["mime_type"],
            status=AssetStatus(row["status"]) if row["status"] else AssetStatus.PENDING,
            indexed_at=_parse_dt(row["indexed_at"]),
            analyzed_at=_parse_dt(row["analyzed_at"]),
            enriched_at=_parse_dt(row["enriched_at"]),
            error_message=row["error_message"],
        )
        # Load exif
        exif_row = conn.execute(
            "SELECT raw_json, structured_json FROM enrich_exif WHERE asset_id = ?", (row["asset_id"],)
        ).fetchone()
        if exif_row:
            raw = json.loads(exif_row["raw_json"]) if exif_row["raw_json"] else {}
            structured = json.loads(exif_row["structured_json"]) if exif_row["structured_json"] else {}
            from .models import ExifMetadata
            if structured:
                asset.exif = ExifMetadata(**structured, raw=raw)
            else:
                asset.exif = ExifMetadata(raw=raw)

        # Load vlm
        vlm_row = conn.execute(
            "SELECT model, caption, description FROM enrich_vlm WHERE asset_id = ?",
            (row["asset_id"],),
        ).fetchone()
        if vlm_row:
            asset.vlm = VLMResult(
                model=vlm_row["model"],
                caption=vlm_row["caption"],
                detailed_description=vlm_row["description"],
            )

        # Load entities
        ent_rows = conn.execute(
            "SELECT * FROM enrich_entities WHERE asset_id = ?", (row["asset_id"],)
        ).fetchall()
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
        face_rows = conn.execute(
            "SELECT * FROM enrich_faces WHERE asset_id = ?", (row["asset_id"],)
        ).fetchall()
        for fr in face_rows:
            from .models import FaceEmbedding
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
                    detected_at=_parse_dt(fr["detected_at"]),
                )
            )

        # Load sidecar
        sc_row = conn.execute(
            "SELECT status FROM enrich_sidecars WHERE asset_id = ?", (row["asset_id"],)
        ).fetchone()
        if sc_row and sc_row["status"]:
            asset.sidecar_status = SidecarStatus(sc_row["status"])

        return asset

    def count(self) -> int:
        """Return total number of indexed assets."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM enrich_assets").fetchone()
            return row["c"] if row else 0

    def delete(self, asset_id: str) -> bool:
        """Delete an asset and all related enrichment data."""
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM enrich_assets WHERE asset_id = ?", (asset_id,)
            )
            return cur.rowcount > 0
