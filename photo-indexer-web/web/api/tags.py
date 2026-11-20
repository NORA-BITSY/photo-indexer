from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from web.auth import get_current_user
from web.db import AsyncPhotoIndexDB, get_db
from web.jobs.manager import get_job_manager


router = APIRouter(prefix="/api/v1/tags", tags=["tags"])


class TagAddRequest(BaseModel):
    tags: List[str]
    source: str = "manual"


class TagMergeRequest(BaseModel):
    from_tags: List[str]
    to_tag: str


class TagAliasRequest(BaseModel):
    from_tag: str
    to_tag: str


class SemanticTagAddRequest(BaseModel):
    namespace: str
    key: str
    value: Optional[str] = None
    confidence: float = 1.0
    source: str = "manual"
    model_version: Optional[str] = None


class SemanticTagSearchRequest(BaseModel):
    filters: List[Dict]  # [{"namespace": "...", "key": "...", "op": "=", "value": "..."}]
    logic: str = "AND"
    limit: int = 50
    offset: int = 0


@router.get("")
async def list_tags(
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """List all unique tags with counts."""
    tags = await db.get_all_tags()
    return {"tags": tags}


@router.get("/namespaces")
async def list_namespaces(
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """List all semantic tag namespaces."""
    from web.config import get_settings
    settings = get_settings()
    sync_db = __import__("photo_indexer.database", fromlist=["PhotoIndexDB"]).PhotoIndexDB(settings.database_path)

    with sync_db._connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT namespace FROM enrich_semantic_tags"
        ).fetchall()

    namespaces = [row["namespace"] for row in rows]
    return {"namespaces": namespaces}


@router.get("/namespaces/{namespace}/keys")
async def get_namespace_keys(
    namespace: str,
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Get all keys in a namespace with counts."""
    keys = await db.get_namespace_keys(namespace)
    return {"namespace": namespace, "keys": keys}


@router.get("/assets/{asset_id}")
async def get_asset_tags(
    asset_id: str,
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Get all tags (simple + semantic) for an asset."""
    simple_tags = await db.get_asset_tags(asset_id)
    semantic_tags = await db.get_semantic_tags_for_asset(asset_id)

    return {
        "simple_tags": simple_tags,
        "semantic_tags": semantic_tags,
    }


@router.post("/assets/{asset_id}")
async def add_asset_tags(
    asset_id: str,
    request: TagAddRequest,
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Add simple tags to an asset."""
    added = []
    for tag in request.tags:
        success = await db.add_tag(asset_id, tag, request.source)
        if success:
            added.append(tag)

    return {"added": added, "count": len(added)}


@router.delete("/assets/{asset_id}/{tag}")
async def remove_asset_tag(
    asset_id: str,
    tag: str,
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Remove a tag from an asset."""
    success = await db.remove_tag(asset_id, tag)
    if not success:
        raise HTTPException(status_code=404, detail="Tag not found on asset")
    return {"message": "Tag removed"}


@router.post("/import")
async def import_tags(
    file: UploadFile = File(...),
    job_manager=Depends(get_job_manager),
    user=Depends(get_current_user),
):
    """Import tags from TSV/CSV file (async job)."""
    # Save uploaded file temporarily
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    job_id = await job_manager.create_job("tag_import", {"file_path": tmp_path})
    return {"job_id": job_id, "message": "Tag import started"}


@router.get("/export")
async def export_tags(
    format: str = Query("csv", pattern="^(csv|json)$"),
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Export all tags as CSV or JSON."""
    tags = await db.get_all_tags()

    if format == "json":
        return {"tags": tags}

    import io
    import csv

    output = io.StringIO()
    if tags:
        writer = csv.DictWriter(output, fieldnames=tags[0].keys())
        writer.writeheader()
        writer.writerows(tags)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=tags.csv"},
    )


@router.post("/merge")
async def merge_tags(
    request: TagMergeRequest,
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Merge multiple tags into one."""
    from web.config import get_settings
    settings = get_settings()
    sync_db = __import__("photo_indexer.database", fromlist=["PhotoIndexDB"]).PhotoIndexDB(settings.database_path)

    merged = 0
    with sync_db.transaction() as conn:
        for from_tag in request.from_tags:
            cursor = conn.execute(
                "UPDATE enrich_tags SET tag_name = ? WHERE tag_name = ?",
                (request.to_tag, from_tag)
            )
            merged += cursor.rowcount

    return {"merged": merged, "to_tag": request.to_tag}


# Semantic tags endpoints

@router.post("/semantic/assets/{asset_id}")
async def add_semantic_tag(
    asset_id: str,
    request: SemanticTagAddRequest,
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Add a semantic tag to an asset."""
    from datetime import datetime, timezone
    import uuid

    tag_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    from web.config import get_settings
    settings = get_settings()
    sync_db = __import__("photo_indexer.database", fromlist=["PhotoIndexDB"]).PhotoIndexDB(settings.database_path)

    with sync_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO enrich_semantic_tags
            (id, asset_id, namespace, key, value, confidence, source, model_version, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (tag_id, asset_id, request.namespace, request.key, request.value,
             request.confidence, request.source, request.model_version, now)
        )

    return {"id": tag_id, "message": "Semantic tag added"}


@router.delete("/semantic/assets/{asset_id}/{namespace}/{key}")
async def remove_semantic_tag(
    asset_id: str,
    namespace: str,
    key: str,
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Remove a semantic tag from an asset."""
    from web.config import get_settings
    settings = get_settings()
    sync_db = __import__("photo_indexer.database", fromlist=["PhotoIndexDB"]).PhotoIndexDB(settings.database_path)

    with sync_db.transaction() as conn:
        cursor = conn.execute(
            "DELETE FROM enrich_semantic_tags WHERE asset_id = ? AND namespace = ? AND key = ?",
            (asset_id, namespace, key)
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Semantic tag not found")

    return {"message": "Semantic tag removed"}


@router.post("/semantic/search")
async def search_by_semantic_tags(
    request: SemanticTagSearchRequest,
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Search assets by semantic tag criteria."""
    asset_ids = await db.search_assets_by_semantic_tags(
        request.filters,
        request.logic,
        request.limit,
        request.offset,
    )

    # Get full asset details
    assets = []
    for aid in asset_ids:
        asset = await db.get_asset(aid)
        if asset:
            assets.append(asset)

    return {"assets": assets, "count": len(assets)}


@router.get("/semantic/facets")
async def get_semantic_facets(
    namespace: Optional[str] = Query(None),
    db: AsyncPhotoIndexDB = Depends(get_db),
    user=Depends(get_current_user),
):
    """Get faceted counts for semantic tags (for filter UI)."""
    from web.config import get_settings
    settings = get_settings()
    sync_db = __import__("photo_indexer.database", fromlist=["PhotoIndexDB"]).PhotoIndexDB(settings.database_path)

    if namespace:
        where = "WHERE namespace = ?"
        params = [namespace]
    else:
        where = ""
        params = []

    with sync_db._connect() as conn:
        rows = conn.execute(
            f"""
            SELECT namespace, key, COUNT(DISTINCT asset_id) as count, AVG(confidence) as avg_confidence
            FROM enrich_semantic_tags
            {where}
            GROUP BY namespace, key
            ORDER BY namespace, count DESC
            """,
            params
        ).fetchall()

    facets = {}
    for row in rows:
        ns = row["namespace"]
        if ns not in facets:
            facets[ns] = []
        facets[ns].append({
            "key": row["key"],
            "count": row["count"],
            "avg_confidence": row["avg_confidence"],
        })

    return {"facets": facets}


@router.post("/semantic/assets/batch/auto-tag")
async def batch_auto_tag(
    asset_ids: List[str],
    namespaces: Optional[List[str]] = None,
    force_recompute: bool = False,
    job_manager=Depends(get_job_manager),
    user=Depends(get_current_user),
):
    """Trigger semantic auto-tagging for multiple assets (async job)."""
    job_id = await job_manager.create_job("semantic_auto_tag", {
        "asset_ids": asset_ids,
        "namespaces": namespaces,
        "force_recompute": force_recompute,
    })
    return {"job_id": job_id, "message": "Semantic auto-tagging started"}


@router.post("/semantic/relations")
async def add_tag_relation(
    from_namespace: str,
    from_key: str,
    to_namespace: str,
    to_key: str,
    relation_type: str,
    weight: float = 1.0,
    user=Depends(get_current_user),
):
    """Add a tag relationship (ontology)."""
    from web.config import get_settings
    settings = get_settings()
    sync_db = __import__("photo_indexer.database", fromlist=["PhotoIndexDB"]).PhotoIndexDB(settings.database_path)

    with sync_db.transaction() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO enrich_tag_relations
            (from_namespace, from_key, to_namespace, to_key, relation_type, weight)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (from_namespace, from_key, to_namespace, to_key, relation_type, weight)
        )

    return {"message": "Tag relation added"}


@router.get("/semantic/ontology/graph")
async def get_ontology_graph(
    user=Depends(get_current_user),
):
    """Get tag ontology as graph for visualization."""
    from web.config import get_settings
    settings = get_settings()
    sync_db = __import__("photo_indexer.database", fromlist=["PhotoIndexDB"]).PhotoIndexDB(settings.database_path)

    with sync_db._connect() as conn:
        rows = conn.execute(
            "SELECT * FROM enrich_tag_relations"
        ).fetchall()

    # Convert to Cytoscape.js format
    nodes = set()
    edges = []

    for row in rows:
        from_id = f"{row['from_namespace']}:{row['from_key']}"
        to_id = f"{row['to_namespace']}:{row['to_key']}"
        nodes.add((row['from_namespace'], row['from_key']))
        nodes.add((row['to_namespace'], row['to_key']))
        edges.append({
            "data": {
                "id": f"{from_id}->{to_id}",
                "source": from_id,
                "target": to_id,
                "label": row["relation_type"],
                "weight": row["weight"],
            }
        })

    node_list = [
        {"data": {"id": f"{ns}:{key}", "namespace": ns, "label": key}}
        for ns, key in nodes
    ]

    return {"elements": {"nodes": node_list, "edges": edges}}