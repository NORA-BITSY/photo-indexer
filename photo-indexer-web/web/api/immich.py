from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from web.auth import get_current_user
from web.immich import get_immich_client


router = APIRouter(prefix="/api/v1/immich", tags=["immich"])


@router.get("/assets/{asset_id}")
async def get_asset(
    asset_id: str,
    user=Depends(get_current_user),
):
    """Proxy to Immich /api/assets/{id}."""
    immich = get_immich_client()
    asset = await immich.get_asset(asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    return asset


@router.get("/assets/{asset_id}/thumbnail")
async def get_thumbnail(
    asset_id: str,
    size: str = Query("preview"),
    user=Depends(get_current_user),
):
    """Proxy to Immich /api/assets/{id}/thumbnail."""
    immich = get_immich_client()
    content = await immich.get_asset_thumbnail(asset_id, size)
    if not content:
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return Response(content=content, media_type="image/jpeg")


@router.get("/assets/{asset_id}/file")
async def get_file(
    asset_id: str,
    user=Depends(get_current_user),
):
    """Proxy to Immich /api/assets/{id}/original (full-res)."""
    immich = get_immich_client()
    content = await immich.get_asset_file(asset_id)
    if not content:
        raise HTTPException(status_code=404, detail="File not found")
    return Response(content=content, media_type="application/octet-stream")


@router.get("/assets/{asset_id}/exif")
async def get_exif(
    asset_id: str,
    user=Depends(get_current_user),
):
    """Proxy to Immich /api/assets/{id}/exif."""
    immich = get_immich_client()
    exif = await immich.get_asset_exif(asset_id)
    if not exif:
        raise HTTPException(status_code=404, detail="EXIF not found")
    return exif


@router.post("/search")
async def search_assets(
    params: Dict[str, Any],
    user=Depends(get_current_user),
):
    """Proxy to Immich /api/search/metadata."""
    immich = get_immich_client()
    assets = await immich.search_assets(params)
    return {"assets": assets}


@router.get("/albums")
async def get_albums(
    user=Depends(get_current_user),
):
    """Proxy to Immich /api/albums."""
    immich = get_immich_client()
    albums = await immich.get_albums()
    return {"albums": albums}


@router.get("/albums/{album_id}/assets")
async def get_album_assets(
    album_id: str,
    user=Depends(get_current_user),
):
    """Proxy to Immich /api/albums/{id}/assets."""
    immich = get_immich_client()
    assets = await immich.get_album_assets(album_id)
    return {"assets": assets}


@router.get("/health")
async def health_check(
    user=Depends(get_current_user),
):
    """Check Immich connectivity."""
    immich = get_immich_client()
    healthy = await immich.health_check()
    return {"healthy": healthy}