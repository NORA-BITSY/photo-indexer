from typing import Any, Dict, List, Optional

import httpx

from web.config import get_settings


class ImmichClient:
    """Async wrapper for Immich API."""

    def __init__(self) -> None:
        self._client: Optional[httpx.AsyncClient] = None
        self._base_url = ""
        self._api_key = ""

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            settings = get_settings()
            self._base_url = settings.immich_url.rstrip("/")
            self._api_key = settings.immich_api_key
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={"x-api-key": self._api_key} if self._api_key else {},
                timeout=httpx.Timeout(30.0, connect=10.0),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def get_asset(self, asset_id: str) -> Optional[Dict]:
        """Get asset metadata."""
        try:
            client = self._ensure_client()
            resp = await client.get(f"/assets/{asset_id}")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError:
            return None

    async def search_assets(self, params: Dict) -> List[Dict]:
        """Search assets via metadata search."""
        try:
            client = self._ensure_client()
            resp = await client.post("/search/metadata", json=params)
            resp.raise_for_status()
            return resp.json().get("assets", [])
        except httpx.HTTPError:
            return []

    async def get_asset_thumbnail(self, asset_id: str, size: str = "preview") -> Optional[bytes]:
        """Get asset thumbnail."""
        try:
            client = self._ensure_client()
            resp = await client.get(f"/assets/{asset_id}/thumbnail", params={"size": size})
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.content
        except httpx.HTTPError:
            return None

    async def get_asset_file(self, asset_id: str) -> Optional[bytes]:
        """Get full-resolution asset file."""
        try:
            client = self._ensure_client()
            resp = await client.get(f"/assets/{asset_id}/original")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.content
        except httpx.HTTPError:
            return None

    async def get_asset_exif(self, asset_id: str) -> Optional[Dict]:
        """Get asset EXIF data."""
        try:
            client = self._ensure_client()
            resp = await client.get(f"/assets/{asset_id}/exif")
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError:
            return None

    async def get_albums(self) -> List[Dict]:
        """Get all albums."""
        try:
            client = self._ensure_client()
            resp = await client.get("/albums")
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError:
            return []

    async def get_album_assets(self, album_id: str) -> List[Dict]:
        """Get assets in an album."""
        try:
            client = self._ensure_client()
            resp = await client.get(f"/albums/{album_id}/assets")
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError:
            return []

    async def health_check(self) -> bool:
        """Check if Immich is reachable."""
        try:
            client = self._ensure_client()
            resp = await client.get("/server/ping", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False


# Global instance
_immich_client: Optional[ImmichClient] = None


def get_immich_client() -> ImmichClient:
    global _immich_client
    if _immich_client is None:
        _immich_client = ImmichClient()
    return _immich_client