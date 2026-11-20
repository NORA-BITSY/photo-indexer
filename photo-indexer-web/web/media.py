import mimetypes
import os
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse

from web.config import get_settings


def get_media_root(name: str) -> Optional[Path]:
    """Get the filesystem path for a media root by name."""
    settings = get_settings()
    return settings.media_root_dict.get(name)


async def serve_media_file(
    root_name: str,
    file_path: str,
    request: Request,
) -> Response:
    """
    Serve a file from a media root with Range request support.
    Prevents path traversal attacks.
    """
    root = get_media_root(root_name)
    if not root:
        raise HTTPException(status_code=404, detail=f"Media root '{root_name}' not found")

    # Resolve the requested path
    try:
        requested = (root / file_path).resolve()
        # Ensure the resolved path is within the root
        if not str(requested).startswith(str(root.resolve())):
            raise HTTPException(status_code=403, detail="Path traversal not allowed")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")

    if not requested.exists() or not requested.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # Determine MIME type
    mime_type, _ = mimetypes.guess_type(str(requested))
    if not mime_type:
        mime_type = "application/octet-stream"

    # Handle Range requests for video/audio seeking
    range_header = request.headers.get("range")
    if range_header:
        return await _serve_range(requested, range_header, mime_type)

    # Full file response
    return FileResponse(
        path=requested,
        media_type=mime_type,
        filename=requested.name,
    )


async def _serve_range(
    file_path: Path,
    range_header: str,
    mime_type: str,
) -> StreamingResponse:
    """Serve a byte range of a file."""
    file_size = file_path.stat().st_size

    # Parse range header (format: "bytes=start-end")
    try:
        range_str = range_header.replace("bytes=", "")
        start_str, end_str = range_str.split("-")
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Range header")

    # Validate range
    if start >= file_size or end >= file_size or start > end:
        raise HTTPException(status_code=416, detail="Range not satisfiable")

    content_length = end - start + 1

    async def file_iterator():
        with open(file_path, "rb") as f:
            f.seek(start)
            remaining = content_length
            chunk_size = 8192
            while remaining > 0:
                read_size = min(chunk_size, remaining)
                data = f.read(read_size)
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
    }

    return StreamingResponse(
        file_iterator(),
        status_code=206,
        media_type=mime_type,
        headers=headers,
    )


def get_media_root_names() -> list[str]:
    """Get list of available media root names."""
    settings = get_settings()
    return list(settings.media_root_dict.keys())