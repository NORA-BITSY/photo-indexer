"""VLM integration: multi-provider client (Ollama, OpenAI-compatible API), image captioning."""

from __future__ import annotations

import base64
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import httpx

from .models import VLMProvider, VLMResult


# Default VLM prompt tuned for rich image understanding with taxonomy coverage.
DEFAULT_VLM_PROMPT = """You are an expert image understanding assistant. Analyze the image carefully and provide:

1. A concise one-sentence caption of the main subject.
2. A detailed description covering:
   - People / faces present (count, approximate age range, gender if apparent, pose, expression)
   - Animals / pets present (species, breed if apparent, behavior)
   - Objects present (everyday items, furniture, electronics, vehicles, etc.)
   - Places / settings (indoor, outdoor, room type, scenery)
   - Locations / landmarks (buildings, signs, geography if recognizable)
   - Text / OCR visible in the image (signs, labels, captions)
   - Colors, lighting, time of day, weather if outdoors
3. A comma-separated list of the most salient entities (people, animals, objects, places).

Respond in plain text, no JSON, no markdown fences. Be factual and avoid hallucination. If unsure, say so."""


def _encode_image(image_path: Union[str, Path]) -> str:
    """Encode an image file as base64."""
    with open(Path(image_path), "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _parse_response(raw: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse raw VLM text into (caption, detailed_description)."""
    if not raw:
        return None, None
    text = raw.strip()
    match = re.match(r"^([^.!?\n]+[.!?])\s+(.*)$", text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    lines = text.split("\n", 1)
    caption = lines[0].strip() if lines else None
    description = lines[1].strip() if len(lines) > 1 else None
    return caption, description


def _guess_mime(image_path: Union[str, Path]) -> str:
    ext = Path(image_path).suffix.lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".gif": "image/gif", ".webp": "image/webp", ".heic": "image/heic",
        ".heif": "image/heif", ".bmp": "image/bmp", ".tiff": "image/tiff", ".tif": "image/tiff",
    }.get(ext, "image/jpeg")


class BaseVLMProvider(ABC):
    """Abstract VLM provider interface."""

    provider: VLMProvider
    model: str

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is reachable/usable."""

    @abstractmethod
    def analyze_image(
        self,
        image_path: Union[str, Path],
        model: Optional[str] = None,
        prompt: str = DEFAULT_VLM_PROMPT,
    ) -> VLMResult:
        """Analyze a single image and return structured result."""

    def list_models(self) -> List[str]:
        return []

    def close(self) -> None:
        return None


class OllamaVLMProvider(BaseVLMProvider):
    """Ollama-compatible VLM API provider (/api/generate with images)."""

    provider = VLMProvider.OLLAMA

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llava",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def is_available(self) -> bool:
        try:
            resp = self._client.get("/api/tags", timeout=10.0)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        try:
            resp = self._client.get("/api/tags", timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            return []

    def analyze_image(
        self,
        image_path: Union[str, Path],
        model: Optional[str] = None,
        prompt: str = DEFAULT_VLM_PROMPT,
    ) -> VLMResult:
        model_name = model or self.model
        image_b64 = _encode_image(image_path)
        start = time.monotonic()
        try:
            resp = self._client.post(
                "/api/generate",
                json={"model": model_name, "prompt": prompt, "images": [image_b64], "stream": False},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed_ms = int((time.monotonic() - start) * 1000)
            raw_text = data.get("response", "")
            caption, description = _parse_response(raw_text)
            return VLMResult(
                provider=VLMProvider.OLLAMA,
                model=model_name,
                caption=caption,
                detailed_description=description,
                raw_response=raw_text,
                processing_time_ms=elapsed_ms,
            )
        except Exception as exc:  # noqa: BLE001
            return VLMResult(provider=VLMProvider.OLLAMA, model=model_name, raw_response="", error_message=str(exc))

    def close(self) -> None:
        self._client.close()


class APIVLMProvider(BaseVLMProvider):
    """Generic OpenAI-compatible VLM API provider (/v1/chat/completions with image_url)."""

    provider = VLMProvider.API

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("VLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, headers=headers)

    def is_available(self) -> bool:
        if not self.api_key and "openai.com" in self.base_url:
            return False
        try:
            # lightweight check: try models endpoint if available, else assume available with key
            resp = self._client.get("/models", timeout=10.0)
            return resp.status_code in (200, 401, 403)  # reachable
        except Exception:
            return bool(self.api_key)

    def list_models(self) -> List[str]:
        try:
            resp = self._client.get("/models", timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
            items = data.get("data", data.get("models", []))
            if isinstance(items, list):
                return [m.get("id", m.get("name", "")) for m in items if isinstance(m, dict)]
        except Exception:
            pass
        return [self.model]

    def analyze_image(
        self,
        image_path: Union[str, Path],
        model: Optional[str] = None,
        prompt: str = DEFAULT_VLM_PROMPT,
    ) -> VLMResult:
        model_name = model or self.model
        image_b64 = _encode_image(image_path)
        mime = _guess_mime(image_path)
        data_url = f"data:{mime};base64,{image_b64}"
        start = time.monotonic()
        try:
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                "max_tokens": 1024,
            }
            resp = self._client.post("/chat/completions", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            elapsed_ms = int((time.monotonic() - start) * 1000)
            raw_text = ""
            choices = data.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content")
                if isinstance(content, str):
                    raw_text = content
                elif isinstance(content, list):
                    raw_text = "\n".join(
                        part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"
                    )
            caption, description = _parse_response(raw_text)
            return VLMResult(
                provider=VLMProvider.API,
                model=model_name,
                caption=caption,
                detailed_description=description,
                raw_response=raw_text,
                processing_time_ms=elapsed_ms,
            )
        except Exception as exc:  # noqa: BLE001
            return VLMResult(provider=VLMProvider.API, model=model_name, raw_response="", error_message=str(exc))

    def close(self) -> None:
        self._client.close()


class VLMClient:
    """Unified VLM client that delegates to a configured provider."""

    def __init__(
        self,
        provider: Union[str, VLMProvider] = VLMProvider.OLLAMA,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        prov = VLMProvider(provider) if isinstance(provider, str) else provider
        self.provider_type = prov
        if prov == VLMProvider.API:
            self._provider: BaseVLMProvider = APIVLMProvider(
                base_url=base_url or os.environ.get("VLM_API_BASE_URL", "https://api.openai.com/v1"),
                model=model or os.environ.get("VLM_API_MODEL", "gpt-4o-mini"),
                api_key=api_key,
                timeout=timeout,
            )
        else:
            self._provider = OllamaVLMProvider(
                base_url=base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                model=model or os.environ.get("VLM_MODEL", "llava"),
                timeout=timeout,
            )

    @property
    def base_url(self) -> str:
        return getattr(self._provider, "base_url", "")

    @property
    def model(self) -> str:
        return self._provider.model

    def is_available(self) -> bool:
        return self._provider.is_available()

    def list_models(self) -> List[str]:
        return self._provider.list_models()

    def analyze_image(
        self,
        image_path: Union[str, Path],
        model: Optional[str] = None,
        prompt: str = DEFAULT_VLM_PROMPT,
    ) -> VLMResult:
        return self._provider.analyze_image(image_path, model=model, prompt=prompt)

    def close(self) -> None:
        self._provider.close()

    def __enter__(self) -> "VLMClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
