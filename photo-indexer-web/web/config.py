from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_path: str = Field(default="/data/photo_index.db", alias="DATABASE_PATH")

    # Immich
    immich_url: str = Field(default="http://immich-server:2283/api", alias="IMMICH_URL")
    immich_api_key: str = Field(default="", alias="IMMICH_API_KEY")

    # Ollama
    ollama_base_url: str = Field(default="http://ollama:11434", alias="OLLAMA_BASE_URL")
    vlm_model: str = Field(default="llava", alias="VLM_MODEL")

    # Redis
    redis_url: str = Field(default="redis://redis:6379", alias="REDIS_URL")

    # Auth
    secret_key: str = Field(default="", alias="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=60, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    web_username: str = Field(default="admin", alias="WEB_USERNAME")
    web_password: str = Field(default="changeme", alias="WEB_PASSWORD")

    # Media roots for direct file serving (comma-separated "name:/path")
    media_roots: str = Field(default="library:/data/library,staging:/data/staging", alias="MEDIA_ROOTS")
    max_upload_size: int = Field(default=10485760, alias="MAX_UPLOAD_SIZE")

    # Job system
    job_timeout_seconds: int = Field(default=3600, alias="JOB_TIMEOUT_SECONDS")
    job_concurrency_vlm: int = Field(default=1, alias="JOB_CONCURRENCY_VLM")
    job_concurrency_face: int = Field(default=2, alias="JOB_CONCURRENCY_FACE")
    job_concurrency_body: int = Field(default=2, alias="JOB_CONCURRENCY_BODY")
    job_concurrency_default: int = Field(default=4, alias="JOB_CONCURRENCY_DEFAULT")

    # CORS
    cors_origins: List[str] = Field(default=["*"], alias="CORS_ORIGINS")

    @property
    def media_root_dict(self) -> dict:
        """Parse MEDIA_ROOTS into {name: Path} dict."""
        result = {}
        for part in self.media_roots.split(","):
            part = part.strip()
            if ":" in part:
                name, path = part.split(":", 1)
                result[name.strip()] = Path(path.strip())
        return result

    @property
    def database_url(self) -> str:
        return f"sqlite+aiosqlite:///{self.database_path}"


@lru_cache
def get_settings() -> Settings:
    return Settings()