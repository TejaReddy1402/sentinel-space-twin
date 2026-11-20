"""Application configuration loaded from environment / .env file."""
from __future__ import annotations

import logging

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Public base URL returned to the frontend for static-file links
    public_base_url: str = "http://localhost:8787"

    # Element84 STAC API — free, no key required
    stac_api_url: str = "https://earth-search.aws.element84.com/v1/search"
    stac_timeout_seconds: int = 35

    # Sentinel Hub — optional, enables real RGB/NDVI renders
    sentinel_hub_client_id: str | None = None
    sentinel_hub_client_secret: SecretStr | None = None
    sentinel_hub_token_url: str = (
        "https://services.sentinel-hub.com/auth/realms/main/protocol/openid-connect/token"
    )
    sentinel_hub_process_url: str = "https://services.sentinel-hub.com/api/v1/process"

    # Ollama VLM — optional, enables image explanation
    ollama_base_url: str = "http://localhost:11434"
    ollama_vlm_model: str = "qwen2.5vl:7b"

    # CORS — plain comma-separated string; pydantic-settings reads it as str (no JSON parsing)
    # e.g. CORS_ORIGINS=https://terraingrid.vercel.app,https://www.terraingrid.io
    cors_origins: str = "*"

    @field_validator("public_base_url", "stac_api_url", mode="before")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return str(v).rstrip("/")

    def log_config(self) -> None:
        """Log active configuration at startup (secrets omitted)."""
        logger.info("STAC API       : %s", self.stac_api_url)
        logger.info("Public base URL: %s", self.public_base_url)
        logger.info(
            "Sentinel Hub   : %s",
            "configured" if self.sentinel_hub_client_id else "not configured",
        )
        logger.info("Ollama VLM     : %s / %s", self.ollama_base_url, self.ollama_vlm_model)


settings = Settings()
