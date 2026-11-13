"""Pydantic request / response models."""
from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, Field, model_validator


class AnalyzeRequest(BaseModel):
    name: str = Field(default="Custom AOI", min_length=1, max_length=120)
    bbox: Annotated[list[float], Field(min_length=4, max_length=4)]
    days_back: int = Field(default=90, ge=1, le=365)
    max_cloud: float = Field(default=40.0, ge=0.0, le=100.0)
    limit: int = Field(default=14, ge=1, le=50)

    @model_validator(mode="after")
    def _validate_bbox(self) -> "AnalyzeRequest":
        min_lon, min_lat, max_lon, max_lat = self.bbox
        if not (-180.0 <= min_lon < max_lon <= 180.0):
            raise ValueError("bbox longitude must satisfy -180 ≤ minLon < maxLon ≤ 180")
        if not (-90.0 <= min_lat < max_lat <= 90.0):
            raise ValueError("bbox latitude must satisfy -90 ≤ minLat < maxLat ≤ 90")
        return self


class SceneSummary(BaseModel):
    id: str
    datetime: str | None = None
    platform: str | None = None
    cloud_cover: float | None = None
    quality_score: Annotated[float, Field(ge=0.0, le=100.0)]
    quality_label: str
    quality_parts: dict[str, float] = Field(default_factory=dict)
    geometry: dict[str, Any] | None = None
    assets: dict[str, str] = Field(default_factory=dict)
    bbox: list[float] | None = None
