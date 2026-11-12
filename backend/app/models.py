from pydantic import BaseModel, Field
from typing import Any


class AnalyzeRequest(BaseModel):
    name: str = "Custom AOI"
    bbox: list[float] = Field(..., min_length=4, max_length=4, description="[minLon, minLat, maxLon, maxLat]")
    days_back: int = Field(90, ge=1, le=365)
    max_cloud: float = Field(40, ge=0, le=100)
    limit: int = Field(12, ge=2, le=50)


class SceneSummary(BaseModel):
    id: str
    datetime: str | None = None
    platform: str | None = None
    cloud_cover: float | None = None
    bbox: list[float] | None = None
    quality_score: float
    quality_label: str
    assets: dict[str, str] = {}
    raw_feature: dict[str, Any] | None = None
