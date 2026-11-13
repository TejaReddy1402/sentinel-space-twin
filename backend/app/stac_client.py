"""STAC API client for Sentinel-2 L2A scene discovery."""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import httpx

from .config import settings
from .constants import (
    CLOUD_SCORE_DIVISOR,
    RECENCY_FLOOR,
    RECENCY_PENALTY_PER_RANK,
    SCORE_STRONG,
    SCORE_USABLE,
    SCORE_WEAK,
    SCORE_WEIGHT_CLOUD,
    SCORE_WEIGHT_COMPLETENESS,
    SCORE_WEIGHT_RECENCY,
    STAC_COMPLETENESS_ASSETS,
    THUMBNAIL_ASSET_PRIORITY,
)

logger = logging.getLogger(__name__)


def _datetime_range(days_back: int) -> str:
    end = dt.datetime.now(dt.timezone.utc)
    start = end - dt.timedelta(days=days_back)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return f"{start.strftime(fmt)}/{end.strftime(fmt)}"


async def search_sentinel2(
    bbox: list[float],
    days_back: int,
    max_cloud: float,
    limit: int,
) -> list[dict[str, Any]]:
    """Search Element84 STAC API for Sentinel-2 L2A scenes within bbox."""
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": _datetime_range(days_back),
        "query": {"eo:cloud_cover": {"lte": max_cloud}},
        "limit": limit,
        "fields": {
            "include": [
                "id",
                "geometry",
                "bbox",
                "properties.datetime",
                "properties.platform",
                "properties.eo:cloud_cover",
                "assets",
            ]
        },
    }
    logger.debug(
        "STAC search | bbox=%s days_back=%d max_cloud=%s limit=%d",
        bbox, days_back, max_cloud, limit,
    )
    async with httpx.AsyncClient(timeout=settings.stac_timeout_seconds) as client:
        try:
            resp = await client.post(settings.stac_api_url, json=payload)
            resp.raise_for_status()
        except httpx.TimeoutException:
            logger.error("STAC API timed out after %ds", settings.stac_timeout_seconds)
            raise
        except httpx.HTTPStatusError as exc:
            logger.error(
                "STAC API HTTP %s: %s",
                exc.response.status_code,
                exc.response.text[:300],
            )
            raise
        except httpx.RequestError as exc:
            logger.error("STAC API network error: %s", exc)
            raise

    features: list[dict[str, Any]] = resp.json().get("features", [])
    logger.info("STAC returned %d feature(s) for bbox=%s", len(features), bbox)
    return features


def extract_asset_urls(feature: dict[str, Any]) -> dict[str, str]:
    """Return {asset_key: href_url} for all assets that expose an href."""
    assets = feature.get("assets", {}) or {}
    out: dict[str, str] = {}
    priority = THUMBNAIL_ASSET_PRIORITY + [
        a for a in STAC_COMPLETENESS_ASSETS if a not in THUMBNAIL_ASSET_PRIORITY
    ]
    for name in priority:
        if name in out:
            continue
        asset = assets.get(name)
        if isinstance(asset, dict) and asset.get("href"):
            out[name] = str(asset["href"])
    for key, asset in assets.items():
        if key not in out and isinstance(asset, dict) and asset.get("href"):
            out[key] = str(asset["href"])
    return out


def score_scene(
    feature: dict[str, Any],
    rank: int,
) -> tuple[float, str, dict[str, float]]:
    """
    Calculate scene quality score (0–100).

    Weights: cloud cover (55%), asset completeness (30%), recency rank (15%).
    """
    props = feature.get("properties", {}) or {}
    raw_cloud = props.get("eo:cloud_cover")
    if raw_cloud is None:
        raw_cloud = props.get("cloud_cover")
    cloud = float(raw_cloud) if raw_cloud is not None else 50.0

    asset_urls = extract_asset_urls(feature)
    completeness = sum(
        1 for k in STAC_COMPLETENESS_ASSETS if k in asset_urls
    ) / len(STAC_COMPLETENESS_ASSETS)

    recency_score = max(RECENCY_FLOOR, 1.0 - rank * RECENCY_PENALTY_PER_RANK)
    cloud_score = max(0.0, 1.0 - cloud / CLOUD_SCORE_DIVISOR)

    raw = (
        cloud_score * SCORE_WEIGHT_CLOUD
        + completeness * SCORE_WEIGHT_COMPLETENESS
        + recency_score * SCORE_WEIGHT_RECENCY
    ) * 100.0
    score = round(min(100.0, max(0.0, raw)), 1)

    if score >= SCORE_STRONG:
        label = "strong"
    elif score >= SCORE_USABLE:
        label = "usable"
    elif score >= SCORE_WEAK:
        label = "weak"
    else:
        label = "poor"

    parts = {
        "cloud_score": round(cloud_score * 100, 1),
        "asset_completeness": round(completeness * 100, 1),
        "recency_score": round(recency_score * 100, 1),
    }
    return score, label, parts


def summarize_feature(feature: dict[str, Any], rank: int) -> dict[str, Any]:
    """Convert a raw STAC feature into a SceneSummary-compatible dict."""
    props = feature.get("properties", {}) or {}
    score, label, parts = score_scene(feature, rank)
    assets = extract_asset_urls(feature)
    return {
        "id": feature.get("id", "unknown"),
        "datetime": props.get("datetime"),
        "platform": props.get("platform"),
        "cloud_cover": props.get("eo:cloud_cover") or props.get("cloud_cover"),
        "quality_score": score,
        "quality_label": label,
        "quality_parts": parts,
        "geometry": feature.get("geometry"),
        "bbox": feature.get("bbox"),
        "assets": assets,
    }
