from __future__ import annotations

import datetime as dt
from typing import Any
import httpx
from .config import settings


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


async def search_sentinel2(bbox: list[float], days_back: int, max_cloud: float, limit: int) -> list[dict[str, Any]]:
    end = utc_now()
    start = end - dt.timedelta(days=days_back)
    payload = {
        "collections": [settings.stac_collection],
        "bbox": bbox,
        "datetime": f"{start.isoformat().replace('+00:00','Z')}/{end.isoformat().replace('+00:00','Z')}",
        "limit": limit,
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
    }
    async with httpx.AsyncClient(timeout=35) as client:
        res = await client.post(settings.stac_api_url, json=payload)
        res.raise_for_status()
        data = res.json()
    return data.get("features", [])


def extract_asset_urls(feature: dict[str, Any]) -> dict[str, str]:
    assets = feature.get("assets", {}) or {}
    names = [
        "thumbnail", "visual", "rendered_preview",
        "red", "green", "blue", "nir", "nir08", "swir16", "swir22", "scl",
        "B02", "B03", "B04", "B08", "B11", "B12", "SCL"
    ]
    out: dict[str, str] = {}
    for name in names:
        a = assets.get(name)
        if isinstance(a, dict) and a.get("href"):
            out[name] = a["href"]
    # Also include every asset href under a safe key, because STAC collections differ.
    for key, asset in assets.items():
        if isinstance(asset, dict) and asset.get("href") and key not in out:
            out[key] = asset["href"]
    return out


def score_scene(feature: dict[str, Any], rank: int) -> tuple[float, str, dict[str, float]]:
    props = feature.get("properties", {}) or {}
    cloud = props.get("eo:cloud_cover")
    if cloud is None:
        cloud = props.get("cloud_cover", 50)
    cloud = float(cloud or 0)

    assets = extract_asset_urls(feature)
    needed = ["thumbnail", "visual", "red", "green", "blue", "nir", "nir08"]
    completeness = sum(1 for k in needed if k in assets) / len(needed)

    # Latest rank helps, but should not dominate cloud and asset quality.
    recency_score = max(0.15, 1.0 - rank * 0.075)
    cloud_score = max(0.0, 1.0 - cloud / 80.0)
    score = (cloud_score * 0.55 + completeness * 0.30 + recency_score * 0.15) * 100
    score = round(min(100.0, max(0.0, score)), 1)
    if score >= 75:
        label = "strong"
    elif score >= 55:
        label = "usable"
    elif score >= 35:
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
    props = feature.get("properties", {}) or {}
    score, label, parts = score_scene(feature, rank)
    assets = extract_asset_urls(feature)
    return {
        "id": feature.get("id"),
        "datetime": props.get("datetime") or feature.get("datetime"),
        "platform": props.get("platform") or props.get("constellation"),
        "cloud_cover": props.get("eo:cloud_cover") or props.get("cloud_cover"),
        "bbox": feature.get("bbox"),
        "geometry": feature.get("geometry"),
        "quality_score": score,
        "quality_label": label,
        "quality_parts": parts,
        "assets": assets,
    }
