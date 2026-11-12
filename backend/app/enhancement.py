from __future__ import annotations

import io
import os
import uuid
from typing import Any

import httpx
import numpy as np
from PIL import Image, ImageOps, ImageDraw

from .config import settings

STATIC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static"))
JOBS_DIR = os.path.join(STATIC_ROOT, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)


def pick_thumbnail(scene: dict[str, Any]) -> str | None:
    assets = scene.get("assets") or {}
    for k in ["thumbnail", "rendered_preview", "overview", "visual"]:
        if assets.get(k):
            return assets[k]
    # Some STAC items only expose COGs. Browsers cannot render COGs directly.
    return None


async def fetch_image(url: str) -> Image.Image | None:
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            res = await client.get(url)
            res.raise_for_status()
            img = Image.open(io.BytesIO(res.content)).convert("RGB")
            return img
    except Exception:
        return None


def _normalise_diff(diff: np.ndarray) -> np.ndarray:
    p95 = np.percentile(diff, 95) if diff.size else 1
    denom = p95 if p95 > 1 else 1
    return np.clip(diff / denom, 0, 1)


def make_heatmap(diff: np.ndarray) -> Image.Image:
    norm = _normalise_diff(diff)
    h, w = norm.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = np.clip(norm * 255, 0, 255).astype(np.uint8)
    rgba[..., 1] = np.clip((1 - np.abs(norm - 0.5) * 2) * 210, 0, 210).astype(np.uint8)
    rgba[..., 2] = 30
    rgba[..., 3] = np.where(norm > 0.28, np.clip(norm * 220, 0, 220), 0).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def cell_geojson(diff: np.ndarray, bbox: list[float], grid: int = 8, threshold: float = 0.38) -> dict[str, Any]:
    norm = _normalise_diff(diff)
    h, w = norm.shape
    minx, miny, maxx, maxy = bbox
    features = []
    for gy in range(grid):
        for gx in range(grid):
            x0 = int(gx * w / grid)
            x1 = int((gx + 1) * w / grid)
            y0 = int(gy * h / grid)
            y1 = int((gy + 1) * h / grid)
            block = norm[y0:y1, x0:x1]
            mean = float(block.mean()) if block.size else 0.0
            hot = float((block > threshold).mean()) if block.size else 0.0
            score = mean * 0.65 + hot * 0.35
            if score < 0.30:
                continue
            lon0 = minx + gx * (maxx - minx) / grid
            lon1 = minx + (gx + 1) * (maxx - minx) / grid
            # Image y increases downward, latitude decreases as y increases.
            lat1 = maxy - gy * (maxy - miny) / grid
            lat0 = maxy - (gy + 1) * (maxy - miny) / grid
            features.append({
                "type": "Feature",
                "properties": {
                    "change_score": round(score, 3),
                    "mean_difference": round(mean, 3),
                    "hot_pixel_ratio": round(hot, 3),
                    "method": "thumbnail_preview_difference",
                    "confidence": "preview_only"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]
                    ]]
                }
            })
    return {"type": "FeatureCollection", "features": features}


def draw_difference_preview(after: Image.Image, diff: np.ndarray) -> Image.Image:
    base = after.resize((720, 480), Image.Resampling.LANCZOS).convert("RGBA")
    heat = make_heatmap(diff).resize(base.size, Image.Resampling.BILINEAR)
    out = Image.alpha_composite(base, heat)
    d = ImageDraw.Draw(out)
    d.rectangle((12, 12, 360, 48), fill=(5, 12, 24, 210), outline=(255, 255, 255, 80))
    d.text((24, 22), "Preview change layer — use band processing for production", fill=(255, 255, 255, 235))
    return out.convert("RGB")


async def preview_change(latest: dict[str, Any], previous: dict[str, Any], bbox: list[float]) -> dict[str, Any]:
    latest_url = pick_thumbnail(latest)
    previous_url = pick_thumbnail(previous)
    if not latest_url or not previous_url:
        return {
            "status": "missing_preview_assets",
            "summary": "No browser-renderable preview asset was available for both scenes. Use Sentinel Hub Process API or COG band rendering.",
            "geojson": {"type": "FeatureCollection", "features": []},
            "image_url": None,
            "method": "none"
        }

    latest_img, previous_img = await fetch_image(latest_url), await fetch_image(previous_url)
    if latest_img is None or previous_img is None:
        return {
            "status": "preview_download_failed",
            "summary": "Preview images could not be downloaded. The scene metadata is still valid, but change preview could not run.",
            "geojson": {"type": "FeatureCollection", "features": []},
            "image_url": None,
            "method": "none"
        }

    size = (360, 240)
    a = ImageOps.autocontrast(latest_img.resize(size, Image.Resampling.LANCZOS).convert("RGB"))
    b = ImageOps.autocontrast(previous_img.resize(size, Image.Resampling.LANCZOS).convert("RGB"))
    aa = np.asarray(a).astype(np.float32)
    bb = np.asarray(b).astype(np.float32)
    diff = np.mean(np.abs(aa - bb), axis=2)
    geojson = cell_geojson(diff, bbox)
    preview = draw_difference_preview(latest_img, diff)

    job_id = str(uuid.uuid4())[:12]
    path = os.path.join(JOBS_DIR, f"{job_id}_change.jpg")
    preview.save(path, quality=88)
    public_url = f"{settings.public_base_url}/static/jobs/{job_id}_change.jpg"

    count = len(geojson["features"])
    if count:
        summary = f"Detected {count} preview-level changed cells. Treat as triage only until confirmed with raw bands/SCL cloud mask."
    else:
        summary = "No strong preview-level changed cells found. This does not rule out spectral changes in raw bands."

    return {
        "status": "ok",
        "summary": summary,
        "geojson": geojson,
        "image_url": public_url,
        "method": "thumbnail_preview_difference",
        "note": "This is a real image comparison, but it uses thumbnails. Production should use raw bands and cloud masks."
    }


def visibility_matrix() -> list[dict[str, Any]]:
    return [
        {
            "class": "visible",
            "items": ["large port structures", "water boundaries", "vegetation fields", "large construction/mining changes", "flood extents", "burn scars"],
        },
        {
            "class": "inferable",
            "items": ["surface clearing", "container yard rearrangement", "reservoir/water level trends", "vegetation stress", "possible ship presence in SAR"],
        },
        {
            "class": "not_visible_from_sentinel",
            "items": ["individual people", "number plates", "exact cargo inside containers", "street-level activity", "continuous live movement"],
        },
        {
            "class": "needs_external_live_api",
            "items": ["AIS vessel tracks", "ADS-B aircraft", "road traffic", "rail positions", "port manifests", "weather nowcasts"],
        },
    ]


def production_indices() -> list[dict[str, str]]:
    return [
        {"name": "NDVI", "formula": "(B08 - B04) / (B08 + B04)", "meaning": "vegetation health / biomass"},
        {"name": "NDWI", "formula": "(B03 - B08) / (B03 + B08)", "meaning": "water / flood signal"},
        {"name": "NBR", "formula": "(B08 - B12) / (B08 + B12)", "meaning": "burn scar / fire recovery"},
        {"name": "NDMI", "formula": "(B08 - B11) / (B08 + B11)", "meaning": "moisture stress"},
        {"name": "SAR VV/VH", "formula": "Sentinel-1 backscatter", "meaning": "cloud-safe water/roughness/ships/structure signal"},
        {"name": "SCL mask", "formula": "Scene Classification Layer", "meaning": "cloud/shadow/snow/water/vegetation quality mask"},
    ]
