"""Image enhancement, change detection and visibility utilities."""
from __future__ import annotations

import io
import logging
import uuid
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .config import settings
from .constants import (
    CHANGE_CELL_MIN_SCORE,
    CHANGE_GRID_CELLS,
    CHANGE_SCORE_HOT_WEIGHT,
    CHANGE_SCORE_MEAN_WEIGHT,
    CHANGE_THRESHOLD,
    DIFF_PERCENTILE,
    FETCH_TIMEOUT_SECONDS,
    PREVIEW_HEIGHT,
    PREVIEW_JPEG_QUALITY,
    PREVIEW_WIDTH,
    THUMBNAIL_ASSET_PRIORITY,
)

logger = logging.getLogger(__name__)

JOBS_DIR = Path(__file__).parent.parent / "static" / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)


def pick_thumbnail(scene: dict[str, Any]) -> str | None:
    """Return the first browser-renderable thumbnail URL from scene assets."""
    assets = scene.get("assets") or {}
    for k in THUMBNAIL_ASSET_PRIORITY:
        if assets.get(k):
            return assets[k]
    return None


async def fetch_image(url: str) -> Image.Image | None:
    """Download and decode an image from a URL; return None on failure."""
    try:
        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT_SECONDS, follow_redirects=True) as client:
            res = await client.get(url)
            res.raise_for_status()
            return Image.open(io.BytesIO(res.content)).convert("RGB")
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        logger.warning("Failed to fetch image %s: %s", url, exc)
        return None


def _normalise_diff(diff: np.ndarray) -> np.ndarray:
    """Normalise difference array to [0, 1] using the given percentile as ceiling."""
    p = np.percentile(diff, DIFF_PERCENTILE) if diff.size else 1
    denom = p if p > 1 else 1
    return np.clip(diff / denom, 0, 1)


def make_heatmap(diff: np.ndarray) -> Image.Image:
    """Convert a normalised difference array into an RGBA heatmap overlay."""
    norm = _normalise_diff(diff)
    h, w = norm.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = np.clip(norm * 255, 0, 255).astype(np.uint8)
    rgba[..., 1] = np.clip((1 - np.abs(norm - 0.5) * 2) * 210, 0, 210).astype(np.uint8)
    rgba[..., 2] = 30
    rgba[..., 3] = np.where(norm > 0.28, np.clip(norm * 220, 0, 220), 0).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def cell_geojson(
    diff: np.ndarray,
    bbox: list[float],
    grid: int = CHANGE_GRID_CELLS,
    threshold: float = CHANGE_THRESHOLD,
) -> dict[str, Any]:
    """Partition the difference array into an NxN grid and emit GeoJSON change cells."""
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
            score = mean * CHANGE_SCORE_MEAN_WEIGHT + hot * CHANGE_SCORE_HOT_WEIGHT
            if score < CHANGE_CELL_MIN_SCORE:
                continue
            lon0 = minx + gx * (maxx - minx) / grid
            lon1 = minx + (gx + 1) * (maxx - minx) / grid
            # Image y increases downward; latitude decreases as y increases.
            lat1 = maxy - gy * (maxy - miny) / grid
            lat0 = maxy - (gy + 1) * (maxy - miny) / grid
            features.append({
                "type": "Feature",
                "properties": {
                    "change_score": round(score, 3),
                    "mean_difference": round(mean, 3),
                    "hot_pixel_ratio": round(hot, 3),
                    "method": "thumbnail_preview_difference",
                    "confidence": "preview_only",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0],
                    ]],
                },
            })
    return {"type": "FeatureCollection", "features": features}


def draw_difference_preview(after: Image.Image, diff: np.ndarray) -> Image.Image:
    """Composite a heatmap overlay onto the latest scene thumbnail."""
    base = after.resize(
        (PREVIEW_WIDTH * 2, PREVIEW_HEIGHT * 2), Image.Resampling.LANCZOS
    ).convert("RGBA")
    heat = make_heatmap(diff).resize(base.size, Image.Resampling.BILINEAR)
    out = Image.alpha_composite(base, heat)
    d = ImageDraw.Draw(out)
    d.rectangle((12, 12, 360, 48), fill=(5, 12, 24, 210), outline=(255, 255, 255, 80))
    d.text(
        (24, 22),
        "Preview change layer — use band processing for production",
        fill=(255, 255, 255, 235),
    )
    return out.convert("RGB")


async def preview_change(
    latest: dict[str, Any],
    previous: dict[str, Any],
    bbox: list[float],
) -> dict[str, Any]:
    """
    Generate a preview-level change detection between two scenes.

    Downloads thumbnails, computes per-pixel absolute difference, produces a
    GeoJSON cell overlay and saves a JPEG composite.  All results are
    preview-quality only; production use requires raw band processing.
    """
    latest_url = pick_thumbnail(latest)
    previous_url = pick_thumbnail(previous)
    if not latest_url or not previous_url:
        return {
            "status": "missing_preview_assets",
            "summary": (
                "No browser-renderable preview asset was available for both scenes. "
                "Use Sentinel Hub Process API or COG band rendering."
            ),
            "geojson": {"type": "FeatureCollection", "features": []},
            "image_url": None,
            "method": "none",
        }

    latest_img, previous_img = (
        await fetch_image(latest_url),
        await fetch_image(previous_url),
    )
    if latest_img is None or previous_img is None:
        return {
            "status": "preview_download_failed",
            "summary": (
                "Preview images could not be downloaded. "
                "The scene metadata is still valid, but change preview could not run."
            ),
            "geojson": {"type": "FeatureCollection", "features": []},
            "image_url": None,
            "method": "none",
        }

    size = (PREVIEW_WIDTH, PREVIEW_HEIGHT)
    a = ImageOps.autocontrast(latest_img.resize(size, Image.Resampling.LANCZOS).convert("RGB"))
    b = ImageOps.autocontrast(previous_img.resize(size, Image.Resampling.LANCZOS).convert("RGB"))
    aa = np.asarray(a).astype(np.float32)
    bb = np.asarray(b).astype(np.float32)
    diff = np.mean(np.abs(aa - bb), axis=2)
    geojson = cell_geojson(diff, bbox)
    preview = draw_difference_preview(latest_img, diff)

    job_id = uuid.uuid4().hex[:12]
    path = JOBS_DIR / f"{job_id}_change.jpg"
    try:
        preview.save(str(path), quality=PREVIEW_JPEG_QUALITY)
    except OSError as exc:
        logger.error("Failed to save change preview to %s: %s", path, exc)
        return {
            "status": "preview_save_failed",
            "summary": "Change detection ran but the preview image could not be saved.",
            "geojson": geojson,
            "image_url": None,
            "method": "thumbnail_preview_difference",
        }

    public_url = f"{settings.public_base_url}/static/jobs/{job_id}_change.jpg"
    count = len(geojson["features"])
    if count:
        summary = (
            f"Detected {count} preview-level changed cells. "
            "Treat as triage only until confirmed with raw bands/SCL cloud mask."
        )
    else:
        summary = (
            "No strong preview-level changed cells found. "
            "This does not rule out spectral changes in raw bands."
        )

    return {
        "status": "ok",
        "summary": summary,
        "geojson": geojson,
        "image_url": public_url,
        "method": "thumbnail_preview_difference",
        "note": (
            "This is a real image comparison, but it uses thumbnails. "
            "Production should use raw bands and cloud masks."
        ),
    }


def visibility_matrix() -> list[dict[str, Any]]:
    """Return the Sentinel-2 visibility classification matrix."""
    return [
        {
            "class": "visible",
            "items": [
                "large port structures",
                "water boundaries",
                "vegetation fields",
                "large construction/mining changes",
                "flood extents",
                "burn scars",
            ],
        },
        {
            "class": "inferable",
            "items": [
                "surface clearing",
                "container yard rearrangement",
                "reservoir/water level trends",
                "vegetation stress",
                "possible ship presence in SAR",
            ],
        },
        {
            "class": "not_visible_from_sentinel",
            "items": [
                "individual people",
                "number plates",
                "exact cargo inside containers",
                "street-level activity",
                "continuous live movement",
            ],
        },
        {
            "class": "needs_external_live_api",
            "items": [
                "AIS vessel tracks",
                "ADS-B aircraft",
                "road traffic",
                "rail positions",
                "port manifests",
                "weather nowcasts",
            ],
        },
    ]


def production_indices() -> list[dict[str, str]]:
    """Return spectral index definitions for the production processing pipeline."""
    return [
        {"name": "NDVI",    "formula": "(B08 - B04) / (B08 + B04)",    "meaning": "vegetation health / biomass"},
        {"name": "NDWI",    "formula": "(B03 - B08) / (B03 + B08)",    "meaning": "water / flood signal"},
        {"name": "NBR",     "formula": "(B08 - B12) / (B08 + B12)",    "meaning": "burn scar / fire recovery"},
        {"name": "NDMI",    "formula": "(B08 - B11) / (B08 + B11)",    "meaning": "moisture stress"},
        {"name": "SAR VV/VH", "formula": "Sentinel-1 backscatter",     "meaning": "cloud-safe water/roughness/ships/structure signal"},
        {"name": "SCL mask",  "formula": "Scene Classification Layer", "meaning": "cloud/shadow/snow/water/vegetation quality mask"},
    ]
