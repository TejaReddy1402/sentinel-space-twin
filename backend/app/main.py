"""
TERRAINGRID — Satellite Intelligence at Grid Scale
FastAPI application entry point.
"""
from __future__ import annotations

import datetime as dt
import logging
import logging.config
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .constellations import CONSTELLATION_DEFS, INDUSTRIES, collection_windows_for_aoi
from .constants import DEFAULT_BBOX
from .enhancement import preview_change, production_indices, visibility_matrix
from .evidence import _aoi_code, generate_record
from .models import AnalyzeRequest, SegmentRequest
from .stac_client import search_sentinel2, summarize_feature
from .vlm import build_vlm_packet

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="TERRAINGRID — Satellite Intelligence at Grid Scale",
    version="2.0.0",
    description=(
        "Evidence-first satellite intelligence platform. "
        "Scene discovery, quality scoring, change detection, "
        "chain-of-custody evidence records and collection window scheduling."
    ),
)

# CORS: allow Vercel preview/production domains and local dev.
# CORS_ORIGINS env var is a plain comma-separated string (e.g. "https://foo.vercel.app,*")
_cors_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (generated change-preview images)
_static_dir = Path(__file__).parent.parent / "static"
_static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# ---------------------------------------------------------------------------
# Static reference data
# ---------------------------------------------------------------------------
PRESETS: dict[str, dict[str, Any]] = {
    "rotterdam_port":  {"name": "Rotterdam Port",       "bbox": [3.90, 51.82, 4.65, 52.08]},
    "london_gateway":  {"name": "London Gateway Port",  "bbox": [0.35, 51.44, 0.58, 51.55]},
    "heathrow":        {"name": "London Heathrow",      "bbox": [-0.55, 51.42, -0.36, 51.51]},
    "mumbai_port":     {"name": "Mumbai Port",          "bbox": [72.78, 18.88, 72.93, 19.05]},
    "singapore_port":  {"name": "Singapore Port",       "bbox": [103.60, 1.15, 104.05, 1.38]},
}

# Illustrative AIS data — replace with a live AIS provider (e.g. Spire, MarineTraffic API)
MARITIME_VESSELS: list[dict[str, Any]] = [
    {"name": "MSC AURORA",     "mmsi": "235678901", "speed": "12.4 kn", "course": "084°", "status": "Underway",  "type": "Container"},
    {"name": "EVER FORWARD",   "mmsi": "477123456", "speed": "0.0 kn",  "course": "270°", "status": "At anchor", "type": "Container"},
    {"name": "NORDIC HAWK",    "mmsi": "257890123", "speed": "8.1 kn",  "course": "192°", "status": "Underway",  "type": "Tanker"},
    {"name": "BW ENTERPRISE",  "mmsi": "563012345", "speed": "14.2 kn", "course": "310°", "status": "Underway",  "type": "LNG"},
    {"name": "CELTIC HORIZON", "mmsi": "232456789", "speed": "0.0 kn",  "course": "000°", "status": "Moored",    "type": "Bulk"},
]


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def _startup() -> None:
    settings.log_config()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health", tags=["system"])
def health() -> dict[str, str]:
    """Service liveness probe."""
    return {"status": "ok", "service": "terraingrid-v2"}


@app.get("/api/presets", tags=["reference"])
def presets() -> dict[str, Any]:
    """Return named AOI presets with bbox coordinates."""
    return PRESETS


@app.get("/api/industries", tags=["reference"])
def industries() -> dict[str, Any]:
    """Return monitored industry sectors with AOI codes."""
    return {"industries": INDUSTRIES}


@app.get("/api/constellations", tags=["collection"])
def constellations(
    bbox: str = Query(
        default=",".join(str(x) for x in DEFAULT_BBOX),
        description="minLon,minLat,maxLon,maxLat",
    ),
) -> dict[str, Any]:
    """Return 24-hour collection windows for the given AOI bounding box."""
    try:
        parts = [float(x) for x in bbox.split(",")]
        if len(parts) != 4:
            raise ValueError("Expected 4 comma-separated floats")
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid bbox '{bbox}': {exc}. Expected minLon,minLat,maxLon,maxLat.",
        ) from exc
    return collection_windows_for_aoi(parts)


@app.get("/api/maritime", tags=["reference"])
def maritime() -> dict[str, Any]:
    """
    Return maritime vessel data.

    Data is illustrative. Integrate a live AIS provider (Spire, MarineTraffic,
    exactEarth) to replace with real vessel positions.
    """
    return {
        "status": "illustrative",
        "note": (
            "Live AIS data requires integration with an AIS provider "
            "(e.g. MarineTraffic, Spire). Data shown is illustrative."
        ),
        "vessels": MARITIME_VESSELS,
        "capabilities": [
            "Maritime domain awareness — live AIS identity, speed, course and route",
            "Vessels detected and tracked in satellite imagery",
            "Port, anchorage and chokepoint monitoring",
            "Alerts for high-risk corridors and supply-chain disruption",
            "Dark vessel detection via SAR when AIS is switched off",
        ],
    }


@app.post("/api/segment", tags=["analysis"])
async def segment(req: SegmentRequest) -> dict[str, Any]:
    """
    Run SAM/SLIC segmentation on a satellite scene thumbnail.

    Downloads the provided URL(s), segments the image, geo-references each
    segment to the bbox, and returns a GeoJSON FeatureCollection with
    land-cover labels, 3-D heights and change scores.
    Also returns a SRTM elevation grid for terrain-aware 3-D rendering.
    """
    import asyncio
    from .elevation import fetch_elevation_grid
    from .enhancement import fetch_image
    from .sam_segmentation import segment_image

    latest_img = await fetch_image(req.latest_url)
    if latest_img is None:
        raise HTTPException(status_code=422, detail=f"Could not download image: {req.latest_url}")

    previous_img = None
    if req.previous_url:
        previous_img = await fetch_image(req.previous_url)

    loop = asyncio.get_running_loop()
    try:
        segments = await loop.run_in_executor(
            None,
            lambda: segment_image(latest_img, req.bbox, previous_img),
        )
    except Exception as exc:
        logger.exception("segment_image failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {exc}") from exc

    elevation = await fetch_elevation_grid(req.bbox)

    logger.info(
        "Segment | bbox=%s method=%s count=%d",
        req.bbox, segments.get("method"), segments.get("segment_count", 0),
    )

    return {
        "segments":      segments,
        "elevation":     elevation,
        "sam_available": segments.get("sam_available", False),
        "method":        segments.get("method", "unknown"),
    }


@app.post("/api/analyze", tags=["analysis"])
async def analyze(req: AnalyzeRequest) -> dict[str, Any]:
    """
    Run a full analysis cycle for the requested AOI:

    1. Search Sentinel-2 L2A STAC catalogue for scenes within bbox/timerange.
    2. Score and rank scenes by cloud cover, asset completeness and recency.
    3. Generate a preview-level change detection between the two best scenes.
    4. Build a VLM input packet for downstream vision-language model analysis.
    5. Compute collection windows for the AOI from the constellation schedule.
    6. Seal a SHA-256 chain-of-custody evidence record.
    """
    try:
        features = await search_sentinel2(
            req.bbox, req.days_back, req.max_cloud, req.limit
        )
    except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.RequestError) as exc:
        logger.error("STAC search failed: %s", exc)
        raise HTTPException(status_code=502, detail="STAC search failed — upstream error.") from exc

    scenes = [summarize_feature(f, i) for i, f in enumerate(features)]
    scenes.sort(key=lambda s: (s["quality_score"], s.get("datetime") or ""), reverse=True)

    latest = scenes[0] if scenes else None
    previous = scenes[1] if len(scenes) > 1 else None

    change: dict[str, Any] = {
        "status": "not_enough_scenes",
        "summary": "Need at least two scenes for comparison.",
        "geojson": {"type": "FeatureCollection", "features": []},
        "image_url": None,
        "method": "none",
    }
    if latest and previous:
        change = await preview_change(latest, previous, req.bbox)

    vlm_packet = build_vlm_packet(req.name, req.bbox, latest, previous, change)
    cw = collection_windows_for_aoi(req.bbox)

    event_desc = (
        f"Scene capture — {len(scenes)} scenes, best quality {latest['quality_score']}"
        if latest
        else f"{len(scenes)} scene(s) captured — change status: {change['status']}"
    )
    evidence = generate_record(
        aoi_name=req.name,
        bbox=req.bbox,
        event=event_desc,
        scene_id=latest["id"] if latest else None,
    )

    logger.info(
        "Analysis complete | aoi=%s scenes=%d change=%s evidence=%s",
        req.name, len(scenes), change["status"], evidence["id"],
    )

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "aoi": {
            "name": req.name,
            "bbox": req.bbox,
            "code": _aoi_code(req.name),
        },
        "scene_count": len(scenes),
        "selected": {"latest": latest, "previous": previous},
        "scenes": scenes,
        "change": change,
        "indices": production_indices(),
        "visibility_matrix": visibility_matrix(),
        "vlm_packet": vlm_packet,
        "evidence_record": evidence,
        "collection_windows": {
            "next_window": cw["next_window"],
            "total_24h": cw["total_24h"],
            "best_resolution": cw["best_resolution"],
        },
        "data_reality": {
            "sentinel_is": "latest available satellite pass via Element84 STAC — near-real-time EO",
            "sentinel_is_not": "continuous live video, street-level surveillance or sub-10m imagery",
            "upgrade_path": [
                "Sentinel Hub Process API for real RGB/NDVI/NDWI/NBR rendered imagery",
                "Raw COG band processing with SCL cloud masks for production change detection",
                "Live AIS/ADS-B/traffic APIs for real movement layers",
                "VLM scene explanation using thumbnails + change GeoJSON",
            ],
        },
    }
