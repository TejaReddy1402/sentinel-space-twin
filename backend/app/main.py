from __future__ import annotations

import datetime as dt
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .models import AnalyzeRequest
from .stac_client import search_sentinel2, summarize_feature
from .enhancement import preview_change, visibility_matrix, production_indices
from .vlm import build_vlm_packet

app = FastAPI(
    title="Sentinel Space Twin Data Engine v2",
    version="2.0.0",
    description="Data-first Sentinel scene discovery, quality scoring, preview change detection and VLM packet generation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

PRESETS = {
    "rotterdam_port": {"name": "Rotterdam Port", "bbox": [3.90, 51.82, 4.65, 52.08]},
    "london_gateway": {"name": "London Gateway Port", "bbox": [0.35, 51.44, 0.58, 51.55]},
    "heathrow": {"name": "London Heathrow", "bbox": [-0.55, 51.42, -0.36, 51.51]},
    "mumbai_port": {"name": "Mumbai Port", "bbox": [72.78, 18.88, 72.93, 19.05]},
    "singapore_port": {"name": "Singapore Port", "bbox": [103.60, 1.15, 104.05, 1.38]},
}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "sentinel-space-twin-data-engine-v2"}


@app.get("/api/presets")
def presets() -> dict[str, Any]:
    return PRESETS


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest) -> dict[str, Any]:
    try:
        features = await search_sentinel2(req.bbox, req.days_back, req.max_cloud, req.limit)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"STAC search failed: {exc}") from exc

    scenes = [summarize_feature(f, i) for i, f in enumerate(features)]
    scenes = sorted(scenes, key=lambda s: (s["quality_score"], s.get("datetime") or ""), reverse=True)

    latest = scenes[0] if scenes else None
    previous = scenes[1] if len(scenes) > 1 else None

    change = {
        "status": "not_enough_scenes",
        "summary": "Need at least two scenes for comparison.",
        "geojson": {"type": "FeatureCollection", "features": []},
        "image_url": None,
        "method": "none"
    }
    if latest and previous:
        change = await preview_change(latest, previous, req.bbox)

    vlm_packet = build_vlm_packet(req.name, req.bbox, latest, previous, change)

    return {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "aoi": {"name": req.name, "bbox": req.bbox},
        "scene_count": len(scenes),
        "selected": {"latest": latest, "previous": previous},
        "scenes": scenes,
        "change": change,
        "indices": production_indices(),
        "visibility_matrix": visibility_matrix(),
        "vlm_packet": vlm_packet,
        "data_reality": {
            "sentinel_is": "latest available satellite pass / near-real-time earth observation",
            "sentinel_is_not": "continuous live video or street-level surveillance",
            "upgrade_path": [
                "Use Sentinel Hub Process API for real RGB/NDVI/NDWI/NBR renders",
                "Use raw COG band processing with SCL cloud masks for production change detection",
                "Fuse AIS/ADS-B/traffic APIs for real live movement layers",
                "Pass evidence crops + metadata into VLM for explanation, not for fake super-resolution"
            ]
        }
    }
