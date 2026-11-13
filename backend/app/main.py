from __future__ import annotations

import datetime as dt
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .models import AnalyzeRequest
from .stac_client import search_sentinel2, summarize_feature
from .enhancement import preview_change, visibility_matrix, production_indices
from .vlm import build_vlm_packet
from .evidence import generate_record, _aoi_code
from .constellations import collection_windows_for_aoi, INDUSTRIES, CONSTELLATION_DEFS

app = FastAPI(
    title="infinityPLEX — Earth Intelligence Platform",
    version="2.0.0",
    description="Integrated Earth Intelligence for Global Infrastructure Operations and Project Assurance.",
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
    "rotterdam_port":   {"name": "Rotterdam Port",        "bbox": [3.90, 51.82, 4.65, 52.08],   "aoi": "AOI-18"},
    "london_gateway":   {"name": "London Gateway Port",   "bbox": [0.35, 51.44, 0.58, 51.55],   "aoi": "AOI-21"},
    "heathrow":         {"name": "London Heathrow",       "bbox": [-0.55, 51.42, -0.36, 51.51], "aoi": "AOI-21"},
    "mumbai_port":      {"name": "Mumbai Port",           "bbox": [72.78, 18.88, 72.93, 19.05], "aoi": "AOI-07"},
    "singapore_port":   {"name": "Singapore Port",        "bbox": [103.60, 1.15, 104.05, 1.38], "aoi": "AOI-18"},
}

MARITIME_VESSELS = [
    {"name": "MSC AURORA",    "mmsi": "235678901", "speed": "12.4 kn", "course": "084°", "status": "Underway", "type": "Container"},
    {"name": "EVER FORWARD",  "mmsi": "477123456", "speed": "0.0 kn",  "course": "270°", "status": "At anchor","type": "Container"},
    {"name": "NORDIC HAWK",   "mmsi": "257890123", "speed": "8.1 kn",  "course": "192°", "status": "Underway", "type": "Tanker"},
    {"name": "BW ENTERPRISE", "mmsi": "563012345", "speed": "14.2 kn", "course": "310°", "status": "Underway", "type": "LNG"},
    {"name": "CELTIC HORIZON", "mmsi": "232456789","speed": "0.0 kn",  "course": "000°", "status": "Moored",   "type": "Bulk"},
]


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "infinityplex-earth-intelligence-v2"}


@app.get("/api/presets")
def presets() -> dict[str, Any]:
    return PRESETS


@app.get("/api/industries")
def industries() -> dict[str, Any]:
    return {"industries": INDUSTRIES}


@app.get("/api/constellations")
def constellations(
    bbox: str = Query(default="4.0,51.9,4.5,52.0", description="minLon,minLat,maxLon,maxLat"),
) -> dict[str, Any]:
    try:
        parts = [float(x) for x in bbox.split(",")]
        if len(parts) != 4:
            raise ValueError
    except ValueError:
        parts = [4.0, 51.9, 4.5, 52.0]
    return collection_windows_for_aoi(parts)


@app.get("/api/maritime")
def maritime() -> dict[str, Any]:
    return {
        "status": "illustrative",
        "note": "Live AIS data requires integration with an AIS provider (e.g. MarineTraffic, Spire). Data shown is illustrative.",
        "vessels": MARITIME_VESSELS,
        "capabilities": [
            "Maritime domain awareness",
            "Live AIS identity, speed, course and route",
            "Vessels detected and tracked in satellite imagery",
            "Port, anchorage and chokepoint monitoring",
            "Alerts for high-risk corridors and supply-chain disruption",
        ],
    }



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

    # Collection windows for this AOI
    cw = collection_windows_for_aoi(req.bbox)

    # Evidence record
    scene_count = len(scenes)
    event_desc = f"{scene_count} scene(s) captured — change status: {change['status']}"
    if latest:
        event_desc = f"Scene capture — {scene_count} scenes, best quality {latest['quality_score']}"
    evidence = generate_record(
        aoi_name=req.name,
        bbox=req.bbox,
        event=event_desc,
        scene_id=latest["id"] if latest else None,
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
            "sentinel_is": "latest available satellite pass / near-real-time earth observation",
            "sentinel_is_not": "continuous live video or street-level surveillance",
            "upgrade_path": [
                "Use Sentinel Hub Process API for real RGB/NDVI/NDWI/NBR renders",
                "Use raw COG band processing with SCL cloud masks for production change detection",
                "Fuse AIS/ADS-B/traffic APIs for real live movement layers",
                "Pass evidence crops + metadata into VLM for explanation, not fake super-resolution",
            ],
        },
    }
