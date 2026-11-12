from __future__ import annotations
from typing import Any


def build_vlm_packet(name: str, bbox: list[float], latest: dict[str, Any] | None, previous: dict[str, Any] | None, change: dict[str, Any]) -> dict[str, Any]:
    latest_assets = latest.get("assets", {}) if latest else {}
    previous_assets = previous.get("assets", {}) if previous else {}
    packet = {
        "task": "satellite_change_interpretation",
        "aoi_name": name,
        "bbox": bbox,
        "inputs": {
            "latest_scene_datetime": latest.get("datetime") if latest else None,
            "previous_scene_datetime": previous.get("datetime") if previous else None,
            "latest_thumbnail": latest_assets.get("thumbnail") or latest_assets.get("rendered_preview"),
            "previous_thumbnail": previous_assets.get("thumbnail") or previous_assets.get("rendered_preview"),
            "change_preview": change.get("image_url"),
            "change_geojson": change.get("geojson"),
        },
        "rules": [
            "Do not claim live video.",
            "Do not identify people, plates, or exact cargo from Sentinel imagery.",
            "Separate visible evidence from inference.",
            "Use timestamps, cloud cover and resolution to qualify confidence.",
            "Recommend follow-up layers: NDVI, NDWI, SAR, SCL cloud mask, AIS/ADS-B if relevant.",
        ],
        "prompt": (
            "You are a geospatial analyst. Compare the latest and previous Sentinel evidence for this AOI. "
            "Explain what visibly changed, what is only inferable, what is not visible from Sentinel, "
            "and what extra data source would confirm it. Return a JSON report with summary, evidence, "
            "confidence, uncertainty, recommended next layer and user-facing explanation."
        )
    }
    return packet
