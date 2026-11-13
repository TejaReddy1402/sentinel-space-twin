"""Build VLM/LLM input packet for vision-language model scene analysis."""
from __future__ import annotations

from typing import Any


def build_vlm_packet(
    aoi_name: str,
    bbox: list[float],
    latest: dict[str, Any] | None,
    previous: dict[str, Any] | None,
    change: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Construct a structured input packet for a vision-language model.

    The packet contains scene metadata, thumbnails, change GeoJSON and strict
    rules to prevent hallucination of resolution or detail beyond what
    Sentinel-2 10m imagery can support.
    """
    latest_assets = latest.get("assets", {}) if latest else {}
    prev_assets = previous.get("assets", {}) if previous else {}

    return {
        "task": "satellite_change_analysis",
        "aoi_name": aoi_name,
        "bbox": bbox,
        "inputs": {
            "latest_thumbnail": latest_assets.get("thumbnail"),
            "previous_thumbnail": prev_assets.get("thumbnail"),
            "latest_datetime": latest.get("datetime") if latest else None,
            "previous_datetime": previous.get("datetime") if previous else None,
            "latest_quality_score": latest.get("quality_score") if latest else None,
            "latest_cloud_cover": latest.get("cloud_cover") if latest else None,
            "change_geojson": change.get("geojson") if change else None,
            "change_status": change.get("status") if change else None,
            "change_method": change.get("method") if change else None,
        },
        "rules": [
            "Only describe what is visible in the provided thumbnail images.",
            "Do not invent or infer sub-pixel detail — Sentinel-2 resolution is 10 m/pixel.",
            "Acknowledge cloud cover and data quality in every observation.",
            "Treat change cells as preview-level triage, not confirmed change events.",
            "Do not name specific people, vehicles or objects unless clearly identifiable at 10 m resolution.",
            "State an explicit confidence level for every observation.",
        ],
        "prompt": (
            f"Analyse the before/after Sentinel-2 thumbnails for '{aoi_name}'. "
            "Describe any observable changes between the two scenes. "
            "Reference the change GeoJSON cell count and status. "
            "State what is visible, what is inferred and what is uncertain. "
            "Do not fabricate detail not present in 10 m satellite imagery."
        ),
    }
