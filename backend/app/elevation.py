"""Elevation grid via OpenTopoData SRTM 30m (free, no key required)."""
from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_API_URL = "https://api.opentopodata.org/v1/srtm30m"
_TIMEOUT = 15.0


async def fetch_elevation_grid(
    bbox: list[float],
    grid_size: int = 8,
) -> dict[str, Any]:
    """Return a grid_size×grid_size elevation grid (metres) over bbox."""
    minx, miny, maxx, maxy = bbox
    lons = [minx + (maxx - minx) * c / (grid_size - 1) for c in range(grid_size)]
    lats = [miny + (maxy - miny) * r / (grid_size - 1) for r in range(grid_size)]
    locations = "|".join(
        f"{lat:.5f},{lon:.5f}" for lat in lats for lon in lons
    )
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(_API_URL, params={"locations": locations})
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("Elevation fetch failed: %s — flat grid", exc)
        return _flat(bbox, grid_size)

    if data.get("status") != "OK" or not data.get("results"):
        return _flat(bbox, grid_size)

    elevs = [float(r.get("elevation") or 0.0) for r in data["results"]]
    grid = [
        [elevs[row * grid_size + col] for col in range(grid_size)]
        for row in range(grid_size)
    ]
    return {
        "grid":            grid,
        "grid_size":       grid_size,
        "min_elevation":   round(min(elevs), 1),
        "max_elevation":   round(max(elevs), 1),
        "elevation_range": round(max(elevs) - min(elevs), 1),
        "bbox":            bbox,
        "source":          "SRTM 30m — opentopodata.org",
    }


def _flat(bbox: list[float], grid_size: int) -> dict[str, Any]:
    return {
        "grid":            [[0.0] * grid_size for _ in range(grid_size)],
        "grid_size":       grid_size,
        "min_elevation":   0.0,
        "max_elevation":   0.0,
        "elevation_range": 0.0,
        "bbox":            bbox,
        "source":          "flat_fallback",
    }
