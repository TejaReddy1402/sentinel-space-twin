from __future__ import annotations

import datetime as dt
from typing import Any

CONSTELLATION_DEFS: list[dict[str, Any]] = [
    {
        "id": "blacksky_gen3",
        "name": "BlackSky Gen-3",
        "resolution": "0.35 m",
        "type": "optical",
        "description": "Very high-resolution optical imagery with rapid revisit and near real-time delivery.",
        "revisit_hours": 1.5,
    },
    {
        "id": "blacksky_gen2",
        "name": "BlackSky Gen-2",
        "resolution": "0.80 m",
        "type": "optical",
        "description": "High-cadence optical monitoring for persistent multi-daily, daily or weekly coverage.",
        "revisit_hours": 2.0,
    },
    {
        "id": "pleiades",
        "name": "Airbus Pléiades",
        "resolution": "0.50 m",
        "type": "optical",
        "description": "Additional optical capacity and revisit for wider-area and time-critical collection.",
        "revisit_hours": 24.0,
    },
    {
        "id": "iceye",
        "name": "ICEYE",
        "resolution": "SAR",
        "type": "sar",
        "description": "All-weather radar imaging that sees through cloud, smoke and darkness.",
        "revisit_hours": 3.5,
    },
]

INDUSTRIES: list[dict[str, Any]] = [
    {"aoi": "AOI-07", "category": "Energy & Industrial", "sector": "Oil, gas and petrochemical", "description": "Terminals, pipelines and processing sites under persistent watch."},
    {"aoi": "AOI-21", "category": "Aviation Security", "sector": "Airports and aviation", "description": "Perimeter, apron and approach monitoring."},
    {"aoi": "AOI-18", "category": "Maritime & Trade", "sector": "Ports and shipping lanes", "description": "Vessel tracking, chokepoints and supply chains."},
    {"aoi": "DD-2291", "category": "Mineral Extraction", "sector": "Mining and critical minerals", "description": "Operational validation and geological insight."},
    {"aoi": "AOI-12", "category": "Energy & Utilities", "sector": "Power generation and grid", "description": "Plants, substations and transmission corridors."},
    {"aoi": "AOI-31", "category": "Real Estate & Commercial", "sector": "Smart buildings and commercial", "description": "Campuses, estates and portfolio sites under persistent watch."},
    {"aoi": "AOI-09", "category": "Transport & Rail", "sector": "Rail networks and telecom", "description": "Linear infrastructure across borders."},
    {"aoi": "AOI-03", "category": "Renewables", "sector": "Solar and wind", "description": "Remote generation sites with change detection."},
    {"aoi": "AOI-14", "category": "Water", "sector": "Treatment and reservoirs", "description": "Critical water assets and dams."},
]


def _daily_passes(revisit_hours: float) -> int:
    return max(1, round(24 / revisit_hours))


def collection_windows_for_aoi(bbox: list[float], hours: int = 24) -> dict[str, Any]:
    now = dt.datetime.now(dt.timezone.utc)
    all_windows: list[dict[str, Any]] = []
    next_win: dict[str, Any] | None = None

    for const in CONSTELLATION_DEFS:
        daily = _daily_passes(const["revisit_hours"])
        interval_minutes = 1440 / daily
        offset = abs(hash(const["id"] + str(round(bbox[0], 1)))) % int(interval_minutes)
        t = now + dt.timedelta(minutes=offset)
        while t < now + dt.timedelta(hours=hours):
            mins = int((t - now).total_seconds() / 60)
            w: dict[str, Any] = {
                "constellation_id": const["id"],
                "constellation_name": const["name"],
                "time_utc": t.strftime("%H:%M"),
                "timestamp": t.isoformat(),
                "resolution": const["resolution"],
                "type": const["type"],
                "minutes_from_now": mins,
                "hour_of_day": round(t.hour + t.minute / 60.0, 3),
            }
            all_windows.append(w)
            if next_win is None or mins < next_win["minutes_from_now"]:
                next_win = w
            t += dt.timedelta(minutes=interval_minutes)

    all_windows.sort(key=lambda w: w["minutes_from_now"])

    if next_win:
        m = next_win["minutes_from_now"]
        label = f"{next_win['constellation_name'].split()[-1]} in {m // 60:02d}:{m % 60:02d}"
    else:
        label = "No windows scheduled"

    return {
        "windows": all_windows[:60],
        "total_24h": len(all_windows),
        "next_window": label,
        "best_resolution": "0.35 m",
        "constellations": CONSTELLATION_DEFS,
        "note": "Illustrative schedule. Real windows depend on orbit, weather and tasking priority.",
    }
