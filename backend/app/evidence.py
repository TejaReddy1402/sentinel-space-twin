from __future__ import annotations

import hashlib
import datetime as dt
import json
import uuid
from typing import Any


def _aoi_code(name: str) -> str:
    # Use SHA-256 for a deterministic (non-randomised) code that survives server restarts.
    digest = int(hashlib.sha256(name.encode()).hexdigest(), 16)
    return f"AOI-{digest % 90 + 1:02d}"


def generate_record(
    aoi_name: str,
    bbox: list[float],
    event: str,
    sensor: str = "EO, space-based ISR",
    scene_id: str | None = None,
) -> dict[str, Any]:
    uid = str(uuid.uuid4()).upper().replace("-", "")
    record_id = f"IPX-{uid[:4]}-{uid[4:8]}"
    captured_at = dt.datetime.now(dt.timezone.utc)
    aoi_code = _aoi_code(aoi_name)

    core: dict[str, Any] = {
        "id": record_id,
        "captured": captured_at.strftime("%Y-%m-%d %H:%M:%SZ"),
        "location": aoi_code,
        "aoi_name": aoi_name,
        "bbox": bbox,
        "sensor": sensor,
        "event": event,
    }
    if scene_id:
        core["scene_id"] = scene_id

    full_hash = hashlib.sha256(
        json.dumps(core, sort_keys=True).encode()
    ).hexdigest()

    return {
        **core,
        "integrity": f"SHA-256 {full_hash[:4]}…{full_hash[-4:]}, sealed",
        "integrity_full": full_hash,
        "chain_of_custody": "sealed",
    }
