"""Chain-of-custody evidence record generation with SHA-256 integrity sealing."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import uuid
from typing import Any

from .constants import AOI_CODE_MODULO, EVIDENCE_SCHEMA_VERSION


def _aoi_code(name: str) -> str:
    """
    Derive a deterministic AOI code from the AOI name using SHA-256.

    The code is stable across server restarts and processes, yielding
    identifiers in the range AOI-01 to AOI-{AOI_CODE_MODULO}.
    """
    digest = int(hashlib.sha256(name.encode()).hexdigest(), 16)
    return f"AOI-{digest % AOI_CODE_MODULO + 1:02d}"


def generate_record(
    aoi_name: str,
    bbox: list[float],
    event: str,
    sensor: str = "EO, space-based ISR",
    scene_id: str | None = None,
) -> dict[str, Any]:
    """
    Generate a SHA-256 sealed chain-of-custody evidence record.

    The record captures the AOI, sensor, event description and scene ID.
    A deterministic integrity hash is computed over the canonical JSON
    representation of the core fields so the record can be independently
    verified.
    """
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
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "integrity": f"SHA-256 {full_hash[:4]}…{full_hash[-4:]}, sealed",
        "integrity_full": full_hash,
        "chain_of_custody": "sealed",
    }
