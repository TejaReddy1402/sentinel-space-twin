"""
Segment Anything Model integration for satellite imagery.

Uses MobileSAM if installed (pip install mobile-sam).
Falls back to SLIC superpixels (scikit-image) automatically.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).parent.parent / "models"
_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── land-cover height + colour ──────────────────────────────────────────────

_CLASS_HEIGHT: dict[str, float] = {
    "water": 0.0,
    "vegetation": 15.0,
    "bare_ground": 5.0,
    "urban": 30.0,
    "cloud_snow": 2.0,
    "unknown": 10.0,
}

# RGBA for deck.gl getFillColor
_CLASS_RGBA: dict[str, list[int]] = {
    "water":       [30,  110, 220, 190],
    "vegetation":  [30,  175,  65, 190],
    "bare_ground": [200, 160,  80, 170],
    "urban":       [150, 150, 160, 190],
    "cloud_snow":  [220, 230, 255, 130],
    "unknown":     [120, 120, 140, 150],
}


def _classify_rgb(mean_rgb: np.ndarray) -> str:
    r, g, b = float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])
    total = r + g + b + 1e-6
    rn, gn, bn = r / total, g / total, b / total
    brightness = (r + g + b) / (3 * 255 + 1e-6)
    if brightness > 0.78:
        return "cloud_snow"
    if bn > rn * 1.15 and bn > gn * 0.95 and brightness < 0.46:
        return "water"
    if gn > rn * 1.05 and gn > bn * 1.05:
        return "vegetation"
    spread = max(rn, gn, bn) - min(rn, gn, bn)
    if spread < 0.12 and 0.3 < brightness < 0.75:
        return "urban"
    if rn > bn * 1.15 and brightness > 0.3:
        return "bare_ground"
    return "unknown"


# ── polygon conversion ──────────────────────────────────────────────────────

def _mask_to_geo_ring(
    mask: np.ndarray,
    img_w: int,
    img_h: int,
    bbox: list[float],
) -> list[list[float]] | None:
    try:
        from skimage.measure import find_contours, approximate_polygon
    except ImportError:
        return None

    contours = find_contours(mask.astype(np.float32), 0.5)
    if not contours:
        return None

    contour = max(contours, key=len)
    tol = max(1.0, min(img_w, img_h) / 160.0)
    contour = approximate_polygon(contour, tolerance=tol)
    if len(contour) < 4:
        return None

    minx, miny, maxx, maxy = bbox
    lon_range = maxx - minx
    lat_range = maxy - miny

    ring: list[list[float]] = []
    for row, col in contour:
        lon = round(minx + (col / img_w) * lon_range, 6)
        lat = round(maxy - (row / img_h) * lat_range, 6)
        ring.append([lon, lat])

    if ring[0] != ring[-1]:
        ring.append(ring[0])

    # Ensure CCW (positive signed area)
    n = len(ring) - 1
    area2 = sum(
        ring[i][0] * ring[(i + 1) % n][1] - ring[(i + 1) % n][0] * ring[i][1]
        for i in range(n)
    )
    if area2 < 0:
        ring = ring[::-1]

    return ring


# ── SAM lazy loader ─────────────────────────────────────────────────────────

_sam_gen = None
_sam_attempted = False


def _load_sam() -> bool:
    global _sam_gen, _sam_attempted
    if _sam_attempted:
        return _sam_gen is not None
    _sam_attempted = True
    try:
        import torch
        from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

        weights = _MODELS_DIR / "mobile_sam.pt"
        if not weights.exists():
            logger.info("Downloading MobileSAM weights (~40 MB) from HuggingFace …")
            try:
                from huggingface_hub import hf_hub_download
                downloaded = hf_hub_download(
                    repo_id="dhkim2810/MobileSAM",
                    filename="mobile_sam.pt",
                    local_dir=str(_MODELS_DIR),
                )
                # hf_hub_download may place the file in a sub-cache; move it
                import shutil
                downloaded_path = Path(downloaded)
                if downloaded_path.resolve() != weights.resolve():
                    shutil.copy2(str(downloaded_path), str(weights))
            except Exception as hf_exc:
                logger.warning("HuggingFace download failed (%s), trying urllib …", hf_exc)
                import urllib.request
                urllib.request.urlretrieve(
                    "https://huggingface.co/dhkim2810/MobileSAM/resolve/main/mobile_sam.pt",
                    str(weights),
                )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        m = sam_model_registry["vit_t"](checkpoint=str(weights))
        m.to(device=device).eval()
        _sam_gen = SamAutomaticMaskGenerator(
            m,
            points_per_side=16,
            pred_iou_thresh=0.87,
            stability_score_thresh=0.93,
            min_mask_region_area=150,
            output_mode="binary_mask",
        )
        logger.info("MobileSAM ready on %s", device)
        return True
    except ModuleNotFoundError as exc:
        if "mobile_sam" in str(exc) or "mobile-sam" in str(exc):
            logger.info("mobile-sam not installed — using SLIC fallback")
        else:
            logger.warning("MobileSAM dependency missing (%s) — using SLIC fallback", exc)
        return False
    except Exception as exc:
        logger.warning("MobileSAM load failed (%s) — using SLIC fallback", exc)
        return False


def _slic_masks(arr: np.ndarray, n: int) -> list[np.ndarray]:
    from skimage.segmentation import slic
    labels = slic(arr, n_segments=n, compactness=10, sigma=1,
                  channel_axis=2, convert2lab=True, start_label=0)
    return [labels == lbl for lbl in np.unique(labels) if (labels == lbl).sum() > 60]


# ── public API ───────────────────────────────────────────────────────────────

def segment_image(
    image: Image.Image,
    bbox: list[float],
    previous_image: Image.Image | None = None,
    max_segments: int = 36,
) -> dict[str, Any]:
    """
    Segment a satellite thumbnail and return a geographic GeoJSON FeatureCollection.

    Each feature has:
      land_cover, height (metres for 3-D extrusion), change_score,
      area_pct, mean_rgb, color (RGBA for deck.gl).
    """
    try:
        from skimage.segmentation import slic as _  # noqa: F401
    except ImportError:
        return {
            "type": "FeatureCollection", "features": [],
            "method": "unavailable", "segment_count": 0, "sam_available": False,
        }

    target_w = 512
    orig_w, orig_h = image.size
    scale = min(target_w / orig_w, 1.0)
    w, h = int(orig_w * scale), int(orig_h * scale)

    img = image.resize((w, h), Image.Resampling.LANCZOS).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)

    prev_arr: np.ndarray | None = None
    if previous_image is not None:
        prev_arr = np.asarray(
            previous_image.resize((w, h), Image.Resampling.LANCZOS).convert("RGB"),
            dtype=np.float32,
        )

    use_sam = _load_sam()
    if use_sam and _sam_gen is not None:
        try:
            raw = _sam_gen.generate(arr)
            raw.sort(key=lambda m: m["area"], reverse=True)
            masks: list[np.ndarray] = [m["segmentation"] for m in raw[:max_segments]]
            method = "mobile_sam"
        except Exception as exc:
            logger.warning("SAM generate failed: %s — SLIC fallback", exc)
            masks = _slic_masks(arr, max_segments)
            method = "slic_fallback"
    else:
        masks = _slic_masks(arr, max_segments)
        method = "slic"

    arr_f = arr.astype(np.float32)
    features: list[dict[str, Any]] = []

    for mask in masks:
        if not mask.any():
            continue
        ring = _mask_to_geo_ring(mask, w, h, bbox)
        if ring is None or len(ring) < 4:
            continue

        region = arr_f[mask]
        mean_rgb = region.mean(axis=0)
        label = _classify_rgb(mean_rgb)
        height = _CLASS_HEIGHT[label]

        change_score = 0.0
        if prev_arr is not None:
            change_score = float(np.abs(arr_f[mask] - prev_arr[mask]).mean()) / 255.0
            height += change_score * 100.0

        pixel_count = int(mask.sum())
        area_pct = round(pixel_count / (w * h) * 100, 2)

        features.append({
            "type": "Feature",
            "properties": {
                "land_cover":   label,
                "height":       round(height, 1),
                "change_score": round(change_score, 4),
                "area_pct":     area_pct,
                "mean_rgb":     [round(float(x), 1) for x in mean_rgb[:3]],
                "color":        list(_CLASS_RGBA[label]),
                "method":       method,
            },
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })

    features.sort(key=lambda f: f["properties"]["area_pct"], reverse=True)

    return {
        "type": "FeatureCollection",
        "features": features,
        "method": method,
        "segment_count": len(features),
        "sam_available": use_sam,
    }
