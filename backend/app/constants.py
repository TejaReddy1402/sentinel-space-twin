"""Project-wide constants — all magic numbers live here."""
from __future__ import annotations

# --- Scoring weights (must sum to 1.0) ---
SCORE_WEIGHT_CLOUD = 0.55
SCORE_WEIGHT_COMPLETENESS = 0.30
SCORE_WEIGHT_RECENCY = 0.15

# --- Score label thresholds (0–100 scale) ---
SCORE_STRONG = 75.0
SCORE_USABLE = 55.0
SCORE_WEAK = 35.0

# --- Scene scoring helpers ---
CLOUD_SCORE_DIVISOR = 80.0          # cloud % at which cloud_score reaches 0
RECENCY_PENALTY_PER_RANK = 0.075
RECENCY_FLOOR = 0.15

# --- Change detection ---
CHANGE_GRID_CELLS = 8               # NxN grid for change heatmap
CHANGE_THRESHOLD = 0.38             # hot-pixel threshold (0–1 normalised diff)
CHANGE_SCORE_MEAN_WEIGHT = 0.65
CHANGE_SCORE_HOT_WEIGHT = 0.35
CHANGE_CELL_MIN_SCORE = 0.30
DIFF_PERCENTILE = 95

# --- Preview image ---
PREVIEW_WIDTH = 360
PREVIEW_HEIGHT = 240
PREVIEW_JPEG_QUALITY = 88
FETCH_TIMEOUT_SECONDS = 30

# --- Evidence records ---
AOI_CODE_MODULO = 90                # yields AOI-01 .. AOI-90
EVIDENCE_SCHEMA_VERSION = "1.0"

# --- Collection windows ---
COLLECTION_WINDOW_HOURS = 24
COLLECTION_WINDOW_LIMIT = 60

# --- Asset selection ---
THUMBNAIL_ASSET_PRIORITY = ["thumbnail", "rendered_preview", "overview", "visual"]
STAC_COMPLETENESS_ASSETS = ["thumbnail", "visual", "red", "green", "blue", "nir", "nir08"]

# --- Fallback bbox (Rotterdam Port, used only when no AOI is specified) ---
DEFAULT_BBOX: list[float] = [4.0, 51.9, 4.5, 52.0]
