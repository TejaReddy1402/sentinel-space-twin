# Sentinel Space Twin — Data Engine v2

This replaces the earlier visual-first mockup with a data-first system.

The goal is not to pretend Sentinel is live CCTV. The goal is to make Sentinel and live geospatial feeds easier to **see, trust, compare and explain**.

## What this version actually does

1. Searches real Sentinel-2 L2A scenes from a STAC catalogue.
2. Selects the best latest/previous clear scenes for an AOI.
3. Scores every scene for usability: cloud cover, recency, asset completeness, comparison suitability.
4. Pulls available scene thumbnails and asset links.
5. Runs a real preview-level before/after change comparison.
6. Produces GeoJSON change cells that can be drawn on the map.
7. Generates a VLM packet: before image, after image, metadata, visible/not-visible limits, and a prompt.
8. Gives a data-first UI with: map, scene cards, thumbnails, quality scores, change overlays, visibility matrix and provenance.

## What this version does not fake

- It does not claim to show real-time video.
- It does not claim to identify people, plates, exact cargo or small vehicles from Sentinel.
- It does not use a generic shiny globe as proof of insight.
- It clearly marks what is visible, inferable, uncertain or needs external APIs.

## Product architecture

```text
Frontend evidence cockpit
        ↓
FastAPI data gateway
        ↓
STAC scene discovery
        ↓
Scene quality scoring
        ↓
Preview enhancement now / real band processing next
        ↓
Change detection GeoJSON
        ↓
VLM packet + analyst explanation
        ↓
Map overlays + evidence report
```

## Quick run

```bash
cd sentinel_space_twin_data_engine_v2
cp .env.example .env
docker compose up --build
```

Open:

```text
Frontend: http://localhost:8080
Backend:  http://localhost:8787/docs
```

## Local backend without Docker

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8787
```

Then open `frontend/index.html` in a browser, or serve it:

```bash
cd frontend
python3 -m http.server 8080
```

## Production upgrades

The code has two processing tiers:

### Tier 1 — runs immediately
Uses STAC metadata + thumbnails. This gives real scene selection, visual comparison, quality scoring and GeoJSON change cells.

### Tier 2 — production processing
Use Sentinel Hub Process API or raw COG processing to calculate real band indices:

- NDVI = vegetation health
- NDWI = water/flood signal
- NBR = burn scar/fire recovery
- NDMI = moisture
- SAR VV/VH = radar/flood/ship/roughness signal
- SCL = cloud/shadow/snow/water/vegetation mask

For Tier 2, configure Sentinel Hub credentials in `.env` or run the raster worker with GDAL/rasterio.

## Team install checklist

- Docker + Docker Compose
- Python 3.11+
- Node only if converting the frontend into React/Vite
- Optional: Sentinel Hub / Copernicus credentials
- Optional: GPU box with Qwen2.5-VL, LLaVA, InternVL or similar
- Optional: AIS/ADS-B/traffic APIs for true live movement layers

## UI principle

The UI should answer four questions immediately:

1. What am I looking at?
2. How fresh is it?
3. How reliable is it?
4. What changed, and what is the evidence?

