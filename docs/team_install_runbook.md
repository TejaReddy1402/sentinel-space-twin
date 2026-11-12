# Team install runbook

## Run now

```bash
cp .env.example .env
docker compose up --build
```

Open:

- Frontend: http://localhost:8080
- API docs: http://localhost:8787/docs

## First acceptance test

1. Select Rotterdam Port.
2. Click Run data enhancement.
3. Confirm scenes appear in the left scene list.
4. Confirm map draws AOI, scene footprint and change cells.
5. Confirm Evidence tab shows before/after thumbnails and a change preview if thumbnails are available.
6. Confirm VLM Packet tab contains JSON with image links, metadata and rules.

## Production deployment

Recommended services:

- API: FastAPI on ECS/Kubernetes/Fly/Render/Azure Container Apps
- Frontend: S3/CloudFront, Vercel or internal static hosting
- Persistent store: PostGIS
- Raster object store: S3/MinIO
- Queue: Redis + RQ/Celery
- Workers: raster processing, SAR processing, VLM report generation
- Observability: OpenTelemetry + structured logs

## Credentials

Add these when ready:

```text
SENTINEL_HUB_CLIENT_ID=
SENTINEL_HUB_CLIENT_SECRET=
```

## Production backlog

- Replace thumbnail diff with raw COG/Sentinel Hub band processing.
- Add SCL cloud/shadow mask.
- Add Sentinel-1 SAR for cloud-safe monitoring.
- Add tiler endpoint for NDVI/NDWI/NBR as map overlays.
- Add live AIS/ADS-B/traffic connectors.
- Add VLM inference worker.
- Add user AOI saving and alert subscriptions.

