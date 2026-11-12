# Data enhancement design

The product should improve the raw data in five ways:

## 1. Scene selection

Bad satellite data is worse than no data. Every scene should be scored before it appears as evidence.

Score components:

- Cloud cover
- Asset completeness
- Recency
- AOI coverage
- Availability of SCL/cloud mask
- Sensor type suitability
- Comparison suitability against previous scene

## 2. Band-level processing

Preview thumbnails are useful for triage, but production decisions need raw bands.

Sentinel-2 bands:

- B02 blue
- B03 green
- B04 red
- B08 near infrared
- B11/B12 short-wave infrared
- SCL scene classification layer

Indices:

- NDVI vegetation
- NDWI water/flood
- NBR burn scar
- NDMI moisture
- bare-soil / brightness deltas

## 3. Cloud and uncertainty handling

Every output must carry uncertainty. Do not show a clean green/red map without telling the user if clouds, shadows, snow, seasonality or sensor angle affected it.

## 4. Change detection

Recommended production path:

1. Get latest clear scene.
2. Get previous comparable clear scene.
3. Align/resample to AOI grid.
4. Apply SCL cloud/shadow mask.
5. Calculate spectral indices for both.
6. Difference each index.
7. Threshold per use case.
8. Polygonise changed pixels.
9. Score each polygon by area, intensity, cloud confidence and persistence.
10. Pass evidence crop + metadata to VLM.

## 5. VLM role

The VLM is not the sensor. It is the analyst assistant.

Good VLM input:

- latest RGB crop
- previous RGB crop
- false colour crop
- NDVI/NDWI/NBR change maps
- GeoJSON changed polygons
- cloud cover and acquisition timestamps
- sensor/resolution metadata
- visibility matrix

Bad VLM input:

- raw promise of “live satellite”
- one screenshot with no timestamp
- no cloud information
- no resolution warning
- asking it to identify small objects below sensor capability

