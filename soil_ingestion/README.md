# KisanShakti source-verified soil ingestion

**Release status: guarded integration candidate; not approved for unrestricted farmer-facing production.** This folder contains the sole new soil ingestion implementation, plus tests and isolated migration. Existing `app.py`, Render files, NDVI jobs, and live soil-health functions are not deleted because other services may reference them.

## Sources and field semantics

| Provider | Operational interface | Stored properties | Confidence limitations |
|---|---|---|---|
| ISRIC SoilGrids | WCS GeoTIFF (primary), REST JSON point query (optional beta) | predicted pH, SOC, total N, sand/silt/clay, bulk density, CEC | static 250 m predicted values, depth and Q0.05/Q0.5/Q0.95 recorded; excludes available NPK |
| NASA POWER | daily point API, `community=AG` | `GWETTOP`, `GWETROOT`, `GWETPROF` | **relative wetness fractions** only, NOT volumetric moisture or SMAP measurements |
| NASA SMAP Level 4 V8 | AppEEARS authenticated point task | surface/root-zone volumetric moisture | 9 km model assimilation, not measured field moisture; asynchronous tasks |

The NASA POWER `SFMC` and `RZMC` request produced `-999` for the tested India location across several date ranges. Those properties are NOT enabled in this importer. ISRIC REST returned null pH for an illustrative Pune-city point, while WCS returned all-zero pixels there. At separate rural Maharashtra test points, both services returned valid pH predictions. The importer rejects unavailable values without replacing them with regional defaults.

## Existing tables checked

- `public.lands` owns the verified location and tenant; only active lands with a stored valid centroid can be processed.
- `public.soil_health` has triggers into `lands` and `env_observations` and its source enum does not distinguish NASA wetness. **Never upsert external model predictions here without changing and fully testing the legacy consumers.**
- `public.env_observations` mixes dynamic forecasts, predictions and environmental calculations; the ingestion cannot safely add new properties without a reviewed registry and source model.
- **Only** `public.soil_source_observations` is permitted as destination. SQL is in `sql/001_soil_source_observations.sql`, with restrictive RLS, service-role-only writes, units, provider/property restrictions, idempotent hashes, and a guard checking the target land's tenant and centroid.

## Installation and test

Python 3.12:

```bash
python -m pip install -r soil_ingestion/requirements.txt
python -m pytest -q soil_ingestion/tests
```

Run only from a trusted CI runner or backend. Never put Supabase service-role secrets in a client or browser.

```bash
export SUPABASE_URL=https://YOUR_PROJECT.supabase.co
export SUPABASE_SERVICE_ROLE_KEY=SERVER_SIDE_SECRET

# No writes; result stored locally:
python -m soil_ingestion.ingest --land-id LAND_UUID --provider power \
  --start 2025-01-01 --end 2025-01-07 --out result.json

python -m soil_ingestion.ingest --land-id LAND_UUID --provider soilgrids \
  --depth 0-5cm --out result_soilgrids.json

# Optional ISRIC beta REST; one request for all properties and quantiles:
python -m soil_ingestion.ingest --land-id LAND_UUID --provider soilgrids \
  --soilgrids-transport rest --depth 0-5cm --out result_soilgrids_rest.json

# Start an authenticated NASA SMAP extraction (no soil writes):
export APPEEARS_TOKEN=SHORT_LIVED_TOKEN
python -m soil_ingestion.ingest --land-id LAND_UUID --provider smap-submit \
  --start 2025-01-01 --end 2025-01-03 --out task.json
# Later retrieve exactly the same location's finished task:
python -m soil_ingestion.ingest --land-id LAND_UUID --provider smap-fetch \
  --task-id ORIGINAL_TASK_UUID --out smap.json
```

**Write requires a migration AND two deliberate gates**:

```bash
export SOIL_INGEST_WRITE_ENABLED=yes
python -m soil_ingestion.ingest --land-id LAND_UUID --provider power \
  --start 2025-01-01 --end 2025-01-07 --write --out result.json
```

All records in a run are validated before any write. For a multiple-batch upsert, a late HTTP error can produce a partial batch; rerunning is safe because each row has an idempotent `record_key`. This is not a two-phase atomic transaction across HTTP batches.

## Release blockers

1. Live SoilGrids REST and WCS retrieval succeeded for illustrative rural Maharashtra coordinates, but exact pixel consistency, urban no-data handling, all requested depths, availability and independent field validation must still be audited.
2. NASA SMAP authenticated AppEEARS task must be tested against a real result bundle, including timestamp and accepted CSV column schema, and missing data.
3. Integration staging test must demonstrate actual tenant guard failures, duplicate upsert idempotency, and read restrictions for `anon` and `authenticated` roles.
4. Secrets, monitoring, rate limits, retry/backoff, batching, quality gates and operational ownership must be approved before automated nationwide ingestion.
5. Never convert `GWETTOP` or `GWETROOT` wetness fractions into volumetric moisture without a validated relationship; do not train or infer available fertilizer NPK from SoilGrids total N.

## Verified production-database source ingestion (2026-11-15)

One authorized source-data canary was executed at the coordinates of an existing active land, with a live HTTPS response from NASA POWER. Nine **real modelled daily wetness fractions**, covering 2026-10-01 through 2026-10-03, were upserted to `public.soil_source_observations`. Provider payload explicitly identified source products `MERRA2` and `GEOSIT`, `time_standard=UTC`, and parameter units `1` (fraction). Readback confirmed exactly 9 records, 9 distinct keys, one land, and source product metadata. The original `soil_health` remained at 36 rows.

The canary used an internal SQL integration path, not the CLI's credentialed write path. The CLI's write path remains **unit-tested but not live-tested end to end**. This is not proof of correctness for all lands, providers, or crop decisions. The dataset's spatial grid is much coarser than typical smallholder farm boundaries.

Do not display NASA `GWET*` numbers as volumetric moisture percentages, nor label them as measured by farm sensors.

## SoilGrids official REST and WRB endpoint validation

- Public API specification: `https://rest.isric.org/soilgrids/v2.0/openapi.json`
- Soil-property endpoint: `https://rest.isric.org/soilgrids/v2.0/properties/query` — latitude/longitude, repeated `property`, `depth`, and `value` parameters; returns GeoJSON with layer/depth/quantile values. New optional CLI `--soilgrids-transport rest`.
- WRB class endpoint: `https://rest.isric.org/soilgrids/v2.0/classification/query` — most probable class plus class probabilities; **not** a pH/OC/NPK endpoint. WRB classes are not currently inserted into the numerical soil observations schema.
- WCS remains primary. REST is ISRIC beta, documented temporary outages and 5 requests/minute fair-use limit. Query multiple properties/quantiles in one REST call. No automatic fallback that converts missing properties to defaults.
- A zero pH response for the urban Pune validation point corresponded to a null REST pH; rural Maharashtra test sites returned valid predictions. Treat absent/invalid pixels as missing, not measured zeros.
- Values are *predicted* and not laboratory observations. Still cannot infer available phosphorus or potassium from SoilGrids.

### Live verification: official REST and WCS (2026-11-15)

A read-only GitHub Actions run returned **24 validated SoilGrids REST estimates**, covering eight soil properties × Q0.05/Q0.5/Q0.95 at 0–5 cm for illustrative latitude 16.75, longitude 74.20. Examples: median pH=6.5, SOC=1.27%, total nitrogen=1.1 g/kg, clay=20.4%, sand=40.4%, silt=16.9%, bulk density=1.40 g/cm3, CEC=20.4 cmol(+)/kg. The WCS returned pH 6.6 for the same geographical point; that difference is still being investigated. No SoilGrids data were written to Supabase during this test.

The linked classification endpoint returned Vertisols as the highest-probability class for the illustrative Kolhapur-area location, with 31% class probability. Soil classes are not stored as pH/chemical values. Temporary diagnostic scripts/workflows were removed after live verification; the core REST parser and regression tests remain in the branch.
