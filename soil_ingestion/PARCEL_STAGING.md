# Parcel soil implementation — incomplete staging candidate

This branch implements and tests native-raster parcel aggregation and static-source
cache decisions. It does **not** implement the complete farmer workflow and is not
approved for production. Do not deploy this as a replacement for the current app.

## Changes

- Added `parcel.py`: canonical geometry fingerprint, native 250 m raster checks,
  tight grid-aligned subsets, per-cell polygon/hole intersections, deterministic
  zigzag traversal, area-weighted statistics, missing coverage, per-cell quantiles,
  three default depths and thickness-weighted concentration summaries.
- Reused the property registry, official scaling, validation and WCS endpoint from
  `ingest.py`. REST remains available there for point diagnostics.
- Added `cache.py`: calendar-six-month eligibility and invalidation by geometry,
  source and algorithm. No imaginary change of measurement date on a cache check.
- Added a read-only GeoJSON diagnostic CLI, tests and pinned Shapely dependency.
- Removed no files; changed no legacy endpoints, deployed services or database rows.

## Verified live audit, 2026-11-15

Backend base: `soil-ingestion-hardened-20261115`, SHA
`b67f3437827ed7f2fdedccd7f12774da78397d63` (draft PR #1).
Frontend checkout inspected: `kisanshakti-ai-update` at
`31f3b301c5eba84bd9b7c4b6354cda4abb8710ea`; main package.json was also read.

Files inspected: backend `app.py`, `render.yaml`,
`kisanshakti_enhanced_api_v5.py`, `soil_ingestion/ingest.py`, ingestion requirements,
tests and CI; frontend `AGENTS.md`, `src/services/landsApi.ts`,
`src/pages/SoilHealthReport.tsx`, `supabase/functions/_shared/sessionVerify.ts`,
README and Hostinger workflow. This is not a completed repository-wide audit.

Live Supabase function bundles inspected for soil references: `lands-api` v919,
`ai-smart-schedule` v799, `weather` v748, `ai-agriculture-chat` v103. Soil readers
still consume `soil_health` or land caches; they do not read the verified-source
observation table. Shared session verification calls `verified_session_context()`.
The farmer app uses a custom session token; an anon bearer token is not the farmer
identity. Do not substitute `auth.uid()` without migrating authentication.

Database audit: all 43 lands have valid `boundary_geom` geometry. The source-table
guard checks tenant then requires centroid-coordinate equality. `soil_health`
updates propagate into `lands` and `env_observations`; source classification treats
strings containing `grid` as SoilGrids and everything else as laboratory data.
That classification is insufficient to distinguish manual/default/lab sources.
These triggers were read, not changed.

Legacy V5 uses random Earth Engine points, derives available N/P/K from proxy
properties and defaults, and writes `soil_health`. Render configuration starts V5,
but its final uvicorn call targets `app:app`. Actual Render runtime ownership and
credentials were not verified. The new code deliberately has no database writer.

## Live source evidence

Official layer definitions: https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_01.html
WCS instructions: https://docs.isric.org/globaldata/soilgrids/WCS_from_Python.html

DescribeCoverage for `phh2o_0-5cm_mean` returned native outer origin
(-19949750, 8361000), centre origin (-19949625, 8360875), offsets (250,0)/(0,-250).
An arbitrary unsnapped subset returned pixels **254.22 by 240.88 m**. This explains
why using nominal 250 m GPS bucketing on those returned rasters is invalid. The
new code snaps subset bounds and rejects wrong resolution or origin.

Illustrative coordinate only, not a registered farmer test land: 16.75 N, 74.35 E.
Polygon spans longitude 74.349–74.351, latitude 16.749–16.751.
REST `properties/query`, `phh2o`, `0-5cm`, `mean`: raw 68 → pH 6.8.
WCS native subset, same coverage: four valid cells, parcel area 47465.008400289305 m².

| Pixel upper-left (IGH metres) | Intersection m² | Raw | pH |
| --- | ---: | ---: | ---: |
| 8067000, 1864750 | 45023.08171724991 | 68 | 6.8 |
| 8067250, 1864750 | 342.24433587267197 | 67 | 6.7 |
| 8067250, 1864500 | 132.5209855978053 | 68 | 6.8 |
| 8067000, 1864500 | 1967.1613616051181 | 68 | 6.8 |

Manual check: sum(area × pH)/sum(area) = **6.799278954439476**. Min 6.7,
max 6.8; spatial standard deviation 0.00846077120314707. This deviation describes
map variation, not model uncertainty. Raster SHA256:
`9ebef2833d11f1683a5dfee4b558f8459a371d9afaf34fa953b44b3add86c03b`.
This point/parcel comparison does not prove REST/WCS equality for every pixel,
property or depth. The source release remains explicitly unverified.

## Local verification and usage

Local result: **57 passed** in 0.86 seconds; compileall and git diff --check passed.
There were 47 affine-operator deprecation warnings, no test failures. These are
unit/regression tests, not evidence of a working authenticated farmer workflow.

```sh
python -m pip install -r soil_ingestion/requirements.txt
python -m pytest -q soil_ingestion/tests
python -m compileall -q soil_ingestion
python -m soil_ingestion.parcel_cli --geometry parcel.geojson \
  --source-version VERIFIED_RELEASE_ID --out report.json
```

The CLI defaults to all eight properties, three depths and four statistics. Those
96 small raster requests need a durable worker before farmer use. Do not run them
inside a browser or Edge Function request. Responses have byte and cell caps with
bounded HTTP retries. Global provider rate limiting and worker lease renewal are
not yet implemented. A partial result cannot be treated as a complete baseline.

pH stays depth-specific. Concentration summaries are thickness-weighted, not
nutrient stocks or laboratory composite tests. Weighted pixel quantiles are labelled
as spatial averages, never falsely presented as parcel prediction intervals.

## Release blockers and continuation

1. No isolated Supabase branch exists. Branch creation confirmation expired;
   no staging schema, test farmer, test land or credentials were created.
2. Confirm deployed farmer repository/branch and current deployment commit.
   The live domain serves `/assets/index-BaL1_XJC.js`, without a verified commit.
   Latest Hostinger run 42804660118 failed on `kisanshakti-ai-update`; the latest
   successful workflow run returned by GitHub was 19581140371 on `main`,
   SHA fea0c12c3232f5e217923b51580c81463efaaab8, dated 2025-11-21.
3. Implement additive job/run/report storage, source-cell storage and transactional
   ownership/geometry rechecks. Review centroid guard adaptation without weakening
   other source invariants. Preserve all lab records and keep predictions out of
   existing soil_health propagation until consumers are explicitly adapted.
4. Implement verified-session Edge API, durable leased worker, concurrency and
   idempotency, rate limits, failure/retry status and backend credential contracts.
5. Integrate the confirmed deployed frontend, including empty-report state (the
   inspected SoilHealthReport currently navigates away when no soil_health row exists).
6. Verify source version, all depths/properties/quantiles, rural point equality,
   nodata coverage and real registered-land E2E in isolated staging; then validate
   RLS, cross-tenant access, worker crash recovery and exact saved UI readback.

No full staging E2E, migration rollback, frontend, authentication or RLS tests have
been executed for the new workflow because that workflow is not implemented yet.
Rollback of this code-only draft is simply not merging it; no database rollback
is needed. Nothing was merged or deployed to production.
