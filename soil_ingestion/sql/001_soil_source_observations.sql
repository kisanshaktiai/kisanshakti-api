-- STAGING-FIRST MIGRATION; do not apply to a live project before integration tests.
-- No triggers, no writes to soil_health, lands, or env_observations.
CREATE TABLE IF NOT EXISTS public.soil_source_observations (
  record_key char(64) PRIMARY KEY,
  land_id uuid NOT NULL REFERENCES public.lands(id) ON DELETE CASCADE,
  tenant_id uuid NOT NULL REFERENCES public.tenants(id) ON DELETE CASCADE,
  provider text NOT NULL CHECK (provider IN ('ISRIC_SOILGRIDS','NASA_POWER','NASA_SMAP_L4_APPEEARS')),
  property_code text NOT NULL CHECK (property_code IN (
    'soil_ph','soil_organic_carbon','soil_total_nitrogen','soil_clay',
    'soil_sand','soil_silt','soil_bulk_density','soil_cec',
    'soil_wetness_surface_fraction','soil_wetness_rootzone_fraction',
    'soil_wetness_profile_fraction','soil_moisture_surface',
    'soil_moisture_rootzone','soil_moisture_profile')),
  value numeric NOT NULL,
  unit text NOT NULL,
  depth_top_cm integer,
  depth_bottom_cm integer,
  observed_at timestamptz,
  statistic text NOT NULL,
  source_reference text NOT NULL,
  source_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  requested_lat numeric NOT NULL CHECK (requested_lat BETWEEN -90 AND 90),
  requested_lon numeric NOT NULL CHECK (requested_lon BETWEEN -180 AND 180),
  retrieved_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT soil_source_depth_check CHECK (
    (depth_top_cm IS NULL AND depth_bottom_cm IS NULL) OR
    (depth_top_cm >= 0 AND depth_bottom_cm > depth_top_cm)),
  CONSTRAINT soil_source_provider_property_check CHECK (
    (provider='ISRIC_SOILGRIDS' AND property_code IN
      ('soil_ph','soil_organic_carbon','soil_total_nitrogen','soil_clay','soil_sand','soil_silt','soil_bulk_density','soil_cec') AND observed_at IS NULL)
    OR
    (provider='NASA_POWER' AND property_code IN
      ('soil_wetness_surface_fraction','soil_wetness_rootzone_fraction','soil_wetness_profile_fraction') AND observed_at IS NOT NULL)
    OR
    (provider='NASA_SMAP_L4_APPEEARS' AND property_code IN
      ('soil_moisture_surface','soil_moisture_rootzone','soil_moisture_profile') AND observed_at IS NOT NULL)),
  CONSTRAINT soil_source_unit_value_check CHECK (
    CASE property_code
      WHEN 'soil_ph' THEN unit='pH' AND value BETWEEN 2 AND 12
      WHEN 'soil_organic_carbon' THEN unit='%' AND value BETWEEN 0 AND 35
      WHEN 'soil_total_nitrogen' THEN unit='g/kg' AND value BETWEEN 0 AND 100
      WHEN 'soil_clay' THEN unit='%' AND value BETWEEN 0 AND 100
      WHEN 'soil_sand' THEN unit='%' AND value BETWEEN 0 AND 100
      WHEN 'soil_silt' THEN unit='%' AND value BETWEEN 0 AND 100
      WHEN 'soil_bulk_density' THEN unit='g/cm3' AND value BETWEEN 0 AND 3
      WHEN 'soil_cec' THEN unit='cmol(+)/kg' AND value BETWEEN 0 AND 200
      WHEN 'soil_wetness_surface_fraction' THEN unit='fraction' AND value BETWEEN 0 AND 1
      WHEN 'soil_wetness_rootzone_fraction' THEN unit='fraction' AND value BETWEEN 0 AND 1
      WHEN 'soil_wetness_profile_fraction' THEN unit='fraction' AND value BETWEEN 0 AND 1
      ELSE unit='m3/m3' AND value BETWEEN 0 AND 1
    END)
);
CREATE INDEX IF NOT EXISTS soil_source_land_time_idx
 ON public.soil_source_observations (tenant_id, land_id, provider, observed_at DESC);
ALTER TABLE public.soil_source_observations ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON TABLE public.soil_source_observations FROM PUBLIC, anon, authenticated;
GRANT SELECT, INSERT, UPDATE ON TABLE public.soil_source_observations TO service_role;
-- No SELECT policy for app roles. Only trusted ingestion service may import.
-- Upsert conflict target: record_key. NEVER insert into soil_health from this pipeline.
