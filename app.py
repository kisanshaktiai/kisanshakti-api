# ============================================================
# KisanShaktiAI - Soil Intelligence API (v4.0.0)
# Author: Amarsinh Patil
# Last updated: 2025-10-13 (v4.0.0)
# Description:
#   - Uses lands.boundary_polygon_old to generate in-field sampling points
#   - Fetches SoilGrids (ISRIC) via Google Earth Engine
#   - Weighted depth (0-30cm) per point, then MEAN across points
#   - Calibrated NPK estimation (SHC/ICAR inspired)
#   - Confidence index from variance + completeness
#   - Saves results to Supabase soil_health table and updates lands summary
# ============================================================

import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import ee
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Supabase client
from supabase import create_client, Client

# -----------------------
# Configuration & Logging
# -----------------------
VERSION = "4.0.0"
SERVICE_ACCOUNT = os.getenv("GEE_SERVICE_ACCOUNT", "kisanshaktiai-n@exalted-legacy-456511-b9.iam.gserviceaccount.com")
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase config missing: set SUPABASE_URL and SUPABASE_SERVICE_KEY (or SERVICE_ROLE_KEY)")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("kisanshakti-api")

# -----------------------
# Earth Engine init
# -----------------------
if not os.path.exists(CREDENTIALS_PATH):
    # allow credentials via env var (service account JSON content)
    creds_content = os.getenv("GEE_SERVICE_ACCOUNT_KEY")
    if creds_content:
        with open("credentials.json", "w") as fh:
            fh.write(creds_content)
        CREDENTIALS_PATH = "credentials.json"

try:
    ee_creds = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, key_file=CREDENTIALS_PATH)
    ee.Initialize(ee_creds)
    logger.info("✅ Earth Engine initialized")
except Exception as e:
    # Still runable for endpoints that don't require EE, but fetch endpoints will fail.
    logger.exception("❌ Earth Engine initialization failed - please check credentials: %s", e)

# -----------------------
# App & middleware
# -----------------------
app = FastAPI(title="KisanShakti Geospatial API", version=VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# -----------------------
# SoilGrids settings
# -----------------------
SOILGRID_LAYERS = {
    "ph": "phh2o",
    "organic_carbon": "ocd",
    "nitrogen": "nitrogen",
    "clay": "clay",
    "sand": "sand",
    "silt": "silt",
    "bulk_density": "bdod",
    "cec": "cec",
}
DEPTH_WEIGHTS = {"0-5cm": 0.35, "5-15cm": 0.35, "15-30cm": 0.30}

# unit conversions
ACRES_TO_HECTARES = 0.404686
GUNTAS_TO_HECTARES = 0.010117
M2_PER_HA = 10000.0

# -----------------------
# Utility: scaling and clamping
# -----------------------
def scale_soilgrid_value(raw_value: float, prefix: str) -> Optional[float]:
    """
    Convert SoilGrids internal units to commonly used units.
    ISRIC/SoilGrids conventions vary; we implement commonly used conversions:
      - phh2o: stored as pH * 10 -> divide by 10
      - ocd: organic carbon density (g/m3) -> convert to percent roughly: /10000
      - bdod: bulk density (kg/m3 or cg/cm3 depending) -> divide by 1000 for g/cm3
      - clay/sand/silt: stored g/kg -> convert to % by /10
      - cec: cmol/kg already
      - nitrogen: depending on SoilGrids may be g/kg; we'll keep as raw and treat carefully
    """
    if raw_value is None:
        return None
    try:
        if prefix == "phh2o":
            return round(raw_value / 10.0, 3)
        if prefix == "ocd":
            # convert g/m3 -> % (approx)
            return round(raw_value / 10000.0, 3)
        if prefix == "bdod":
            # convert cg/cm3 or kg/m3 to g/cm3: We assume stored in kg/m3 -> /1000
            return round(raw_value / 1000.0, 3)
        if prefix in ("clay", "sand", "silt"):
            return round(raw_value / 10.0, 3)
        if prefix == "cec":
            return round(raw_value, 3)
        # nitrogen layer sometimes in g/kg - convert to % style or keep raw
        return round(raw_value, 3)
    except Exception:
        return None

def clamp(v: Optional[float], lo: float, hi: float, digits: int = 3) -> Optional[float]:
    if v is None:
        return None
    try:
        v2 = max(lo, min(hi, v))
        return round(v2, digits)
    except Exception:
        return None

# -----------------------
# Classifications (farmer friendly & SHC)
# -----------------------
def classify_ph(ph: Optional[float]) -> Tuple[Optional[str], Optional[str]]:
    if ph is None:
        return None, None
    if ph < 5.5: return "acidic", "Strongly acidic — lime recommended"
    if ph < 6.5: return "slightly_acidic", "Slightly acidic — good for many crops"
    if ph < 7.5: return "neutral", "Neutral — ideal for most crops"
    if ph < 8.5: return "slightly_alkaline", "Slightly alkaline"
    return "alkaline", "Alkaline — may need gypsum"

def classify_oc(oc: Optional[float]) -> Tuple[Optional[str], Optional[str]]:
    if oc is None:
        return None, None
    if oc < 0.5: return "very_low", "Very low — add organic matter (FYM/compost)"
    if oc < 0.75: return "low", "Low — apply compost/FYM"
    if oc < 1.5: return "medium", "Medium — soil health acceptable"
    if oc < 3.0: return "high", "High — good organic matter"
    return "very_high", "Very high — excellent soil"

def classify_nutrient(kg_ha: Optional[float], nutrient: str) -> Tuple[Optional[str], Optional[str]]:
    if kg_ha is None:
        return None, None
    if nutrient == "nitrogen":
        if kg_ha < 180: return "low", "Low — apply N fertilizer (urea/organic manure)"
        if kg_ha < 360: return "medium", "Medium — maintain balanced fertility"
        return "high", "High — only maintenance required"
    if nutrient == "phosphorus":
        if kg_ha < 12: return "low", "Low — apply P fertilizer (DAP/RP)"
        if kg_ha < 24: return "medium", "Medium — maintain levels"
        return "high", "High — sufficient"
    if nutrient == "potassium":
        if kg_ha < 120: return "low", "Low — apply potash (MOP)"
        if kg_ha < 280: return "medium", "Medium — maintain"
        return "high", "High — adequate"
    return None, None

# -----------------------
# Earth Engine robust image loader
# -----------------------
def load_soilgrids_image(prefix: str):
    """
    Attempt multiple ways to load SoilGrids asset:
    1) direct image name projects/soilgrids-isric/{prefix}_mean
    2) image collection projects/soilgrids-isric/{prefix} -> select_mean
    This increases robustness across different SoilGrids deployments.
    """
    try:
        # Attempt common direct image (works in some deployments)
        img = ee.Image(f"projects/soilgrids-isric/{prefix}_mean")
        # quick check: does it have any bands? We'll attempt to select a depth band to verify
        # but selection will happen later; just return img and rely on reduceRegion exceptions handling
        return img
    except Exception:
        # fallback to collection
        try:
            coll = ee.ImageCollection(f"projects/soilgrids-isric/{prefix}")
            img = coll.mosaic().select([s for s in coll.first().bandNames().getInfo()]) if coll.size().getInfo() > 0 else coll.first()
            return img
        except Exception:
            # As last resort return None (caller will handle)
            logger.debug(f"Could not load soilgrids asset for prefix {prefix}")
            return None

# -----------------------
# Weighted per-point fetch
# -----------------------
def get_soil_value_weighted(lat: float, lon: float, prefix: str, scale: int = 250) -> Optional[float]:
    """
    For a sample point (lat, lon) compute weighted average across DEPTH_WEIGHTS (0-30cm).
    Returns scaled & clamped values depending on prefix.
    """
    try:
        img = load_soilgrids_image(prefix)
        if img is None:
            return None

        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(200).bounds()  # smaller buffer to avoid mixing distant pixels for small fields

        values = []
        weights = []
        for depth, w in DEPTH_WEIGHTS.items():
            band = f"{prefix}_{depth}_mean"
            try:
                # compute region pixel value
                rr = img.select(band).reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=region,
                    scale=scale,
                    bestEffort=True,
                    maxPixels=1e8
                )
                res = rr.getInfo()
            except Exception:
                res = None

            if res and band in res and res[band] is not None:
                raw = float(res[band])
                scaled = scale_soilgrid_value(raw, prefix)
                if scaled is not None:
                    values.append(scaled)
                    weights.append(w)

        if not values:
            return None

        # weighted mean
        weighted = sum(v * wt for v, wt in zip(values, weights)) / sum(weights)

        # prefix-specific clamping & rounding
        if prefix == "phh2o":
            return clamp(weighted, 3.0, 10.0, digits=3)
        if prefix == "ocd":
            return clamp(weighted, 0.0, 15.0, digits=3)
        if prefix in ("clay", "sand", "silt"):
            return clamp(weighted, 0.0, 100.0, digits=3)
        if prefix == "bdod":
            return clamp(weighted, 0.5, 2.5, digits=3)
        if prefix == "cec":
            return clamp(weighted, 0.0, 100.0, digits=3)
        # default rounding
        return round(weighted, 3)
    except Exception as e:
        logger.warning("EE fetch error for %s at (%s,%s): %s", prefix, lat, lon, e)
        return None

# -----------------------
# Sampling generator inside polygon
# -----------------------
def calculate_area_from_polygon(polygon_geojson: dict) -> Tuple[float, float]:
    """
    Compute polygon area using Earth Engine.
    Returns (area_ha, area_m2). If there's a failure returns (0.0, 0.0).
    """
    try:
        geom = ee.Geometry(polygon_geojson)
        area_m2 = geom.area().getInfo()
        area_ha = area_m2 / M2_PER_HA
        return round(area_ha, 4), round(area_m2, 2)
    except Exception as e:
        logger.warning("Area calc fail: %s", e)
        return 0.0, 0.0

def generate_sample_points_from_polygon(polygon_geojson: dict, area_m2: float) -> List[Tuple[float, float]]:
    """
    Adaptive sampling count:
      area < 2000 m2 -> 1 (centroid)
      2000-10000 -> 1
      10000-50000 -> 3
      >50000 -> 5
    Use ee.FeatureCollection.randomPoints(region, points, seed) if available; fallback to centroid(s).
    Returns list of (lon, lat) tuples.
    """
    try:
        geom = ee.Geometry(polygon_geojson)
        if area_m2 < 2000:
            n = 1
        elif area_m2 <= 10000:
            n = 1
        elif area_m2 <= 50000:
            n = 3
        else:
            n = 5

        try:
            pts_fc = ee.FeatureCollection.randomPoints(region=geom, points=n, seed=42)
            feats = pts_fc.getInfo().get("features", [])
            coords = [tuple(f["geometry"]["coordinates"]) for f in feats if f and "geometry" in f]
            if coords:
                return coords
        except Exception as e:
            logger.debug("randomPoints failed: %s - fallback to centroid", e)

        # centroid fallback: return a single centroid coordinate
        centroid = geom.centroid().coordinates().getInfo()
        return [tuple(centroid)]
    except Exception as e:
        logger.error("Sample point generation failed: %s", e)
        return []

# -----------------------
# Aggregate across samples (MEAN) + statistics
# -----------------------
def aggregate_mean_with_stats(samples: List[Dict[str, Optional[float]]]) -> Dict[str, Any]:
    """
    Compute mean and standard deviation (and count) across sample dictionaries.
    Returns dict with keys for mean and appended keys `_std` and `_count`.
    """
    from math import sqrt
    agg = {}
    keys = set().union(*[set(s.keys()) for s in samples]) if samples else set()

    for k in keys:
        vals = [s.get(k) for s in samples if s.get(k) is not None and isinstance(s.get(k), (int, float))]
        if not vals:
            agg[k] = None
            agg[f"{k}_std"] = None
            agg[f"{k}_count"] = 0
            continue
        n = len(vals)
        mean = sum(vals) / n
        # sample standard deviation
        if n > 1:
            var = sum((x - mean) ** 2 for x in vals) / (n - 1)
            std = sqrt(var)
        else:
            std = 0.0
        agg[k] = round(mean, 3)
        agg[f"{k}_std"] = round(std, 4)
        agg[f"{k}_count"] = n

    return agg

# -----------------------
# Improved NPK estimator (SHC/ICAR inspired, calibrated)
# -----------------------
def estimate_npk_from_aggregates(agg: Dict[str, Optional[float]], field_area_ha: float) -> Dict[str, Optional[float]]:
    """
    Calibrated empirical formulas (SHC/ICAR inspired). More realistic ranges:
      - N (kg/ha) = 224 * OC% + 60  (gives typical Indian ranges)
      - P (kg/ha) = (6 + 2.5*OC%) * (7.5 - |7 - pH|)  (incorporates pH effect)
      - K (kg/ha) = 5 * CEC  (approximate exchangeable K)
    These are approximate advisory estimates; lab tests should be used for fine management.
    """
    oc = agg.get("organic_carbon")
    ph = agg.get("ph")
    cec = agg.get("cec")
    bd = agg.get("bulk_density") or 1.3

    # defaults if missing - keep None if core missing
    n_kg_ha = None
    p_kg_ha = None
    k_kg_ha = None

    if oc is not None:
        # ensure oc is percent (0.0 - 5.0)
        oc_val = max(0.0, float(oc))
        n_kg_ha = 224.0 * oc_val + 60.0
        n_kg_ha = round(max(0.0, n_kg_ha), 2)

    if oc is not None and ph is not None:
        oc_val = float(oc)
        ph_val = float(ph)
        p_kg_ha = (6.0 + 2.5 * oc_val) * max(0.0, (7.5 - abs(7.0 - ph_val)))
        p_kg_ha = round(max(0.0, p_kg_ha), 2)

    if cec is not None:
        cec_val = float(cec)
        k_kg_ha = 5.0 * cec_val  # heuristic
        k_kg_ha = round(max(0.0, k_kg_ha), 2)

    # compute totals
    n_total = round(n_kg_ha * field_area_ha, 2) if (n_kg_ha is not None and field_area_ha is not None) else None
    p_total = round(p_kg_ha * field_area_ha, 2) if (p_kg_ha is not None and field_area_ha is not None) else None
    k_total = round(k_kg_ha * field_area_ha, 2) if (k_kg_ha is not None and field_area_ha is not None) else None

    return {
        "nitrogen_kg_per_ha": n_kg_ha,
        "phosphorus_kg_per_ha": p_kg_ha,
        "potassium_kg_per_ha": k_kg_ha,
        "nitrogen_total_kg": n_total,
        "phosphorus_total_kg": p_total,
        "potassium_total_kg": k_total,
    }

# -----------------------
# Confidence & completeness
# -----------------------
def compute_confidence_and_completeness(agg_stats: Dict[str, Any], required_fields: List[str]) -> Tuple[str, float, List[str], List[str]]:
    """
    - Completeness: fraction of required fields that are non-null (0-100).
    - Confidence: low/medium/high based on completeness and coefficient of variation across samples.
    Returns (confidence_level, completeness_percent, flags, warnings)
    """
    flags = []
    warnings = []

    # completeness
    present = sum(1 for f in required_fields if agg_stats.get(f) is not None)
    completeness = round((present / len(required_fields)) * 100.0, 1) if required_fields else 0.0

    # compute CV (coefficient of variation) from std and mean for key variables (ph, organic_carbon, cec)
    cv_values = []
    for k in ["ph", "organic_carbon", "cec"]:
        mean = agg_stats.get(k)
        std = agg_stats.get(f"{k}_std")
        if mean is not None and std is not None and mean != 0:
            cv_values.append(std / abs(mean))
    avg_cv = sum(cv_values) / len(cv_values) if cv_values else 0.0

    # decide confidence
    if completeness < 50 or avg_cv > 0.3:
        confidence = "low"
    elif completeness < 80 or avg_cv > 0.15:
        confidence = "medium"
    else:
        confidence = "high"

    # warnings/flags
    if avg_cv > 0.25:
        warnings.append("high_spatial_variability")
    if completeness < 60:
        warnings.append("incomplete_data")
    # pH out-of-range flag
    ph_mean = agg_stats.get("ph")
    if ph_mean is not None and (ph_mean < 3.5 or ph_mean > 10.0):
        flags.append("ph_out_of_range")

    return confidence, completeness, flags, warnings

# -----------------------
# Core process: process a single land
# -----------------------
def process_land_soilgrid(land_id: str, tenant_id: str) -> Dict[str, Any]:
    """
    Full pipeline for a single land:
      - validate tenant
      - check existing soilgrid record (idempotency)
      - fetch polygon & area
      - generate sample points
      - fetch per-point weighted soil properties
      - aggregate by MEAN with stats
      - estimate NPK and totals
      - compute confidence & completeness
      - insert soil_health record and update lands summary
    """
    try:
        # tenant exists?
        tcheck = supabase.table("tenants").select("id").eq("id", tenant_id).execute()
        if not tcheck.data:
            return {"land_id": land_id, "status": "error", "error": "invalid_tenant"}

        # idempotent: skip if source=soilgrid exists for this land
        existing = supabase.table("soil_health").select("id,source").eq("land_id", land_id).eq("source", "soilgrid").execute()
        if existing.data and len(existing.data) > 0:
            return {"land_id": land_id, "status": "already_exists"}

        # fetch land polygon & metadata
        land_row = supabase.table("lands").select("boundary_polygon_old, farmer_id, area_acres, area_guntas").eq("id", land_id).eq("tenant_id", tenant_id).single().execute()
        if not land_row.data:
            return {"land_id": land_id, "status": "error", "error": "land_not_found"}

        polygon_geojson = land_row.data.get("boundary_polygon_old")
        farmer_id = land_row.data.get("farmer_id")
        area_acres = land_row.data.get("area_acres")
        area_guntas = land_row.data.get("area_guntas")

        if not polygon_geojson:
            return {"land_id": land_id, "status": "error", "error": "no_polygon_data"}

        # compute area
        area_ha, area_m2 = calculate_area_from_polygon(polygon_geojson)
        if area_ha == 0.0:
            if area_acres:
                area_ha = round(area_acres * ACRES_TO_HECTARES, 4)
                area_m2 = round(area_acres * 4046.86, 2)
            elif area_guntas:
                area_ha = round(area_guntas * GUNTAS_TO_HECTARES, 4)
                area_m2 = round(area_guntas * 101.17, 2)
            else:
                # fallback default to 1 ha
                area_ha = 1.0
                area_m2 = 10000.0

        # sample points
        points = generate_sample_points_from_polygon(polygon_geojson, area_m2)
        if not points:
            return {"land_id": land_id, "status": "error", "error": "no_sample_points"}

        # fetch soil properties per point
        samples = []
        for lon, lat in points:
            sample = {}
            for logical_key, prefix in SOILGRID_LAYERS.items():
                val = get_soil_value_weighted(lat, lon, prefix)
                # map logical keys to friendly names
                if logical_key == "ph":
                    sample["ph"] = val
                elif logical_key == "organic_carbon":
                    sample["organic_carbon"] = val
                elif logical_key == "nitrogen":
                    sample["nitrogen_raw"] = val
                elif logical_key == "bulk_density":
                    sample["bulk_density"] = val
                elif logical_key == "cec":
                    sample["cec"] = val
                elif logical_key == "clay":
                    sample["clay"] = val
                elif logical_key == "sand":
                    sample["sand"] = val
                elif logical_key == "silt":
                    sample["silt"] = val
            samples.append(sample)

        if not samples:
            return {"land_id": land_id, "status": "error", "error": "no_sample_values"}

        # aggregate mean + stats
        agg_stats = aggregate_mean_with_stats(samples)

        # estimate NPK using calibrated model
        npk = estimate_npk_from_aggregates(agg_stats, area_ha)

        # classifications & texts
        ph_lvl, ph_txt = classify_ph(agg_stats.get("ph"))
        oc_lvl, oc_txt = classify_oc(agg_stats.get("organic_carbon"))
        n_lvl, n_txt = classify_nutrient(npk.get("nitrogen_kg_per_ha"), "nitrogen")
        p_lvl, p_txt = classify_nutrient(npk.get("phosphorus_kg_per_ha"), "phosphorus")
        k_lvl, k_txt = classify_nutrient(npk.get("potassium_kg_per_ha"), "potassium")

        # texture (USDA simplified)
        texture = None
        if agg_stats.get("clay") is not None and agg_stats.get("sand") is not None and agg_stats.get("silt") is not None:
            cl = agg_stats.get("clay"); sa = agg_stats.get("sand"); si = agg_stats.get("silt")
            if cl >= 40:
                texture = "Clay"
            elif cl >= 27:
                texture = "Clay Loam"
            elif sa > 70:
                texture = "Sandy Loam"
            elif si > 50:
                texture = "Silt Loam"
            else:
                texture = "Loam"

        # compute confidence & completeness
        required = ["ph", "organic_carbon", "cec"]
        confidence, completeness, flags, warnings = compute_confidence_and_completeness(agg_stats, required)

        # prepare record
        record = {
            "land_id": land_id,
            "tenant_id": tenant_id,
            "farmer_id": farmer_id,
            "source": "soilgrid",
            "test_date": datetime.utcnow().date().isoformat(),
            # aggregated means
            "ph_level": agg_stats.get("ph"),
            "organic_carbon": agg_stats.get("organic_carbon"),
            "bulk_density": agg_stats.get("bulk_density"),
            "cec": agg_stats.get("cec"),
            "clay_percent": agg_stats.get("clay"),
            "sand_percent": agg_stats.get("sand"),
            "silt_percent": agg_stats.get("silt"),
            "texture": texture,
            "field_area_ha": area_ha,
            # npk
            "nitrogen_kg_per_ha": npk.get("nitrogen_kg_per_ha"),
            "phosphorus_kg_per_ha": npk.get("phosphorus_kg_per_ha"),
            "potassium_kg_per_ha": npk.get("potassium_kg_per_ha"),
            "nitrogen_total_kg": npk.get("nitrogen_total_kg"),
            "phosphorus_total_kg": npk.get("phosphorus_total_kg"),
            "potassium_total_kg": npk.get("potassium_total_kg"),
            # classification texts
            "ph_text": ph_txt,
            "organic_carbon_text": oc_txt,
            "nitrogen_text": n_txt,
            "phosphorus_text": p_txt,
            "potassium_text": k_txt,
            "nitrogen_level": n_lvl,
            "phosphorus_level": p_lvl,
            "potassium_level": k_lvl,
            # data quality & metadata
            "data_quality_flags": flags or [],
            "data_quality_warnings": warnings or [],
            "data_completeness": completeness,
            "confidence_level": confidence,
            "note": "Generated from SoilGrids (weighted 0-30cm). Mean across in-field samples used.",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

        # remove None values & ensure rounding to DB-friendly precision
        record_clean = {}
        for k, v in record.items():
            if v is None:
                continue
            if isinstance(v, float):
                # reasonable rounding for DB columns
                record_clean[k] = round(v, 4)
            else:
                record_clean[k] = v

               # Insert into soil_health
        ins = supabase.table("soil_health").insert(record_clean).execute()

        # ✅ Fixed Supabase response check
        if getattr(ins, "error", None):
            logger.error(f"DB insert failed for land {land_id}: {ins.error}")
            return {"land_id": land_id, "status": "error", "error": str(ins.error)}

        logger.info(f"✅ Soil data inserted for land {land_id}")

        # Update lands summary columns
        update_payload = {}
        if record_clean.get("ph_level") is not None:
            update_payload["soil_ph"] = record_clean["ph_level"]
        if record_clean.get("organic_carbon") is not None:
            update_payload["organic_carbon_percent"] = record_clean["organic_carbon"]
        if record_clean.get("nitrogen_kg_per_ha") is not None:
            update_payload["nitrogen_kg_per_ha"] = record_clean["nitrogen_kg_per_ha"]
        if record_clean.get("phosphorus_kg_per_ha") is not None:
            update_payload["phosphorus_kg_per_ha"] = record_clean["phosphorus_kg_per_ha"]
        if record_clean.get("potassium_kg_per_ha") is not None:
            update_payload["potassium_kg_per_ha"] = record_clean["potassium_kg_per_ha"]
        update_payload["last_soil_test_date"] = record_clean.get("test_date", datetime.utcnow().date().isoformat())

        if update_payload:
            supabase.table("lands").update(update_payload).eq("id", land_id).eq("tenant_id", tenant_id).execute()


        # return success, include a short summary of computed values & confidence
        return {
            "land_id": land_id,
            "status": "saved",
            "record_id": ins.data[0]["id"] if ins.data else None,
            "summary": {
                "ph": record_clean.get("ph_level"),
                "organic_carbon": record_clean.get("organic_carbon"),
                "N_kg_ha": record_clean.get("nitrogen_kg_per_ha"),
                "P_kg_ha": record_clean.get("phosphorus_kg_per_ha"),
                "K_kg_ha": record_clean.get("potassium_kg_per_ha"),
                "confidence": confidence,
                "completeness": completeness
            }
        }

    except Exception as e:
        logger.exception("Error processing land %s: %s", land_id, e)
        return {"land_id": land_id, "status": "error", "error": str(e)}

# -----------------------
# API Models & Endpoints
# -----------------------
class BatchRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant UUID")
    land_ids: List[str] = Field(..., description="List of land UUIDs to process")

@app.post("/soil/save", tags=["Soil Analysis"])
async def soil_save_endpoint(request: Request, tenant_id: Optional[str] = Query(None)):
    """
    POST /soil/save
    Modes:
      - Query param tenant_id and JSON body { "land_ids": [...] }  => batch
      - OR send single-land body { "tenant_id", "land_id" } to process one land.
    Response: summary + per-land results
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    tenant = tenant_id or body.get("tenant_id")
    if not tenant:
        raise HTTPException(status_code=400, detail="tenant_id required (query or body)")

    land_ids = body.get("land_ids") or ([body.get("land_id")] if body.get("land_id") else None)
    if not land_ids:
        raise HTTPException(status_code=400, detail="land_ids required in body (array) or use land_id")

    results = []
    for lid in land_ids:
        res = process_land_soilgrid(lid, tenant)
        results.append(res)

    summary = {
        "processed": len(land_ids),
        "saved": sum(1 for r in results if r.get("status") == "saved"),
        "skipped": sum(1 for r in results if r.get("status") == "already_exists"),
        "failed": sum(1 for r in results if r.get("status") == "error"),
    }
    return {"status": "done", "summary": summary, "details": results, "timestamp": datetime.utcnow().isoformat()}

@app.get("/soil/preview", tags=["Soil Analysis"])
async def soil_preview(lat: float = Query(...), lon: float = Query(...), include_classifications: bool = Query(True)):
    """
    GET /soil/preview?lat={lat}&lon={lon}
    Quick preview endpoint: returns weighted 0-30cm mean for a single coordinate (no polygon sampling).
    """
    try:
        result = {}
        for k, prefix in SOILGRID_LAYERS.items():
            v = get_soil_value_weighted(lat, lon, prefix)
            result[k] = v
        # compute texture if possible
        texture = None
        if result.get("clay") is not None and result.get("sand") is not None and result.get("silt") is not None:
            cl = result["clay"]; sa = result["sand"]; si = result["silt"]
            if cl >= 40:
                texture = "Clay"
            elif cl >= 27:
                texture = "Clay Loam"
            elif sa > 70:
                texture = "Sandy Loam"
            elif si > 50:
                texture = "Silt Loam"
            else:
                texture = "Loam"
        out = {
            "latitude": lat,
            "longitude": lon,
            "data": {
                "ph_level": result.get("ph"),
                "organic_carbon": result.get("organic_carbon"),
                "clay_percent": result.get("clay"),
                "sand_percent": result.get("sand"),
                "silt_percent": result.get("silt"),
                "bulk_density": result.get("bulk_density"),
                "cec": result.get("cec"),
                "texture": texture,
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        if include_classifications:
            out["classifications"] = {
                "ph": classify_ph(result.get("ph")),
                "organic_carbon": classify_oc(result.get("organic_carbon")),
            }
        return out
    except Exception as e:
        logger.exception("Preview error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", tags=["Health"])
async def health():
    try:
        ee.Number(1).getInfo()
        ee_status = "connected"
    except Exception as e:
        ee_status = f"disconnected: {e}"
    return {"status": "ok", "version": VERSION, "earth_engine": ee_status, "timestamp": datetime.utcnow().isoformat()}

@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse(status_code=404, content={"error": "Endpoint not found", "available": ["/soil/save", "/soil/preview", "/health"]})

# -----------------------
# Run server
# -----------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    logger.info(f"Starting KisanShakti API v{VERSION} on port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")
