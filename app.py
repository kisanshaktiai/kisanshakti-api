# ============================================================
# 🌾 KisanShaktiAI - Soil Intelligence API
# Version: v3.7.1
# Author: Amarsinh Patil
# ------------------------------------------------------------
# Notes:
# - Uses lands.boundary_polygon_old to generate sampling points
# - Samples = adaptive (1 / 3 / 5) based on area; default 5 for large fields
# - For each sample point: fetch SoilGrids weighted 0-30cm values
# - Aggregation: save MIN value across samples (conservative)
# - Updates lands summary fields and inserts soil_health with source 'soilgrid'
# - Idempotent: skips if soilgrid record already exists for the land
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

# ======================
# Config & Logging
# ======================
VERSION = "3.7.1"
SERVICE_ACCOUNT = "kisanshaktiai-n@exalted-legacy-456511-b9.iam.gserviceaccount.com"
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase config missing: set SUPABASE_URL and SUPABASE_SERVICE_KEY (or SERVICE_ROLE_KEY)")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("kisanshakti-api")

# ======================
# Initialize Earth Engine
# ======================
if not os.path.exists(CREDENTIALS_PATH):
    creds = os.getenv("GEE_SERVICE_ACCOUNT_KEY")
    if creds:
        with open("credentials.json", "w") as fh:
            fh.write(creds)
        CREDENTIALS_PATH = "credentials.json"

try:
    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, key_file=CREDENTIALS_PATH)
    ee.Initialize(credentials)
    logger.info("✅ Earth Engine initialized")
except Exception as e:
    logger.error(f"❌ Earth Engine init failed: {e}")

# ======================
# FastAPI App
# ======================
app = FastAPI(title="KisanShakti Geospatial API", version=VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ======================
# SoilGrids config
# ======================
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
# Weighted agricultural depths (0-30 cm)
DEPTH_WEIGHTS = {"0-5cm": 0.35, "5-15cm": 0.35, "15-30cm": 0.30}

# unit conversions
ACRES_TO_HECTARES = 0.404686
GUNTAS_TO_HECTARES = 0.010117
M2_PER_HA = 10000.0

# ======================
# Utilities: scaling, clamping, classification
# ======================
def scale_soilgrid_value(raw_value: float, prefix: str) -> Optional[float]:
    """Scale SoilGrids raw values to human units (ISRIC conventions)."""
    if raw_value is None:
        return None
    try:
        if prefix == "phh2o":
            return round(raw_value / 10.0, 2)  # pH stored as pH * 10
        if prefix == "ocd":
            return round(raw_value / 10000.0, 3)  # g/m3 -> % (approx)
        if prefix == "bdod":
            return round(raw_value / 1000.0, 3)  # cg/cm3 -> g/cm3
        if prefix in ("clay", "sand", "silt"):
            return round(raw_value / 10.0, 2)  # g/kg -> %
        return round(raw_value, 3)
    except Exception:
        return None

def clamp_value(v: Optional[float], lo: float, hi: float) -> Optional[float]:
    if v is None:
        return None
    return round(max(lo, min(hi, v)), 3)

def classify_ph(ph: Optional[float]) -> Tuple[Optional[str], Optional[str]]:
    if ph is None:
        return None, None
    if ph < 5.5:
        return "acidic", "Strongly acidic - lime required"
    if ph < 6.5:
        return "slightly_acidic", "Slightly acidic - good for most crops"
    if ph < 7.5:
        return "neutral", "Neutral - ideal for many crops"
    if ph < 8.5:
        return "slightly_alkaline", "Slightly alkaline"
    return "alkaline", "Alkaline - consider gypsum"

def classify_oc(oc: Optional[float]) -> Tuple[Optional[str], Optional[str]]:
    if oc is None:
        return None, None
    if oc < 0.5:
        return "very_low", "Very low - add organic matter"
    if oc < 0.75:
        return "low", "Low - apply compost/FYM"
    if oc < 1.5:
        return "medium", "Medium - acceptable"
    if oc < 3.0:
        return "high", "High - good"
    return "very_high", "Very high - excellent"

def classify_nutrient(kg_ha: Optional[float], nutrient: str) -> Tuple[Optional[str], Optional[str]]:
    if kg_ha is None:
        return None, None
    if nutrient == "nitrogen":
        if kg_ha < 280: return "low", "Low - apply urea or organic N"
        if kg_ha < 560: return "medium", "Medium - maintain"
        return "high", "High - sufficient"
    if nutrient == "phosphorus":
        if kg_ha < 12: return "low", "Low - apply P fertilizer"
        if kg_ha < 24: return "medium", "Medium"
        return "high", "High"
    if nutrient == "potassium":
        if kg_ha < 120: return "low", "Low - apply potassium fertilizer"
        if kg_ha < 280: return "medium", "Medium"
        return "high", "High"
    return None, None

# ======================
# Earth Engine fetch (weighted average per parameter)
# ======================
def get_soil_value_weighted(lat: float, lon: float, prefix: str, scale: int = 250) -> Optional[float]:
    """
    Weighted average over DEPTH_WEIGHTS for agricultural relevance (0-30 cm).
    Returns scaled & clamped value.
    """
    try:
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(300).bounds()
        img = ee.Image(f"projects/soilgrids-isric/{prefix}_mean")

        values = []
        weights = []
        for depth, w in DEPTH_WEIGHTS.items():
            band = f"{prefix}_{depth}_mean"
            try:
                res = img.select(band).reduceRegion(
                    reducer=ee.Reducer.first(), geometry=region, scale=scale, bestEffort=True, maxPixels=1e8
                ).getInfo()
            except Exception:
                res = None
            if res and band in res and res[band] is not None:
                scaled = scale_soilgrid_value(float(res[band]), prefix)
                if scaled is not None:
                    values.append(scaled)
                    weights.append(w)
        if not values:
            return None
        # weighted average
        weighted = sum(v * wt for v, wt in zip(values, weights)) / sum(weights)
        # additional clamping for known properties
        if prefix == "phh2o":
            return clamp_value(round(weighted, 2), 3.0, 10.0)
        if prefix == "ocd":
            return clamp_value(round(weighted, 3), 0.0, 15.0)
        if prefix in ("clay", "sand", "silt"):
            return clamp_value(round(weighted, 2), 0.0, 100.0)
        if prefix == "bdod":
            return clamp_value(round(weighted, 3), 0.5, 2.5)
        if prefix == "cec":
            return clamp_value(round(weighted, 2), 0.0, 100.0)
        return round(weighted, 3)
    except Exception as e:
        logger.warning(f"EE fetch error for {prefix} at ({lat},{lon}): {e}")
        return None

# ======================
# Sampling inside polygon
# ======================
def calculate_area_from_polygon(polygon_geojson: dict) -> Tuple[float, float]:
    """
    Returns (area_ha, area_m2). Uses EE geometry to compute polygon area (m2).
    """
    try:
        geom = ee.Geometry(polygon_geojson)
        area_m2 = geom.area().getInfo()
        area_ha = area_m2 / M2_PER_HA
        return round(area_ha, 4), round(area_m2, 2)
    except Exception as e:
        logger.warning(f"Polygon area calc failed: {e}")
        return 0.0, 0.0

def generate_sample_points_from_polygon(polygon_geojson: dict, area_m2: float) -> List[Tuple[float, float]]:
    """
    Adaptive sampling:
      - area < 2000 m2 -> centroid only
      - 2000 < area <= 10000 -> 1
      - 10000 < area <= 50000 -> 3
      - >50000 -> 5
    Returns list of (lon, lat) tuples.
    """
    try:
        geom = ee.Geometry(polygon_geojson)
        # choose number of points
        if area_m2 < 2000:
            n = 1
        elif area_m2 <= 10000:
            n = 1
        elif area_m2 <= 50000:
            n = 3
        else:
            n = 5
        try:
            # ee.FeatureCollection.randomPoints accepts (region, points, seed)
            pts_fc = ee.FeatureCollection.randomPoints(region=geom, points=n, seed=42)
            feats = pts_fc.getInfo()["features"]
            coords = [tuple(feat["geometry"]["coordinates"]) for feat in feats]
            if coords:
                return coords
        except Exception as e:
            logger.warning(f"randomPoints failed: {e} - falling back to centroid(s)")
        # centroid fallback (single)
        centroid = geom.centroid().coordinates().getInfo()
        return [tuple(centroid)]
    except Exception as e:
        logger.error(f"Failed generate sample points from polygon: {e}")
        return []

# ======================
# NPK estimation using OC, CEC, bulk_density and field area
# ======================
def estimate_npk_from_aggregates(agg: Dict[str, Optional[float]], field_area_ha: float) -> Dict[str, Optional[float]]:
    """
    Returns per-ha and total values: nitrogen_kg_per_ha, phosphorus_kg_per_ha, potassium_kg_per_ha
    and nitrogen_total_kg, phosphorus_total_kg, potassium_total_kg
    """
    oc = agg.get("organic_carbon")
    ph = agg.get("ph")
    cec = agg.get("cec")
    clay = agg.get("clay")
    bd = agg.get("bulk_density") or 1.3

    # Nitrogen (from OC using C:N ~10:1 and 2.5% mineralization)
    if oc is not None:
        total_n_g_kg = oc / 10.0  # g/kg
        available_n_kg_ha = total_n_g_kg * bd * 1500 * 0.025  # 0-15cm
        nitrogen_kg_per_ha = round(available_n_kg_ha, 2)
        nitrogen_total_kg = round(nitrogen_kg_per_ha * field_area_ha, 2)
    else:
        nitrogen_kg_per_ha = nitrogen_total_kg = None

    # Phosphorus - approximate based on pH & OC (conservative defaults)
    if ph is not None and oc is not None:
        if ph < 5.5 or ph > 8.0:
            phosphorus_kg_per_ha = 10.0
            p_level = "low"
        elif 6.0 <= ph <= 7.5 and oc > 1.5:
            phosphorus_kg_per_ha = 25.0
            p_level = "medium-high"
        else:
            phosphorus_kg_per_ha = 15.0
            p_level = "medium"
        phosphorus_total_kg = round(phosphorus_kg_per_ha * field_area_ha, 2)
    else:
        phosphorus_kg_per_ha = phosphorus_total_kg = None

    # Potassium - from CEC assumption ~3.5% saturation
    if cec is not None:
        k_cmol_kg = cec * 0.035
        k_kg_ha = k_cmol_kg * 39.1 * 15 * bd  # convert to kg/ha top 15cm
        potassium_kg_per_ha = round(k_kg_ha, 2)
        potassium_total_kg = round(potassium_kg_per_ha * field_area_ha, 2)
    else:
        potassium_kg_per_ha = potassium_total_kg = None

    return {
        "nitrogen_kg_per_ha": nitrogen_kg_per_ha,
        "phosphorus_kg_per_ha": phosphorus_kg_per_ha,
        "potassium_kg_per_ha": potassium_kg_per_ha,
        "nitrogen_total_kg": nitrogen_total_kg,
        "phosphorus_total_kg": phosphorus_total_kg,
        "potassium_total_kg": potassium_total_kg,
    }

# ======================
# Aggregation helpers
# ======================
def aggregate_min(samples: List[Dict[str, Optional[float]]]) -> Dict[str, Optional[float]]:
    """Take min across sample dicts for keys present in SOILGRID_LAYERS and others."""
    keys = set().union(*[set(s.keys()) for s in samples]) if samples else set()
    agg = {}
    for k in keys:
        vals = [v for s in samples for v in ([s.get(k)] if k in s else []) if isinstance(v, (int, float))]
        agg[k] = round(min(vals), 3) if vals else None
    return agg

# ======================
# Core process: single land
# ======================
def process_land_soilgrid(land_id: str, tenant_id: str) -> Dict[str, Any]:
    """
    - Fetch land.boundary_polygon_old & area fields
    - Generate sample points
    - For each point fetch weighted SoilGrids values
    - Aggregate using MIN across points
    - Estimate NPK (per ha & total)
    - Insert soil_health (if not exists) and update lands summary
    """
    try:
        # Tenant validation
        tcheck = supabase.table("tenants").select("id").eq("id", tenant_id).execute()
        if not tcheck.data:
            return {"land_id": land_id, "status": "error", "error": "invalid_tenant"}

        # Skip if already exists (avoid duplicates)
        existing = supabase.table("soil_health").select("id").eq("land_id", land_id).eq("source", "soilgrid").execute()
        if existing.data and len(existing.data) > 0:
            return {"land_id": land_id, "status": "already_exists"}

        # Fetch land polygon & metadata
        land_row = supabase.table("lands").select(
            "boundary_polygon_old, farmer_id, area_acres, area_guntas"
        ).eq("id", land_id).eq("tenant_id", tenant_id).single().execute()

        if not land_row.data:
            return {"land_id": land_id, "status": "error", "error": "land_not_found"}

        polygon_geojson = land_row.data.get("boundary_polygon_old")
        farmer_id = land_row.data.get("farmer_id")
        area_acres = land_row.data.get("area_acres")
        area_guntas = land_row.data.get("area_guntas")

        if not polygon_geojson:
            return {"land_id": land_id, "status": "error", "error": "no_polygon"}

        # area calculation
        area_ha, area_m2 = calculate_area_from_polygon(polygon_geojson)
        if area_ha == 0.0:
            # fallback to acres/guntas
            if area_acres:
                area_ha = round(area_acres * ACRES_TO_HECTARES, 4)
                area_m2 = round(area_acres * 4046.86, 2)
            elif area_guntas:
                area_ha = round(area_guntas * GUNTAS_TO_HECTARES, 4)
                area_m2 = round(area_guntas * 101.17, 2)
            else:
                # default 1 ha to avoid divide by zero
                area_ha = 1.0
                area_m2 = 10000.0

        # sample points
        points = generate_sample_points_from_polygon(polygon_geojson, area_m2)
        if not points:
            return {"land_id": land_id, "status": "error", "error": "no_sample_points"}

        # For each point fetch soil parameters
        samples = []
        for lon, lat in points:
            sample = {}
            # Use weighted retrieval for each prefix
            for k, prefix in SOILGRID_LAYERS.items():
                val = get_soil_value_weighted(lat, lon, prefix)
                # map keys to convenient names
                if k == "ph":
                    sample["ph"] = val
                elif k == "organic_carbon":
                    sample["organic_carbon"] = val
                elif k == "nitrogen":
                    sample["nitrogen"] = val
                elif k == "bulk_density":
                    sample["bulk_density"] = val
                elif k == "cec":
                    sample["cec"] = val
                elif k == "clay":
                    sample["clay"] = val
                elif k == "sand":
                    sample["sand"] = val
                elif k == "silt":
                    sample["silt"] = val
            samples.append(sample)

        # Aggregate (min across points)
        agg = aggregate_min(samples)

        # Estimate NPK based on aggregate and area
        npk = estimate_npk_from_aggregates(
            {
                "organic_carbon": agg.get("organic_carbon"),
                "ph": agg.get("ph"),
                "cec": agg.get("cec"),
                "clay": agg.get("clay"),
                "bulk_density": agg.get("bulk_density"),
            },
            area_ha
        )

        # classification texts
        ph_level, ph_text = classify_ph(agg.get("ph"))
        oc_level, oc_text = classify_oc(agg.get("organic_carbon"))

        n_level, n_text = classify_nutrient(npk.get("nitrogen_kg_per_ha"), "nitrogen")
        p_level, p_text = classify_nutrient(npk.get("phosphorus_kg_per_ha"), "phosphorus")
        k_level, k_text = classify_nutrient(npk.get("potassium_kg_per_ha"), "potassium")

        texture = None
        if agg.get("clay") is not None and agg.get("sand") is not None and agg.get("silt") is not None:
            # simple USDA triangle classification
            cl = agg.get("clay"); sa = agg.get("sand"); si = agg.get("silt")
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

        # prepare record to insert; follow your soil_health schema column names
        record = {
            "land_id": land_id,
            "tenant_id": tenant_id,
            "farmer_id": farmer_id,
            "source": "soilgrid",
            "test_date": datetime.utcnow().date().isoformat(),
            "ph_level": agg.get("ph"),
            "organic_carbon": agg.get("organic_carbon"),
            "bulk_density": agg.get("bulk_density"),
            "cec": agg.get("cec"),
            "clay_percent": agg.get("clay"),
            "sand_percent": agg.get("sand"),
            "silt_percent": agg.get("silt"),
            "texture": texture,
            "field_area_ha": area_ha,
            # NPK per ha and totals
            "nitrogen_kg_per_ha": npk.get("nitrogen_kg_per_ha"),
            "phosphorus_kg_per_ha": npk.get("phosphorus_kg_per_ha"),
            "potassium_kg_per_ha": npk.get("potassium_kg_per_ha"),
            "nitrogen_total_kg": npk.get("nitrogen_total_kg"),
            "phosphorus_total_kg": npk.get("phosphorus_total_kg"),
            "potassium_total_kg": npk.get("potassium_total_kg"),
            # classification texts & levels
            "ph_text": ph_text,
            "organic_carbon_text": oc_text,
            "nitrogen_text": n_text,
            "phosphorus_text": p_text,
            "potassium_text": k_text,
            "nitrogen_level": n_level,
            "phosphorus_level": p_level,
            "potassium_level": k_level,
            # validation / confidence
            "data_quality_flags": [],  # could be filled by additional checks
            "data_quality_warnings": [],
            "data_completeness": None,
            "confidence_level": "medium",
            "note": "Generated from SoilGrids (weighted 0-30cm). Min across samples used (conservative).",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }

        # Remove None values to avoid DB type issues
        record_clean = {k: v for k, v in record.items() if v is not None}

        # Insert into soil_health
        ins = supabase.table("soil_health").insert(record_clean).execute()
        if ins.status_code not in (200, 201):
            logger.error(f"DB insert failed for land {land_id}: {ins.error_message if hasattr(ins, 'error_message') else ins}")
            return {"land_id": land_id, "status": "error", "error": "db_insert_failed"}

        # Update lands summary columns (tenant + land isolation)
        update_payload = {}
        if record.get("ph_level") is not None:
            update_payload["soil_ph"] = record["ph_level"]
        if record.get("organic_carbon") is not None:
            update_payload["organic_carbon_percent"] = record["organic_carbon"]
        if record.get("nitrogen_kg_per_ha") is not None:
            update_payload["nitrogen_kg_per_ha"] = record["nitrogen_kg_per_ha"]
        if record.get("phosphorus_kg_per_ha") is not None:
            update_payload["phosphorus_kg_per_ha"] = record["phosphorus_kg_per_ha"]
        if record.get("potassium_kg_per_ha") is not None:
            update_payload["potassium_kg_per_ha"] = record["potassium_kg_per_ha"]
        update_payload["last_soil_test_date"] = record["test_date"]

        if update_payload:
            supabase.table("lands").update(update_payload).eq("id", land_id).eq("tenant_id", tenant_id).execute()

        return {"land_id": land_id, "status": "saved", "record_id": (ins.data[0]["id"] if ins.data else None)}

    except Exception as e:
        logger.error(f"Error processing land {land_id}: {e}")
        return {"land_id": land_id, "status": "error", "error": str(e)}

# ======================
# API endpoints
# ======================
class BatchRequest(BaseModel):
    tenant_id: str = Field(..., description="Tenant UUID")
    land_ids: List[str] = Field(..., description="List of land UUIDs to process")

@app.post("/soil/save", tags=["Soil Analysis"])
async def soil_save_endpoint(req: Request, tenant_id: Optional[str] = Query(None)):
    """
    POST /soil/save
    Modes:
      - Query param tenant_id and JSON body { "land_ids": [...] }  => batch
      - OR send single-land body to process one land.
    """
    try:
        body = await req.json()
    except Exception:
        body = {}
    # Determine tenant_id
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
    return JSONResponse(status_code=404, content={"error": "Endpoint not found", "available": ["/soil", "/soil/save", "/health"]})

# ======================
# Runner
# ======================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    logger.info(f"Starting KisanShakti API v{VERSION} on port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")
