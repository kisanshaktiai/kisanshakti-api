# ============================================================
# 🌱 KisanShaktiAI - Soil Intelligence API
# Version: v3.4.0
# Author: Amarsinh Patil
# ------------------------------------------------------------
# Description:
# Fetches soil property data from SoilGrids (ISRIC) via GEE,
# calculates weighted depth averages, estimates NPK per hectare
# AND total NPK for actual field area (from area_acres/area_guntas),
# validates data quality, and updates Supabase tables.
# ============================================================

import os
import logging
import asyncio
import numpy as np
from datetime import datetime
from typing import Any, List, Dict, Optional

import ee
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from supabase import create_client, Client

# ======================
# Logging Setup
# ======================
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("kisanshakti-api")

# ======================
# Environment Config
# ======================
SERVICE_ACCOUNT = "kisanshaktiai-n@exalted-legacy-456511-b9.iam.gserviceaccount.com"
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "❌ Missing Supabase configuration! "
        "Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ======================
# Earth Engine Initialization
# ======================
try:
    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, key_file=CREDENTIALS_PATH)
    ee.Initialize(credentials)
    logger.info("✅ Earth Engine initialized successfully.")
except Exception as e:
    logger.error(f"❌ Earth Engine initialization failed: {str(e)}")

# ======================
# FastAPI Initialization
# ======================
app = FastAPI(
    title="KisanShakti Geospatial API",
    description="Advanced soil analysis with area-adjusted NPK calculations",
    version="3.4.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# SoilGrids Configuration
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

# Weighted agricultural depth ranges (0–30 cm most relevant for crops)
DEPTH_WEIGHTS = {
    "0-5cm": 0.35,
    "5-15cm": 0.35,
    "15-30cm": 0.30
}

# Unit conversions
ACRES_TO_HECTARES = 0.404686
GUNTAS_TO_HECTARES = 0.010117  # 1 gunta = 0.010117 ha
ACRES_TO_M2 = 4046.86
GUNTAS_TO_M2 = 101.17

# ======================
# Helper Functions
# ======================

def calculate_field_area(area_acres: Optional[float] = None, 
                        area_guntas: Optional[float] = None,
                        polygon_geojson: Optional[Dict] = None) -> tuple:
    """
    Calculate field area in hectares and m² from multiple sources.
    Priority: polygon > acres > guntas > default
    Returns: (area_ha, area_m2)
    """
    # Try polygon first (most accurate)
    if polygon_geojson:
        try:
            polygon = ee.Geometry(polygon_geojson)
            area_m2 = polygon.area().getInfo()
            area_ha = area_m2 / 10000
            return round(area_ha, 4), round(area_m2, 2)
        except Exception as e:
            logger.warning(f"⚠️ Polygon area calculation failed: {e}")
    
    # Try acres
    if area_acres and area_acres > 0:
        area_ha = area_acres * ACRES_TO_HECTARES
        area_m2 = area_acres * ACRES_TO_M2
        return round(area_ha, 4), round(area_m2, 2)
    
    # Try guntas
    if area_guntas and area_guntas > 0:
        area_ha = area_guntas * GUNTAS_TO_HECTARES
        area_m2 = area_guntas * GUNTAS_TO_M2
        return round(area_ha, 4), round(area_m2, 2)
    
    # Default to 1 hectare
    logger.warning("⚠️ No valid area found, defaulting to 1 hectare")
    return 1.0, 10000.0


def get_soil_value_weighted(lat: float, lon: float, prefix: str, 
                            retries: int = 2) -> Optional[float]:
    """
    Fetch soil property with weighted average across 0–30cm depths.
    Includes retry logic for API timeouts.
    """
    for attempt in range(retries):
        try:
            point = ee.Geometry.Point([lon, lat])
            region = point.buffer(300).bounds()
            img = ee.Image(f"projects/soilgrids-isric/{prefix}_mean")

            values, weights = [], []
            for depth, weight in DEPTH_WEIGHTS.items():
                band = f"{prefix}_{depth}_mean"
                val = img.select(band).reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=region,
                    scale=250,
                    bestEffort=True,
                    maxPixels=1e8
                ).getInfo()
                
                if val and band in val and val[band] is not None:
                    values.append(float(val[band]))
                    weights.append(weight)

            if values:
                weighted_avg = sum(v * w for v, w in zip(values, weights)) / sum(weights)
                return round(weighted_avg, 2)
            return None
            
        except Exception as e:
            if attempt < retries - 1:
                logger.warning(f"⚠️ Retry {attempt + 1}/{retries} for {prefix} at ({lat:.4f}, {lon:.4f})")
                continue
            else:
                logger.error(f"❌ Failed to fetch {prefix} after {retries} attempts: {str(e)}")
                return None


def generate_sample_points(polygon_geojson: Optional[Dict[str, Any]], 
                          area_m2: float) -> List[List[float]]:
    """
    Generate sample points within polygon based on area.
    Small fields: centroid only
    Medium fields: 3 points
    Large fields: 5 points
    """
    if not polygon_geojson:
        return []
    
    try:
        polygon = ee.Geometry(polygon_geojson)
        
        # For very small fields, use centroid
        if area_m2 < 2000:
            centroid = polygon.centroid().coordinates().getInfo()
            return [centroid]
        
        # Determine number of sample points based on area
        if area_m2 > 50000:  # > 5 hectares
            n_points = 5
        elif area_m2 > 10000:  # 1-5 hectares
            n_points = 3
        else:
            n_points = 1
        
        pts = polygon.randomPoints(maxPoints=n_points, seed=42)
        coords = pts.getInfo()["features"]
        return [feat["geometry"]["coordinates"] for feat in coords]
        
    except Exception as e:
        logger.warning(f"⚠️ Point generation failed: {str(e)} — using centroid")
        try:
            centroid = ee.Geometry(polygon_geojson).centroid().coordinates().getInfo()
            return [centroid]
        except Exception:
            return []


def classify_texture(sand: Optional[float], silt: Optional[float], 
                    clay: Optional[float]) -> Optional[str]:
    """Classify soil texture using USDA soil texture triangle."""
    if None in (sand, silt, clay):
        return None
    
    if clay >= 40:
        return "Clay"
    elif clay >= 27 and silt < 28:
        return "Clay Loam"
    elif clay >= 20 and sand >= 45:
        return "Sandy Clay Loam"
    elif clay >= 7 and silt >= 28 and sand <= 52:
        return "Loam"
    elif silt >= 50 and clay < 27:
        return "Silt Loam"
    elif sand >= 85 and clay < 10:
        return "Sand"
    elif sand >= 70 and clay < 15:
        return "Loamy Sand"
    elif sand >= 43 and clay < 20:
        return "Sandy Loam"
    return "Loam"


def estimate_npk_for_field(oc: Optional[float], ph: Optional[float], 
                           cec: Optional[float], clay: Optional[float],
                           bulk_density: Optional[float], 
                           field_area_ha: float) -> Dict[str, Any]:
    """
    Estimate NPK values per hectare AND total for field area.
    
    Returns:
    - nitrogen_kg_ha: Available N per hectare
    - nitrogen_total_kg: Total N for entire field
    - phosphorus_kg_ha: Extractable P per hectare
    - phosphorus_total_kg: Total P for entire field
    - potassium_kg_ha: Exchangeable K per hectare
    - potassium_total_kg: Total K for entire field
    - Classification levels (low/medium/high)
    """
    bd = bulk_density or 1.3  # Default bulk density
    
    # ========== NITROGEN ESTIMATION ==========
    # Based on organic carbon (C:N ratio ≈ 10:1)
    if oc:
        total_n_g_kg = oc / 10  # Total N in g/kg
        # Available N (2.5% mineralization per season, 0-15cm depth)
        available_n_kg_ha = total_n_g_kg * bd * 1500 * 0.025
        n_kg_ha = round(available_n_kg_ha, 2)
        n_total_kg = round(n_kg_ha * field_area_ha, 2)
    else:
        n_kg_ha = n_total_kg = None
    
    # N classification (Indian agricultural standards)
    if n_kg_ha:
        if n_kg_ha < 280:
            n_level = "low"
        elif n_kg_ha < 560:
            n_level = "medium"
        else:
            n_level = "high"
    else:
        n_level = "unknown"
    
    # ========== PHOSPHORUS ESTIMATION ==========
    # Note: Without actual P2O5 data, this is approximate
    # Based on pH and organic matter correlation
    if ph and oc:
        if ph < 5.5 or ph > 8.0:
            # P fixation likely in acidic/alkaline soils
            p_kg_ha = 10.0  # Low availability
            p_level = "low"
        elif 6.0 <= ph <= 7.5 and oc > 1.5:
            # Optimal pH range with good OM
            p_kg_ha = 25.0  # Medium-high availability
            p_level = "medium-high"
        else:
            # Average conditions
            p_kg_ha = 15.0
            p_level = "medium"
        p_total_kg = round(p_kg_ha * field_area_ha, 2)
    else:
        p_kg_ha = p_total_kg = None
        p_level = "unknown"
    
    # ========== POTASSIUM ESTIMATION ==========
    # Based on CEC (Cation Exchange Capacity)
    if cec:
        # Assume K saturation is 3.5% of CEC
        k_cmol_kg = cec * 0.035
        # Convert to kg/ha (K atomic weight = 39.1, 0-15cm depth)
        k_kg_ha = k_cmol_kg * 39.1 * 15 * bd
        k_kg_ha = round(k_kg_ha, 2)
        k_total_kg = round(k_kg_ha * field_area_ha, 2)
    else:
        k_kg_ha = k_total_kg = None
    
    # K classification (Indian standards)
    if k_kg_ha:
        if k_kg_ha < 110:
            k_level = "low"
        elif k_kg_ha < 280:
            k_level = "medium"
        else:
            k_level = "high"
    else:
        k_level = "unknown"
    
    return {
        # Per hectare values
        "nitrogen_kg_ha": n_kg_ha,
        "phosphorus_kg_ha": p_kg_ha,
        "potassium_kg_ha": k_kg_ha,
        
        # Total field values
        "nitrogen_total_kg": n_total_kg,
        "phosphorus_total_kg": p_total_kg,
        "potassium_total_kg": k_total_kg,
        
        # Classification levels
        "nitrogen_level": n_level,
        "phosphorus_level": p_level,
        "potassium_level": k_level,
        
        # Metadata
        "field_area_ha": field_area_ha,
        "calculation_note": "NPK values are estimates based on SoilGrids data. Lab testing recommended for precision farming."
    }


def validate_soil_data(record: dict) -> dict:
    """
    Validate soil data and add quality flags.
    Returns record with data_quality_flags and confidence_level.
    """
    flags = []
    warnings = []
    
    # pH validation
    ph = record.get("ph_level")
    if ph:
        if not (3 <= ph <= 11):
            flags.append("pH_out_of_range")
        elif ph < 4.5 or ph > 9.0:
            warnings.append("extreme_pH")
    
    # Texture sum validation
    clay = record.get("clay_percent") or 0
    sand = record.get("sand_percent") or 0
    silt = record.get("silt_percent") or 0
    total_texture = clay + sand + silt
    
    if total_texture > 105:
        flags.append("texture_sum_invalid")
    elif total_texture < 95:
        warnings.append("texture_sum_low")
    
    # Organic carbon validation
    oc = record.get("organic_carbon")
    if oc and oc > 10:
        warnings.append("very_high_organic_carbon")
    
    # CEC validation
    cec = record.get("cec")
    if cec and cec > 50:
        warnings.append("very_high_cec")
    
    # Calculate data completeness
    non_null_count = sum(1 for v in record.values() if v is not None)
    total_fields = len(record)
    completeness = non_null_count / total_fields if total_fields > 0 else 0
    
    # Assign confidence level
    if flags:
        confidence = "low"
    elif completeness > 0.8 and not warnings:
        confidence = "high"
    elif completeness > 0.6:
        confidence = "medium"
    else:
        confidence = "low"
    
    record["data_quality_flags"] = flags
    record["data_quality_warnings"] = warnings
    record["confidence_level"] = confidence
    record["data_completeness"] = round(completeness * 100, 1)
    
    return record


async def process_single_land(land_id: str, tenant_id: str) -> Dict[str, Any]:
    """
    Process a single land parcel: fetch soil data, calculate NPK, 
    validate, and store in database.
    """
    try:
        # Check if already processed
        existing = supabase.table("soil_health").select("id").eq("land_id", land_id).execute()
        if existing.data and len(existing.data) > 0:
            return {"land_id": land_id, "status": "already_exists"}

        # Fetch land data
        land_resp = supabase.table("lands").select(
            "boundary_polygon_old, farmer_id, area_acres, area_guntas"
        ).eq("id", land_id).eq("tenant_id", tenant_id).single().execute()
        
        if not land_resp.data:
            return {"land_id": land_id, "status": "not_found"}

        polygon_geojson = land_resp.data.get("boundary_polygon_old")
        farmer_id = land_resp.data.get("farmer_id")
        area_acres = land_resp.data.get("area_acres")
        area_guntas = land_resp.data.get("area_guntas")

        # Calculate field area
        area_ha, area_m2 = calculate_field_area(area_acres, area_guntas, polygon_geojson)

        # Generate sampling points
        points = generate_sample_points(polygon_geojson, area_m2)
        if not points:
            return {
                "land_id": land_id, 
                "status": "failed", 
                "error": "no_valid_sampling_points"
            }

        # Fetch soil data for all sample points
        samples = []
        for lon_pt, lat_pt in points:
            sample_data = {
                key: get_soil_value_weighted(lat_pt, lon_pt, prefix)
                for key, prefix in SOILGRID_LAYERS.items()
            }
            samples.append(sample_data)

        # Aggregate samples (mean of all points)
        agg = {
            key: round(float(np.mean([s[key] for s in samples if s[key] is not None])), 2)
            if any(s[key] is not None for s in samples) else None
            for key in SOILGRID_LAYERS.keys()
        }

        # Classify texture
        texture = classify_texture(
            agg.get("sand"), 
            agg.get("silt"), 
            agg.get("clay")
        )

        # Estimate NPK for this specific field area
        npk_data = estimate_npk_for_field(
            oc=agg.get("organic_carbon"),
            ph=agg.get("ph"),
            cec=agg.get("cec"),
            clay=agg.get("clay"),
            bulk_density=agg.get("bulk_density"),
            field_area_ha=area_ha
        )

        # Prepare soil health record
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
            **npk_data,
        }

        # Validate data quality
        record = validate_soil_data(record)

        # Insert into soil_health table
        supabase.table("soil_health").insert(record).execute()

        # Update lands table with summary
        supabase.table("lands").update({
            "soil_ph": record["ph_level"],
            "organic_carbon_percent": record["organic_carbon"],
            "nitrogen_kg_per_ha": record["nitrogen_kg_ha"],
            "phosphorus_kg_per_ha": record["phosphorus_kg_ha"],
            "potassium_kg_per_ha": record["potassium_kg_ha"],
            "last_soil_test_date": record["test_date"],
        }).eq("id", land_id).eq("tenant_id", tenant_id).execute()

        return {
            "land_id": land_id,
            "status": "saved",
            "area_ha": area_ha,
            "confidence": record["confidence_level"],
            "npk_summary": {
                "N_total_kg": record["nitrogen_total_kg"],
                "P_total_kg": record["phosphorus_total_kg"],
                "K_total_kg": record["potassium_total_kg"]
            }
        }

    except Exception as e:
        logger.error(f"❌ Error processing land {land_id}: {str(e)}")
        return {
            "land_id": land_id,
            "status": "error",
            "error": str(e)
        }


# ======================
# API Routes
# ======================

@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "KisanShakti Geospatial API",
        "version": "3.4.0",
        "status": "operational",
        "features": [
            "Area-adjusted NPK calculations",
            "Multi-point soil sampling",
            "Data quality validation",
            "Async batch processing"
        ],
        "endpoints": {
            "soil_analysis": "/soil/save",
            "health_check": "/health",
            "documentation": "/docs"
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Check API and Earth Engine connectivity."""
    try:
        ee.Number(1).getInfo()
        ee_status = "connected"
    except Exception as e:
        ee_status = f"disconnected: {str(e)}"
    
    return {
        "api_status": "healthy",
        "earth_engine": ee_status,
        "version": "3.4.0",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/soil/save", tags=["Soil Analysis"])
async def save_soil_health(request: Request, tenant_id: str = Query(...)):
    """
    Fetch and store SoilGrids data for one or multiple land parcels.
    
    Request body:
    {
        "land_ids": ["land_id_1", "land_id_2", ...]
    }
    
    Features:
    - Async batch processing
    - Area-adjusted NPK calculations (per-ha + total)
    - Multi-point sampling based on field size
    - Data quality validation
    - Automatic lands table updates
    """
    try:
        body = await request.json()
        land_ids = body.get("land_ids", [])
        
        if not land_ids:
            raise HTTPException(
                status_code=400,
                detail="Missing required field: land_ids (array)"
            )

        # Validate tenant exists
        tenant_check = supabase.table("tenants").select("id").eq("id", tenant_id).execute()
        if not tenant_check.data:
            raise HTTPException(status_code=404, detail="Invalid tenant_id")

        # Process all lands concurrently
        tasks = [process_single_land(lid, tenant_id) for lid in land_ids]
        results = await asyncio.gather(*tasks)

        # Calculate summary statistics
        saved = sum(1 for r in results if r["status"] == "saved")
        skipped = sum(1 for r in results if r["status"] == "already_exists")
        failed = sum(1 for r in results if r["status"] in ["failed", "error", "not_found"])

        return {
            "status": "success",
            "summary": {
                "total_processed": len(land_ids),
                "saved": saved,
                "skipped": skipped,
                "failed": failed
            },
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in /soil/save: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "available_endpoints": ["/", "/health", "/soil/save", "/docs"],
            "documentation": "/docs"
        }
    )


# ============================================================
# Server Entry Point
# ============================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🚀 Starting KisanShakti API v3.4.0 on port {port}")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False
    )
