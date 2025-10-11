import os
import logging
import numpy as np
from datetime import datetime
from typing import Any, List, Dict

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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
    logger.error("❌ Missing Supabase configuration! Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.")
    raise RuntimeError("Supabase configuration missing")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ======================
# Earth Engine Initialization
# ======================
try:
    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, key_file=CREDENTIALS_PATH)
    ee.Initialize(credentials)
    logger.info(f"✅ Earth Engine initialized successfully with {CREDENTIALS_PATH}")
except Exception as e:
    logger.error(f"❌ Earth Engine initialization failed: {str(e)}")

# ======================
# FastAPI Initialization
# ======================
app = FastAPI(
    title="KisanShakti Geospatial API",
    description="Soil analysis, NDVI, and tile services for agricultural intelligence",
    version="2.3.1",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
# SoilGrids Config
# ======================
SOILGRID_LAYERS = {
    "ph": "phh2o",
    "organic_carbon": "ocd",
    "clay": "clay",
    "sand": "sand",
    "silt": "silt",
    "bulk_density": "bdod",
    "cec": "cec",
}
DEPTHS = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]

# ======================
# Helper Functions
# ======================
def get_soil_value(lat: float, lon: float, prefix: str) -> Any:
    """Fetch soil property value at a location with depth fallback."""
    try:
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(300).bounds()
        img = ee.Image(f"projects/soilgrids-isric/{prefix}_mean")

        for depth in DEPTHS:
            band = f"{prefix}_{depth}_mean"
            try:
                val = img.select(band).reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=region,
                    scale=250
                ).getInfo()
                if val and band in val and val[band] is not None:
                    return round(float(val[band]), 2)
            except Exception:
                continue
        return None
    except Exception as e:
        logger.error(f"Error in get_soil_value: {str(e)}")
        return None


def generate_sample_points(polygon_geojson: Dict[str, Any], n_points: int = 5) -> List[List[float]]:
    """Generate random sample points inside a polygon (lon, lat)."""
    try:
        polygon = ee.Geometry(polygon_geojson)
        pts = polygon.randomPoints(maxPoints=n_points, seed=42)
        coords = pts.getInfo()["features"]
        return [feat["geometry"]["coordinates"] for feat in coords]
    except Exception as e:
        logger.error(f"Failed to generate sample points: {str(e)}")
        return []


def aggregate_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate multiple soil samples into mean, min, max, stddev."""
    agg = {}
    for key in SOILGRID_LAYERS.keys():
        values = [s["data"][key] for s in samples if isinstance(s["data"][key], (int, float))]
        if values:
            agg[key] = {
                "mean": round(float(np.mean(values)), 2),
                "min": round(float(np.min(values)), 2),
                "max": round(float(np.max(values)), 2),
                "stddev": round(float(np.std(values)), 2),
                "samples": len(values)
            }
        else:
            agg[key] = "No data"
    return agg

# ======================
# Routes
# ======================
@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "KisanShakti Geospatial API",
        "version": "2.3.1",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "soil": "/soil",
            "soil/save": "/soil/save",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health", tags=["Health"])
async def health_check():
    try:
        ee.Number(1).getInfo()
        ee_status = "connected"
    except Exception as e:
        logger.error(f"Earth Engine health check failed: {str(e)}")
        ee_status = "disconnected"
    return {
        "status": "healthy",
        "earth_engine": ee_status,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/soil", tags=["Soil Analysis"])
async def get_soil_data(request: Request, lat: float = Query(None), lon: float = Query(None), locations: str = Query(None)):
    """Fetch soil data using lat/lon, multiple points, or polygon."""
    try:
        samples = []
        body = await request.json() if request.method == "POST" else {}

        # Polygon mode
        if "polygon" in body:
            polygon = body["polygon"]
            points = generate_sample_points(polygon, n_points=5)
            if not points:
                raise HTTPException(status_code=400, detail="Invalid polygon or could not generate points")

            for lon_pt, lat_pt in points:
                data = {k: get_soil_value(lat_pt, lon_pt, p) or "No data" for k, p in SOILGRID_LAYERS.items()}
                samples.append({"latitude": lat_pt, "longitude": lon_pt, "data": data})
            method = "Polygon composite sampling"

        # Multiple points mode
        elif locations:
            locs = locations.split(";")
            if len(locs) > 15:
                raise HTTPException(status_code=400, detail="Maximum 15 locations allowed per request")
            for loc in locs:
                lat_str, lon_str = loc.strip().split(",")
                lt, ln = float(lat_str), float(lon_str)
                data = {k: get_soil_value(lt, ln, p) or "No data" for k, p in SOILGRID_LAYERS.items()}
                samples.append({"latitude": lt, "longitude": ln, "data": data})
            method = "Multiple point composite"

        # Single point mode
        elif lat is not None and lon is not None:
            data = {k: get_soil_value(lat, lon, p) or "No data" for k, p in SOILGRID_LAYERS.items()}
            samples.append({"latitude": lat, "longitude": lon, "data": data})
            method = "Single point"

        else:
            raise HTTPException(status_code=400, detail="Provide lat/lon, locations, or polygon")

        response = {
            "timestamp": datetime.utcnow().isoformat(),
            "sample_count": len(samples),
            "samples": samples,
            "method": method
        }
        if len(samples) > 1:
            response["aggregated"] = aggregate_samples(samples)

        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /soil: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch soil data: {str(e)}")


@app.post("/soil/save", tags=["Soil Analysis"])
async def save_soil_health(request: Request, land_id: str = Query(...), tenant_id: str = Query(...)):
    """
    Fetch soil data for a land boundary and save it to Supabase.
    Updates both soil_health and lands tables.
    """
    try:
        # Step 1: Fetch land boundary
        land_resp = supabase.table("lands").select("boundary_polygon_old").eq("id", land_id).single().execute()
        if not land_resp.data or not land_resp.data.get("boundary_polygon_old"):
            raise HTTPException(status_code=404, detail=f"No boundary found for land {land_id}")

        polygon_geojson = land_resp.data["boundary_polygon_old"]

        # Step 2: Generate soil samples
        points = generate_sample_points(polygon_geojson, n_points=5)
        if not points:
            raise HTTPException(status_code=400, detail="Could not generate sample points")

        samples = []
        for lon_pt, lat_pt in points:
            data = {k: get_soil_value(lat_pt, lon_pt, p) or None for k, p in SOILGRID_LAYERS.items()}
            samples.append(data)

        # Step 3: Aggregate results
        agg = {key: round(float(np.mean([s[key] for s in samples if s[key] is not None])), 2)
               if any(s[key] is not None for s in samples) else None
               for key in SOILGRID_LAYERS.keys()}

        # Step 4: Insert into soil_health
        record = {
            "land_id": land_id,
            "tenant_id": tenant_id,
            "source": "soilgrid",
            "ph_level": agg.get("ph"),
            "organic_carbon": agg.get("organic_carbon"),
            "bulk_density": agg.get("bulk_density"),
            "test_date": datetime.utcnow().date().isoformat(),
        }
        supabase.table("soil_health").insert(record).execute()

        # Step 5: Update lands table with summary
        supabase.table("lands").update({
            "soil_ph": record["ph_level"],
            "organic_carbon_percent": record["organic_carbon"],
            "last_soil_test_date": record["test_date"]
        }).eq("id", land_id).execute()

        return {
            "status": "success",
            "message": "Soil data fetched and saved successfully",
            "data": record
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save soil data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested endpoint does not exist",
            "available_endpoints": ["/", "/health", "/soil", "/soil/save", "/docs"]
        }
    )

# ======================
# Run App
# ======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info", access_log=True)
