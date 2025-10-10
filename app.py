import ee
import os
import logging
import numpy as np
from typing import Any, List, Dict
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# ======================
# Logging Setup
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ======================
# Earth Engine Initialization
# ======================
SERVICE_ACCOUNT = "kisanshaktiai-n@exalted-legacy-456511-b9.iam.gserviceaccount.com"
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json")

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
    version="2.3.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
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

# ======================
# Routes
# ======================
@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "KisanShakti Geospatial API",
        "version": "2.3.0",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "soil": "/soil - Soil properties analysis (lat/lon, multiple, or polygon)",
            "ndvi": "/ndvi - Coming soon",
            "tiles": "/tiles - Coming soon",
            "docs": "/docs - API documentation"
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
async def get_soil_data(
    request: Request,
    lat: float = Query(None, description="Latitude", ge=-90, le=90),
    lon: float = Query(None, description="Longitude", ge=-180, le=180),
    locations: str = Query(
        None,
        description="Semicolon-separated lat,lon pairs. Example: 16.7,74.2;17.5,75.3"
    )
):
    """
    Get soil data in 3 modes:
    - Single point (?lat=..&lon=..)
    - Multiple points (?locations=lat1,lon1;lat2,lon2)
    - Polygon (POST body with GeoJSON {\"polygon\": {...}})
    """
    try:
        samples = []
        body = await request.json() if request.method == "POST" else {}

        # Case 1: Polygon mode
        if "polygon" in body:
            polygon = body["polygon"]
            points = generate_sample_points(polygon, n_points=5)
            if not points:
                raise HTTPException(status_code=400, detail="Invalid polygon or could not generate points")

            for lon_pt, lat_pt in points:
                data = {k: get_soil_value(lat_pt, lon_pt, p) or "No data" for k, p in SOILGRID_LAYERS.items()}
                samples.append({"latitude": lat_pt, "longitude": lon_pt, "data": data})
            method = "Composite sampling (5 random points inside polygon)"

        # Case 2: Multiple locations mode
        elif locations:
            locs = locations.split(";")
            if len(locs) > 15:
                raise HTTPException(status_code=400, detail="Maximum 15 locations allowed per request")
            for loc in locs:
                try:
                    lat_str, lon_str = loc.strip().split(",")
                    lt, ln = float(lat_str), float(lon_str)
                    data = {k: get_soil_value(lt, ln, p) or "No data" for k, p in SOILGRID_LAYERS.items()}
                    samples.append({"latitude": lt, "longitude": ln, "data": data})
                except Exception:
                    samples.append({"location": loc, "error": "Invalid format. Use: lat,lon"})
            method = "Composite sampling (averaged across multiple points)"

        # Case 3: Single point mode
        elif lat is not None and lon is not None:
            data = {k: get_soil_value(lat, lon, p) or "No data" for k, p in SOILGRID_LAYERS.items()}
            samples.append({"latitude": lat, "longitude": lon, "data": data})
            method = "Single point sampling (not recommended in soil science)"

        else:
            raise HTTPException(status_code=400, detail="Provide lat/lon, locations, or polygon in request")

        # Build response
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
        logger.error(f"Error in /soil endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch soil data: {str(e)}")

@app.get("/ndvi", tags=["NDVI Analysis"])
async def get_ndvi_data():
    return {
        "status": "coming_soon",
        "message": "NDVI analysis will be available soon",
        "features": ["Vegetation health monitoring", "Time-series analysis", "Crop stress detection"]
    }

@app.get("/tiles", tags=["Tile Services"])
async def get_tiles():
    return {
        "status": "coming_soon",
        "message": "Tile services will be available soon",
        "features": ["Map tiles generation", "Custom layer rendering", "High-resolution imagery"]
    }

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested endpoint does not exist",
            "available_endpoints": ["/", "/health", "/soil", "/docs"]
        }
    )

# ======================
# Run App
# ======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info", access_log=True)
