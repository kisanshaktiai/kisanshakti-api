# -------------------------------
# FastAPI Gateway for NDVI Worker v.1.0.0
# update the status Endpoint 
# -------------------------------
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import logging
import tile_fetch_worker
import os
from supabase import create_client

# ---------------- Configuration ----------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="NDVI Tile Worker API", version="1.0.1")

# ✅ Allow CORS for your Lovable frontend
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["authorization", "x-client-info", "apikey", "content-type"],
)

# ---------------- Health Check ----------------
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "tile-fetch-worker", "version": "1.0.1"}

# ---------------- Run Worker ----------------
@app.post("/run")
async def run_worker(request: Request):
    """
    Trigger NDVI tile fetch worker manually.
    The frontend (NDVI Data Status page) sends {cloud_cover, lookback_days}.
    """
    try:
        body = await request.json()
        cloud_cover = int(body.get("cloud_cover", 30))
        lookback_days = int(body.get("lookback_days", 5))

        logging.info(f"🚀 NDVI Worker Triggered | cloud_cover={cloud_cover}, lookback_days={lookback_days}")
        processed_count = tile_fetch_worker.main(cloud_cover=cloud_cover, lookback_days=lookback_days)

        return {
            "status": "success",
            "message": f"Worker completed successfully",
            "cloud_cover": cloud_cover,
            "lookback_days": lookback_days,
            "processed_tiles": processed_count
        }
    except Exception as e:
        logging.error(f"❌ Worker failed: {e}")
        return {"status": "error", "message": str(e)}

# ---------------- Tiles Preview (for NDVI UI) ----------------
@app.get("/tiles/status")
def tiles_status_preview():
    """
    Frontend NDVI Data Status page calls this endpoint
    to preview current tile coverage, counts, and readiness.
    """
    try:
        resp = supabase.table("mgrs_tiles") \
            .select("tile_id, is_agri, is_land_contain, is_ndvi_ready, total_lands_count, last_land_check, updated_at") \
            .eq("is_agri", True) \
            .eq("is_land_contain", True) \
            .order("last_land_check", desc=True) \
            .limit(100) \
            .execute()

        data = resp.data or []
        total = len(data)
        ready = sum(1 for r in data if r.get("is_ndvi_ready"))
        not_ready = total - ready

        return {
            "status": "success",
            "summary": {
                "total_tiles": total,
                "ready": ready,
                "pending": not_ready,
            },
            "data": data
        }

    except Exception as e:
        logging.error(f"Failed to load tile status: {e}")
        return {"status": "error", "message": str(e)}

# ---------------- Optional: Single Land Tile Check ----------------
@app.get("/lands/{land_id}/tiles")
def get_tiles_for_land(land_id: str):
    """
    For verification or debugging:
    Fetch which tiles intersect with a specific land.
    Useful for frontend NDVI Data Status → 'Inspect Land' feature.
    """
    try:
        resp = supabase.table("land_tile_intersections") \
            .select("tile_id, created_at") \
            .eq("land_id", land_id) \
            .execute()

        intersections = resp.data or []
        tile_ids = [r["tile_id"] for r in intersections]

        return {"status": "success", "land_id": land_id, "intersecting_tiles": tile_ids}
    except Exception as e:
        logging.error(f"Failed to fetch tiles for land {land_id}: {e}")
        return {"status": "error", "message": str(e)}
