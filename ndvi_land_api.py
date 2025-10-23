"""
NDVI Land API v4.1.0 — Unified Table Fix
-----------------------------------------
✅ Uses unified table: ndvi_micro_tiles
✅ Backward compatible with v4.0.0 queue
✅ Safe /stats handling for empty tenants
✅ Improved diagnostics endpoints

© 2025 KisanShaktiAI
"""

import os, datetime, logging
from typing import Optional
from fastapi import FastAPI, HTTPException, Query, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from supabase import create_client

# ============ CONFIG ============
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("❌ Missing Supabase credentials")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ndvi-api-v4.1")

# ============ FASTAPI ============
app = FastAPI(
    title="KisanShakti NDVI API",
    description="Unified NDVI microtile API (v4.1.0)",
    version="4.1.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ MODELS ============
class NDVIRequest(BaseModel):
    land_ids: list[str]
    tile_id: Optional[str] = None

# ============ HEALTH ============
@app.get("/api/v1/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "service": "ndvi-land-api", "version": "4.1.0"}

# ============ NDVI DATA (UNIFIED) ============
@app.get("/api/v1/ndvi/data", tags=["NDVI Data"])
async def list_ndvi_data(tenant_id: str = Query(...), land_id: Optional[str] = None, limit: int = 100):
    """Retrieve NDVI data from unified table."""
    try:
        query = supabase.table("ndvi_micro_tiles").select("*").eq("tenant_id", tenant_id)
        if land_id:
            query = query.eq("land_id", land_id)
        query = query.order("acquisition_date", desc=True).limit(limit)
        resp = query.execute()

        return {
            "status": "success",
            "count": len(resp.data or []),
            "data": resp.data or [],
        }
    except Exception as e:
        logger.error(f"❌ Error fetching NDVI data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ndvi/data/{land_id}/latest", tags=["NDVI Data"])
async def get_latest_ndvi(land_id: str, tenant_id: str = Query(...)):
    """Get latest NDVI for a land."""
    try:
        resp = (
            supabase.table("ndvi_micro_tiles")
            .select("*")
            .eq("tenant_id", tenant_id)
            .eq("land_id", land_id)
            .order("acquisition_date", desc=True)
            .limit(1)
            .execute()
        )

        if not resp.data:
            raise HTTPException(status_code=404, detail="No NDVI data found")

        return {"status": "success", "data": resp.data[0]}
    except Exception as e:
        logger.error(f"❌ Error fetching latest NDVI: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ndvi/data/{land_id}/history", tags=["NDVI Data"])
async def get_ndvi_history(land_id: str, tenant_id: str = Query(...), limit: int = 30):
    """Get NDVI history for a land."""
    try:
        resp = (
            supabase.table("ndvi_micro_tiles")
            .select("*")
            .eq("tenant_id", tenant_id)
            .eq("land_id", land_id)
            .order("acquisition_date", desc=True)
            .limit(limit)
            .execute()
        )

        return {
            "status": "success",
            "land_id": land_id,
            "count": len(resp.data or []),
            "history": resp.data or [],
        }
    except Exception as e:
        logger.error(f"❌ Error fetching NDVI history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ NDVI STATS ============
@app.get("/api/v1/ndvi/requests/stats", tags=["NDVI Statistics"])
async def get_ndvi_stats(tenant_id: Optional[str] = Query(None)):
    """Get NDVI processing statistics."""
    try:
        base = supabase.table("ndvi_request_queue").select("id", count="exact")
        if tenant_id:
            base = base.eq("tenant_id", tenant_id)

        total = base.execute()
        queued = base.eq("status", "queued").execute()
        processing = base.eq("status", "processing").execute()
        completed = base.eq("status", "completed").execute()
        failed = base.eq("status", "failed").execute()

        return {
            "status": "success",
            "tenant_id": tenant_id or "all",
            "stats": {
                "total_requests": total.count or 0,
                "queued": queued.count or 0,
                "processing": processing.count or 0,
                "completed": completed.count or 0,
                "failed": failed.count or 0,
            },
        }
    except Exception as e:
        logger.error(f"❌ Stats error: {e}")
        return {"status": "error", "message": str(e), "stats": {}}


# ============ DIAGNOSTICS ============
@app.get("/api/v1/ndvi/diagnostics/land/{land_id}", tags=["Diagnostics"])
async def diagnose_land(land_id: str, tenant_id: str = Query(...)):
    """Get diagnostic summary for a land."""
    try:
        land = (
            supabase.table("lands")
            .select("*")
            .eq("id", land_id)
            .eq("tenant_id", tenant_id)
            .execute()
        )
        if not land.data:
            raise HTTPException(status_code=404, detail="Land not found")

        ndvi = (
            supabase.table("ndvi_micro_tiles")
            .select("*")
            .eq("land_id", land_id)
            .order("acquisition_date", desc=True)
            .limit(5)
            .execute()
        )

        logs = (
            supabase.table("ndvi_processing_logs")
            .select("*")
            .eq("land_id", land_id)
            .order("completed_at", desc=True)
            .limit(5)
            .execute()
        )

        land_info = land.data[0]
        return {
            "status": "success",
            "land": {
                "id": land_info["id"],
                "name": land_info["name"],
                "area_acres": land_info.get("area_acres"),
                "has_boundary": bool(land_info.get("boundary_polygon_old")),
                "ndvi_tested": land_info.get("ndvi_tested"),
                "last_ndvi_value": land_info.get("last_ndvi_value"),
                "last_processed": land_info.get("last_processed_at"),
            },
            "recent_ndvi": ndvi.data or [],
            "processing_logs": logs.data or [],
        }
    except Exception as e:
        logger.error(f"❌ Diagnostics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ ENTRY ============
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🚀 Running NDVI API v4.1.0 on port {port}")
    uvicorn.run("ndvi_land_api_fixed:app", host="0.0.0.0", port=port, log_level="info")
