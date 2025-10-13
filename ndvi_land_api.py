"""
NDVI Land API (v3.9)
--------------------
Unified API for NDVI processing and data management.

✅ Features & Fixes (v3.9)
- Keeps all previous v3.8 endpoints (lands/analyze, requests/queue, requests/stats)
- Adds Lovable-compatible routes:
    • /api/v1/ndvi/data
    • /api/v1/ndvi/requests
    • /api/v1/ndvi/stats/global
    • /api/v1/ndvi/queue/status
- Improved error handling, consistent JSON response format
- Compatible with Render cron jobs & NDVI worker (v3.9)
- Works with Supabase + B2 + Lovable NDVI dashboard

© 2025 KisanShaktiAI
"""

import os
import datetime
import logging
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from supabase import create_client

# =====================================================
# CONFIGURATION
# =====================================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("❌ Missing Supabase credentials!")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ndvi-api-v3.9")


# =====================================================
# FASTAPI INITIALIZATION
# =====================================================
app = FastAPI(
    title="KisanShakti NDVI API",
    description="Handles NDVI request queueing, data retrieval, and statistics.",
    version="3.9.0",
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


# =====================================================
# ROOT & HEALTH
# =====================================================
@app.get("/api/v1/", tags=["Health"])
async def root():
    """Root service metadata"""
    return {
        "service": "KisanShakti NDVI API",
        "version": "3.9.0",
        "status": "operational",
        "features": [
            "NDVI request queue",
            "NDVI processing pipeline",
            "Lovable dashboard integration",
        ],
        "endpoints": {
            "create_request": "/api/v1/ndvi/lands/analyze",
            "get_queue": "/api/v1/ndvi/requests/queue",
            "get_stats": "/api/v1/ndvi/requests/stats",
            "get_data": "/api/v1/ndvi/data",
            "lovable_stats": "/api/v1/ndvi/stats/global",
        },
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }


@app.get("/api/v1/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": "ndvi-land-api",
        "version": "3.9.0",
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }


# =====================================================
# NDVI REQUEST MANAGEMENT (existing v3.8)
# =====================================================
@app.post("/api/v1/ndvi/lands/analyze", tags=["NDVI Analysis"])
async def create_ndvi_request(request: Request, tenant_id: str = Query(...)):
    """Create NDVI processing request for one or more lands."""
    try:
        body = await request.json()
        land_ids = body.get("land_ids", [])
        tile_id = body.get("tile_id")

        if not land_ids or not tile_id:
            raise HTTPException(status_code=400, detail="Missing required fields: land_ids or tile_id")

        payload = {
            "tenant_id": tenant_id,
            "land_ids": land_ids,
            "tile_id": tile_id,
            "status": "queued",
            "requested_at": datetime.datetime.utcnow().isoformat(),
        }

        supabase.table("ndvi_request_queue").insert(payload).execute()
        return {
            "status": "success",
            "message": "NDVI request created successfully",
            "data": payload,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error creating NDVI request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ndvi/requests/queue", tags=["NDVI Queue"])
async def get_ndvi_queue(tenant_id: str = Query(...)):
    """Get NDVI request queue for a tenant."""
    try:
        resp = supabase.table("ndvi_request_queue").select("*").eq("tenant_id", tenant_id).execute()
        return {
            "status": "success",
            "count": len(resp.data or []),
            "queue": resp.data or [],
        }
    except Exception as e:
        logger.error(f"❌ Error fetching NDVI queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ndvi/requests/stats", tags=["NDVI Statistics"])
async def get_ndvi_stats():
    """Retrieve NDVI processing statistics."""
    try:
        total = supabase.table("ndvi_request_queue").select("id", count="exact").execute()
        queued = supabase.table("ndvi_request_queue").select("id", count="exact").eq("status", "queued").execute()
        processing = supabase.table("ndvi_request_queue").select("id", count="exact").eq("status", "processing").execute()
        completed = supabase.table("ndvi_request_queue").select("id", count="exact").eq("status", "completed").execute()

        return {
            "status": "success",
            "stats": {
                "total_requests": total.count or 0,
                "queued": queued.count or 0,
                "processing": processing.count or 0,
                "completed": completed.count or 0,
            },
        }

    except Exception as e:
        logger.error(f"❌ Error fetching NDVI stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# LOVABLE-COMPATIBLE ENDPOINTS (new additions)
# =====================================================

@app.get("/api/v1/ndvi/requests", tags=["Lovable Compatibility"])
async def list_requests(tenant_id: str = Query(...), limit: int = Query(50)):
    """List NDVI requests (Lovable frontend expects this)."""
    try:
        resp = (
            supabase.table("ndvi_request_queue")
            .select("*")
            .eq("tenant_id", tenant_id)
            .order("requested_at", desc=True)
            .limit(limit)
            .execute()
        )
        return {"status": "success", "requests": resp.data or []}
    except Exception as e:
        logger.error(f"❌ Error listing NDVI requests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ndvi/data", tags=["Lovable Compatibility"])
async def list_ndvi_data(tenant_id: str = Query(...), limit: int = Query(100)):
    """Retrieve NDVI data history for Lovable dashboard."""
    try:
        resp = (
            supabase.table("ndvi_data")
            .select("*")
            .eq("tenant_id", tenant_id)
            .order("date", desc=True)
            .limit(limit)
            .execute()
        )
        return {"status": "success", "count": len(resp.data or []), "data": resp.data or []}
    except Exception as e:
        logger.error(f"❌ Error fetching NDVI data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ndvi/stats/global", tags=["Lovable Compatibility"])
async def global_ndvi_stats():
    """Return global NDVI processing summary for Lovable dashboard."""
    try:
        completed = supabase.table("ndvi_request_queue").select("id", count="exact").eq("status", "completed").execute()
        queued = supabase.table("ndvi_request_queue").select("id", count="exact").eq("status", "queued").execute()
        processing = supabase.table("ndvi_request_queue").select("id", count="exact").eq("status", "processing").execute()
        failed = supabase.table("ndvi_request_queue").select("id", count="exact").eq("status", "failed").execute()

        return {
            "status": "success",
            "stats": {
                "completed": completed.count or 0,
                "queued": queued.count or 0,
                "processing": processing.count or 0,
                "failed": failed.count or 0,
            },
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"❌ Error fetching global NDVI stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ndvi/queue/status", tags=["Lovable Compatibility"])
async def ndvi_queue_status():
    """Return queue status summary (active and processing counts)."""
    try:
        active = supabase.table("ndvi_request_queue").select("*").in_("status", ["queued", "processing"]).execute()
        return {
            "status": "success",
            "active_jobs": len(active.data or []),
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"❌ Error fetching queue status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# ERROR HANDLING
# =====================================================
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "available_endpoints": [
                "/api/v1/",
                "/api/v1/health",
                "/api/v1/ndvi/lands/analyze",
                "/api/v1/ndvi/requests",
                "/api/v1/ndvi/data",
                "/api/v1/ndvi/stats/global",
                "/api/v1/ndvi/queue/status",
                "/api/v1/ndvi/requests/queue",
                "/api/v1/ndvi/requests/stats",
                "/docs",
            ],
            "documentation": "/docs",
        },
    )


# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🚀 Starting NDVI Land API v3.9 on port {port}")
    import uvicorn

    uvicorn.run("ndvi_land_api:app", host="0.0.0.0", port=port, log_level="info")
