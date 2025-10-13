"""
NDVI Land API (v3.8)
--------------------
Processes and manages NDVI requests for KisanShaktiAI.

✅ Improvements:
- RESTful endpoint naming
- Versioned base path `/api/v1/`
- Consistent resource naming with Soil API
- Backward-compatible health and docs URLs
- Core NDVI logic unchanged
"""

import os
import datetime
import logging
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from supabase import create_client

# ==========================
# Configuration
# ==========================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("❌ Missing Supabase credentials!")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ndvi-api")

# ==========================
# FastAPI Setup
# ==========================
app = FastAPI(
    title="KisanShakti NDVI API",
    description="Handles NDVI request queueing and retrieval for lands.",
    version="3.8.0",
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

# ==========================
# Root and Health Endpoints
# ==========================

@app.get("/api/v1/", tags=["Health"])
async def root():
    return {
        "service": "KisanShakti NDVI API",
        "version": "3.8.0",
        "status": "operational",
        "features": [
            "NDVI request creation",
            "NDVI queue management",
            "NDVI statistics retrieval",
        ],
        "endpoints": {
            "create_request": "/api/v1/ndvi/lands/analyze",
            "queue_status": "/api/v1/ndvi/requests/queue",
            "queue_stats": "/api/v1/ndvi/requests/stats",
            "health_check": "/api/v1/health",
        },
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }


@app.get("/api/v1/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": "ndvi-land-api",
        "version": "3.8.0",
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }

# ==========================
# NDVI Request Management
# ==========================

@app.post("/api/v1/ndvi/lands/analyze", tags=["NDVI Analysis"])
async def create_ndvi_request(request: Request, tenant_id: str = Query(...)):
    """
    Create NDVI processing request for one or more lands.
    Replaces: /requests
    """
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
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error creating NDVI request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ndvi/requests/queue", tags=["NDVI Queue"])
async def get_ndvi_queue(tenant_id: str = Query(...)):
    """
    Get NDVI request queue for a tenant.
    Replaces: /queue
    """
    try:
        resp = supabase.table("ndvi_request_queue").select("*").eq("tenant_id", tenant_id).execute()
        return {
            "status": "success",
            "count": len(resp.data or []),
            "queue": resp.data or [],
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"❌ Error fetching NDVI queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ndvi/requests/stats", tags=["NDVI Statistics"])
async def get_ndvi_stats():
    """
    Retrieve NDVI processing statistics.
    Replaces: /stats
    """
    try:
        total = supabase.table("ndvi_request_queue").select("count").execute()
        queued = supabase.table("ndvi_request_queue").select("count").eq("status", "queued").execute()
        processing = supabase.table("ndvi_request_queue").select("count").eq("status", "processing").execute()
        completed = supabase.table("ndvi_request_queue").select("count").eq("status", "completed").execute()

        return {
            "status": "success",
            "stats": {
                "total_requests": total.data[0]["count"] if total.data else 0,
                "queued": queued.data[0]["count"] if queued.data else 0,
                "processing": processing.data[0]["count"] if processing.data else 0,
                "completed": completed.data[0]["count"] if completed.data else 0,
            },
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"❌ Error fetching NDVI stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================
# Error Handling
# ==========================
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
                "/api/v1/ndvi/requests/queue",
                "/api/v1/ndvi/requests/stats",
                "/docs",
            ],
            "documentation": "/docs",
        },
    )


# ==========================
# Server Entry Point
# ==========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🚀 Starting NDVI Land API v3.8 on port {port}")
    import uvicorn
    uvicorn.run("ndvi_land_api:app", host="0.0.0.0", port=port, log_level="info")
