# ndvi_land_api.py
"""
NDVI Land API v4.1 - Multi-tenant, instant-trigger + cron-compatible FastAPI service

Roles:
- Handles NDVI request creation
- Optionally triggers NDVI processing instantly (if not deferred)
- Provides queue/data/stats endpoints for dashboards
"""

import os
import datetime
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from supabase import create_client

# -----------------------------
# Logging config
# -----------------------------
logging.basicConfig(
    level=os.getenv("NDVI_API_LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("ndvi-api-v4")

# -----------------------------
# Config & Supabase client
# -----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://qfklkkzxemsbeniyugiz.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_NDVI_BUCKET = os.getenv("SUPABASE_NDVI_BUCKET", "ndvi-thumbnails")

if not SUPABASE_KEY:
    logger.error("Missing SUPABASE_SERVICE_ROLE_KEY environment variable")
    raise RuntimeError("Missing SUPABASE_SERVICE_ROLE_KEY environment variable")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# FastAPI app setup
# -----------------------------
app = FastAPI(title="NDVI Land API v4.1", version="4.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# -----------------------------
# Models
# -----------------------------
class CreateNDVIRequestBody(BaseModel):
    land_ids: List[str]
    tile_id: str
    statistics_only: Optional[bool] = False
    priority: Optional[int] = 5
    farmer_id: Optional[str] = None
    metadata: Optional[dict] = None
    instant: Optional[bool] = False  # 🔥 new field - trigger processing immediately

# -----------------------------
# Helpers
# -----------------------------
def now_iso():
    return datetime.datetime.utcnow().isoformat()

# -----------------------------
# Health & Root
# -----------------------------
@app.get("/api/v1/health", tags=["Health"])
async def health_check():
    logger.info("Health check requested")
    return {
        "status": "healthy",
        "service": "ndvi-land-api",
        "version": "4.1.0",
        "timestamp": now_iso(),
    }

@app.get("/api/v1/", tags=["Health"])
async def root():
    return {
        "service": "ndvi-land-api",
        "version": "4.1.0",
        "status": "operational",
        "timestamp": now_iso(),
    }

# -----------------------------
# Create NDVI request + optional instant trigger
# -----------------------------
@app.post("/api/v1/ndvi/lands/analyze", tags=["NDVI Analysis"])
async def create_ndvi_request(body: CreateNDVIRequestBody, tenant_id: str = Query(...)):
    """
    Creates NDVI processing request for lands.
    If body.instant=True → immediately triggers ndvi_land_worker.process_request_sync
    Otherwise → job will be picked up later by cron worker.
    """
    try:
        logger.info(
            "Create NDVI request",
            extra={
                "tenant_id": tenant_id,
                "tile_id": body.tile_id,
                "lands": len(body.land_ids),
                "instant": body.instant,
            },
        )

        if not body.land_ids or not body.tile_id:
            raise HTTPException(status_code=400, detail="Missing land_ids or tile_id")

        payload = {
            "tenant_id": tenant_id,
            "land_ids": body.land_ids,
            "tile_id": body.tile_id,
            "statistics_only": body.statistics_only,
            "priority": body.priority,
            "farmer_id": body.farmer_id,
            "metadata": body.metadata or {},
            "status": "queued",
            "requested_at": now_iso(),
            "created_at": now_iso(),
        }

        # 1️⃣ Enqueue job
        res = supabase.table("ndvi_request_queue").insert(payload).execute()
        if getattr(res, "error", None):
            logger.error("Supabase insert error", extra={"error": res.error})
            raise HTTPException(status_code=500, detail="Failed to enqueue request")

        queue_id = res.data[0]["id"]
        logger.info("✅ NDVI request enqueued", extra={"queue_id": queue_id})

        # 2️⃣ If instant flag is set → immediately trigger processing
        result = None
        if body.instant:
            try:
                from ndvi_land_worker import process_request_sync
                logger.info("⚡ Instant NDVI processing triggered", extra={"queue_id": queue_id})
                result = process_request_sync(
                    queue_id=queue_id,
                    tenant_id=tenant_id,
                    land_ids=body.land_ids,
                    tile_id=body.tile_id,
                )

                final_status = "completed" if result.get("processed_count", 0) > 0 else "failed"
                supabase.table("ndvi_request_queue").update(
                    {
                        "status": final_status,
                        "processed_count": result.get("processed_count", 0),
                        "completed_at": now_iso(),
                        "last_error": result.get("last_error"),
                    }
                ).eq("id", queue_id).execute()

                logger.info(
                    "✅ Instant NDVI job finished",
                    extra={"queue_id": queue_id, "status": final_status, "result": result},
                )

            except Exception as e:
                logger.exception("❌ Instant NDVI processing failed")
                supabase.table("ndvi_request_queue").update(
                    {"status": "failed", "last_error": str(e)[:500]}
                ).eq("id", queue_id).execute()
                raise HTTPException(status_code=500, detail=f"NDVI processing failed: {e}")

        # 3️⃣ Respond to frontend
        return {
            "status": "success",
            "message": "NDVI request created successfully"
            + (" and processed instantly" if body.instant else ""),
            "queue_id": queue_id,
            "instant_result": result if body.instant else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error creating NDVI request")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# Query endpoints (queue, data, stats)
# -----------------------------
@app.get("/api/v1/ndvi/requests/queue", tags=["NDVI Queue"])
async def get_ndvi_queue(tenant_id: str = Query(...)):
    try:
        resp = (
            supabase.table("ndvi_request_queue")
            .select("*")
            .eq("tenant_id", tenant_id)
            .order("created_at", desc=True)
            .execute()
        )
        queue = resp.data or []
        logger.info("Fetched ndvi_request_queue", extra={"tenant_id": tenant_id, "count": len(queue)})
        return {"status": "success", "count": len(queue), "queue": queue}
    except Exception as e:
        logger.exception("Error fetching ndvi queue")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/ndvi/data", tags=["NDVI Data"])
async def list_ndvi_data(tenant_id: str = Query(...), limit: int = Query(100)):
    try:
        resp = (
            supabase.table("ndvi_data")
            .select("*")
            .eq("tenant_id", tenant_id)
            .order("date", desc=True)
            .limit(limit)
            .execute()
        )
        logger.info("Fetched ndvi data", extra={"tenant_id": tenant_id, "count": len(resp.data or [])})
        return {"status": "success", "count": len(resp.data or []), "data": resp.data or []}
    except Exception as e:
        logger.exception("Error fetching ndvi data")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/ndvi/stats/global", tags=["NDVI Stats"])
async def global_ndvi_stats():
    try:
        completed = supabase.table("ndvi_request_queue").select("id", count="exact").eq("status", "completed").execute()
        queued = supabase.table("ndvi_request_queue").select("id", count="exact").eq("status", "queued").execute()
        processing = supabase.table("ndvi_request_queue").select("id", count="exact").eq("status", "processing").execute()
        failed = supabase.table("ndvi_request_queue").select("id", count="exact").eq("status", "failed").execute()

        stats = {
            "completed": getattr(completed, "count", 0) or 0,
            "queued": getattr(queued, "count", 0) or 0,
            "processing": getattr(processing, "count", 0) or 0,
            "failed": getattr(failed, "count", 0) or 0,
        }
        logger.info("Fetched global ndvi stats", extra={"stats": stats})
        return {"status": "success", "stats": stats, "timestamp": now_iso()}
    except Exception as e:
        logger.exception("Error fetching global ndvi stats")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# 404 handler
# -----------------------------
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(status_code=404, content={"error": "Endpoint not found", "documentation": "/docs"})

# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting NDVI Land API v4.1 on port {port}")
    uvicorn.run("ndvi_land_api:app", host="0.0.0.0", port=port, log_level="info")
