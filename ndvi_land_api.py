# ndvi_land_api_v4.py
"""
NDVI Land API v4 - Multi-tenant, log-rich FastAPI service
- Endpoints:
  - GET /api/v1/health
  - POST /api/v1/ndvi/lands/analyze  (creates queue item)
  - POST /api/v1/ndvi/process-queue  (called by Edge/worker orchestrator to process a specific queue item)
  - GET /api/v1/ndvi/requests/queue
  - GET /api/v1/ndvi/requests/stats
  - GET /api/v1/ndvi/requests
  - GET /api/v1/ndvi/data
  - GET /api/v1/ndvi/stats/global
  - GET /api/v1/ndvi/queue/status
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

# ----------------------
# Logging config (rich, structured-ish)
# ----------------------
logging.basicConfig(
    level=os.getenv("NDVI_API_LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("ndvi-api-v4")

# ----------------------
# Config & defaults (override with env)
# ----------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://qfklkkzxemsbeniyugiz.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_NDVI_BUCKET = os.getenv("SUPABASE_NDVI_BUCKET", "ndvi-thumbnails")

if not SUPABASE_KEY:
    logger.error("Missing SUPABASE_SERVICE_ROLE_KEY environment variable")
    raise RuntimeError("Missing SUPABASE_SERVICE_ROLE_KEY environment variable")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------------------
# App
# ----------------------
app = FastAPI(title="NDVI Land API v4", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ----------------------
# Models
# ----------------------
class CreateNDVIRequestBody(BaseModel):
    land_ids: List[str]
    tile_id: str
    statistics_only: Optional[bool] = False
    priority: Optional[int] = 5
    farmer_id: Optional[str] = None
    metadata: Optional[dict] = None

# ----------------------
# Helpers
# ----------------------
def now_iso():
    return datetime.datetime.utcnow().isoformat()

# ----------------------
# Health & Root
# ----------------------
@app.get("/api/v1/health", tags=["Health"])
async def health_check():
    logger.info("Health check requested")
    return {"status": "healthy", "service": "ndvi-land-api", "version": "4.0.0", "timestamp": now_iso()}

@app.get("/api/v1/", tags=["Health"])
async def root():
    return {
        "service": "ndvi-land-api",
        "version": "4.0.0",
        "status": "operational",
        "timestamp": now_iso(),
    }

# ----------------------
# Create NDVI request (queue)
# ----------------------
@app.post("/api/v1/ndvi/lands/analyze", tags=["NDVI Analysis"])
async def create_ndvi_request(body: CreateNDVIRequestBody, tenant_id: str = Query(...)):
    """
    Create NDVI processing request for lands under a tenant.
    """
    try:
        logger.info("Create NDVI request", extra={"tenant_id": tenant_id, "tile_id": body.tile_id, "lands": len(body.land_ids)})
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

        res = supabase.table("ndvi_request_queue").insert(payload).execute()
        if getattr(res, "error", None):
            logger.error("Supabase insert error", extra={"error": res.error})
            raise HTTPException(status_code=500, detail="Failed to enqueue request")

        logger.info("NDVI request enqueued", extra={"tenant_id": tenant_id, "rows": res.data})
        return {"status": "success", "message": "NDVI request created successfully", "data": payload}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error creating NDVI request")
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------
# Process queue item (synchronous call used by orchestrator/edge)
# ----------------------
@app.post("/api/v1/ndvi/process-queue", tags=["NDVI Processing"])
async def process_queue_item(request: Request):
    """
    Synchronously process a specific queue item.
    The actual heavy work is in the worker module; this endpoint imports and reuses that logic.
    """
    # This endpoint is intentionally sync-ish: the caller expects the API to return the result of processing.
    # It should be called by orchestrator or small worker (or use to test).
    try:
        body = await request.json()
        queue_id = body.get("queue_id")
        tenant_id = body.get("tenant_id")
        land_ids = body.get("land_ids", [])
        tile_id = body.get("tile_id")

        logger.info("process-queue called", extra={"queue_id": queue_id, "tenant_id": tenant_id, "tile_count": len(land_ids)})

        if not all([queue_id, tenant_id, land_ids, tile_id]):
            logger.warning("Missing required fields in process-queue", extra={"payload": body})
            raise HTTPException(status_code=400, detail="Missing required fields: queue_id, tenant_id, land_ids, tile_id")

        # verify queue item exists
        q = supabase.table("ndvi_request_queue").select("*").eq("id", queue_id).limit(1).execute()
        if not q.data:
            logger.warning("Queue item not found", extra={"queue_id": queue_id})
            raise HTTPException(status_code=404, detail="Queue item not found")

        # mark processing
        supabase.table("ndvi_request_queue").update({"status": "processing", "started_at": now_iso()}).eq("id", queue_id).execute()

        # import worker processing function
        try:
            # worker file expected in same project: ndvi_land_worker_v4.py
            from ndvi_land_worker_v4 import process_request_sync  # type: ignore
        except Exception as ex:
            logger.exception("Worker module import failed")
            raise HTTPException(status_code=500, detail="Worker module not available")

        # call worker sync helper
        result = process_request_sync(queue_id=queue_id, tenant_id=tenant_id, land_ids=land_ids, tile_id=tile_id)

        # update queue final status
        final_status = "completed" if result.get("processed_count", 0) > 0 else "failed"
        supabase.table("ndvi_request_queue").update({
            "status": final_status,
            "processed_count": result.get("processed_count", 0),
            "completed_at": now_iso(),
            "last_error": result.get("last_error", None),
        }).eq("id", queue_id).execute()

        logger.info("process-queue completed", extra={"queue_id": queue_id, "result": result})
        return {"status": "success", "result": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("process-queue failed")
        # attempt mark failed
        try:
            if 'queue_id' in locals():
                supabase.table("ndvi_request_queue").update({"status": "failed", "last_error": str(e)[:500], "completed_at": now_iso()}).eq("id", queue_id).execute()
        except Exception:
            logger.exception("Failed to update queue failure state")
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------
# Query endpoints (queue, stats, data) - Lovable compatible
# ----------------------
@app.get("/api/v1/ndvi/requests/queue", tags=["NDVI Queue"])
async def get_ndvi_queue(tenant_id: str = Query(...)):
    try:
        resp = supabase.table("ndvi_request_queue").select("*").eq("tenant_id", tenant_id).order("created_at", desc=True).execute()
        queue = resp.data or []
        logger.info("Fetched ndvi_request_queue", extra={"tenant_id": tenant_id, "count": len(queue)})
        return {"status": "success", "count": len(queue), "queue": queue}
    except Exception as e:
        logger.exception("Error fetching ndvi queue")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/ndvi/requests/stats", tags=["NDVI Stats"])
async def get_ndvi_stats(tenant_id: Optional[str] = Query(None)):
    """
    Returns counts by status for a tenant (if tenant_id provided) or global otherwise.
    """
    try:
        q = supabase.table("ndvi_request_queue")
        if tenant_id:
            total = q.select("id", count="exact").eq("tenant_id", tenant_id).execute()
            queued = q.select("id", count="exact").eq("tenant_id", tenant_id).eq("status", "queued").execute()
            processing = q.select("id", count="exact").eq("tenant_id", tenant_id).eq("status", "processing").execute()
            completed = q.select("id", count="exact").eq("tenant_id", tenant_id).eq("status", "completed").execute()
        else:
            total = q.select("id", count="exact").execute()
            queued = q.select("id", count="exact").eq("status", "queued").execute()
            processing = q.select("id", count="exact").eq("status", "processing").execute()
            completed = q.select("id", count="exact").eq("status", "completed").execute()

        stats = {
            "total_requests": getattr(total, "count", 0) or 0,
            "queued": getattr(queued, "count", 0) or 0,
            "processing": getattr(processing, "count", 0) or 0,
            "completed": getattr(completed, "count", 0) or 0,
        }
        logger.info("Fetched ndvi stats", extra={"tenant_id": tenant_id, "stats": stats})
        return {"status": "success", "stats": stats}
    except Exception as e:
        logger.exception("Error fetching ndvi stats")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/ndvi/requests", tags=["Lovable Compatibility"])
async def list_requests(tenant_id: str = Query(...), limit: int = Query(50)):
    try:
        resp = (supabase.table("ndvi_request_queue").select("*").eq("tenant_id", tenant_id).order("created_at", desc=True).limit(limit).execute())
        return {"status": "success", "requests": resp.data or []}
    except Exception as e:
        logger.exception("Error listing requests")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/ndvi/data", tags=["Lovable Compatibility"])
async def list_ndvi_data(tenant_id: str = Query(...), limit: int = Query(100)):
    try:
        resp = supabase.table("ndvi_data").select("*").eq("tenant_id", tenant_id).order("date", desc=True).limit(limit).execute()
        logger.info("Fetched ndvi data", extra={"tenant_id": tenant_id, "count": len(resp.data or [])})
        return {"status": "success", "count": len(resp.data or []), "data": resp.data or []}
    except Exception as e:
        logger.exception("Error fetching ndvi data")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/ndvi/stats/global", tags=["Lovable Compatibility"])
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

@app.get("/api/v1/ndvi/queue/status", tags=["Lovable Compatibility"])
async def ndvi_queue_status():
    try:
        active = supabase.table("ndvi_request_queue").select("*").in_("status", ["queued", "processing"]).execute()
        active_count = len(active.data or [])
        logger.info("Queue status", extra={"active_jobs": active_count})
        return {"status": "success", "active_jobs": active_count, "timestamp": now_iso()}
    except Exception as e:
        logger.exception("Error fetching queue status")
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------
# 404 handler
# ----------------------
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(status_code=404, content={"error": "Endpoint not found", "documentation": "/docs"})

# ----------------------
# run
# ----------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting NDVI Land API v4 on port {port}")
    uvicorn.run("ndvi_land_api_v4:app", host="0.0.0.0", port=port, log_level="info")
