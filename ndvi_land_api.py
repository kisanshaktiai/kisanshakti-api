"""
NDVI Land API v4.2.0 — Analysis + Worker Trigger
-------------------------------------------------
✅ Uses unified ndvi_micro_tiles table
✅ Triggers NDVI Land Worker v5.1 in background
✅ Handles Supabase queue updates
✅ Safe error logging + full trace
✅ Compatible with frontend NDVI monitor

© 2025 KisanShaktiAI
"""

import os, datetime, logging, traceback
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client

# ---------------- CONFIG ----------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("❌ Missing Supabase credentials")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ndvi-api-v4.2")

# ---------------- FASTAPI ----------------
app = FastAPI(
    title="KisanShakti NDVI API",
    description="NDVI microtile API (v4.2.0) with background worker",
    version="4.2.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MODELS ----------------
class NDVIRequest(BaseModel):
    tenant_id: str
    land_ids: List[str]
    tile_id: Optional[str] = None


# ---------------- HEALTH ----------------
@app.get("/api/v1/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": "ndvi-land-api",
        "version": "4.2.0",
        "worker": "v5.1",
    }


# ---------------- IMPORT WORKER ----------------
try:
    from ndvi_land_worker_v5 import process_land_ndvi
except ImportError:
    logger.warning("⚠️ NDVI Worker module not found — background processing disabled")
    process_land_ndvi = None


# ---------------- WORKER RUNNER ----------------
def run_ndvi_worker(queue_id: str, tenant_id: str, land_ids: list, tile_id: Optional[str]):
    """Executes NDVI computation for all lands in queue (background)."""
    logger.info(f"🔧 [WORKER START] queue_id={queue_id}, tenant={tenant_id}, lands={len(land_ids)}")

    try:
        # Fetch lands
        lands_resp = (
            supabase.table("lands")
            .select("*")
            .in_("id", land_ids)
            .eq("tenant_id", tenant_id)
            .execute()
        )
        lands = lands_resp.data or []
        if not lands:
            raise ValueError("No lands found for NDVI processing")

        # Fetch NDVI tiles
        tiles_resp = (
            supabase.table("ndvi_micro_tiles")
            .select("*")
            .eq("tenant_id", tenant_id)
            .execute()
        )
        all_tiles = tiles_resp.data or []

        # Update queue → processing
        supabase.table("ndvi_request_queue").update(
            {"status": "processing", "started_at": datetime.datetime.utcnow().isoformat()}
        ).eq("id", queue_id).execute()

        success, failed = 0, 0
        for land in lands:
            ok = process_land_ndvi(land, all_tiles)
            if ok:
                success += 1
            else:
                failed += 1

        # Finalize queue record
        supabase.table("ndvi_request_queue").update(
            {
                "status": "completed" if failed == 0 else "failed",
                "completed_at": datetime.datetime.utcnow().isoformat(),
                "processed_count": success,
                "failed_count": failed,
            }
        ).eq("id", queue_id).execute()

        logger.info(
            f"✅ [WORKER DONE] queue_id={queue_id} | processed={success} | failed={failed}"
        )

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"❌ [WORKER ERROR] queue_id={queue_id}: {e}")
        logger.debug(tb)

        # Mark queue failed
        supabase.table("ndvi_request_queue").update(
            {
                "status": "failed",
                "completed_at": datetime.datetime.utcnow().isoformat(),
                "error_message": str(e)[:400],
                "error_trace": tb[:1000],
            }
        ).eq("id", queue_id).execute()


# ---------------- ANALYZE ENDPOINT ----------------
@app.post("/api/v1/ndvi/lands/analyze", tags=["NDVI Analysis"])
async def analyze_lands(request: NDVIRequest, background_tasks: BackgroundTasks):
    """Queue NDVI processing and run background worker."""
    try:
        if not process_land_ndvi:
            raise RuntimeError("NDVI Worker not available in runtime")

        tenant_id = request.tenant_id
        land_ids = request.land_ids
        tile_id = request.tile_id

        logger.info(f"📩 NDVI analyze request: tenant={tenant_id}, lands={len(land_ids)}")

        # Insert into request queue
        queue_id = supabase.table("ndvi_request_queue").insert(
            {
                "tenant_id": tenant_id,
                "status": "queued",
                "land_ids": land_ids,
                "tile_id": tile_id,
                "requested_at": datetime.datetime.utcnow().isoformat(),
            }
        ).execute().data[0]["id"]

        logger.info(f"🧾 Queue created: {queue_id}")

        # Schedule background worker
        background_tasks.add_task(run_ndvi_worker, queue_id, tenant_id, land_ids, tile_id)
        logger.info(f"🚀 NDVI worker task registered for {queue_id}")

        return {"queue_id": queue_id, "status": "queued", "tenant": tenant_id}

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"❌ Failed to queue NDVI analysis: {e}")
        logger.debug(tb)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------- STATS + DATA (existing endpoints) ----------------
@app.get("/api/v1/ndvi/data", tags=["NDVI Data"])
async def list_ndvi_data(
    tenant_id: str = Query(...), land_id: Optional[str] = None, limit: int = 100
):
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


# ---------------- ENTRY ----------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🚀 Running NDVI API v4.2.0 on port {port}")
    uvicorn.run("ndvi_land_api_v4_2:app", host="0.0.0.0", port=port, log_level="info")
