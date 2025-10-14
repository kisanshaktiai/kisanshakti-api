"""
NDVI Land API (v3.9.1-FIXED)
-----------------------------
✅ CRITICAL FIX: Auto-triggers worker after queueing requests

Changes from v3.9:
1. Added automatic worker invocation in /api/v1/ndvi/lands/analyze
2. New background task processor to avoid blocking
3. Enhanced diagnostics in /api/v1/ndvi/process-queue
4. All core NDVI logic preserved

© 2025 KisanShaktiAI
"""

import os
import datetime
import logging
import asyncio
from fastapi import FastAPI, HTTPException, Query, Request, BackgroundTasks
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
logger = logging.getLogger("ndvi-api-v3.9.1")


# =====================================================
# FASTAPI INITIALIZATION
# =====================================================
app = FastAPI(
    title="KisanShakti NDVI API",
    description="Handles NDVI request queueing, data retrieval, and statistics.",
    version="3.9.1",
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
# BACKGROUND WORKER CALLER
# =====================================================
async def process_queue_item_background(queue_id: str, tenant_id: str, land_ids: list, tile_id: str):
    """
    Background task that calls the worker to process NDVI.
    This runs asynchronously so the API returns immediately.
    """
    try:
        logger.info(f"🔄 [BACKGROUND] Starting NDVI processing for queue_id={queue_id}")
        
        # Import worker (do this inside function to avoid import errors)
        try:
            from ndvi_land_worker import process_farmer_land
        except ImportError as ie:
            logger.error(f"❌ [BACKGROUND] Worker import failed: {ie}")
            supabase.table("ndvi_request_queue").update({
                "status": "failed",
                "last_error": f"Worker import failed: {str(ie)}",
                "completed_at": datetime.datetime.utcnow().isoformat(),
            }).eq("id", queue_id).execute()
            return

        # Update status to processing
        supabase.table("ndvi_request_queue").update({
            "status": "processing",
            "started_at": datetime.datetime.utcnow().isoformat(),
        }).eq("id", queue_id).execute()

        # Fetch lands
        lands_resp = supabase.table("lands").select("*").eq("tenant_id", tenant_id).in_("id", land_ids).execute()
        lands = lands_resp.data or []
        
        if not lands:
            logger.error(f"❌ [BACKGROUND] No lands found for queue_id={queue_id}")
            supabase.table("ndvi_request_queue").update({
                "status": "failed",
                "last_error": "No lands found",
                "completed_at": datetime.datetime.utcnow().isoformat(),
            }).eq("id", queue_id).execute()
            return

        # Fetch tile metadata
        tile_resp = supabase.table("satellite_tiles").select("*").eq("tile_id", tile_id).limit(1).execute()
        if not tile_resp.data:
            logger.error(f"❌ [BACKGROUND] Tile metadata not found for tile_id={tile_id}")
            supabase.table("ndvi_request_queue").update({
                "status": "failed",
                "last_error": f"Tile {tile_id} not found in satellite_tiles",
                "completed_at": datetime.datetime.utcnow().isoformat(),
            }).eq("id", queue_id).execute()
            return

        tile = tile_resp.data[0]
        logger.info(f"✅ [BACKGROUND] Processing {len(lands)} lands with tile {tile_id}")

        # Process each land
        processed_count = 0
        failed_lands = []

        for land in lands:
            try:
                logger.info(f"▶️ [BACKGROUND] Processing land {land['id']}")
                success = process_farmer_land(land, tile)
                
                if success:
                    processed_count += 1
                    logger.info(f"✅ [BACKGROUND] Land {land['id']} processed successfully")
                else:
                    failed_lands.append(land['id'])
                    logger.warning(f"⚠️ [BACKGROUND] Land {land['id']} failed (worker returned False)")
                    
            except Exception as land_error:
                failed_lands.append(land['id'])
                logger.error(f"❌ [BACKGROUND] Exception processing land {land['id']}: {land_error}")

        # Update queue status
        final_status = "completed" if processed_count > 0 else "failed"
        update_data = {
            "status": final_status,
            "processed_count": processed_count,
            "completed_at": datetime.datetime.utcnow().isoformat(),
        }
        
        if failed_lands:
            update_data["last_error"] = f"Failed lands: {failed_lands}"

        supabase.table("ndvi_request_queue").update(update_data).eq("id", queue_id).execute()

        # Verify insertions
        ndvi_check = supabase.table("ndvi_data").select("id").in_("land_id", land_ids).execute()
        micro_check = supabase.table("ndvi_micro_tiles").select("id").in_("land_id", land_ids).execute()
        
        logger.info(
            f"🎯 [BACKGROUND] Queue {queue_id} completed | "
            f"processed={processed_count}/{len(lands)} | "
            f"ndvi_data={len(ndvi_check.data or [])} | "
            f"micro_tiles={len(micro_check.data or [])}"
        )

    except Exception as e:
        logger.error(f"❌ [BACKGROUND] Fatal error in background processor: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Update queue to failed
        try:
            supabase.table("ndvi_request_queue").update({
                "status": "failed",
                "last_error": str(e)[:500],
                "completed_at": datetime.datetime.utcnow().isoformat(),
            }).eq("id", queue_id).execute()
        except Exception as update_err:
            logger.error(f"❌ [BACKGROUND] Failed to update queue status: {update_err}")


# =====================================================
# ROOT & HEALTH
# =====================================================
@app.get("/api/v1/", tags=["Health"])
async def root():
    """Root service metadata"""
    return {
        "service": "KisanShakti NDVI API",
        "version": "3.9.1",
        "status": "operational",
        "changelog": "Fixed: Auto-invokes worker after queueing",
        "features": [
            "NDVI request queue",
            "Automatic background processing",
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
        "version": "3.9.1",
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }


# =====================================================
# NDVI REQUEST MANAGEMENT (FIXED)
# =====================================================
@app.post("/api/v1/ndvi/lands/analyze", tags=["NDVI Analysis"])
async def create_ndvi_request(
    request: Request, 
    background_tasks: BackgroundTasks,
    tenant_id: str = Query(...)
):
    """
    Create NDVI processing request and automatically trigger processing.
    
    ✅ FIX: Now automatically invokes worker via background task
    """
    try:
        body = await request.json()
        land_ids = body.get("land_ids", [])
        tile_id = body.get("tile_id")

        if not land_ids or not tile_id:
            raise HTTPException(status_code=400, detail="Missing required fields: land_ids or tile_id")

        logger.info(f"📥 Creating NDVI request: tenant={tenant_id}, lands={len(land_ids)}, tile={tile_id}")

        # Insert into queue
        payload = {
            "tenant_id": tenant_id,
            "land_ids": land_ids,
            "tile_id": tile_id,
            "status": "queued",
            "requested_at": datetime.datetime.utcnow().isoformat(),
        }

        insert_resp = supabase.table("ndvi_request_queue").insert(payload).execute()
        
        if not insert_resp.data:
            raise HTTPException(status_code=500, detail="Failed to insert into queue")

        queue_id = insert_resp.data[0]["id"]
        logger.info(f"✅ Queue entry created: queue_id={queue_id}")

        # 🔥 CRITICAL FIX: Automatically trigger background processing
        background_tasks.add_task(
            process_queue_item_background,
            queue_id=queue_id,
            tenant_id=tenant_id,
            land_ids=land_ids,
            tile_id=tile_id
        )
        
        logger.info(f"🚀 Background worker scheduled for queue_id={queue_id}")

        return {
            "status": "success",
            "message": "NDVI request created and processing started",
            "data": {
                **payload,
                "id": queue_id,
                "processing": "background",
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error creating NDVI request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# MANUAL PROCESSING ENDPOINT (for debugging)
# =====================================================
@app.post("/api/v1/ndvi/process-queue", tags=["NDVI Processing"])
async def process_queue_item(request: Request):
    """
    Manually process a specific queue item (for debugging).
    Note: /api/v1/ndvi/lands/analyze now auto-triggers this via background task.
    """
    try:
        body = await request.json()
        queue_id = body.get("queue_id")
        tenant_id = body.get("tenant_id")
        land_ids = body.get("land_ids", [])
        tile_id = body.get("tile_id")

        logger.info(f"🔄 [MANUAL] Processing queue_id={queue_id}")

        if not all([queue_id, tenant_id, land_ids, tile_id]):
            raise HTTPException(status_code=400, detail="Missing required fields")

        # Import worker
        try:
            from ndvi_land_worker import process_farmer_land
        except ImportError as e:
            logger.error(f"❌ Worker import failed: {e}")
            raise HTTPException(status_code=500, detail="Worker module not found")

        # Verify queue entry
        queue_resp = supabase.table("ndvi_request_queue").select("*").eq("id", queue_id).execute()
        if not queue_resp.data:
            raise HTTPException(status_code=404, detail=f"Queue item {queue_id} not found")

        # Update to processing
        supabase.table("ndvi_request_queue").update({
            "status": "processing",
            "started_at": datetime.datetime.utcnow().isoformat(),
        }).eq("id", queue_id).execute()

        # Fetch lands
        lands_resp = supabase.table("lands").select("*").eq("tenant_id", tenant_id).in_("id", land_ids).execute()
        lands = lands_resp.data or []
        
        if not lands:
            raise HTTPException(status_code=404, detail="No lands found for given IDs")

        # Fetch tile
        tile_resp = supabase.table("satellite_tiles").select("*").eq("tile_id", tile_id).limit(1).execute()
        if not tile_resp.data:
            raise HTTPException(status_code=404, detail=f"Tile metadata not found for {tile_id}")

        tile = tile_resp.data[0]

        # Process lands
        processed_count = 0
        failed_lands = []
        results_summary = []

        for land in lands:
            try:
                logger.info(f"▶️ Processing land_id={land['id']}")
                success = process_farmer_land(land, tile)

                if success:
                    processed_count += 1
                    logger.info(f"✅ Land {land['id']} processed")
                else:
                    failed_lands.append(land['id'])
                    logger.warning(f"⚠️ Land {land['id']} failed")

                results_summary.append({
                    "land_id": land["id"],
                    "result": "success" if success else "failed",
                })

            except Exception as land_error:
                failed_lands.append(land["id"])
                logger.error(f"❌ Exception processing land {land['id']}: {land_error}")

        # Update queue status
        final_status = "completed" if processed_count > 0 else "failed"
        supabase.table("ndvi_request_queue").update({
            "status": final_status,
            "processed_count": processed_count,
            "completed_at": datetime.datetime.utcnow().isoformat(),
            "last_error": f"Failed lands: {failed_lands}" if failed_lands else None,
        }).eq("id", queue_id).execute()

        # Verify NDVI insertions
        ndvi_check = supabase.table("ndvi_data").select("id").in_("land_id", land_ids).execute()
        micro_check = supabase.table("ndvi_micro_tiles").select("id").in_("land_id", land_ids).execute()

        logger.info(f"📊 Verification: ndvi_data={len(ndvi_check.data or [])}, micro_tiles={len(micro_check.data or [])}")

        return {
            "status": "success",
            "queue_id": queue_id,
            "processed_count": processed_count,
            "failed_count": len(failed_lands),
            "failed_lands": failed_lands,
            "verify": {
                "ndvi_data": len(ndvi_check.data or []),
                "ndvi_micro_tiles": len(micro_check.data or []),
            },
            "results": results_summary,
        }

    except Exception as e:
        logger.error(f"❌ Fatal error in process_queue_item: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# QUEUE MANAGEMENT
# =====================================================
@app.get("/api/v1/ndvi/requests/queue", tags=["NDVI Queue"])
async def get_ndvi_queue(tenant_id: str = Query(...)):
    """Get NDVI request queue for a tenant."""
    try:
        resp = supabase.table("ndvi_request_queue").select("*").eq("tenant_id", tenant_id).order("created_at", desc=True).execute()
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
# LOVABLE-COMPATIBLE ENDPOINTS
# =====================================================
@app.get("/api/v1/ndvi/requests", tags=["Lovable Compatibility"])
async def list_requests(tenant_id: str = Query(...), limit: int = Query(50)):
    """List NDVI requests (Lovable frontend expects this)."""
    try:
        resp = (
            supabase.table("ndvi_request_queue")
            .select("*")
            .eq("tenant_id", tenant_id)
            .order("created_at", desc=True)
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
    logger.info(f"🚀 Starting NDVI Land API v3.9.1 (FIXED) on port {port}")
    import uvicorn

    uvicorn.run("ndvi_land_api:app", host="0.0.0.0", port=port, log_level="info")
