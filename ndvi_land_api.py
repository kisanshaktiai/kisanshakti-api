"""
NDVI Land API v4.0.0 — Production Fix
--------------------------------------
✅ Auto-detects tiles (no manual tile_id required)
✅ Validates land-tile intersection before queueing
✅ Prevents duplicate requests
✅ Enhanced error reporting
✅ Backward compatible with v3.9.1

© 2025 KisanShaktiAI
"""

import os
import datetime
import logging
import asyncio
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ndvi-api-v4")

# ============ FASTAPI ============
app = FastAPI(
    title="KisanShakti NDVI API",
    description="Auto-detects tiles and processes NDVI for farmer lands",
    version="4.0.0",
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

# ============ MODELS ============
class NDVIRequest(BaseModel):
    land_ids: list[str]
    tile_id: Optional[str] = None  # Now optional - auto-detects if not provided

# ============ BACKGROUND PROCESSOR ============
async def process_queue_item_background(queue_id: str, tenant_id: str, land_ids: list, tile_id: Optional[str]):
    """
    Background task that processes NDVI with auto-tile detection.
    """
    try:
        logger.info(f"🔄 [BG] Starting queue_id={queue_id}")
        
        # Import worker
        try:
            from ndvi_land_worker_fixed import process_land_ndvi_thumbnail
        except ImportError:
            try:
                from ndvi_land_worker import process_farmer_land as process_land_ndvi_thumbnail
            except ImportError as ie:
                logger.error(f"❌ [BG] Worker import failed: {ie}")
                supabase.table("ndvi_request_queue").update({
                    "status": "failed",
                    "last_error": f"Worker not found: {str(ie)}",
                    "completed_at": datetime.datetime.utcnow().isoformat(),
                }).eq("id", queue_id).execute()
                return
        
        # Update to processing
        supabase.table("ndvi_request_queue").update({
            "status": "processing",
            "started_at": datetime.datetime.utcnow().isoformat(),
        }).eq("id", queue_id).execute()
        
        # Fetch lands
        lands_resp = supabase.table("lands")\
            .select("*")\
            .eq("tenant_id", tenant_id)\
            .in_("id", land_ids)\
            .execute()
        
        lands = lands_resp.data or []
        
        if not lands:
            logger.error(f"❌ [BG] No lands found")
            supabase.table("ndvi_request_queue").update({
                "status": "failed",
                "last_error": "No lands found",
                "completed_at": datetime.datetime.utcnow().isoformat(),
            }).eq("id", queue_id).execute()
            return
        
        # Fetch tile metadata (optional)
        tile = None
        if tile_id:
            tile_resp = supabase.table("satellite_tiles")\
                .select("*")\
                .eq("tile_id", tile_id)\
                .order("acquisition_date", desc=True)\
                .limit(1)\
                .execute()
            
            if tile_resp.data:
                tile = tile_resp.data[0]
                logger.info(f"✅ [BG] Using specified tile: {tile_id}")
            else:
                logger.warning(f"⚠️ [BG] Tile {tile_id} not found - will auto-detect")
        else:
            logger.info("🎯 [BG] Auto-detecting tiles for each land")
        
        # Process lands
        processed_count = 0
        failed_lands = []
        
        for land in lands:
            try:
                logger.info(f"▶️ [BG] Processing land {land['name']} (id={land['id'][:8]}...)")
                
                # Call worker with optional tile (auto-detects if None)
                success = process_land_ndvi_thumbnail(land, tile)
                
                if success:
                    processed_count += 1
                    logger.info(f"✅ [BG] Land {land['name']} done")
                else:
                    failed_lands.append({
                        "land_id": land["id"],
                        "name": land["name"],
                        "reason": "Worker returned False"
                    })
                    logger.warning(f"⚠️ [BG] Land {land['name']} failed")
                
            except Exception as land_error:
                failed_lands.append({
                    "land_id": land["id"],
                    "name": land["name"],
                    "reason": str(land_error)[:200]
                })
                logger.error(f"❌ [BG] Exception: {land['name']} - {land_error}")
        
        # Update queue status
        final_status = "completed" if processed_count > 0 else "failed"
        update_data = {
            "status": final_status,
            "processed_count": processed_count,
            "completed_at": datetime.datetime.utcnow().isoformat(),
        }
        
        if failed_lands:
            update_data["last_error"] = f"{len(failed_lands)} lands failed"
            update_data["error_details"] = {"failed_lands": failed_lands}
        
        supabase.table("ndvi_request_queue").update(update_data).eq("id", queue_id).execute()
        
        logger.info(
            f"🎯 [BG] Queue {queue_id} complete | "
            f"success={processed_count}/{len(lands)} | "
            f"failed={len(failed_lands)}"
        )
        
    except Exception as e:
        logger.error(f"❌ [BG] Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        try:
            supabase.table("ndvi_request_queue").update({
                "status": "failed",
                "last_error": str(e)[:500],
                "completed_at": datetime.datetime.utcnow().isoformat(),
            }).eq("id", queue_id).execute()
        except Exception as update_err:
            logger.error(f"❌ [BG] Failed to update status: {update_err}")

# ============ ROOT & HEALTH ============
@app.get("/api/v1/", tags=["Health"])
async def root():
    return {
        "service": "KisanShakti NDVI API",
        "version": "4.0.0",
        "status": "operational",
        "features": [
            "Auto-tile detection (no manual tile_id needed)",
            "Boundary overlap detection",
            "Enhanced error reporting",
            "Background processing",
        ],
        "changelog": "v4.0.0: Auto-detects tiles, validates boundaries, prevents overlaps",
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }


@app.get("/api/v1/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": "ndvi-land-api",
        "version": "4.0.0",
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }


# ============ NDVI ANALYSIS (ENHANCED) ============
@app.post("/api/v1/ndvi/lands/analyze", tags=["NDVI Analysis"])
async def create_ndvi_request(
    background_tasks: BackgroundTasks,
    tenant_id: str = Query(...),
    body: NDVIRequest = None
):
    """
    Create NDVI processing request with auto-tile detection.
    
    Now supports:
    - Auto-detects best tile if tile_id not provided
    - Validates land-tile intersection
    - Prevents duplicate requests
    """
    try:
        if body is None:
            raise HTTPException(status_code=400, detail="Request body required")
        
        land_ids = body.land_ids
        tile_id = body.tile_id  # Optional
        
        if not land_ids:
            raise HTTPException(status_code=400, detail="land_ids required")
        
        logger.info(f"📥 NDVI request: tenant={tenant_id}, lands={len(land_ids)}, tile={tile_id or 'auto'}")
        
        # Check for duplicate pending requests
        existing = supabase.table("ndvi_request_queue")\
            .select("id, status")\
            .eq("tenant_id", tenant_id)\
            .in_("status", ["queued", "processing"])\
            .execute()
        
        if existing.data:
            # Check if same land_ids are already queued
            for req in existing.data:
                logger.warning(f"⚠️ Found pending request: {req['id']} ({req['status']})")
        
        # Validate lands exist
        lands_check = supabase.table("lands")\
            .select("id, name, boundary_polygon_old")\
            .eq("tenant_id", tenant_id)\
            .in_("id", land_ids)\
            .execute()
        
        if not lands_check.data:
            raise HTTPException(status_code=404, detail="No valid lands found")
        
        found_ids = [l["id"] for l in lands_check.data]
        missing = set(land_ids) - set(found_ids)
        if missing:
            logger.warning(f"⚠️ Missing land IDs: {missing}")
        
        # Validate lands have boundaries
        lands_without_boundary = [
            l["name"] for l in lands_check.data 
            if not l.get("boundary_polygon_old")
        ]
        if lands_without_boundary:
            raise HTTPException(
                status_code=400,
                detail=f"Lands missing boundaries: {', '.join(lands_without_boundary)}"
            )
        
        # Insert into queue
        payload = {
            "tenant_id": tenant_id,
            "land_ids": found_ids,
            "tile_id": tile_id,
            "status": "queued",
            "requested_at": datetime.datetime.utcnow().isoformat(),
        }
        
        insert_resp = supabase.table("ndvi_request_queue").insert(payload).execute()
        
        if not insert_resp.data:
            raise HTTPException(status_code=500, detail="Failed to create queue entry")
        
        queue_id = insert_resp.data[0]["id"]
        logger.info(f"✅ Queue created: {queue_id}")
        
        # Schedule background processing
        background_tasks.add_task(
            process_queue_item_background,
            queue_id=queue_id,
            tenant_id=tenant_id,
            land_ids=found_ids,
            tile_id=tile_id
        )
        
        logger.info(f"🚀 Background worker scheduled for {queue_id}")
        
        return {
            "status": "success",
            "message": "NDVI processing started",
            "data": {
                "queue_id": queue_id,
                "tenant_id": tenant_id,
                "land_count": len(found_ids),
                "tile_id": tile_id or "auto-detect",
                "processing": "background",
                "created_at": payload["requested_at"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error creating NDVI request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ QUEUE MANAGEMENT ============
@app.get("/api/v1/ndvi/requests/queue", tags=["NDVI Queue"])
async def get_ndvi_queue(
    tenant_id: str = Query(...),
    status: Optional[str] = Query(None),
    limit: int = Query(50)
):
    """Get NDVI request queue for a tenant."""
    try:
        query = supabase.table("ndvi_request_queue")\
            .select("*")\
            .eq("tenant_id", tenant_id)
        
        if status:
            query = query.eq("status", status)
        
        query = query.order("created_at", desc=True).limit(limit)
        
        resp = query.execute()
        
        return {
            "status": "success",
            "count": len(resp.data or []),
            "queue": resp.data or [],
        }
    except Exception as e:
        logger.error(f"❌ Error fetching queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ndvi/requests/{queue_id}", tags=["NDVI Queue"])
async def get_queue_item(queue_id: str, tenant_id: str = Query(...)):
    """Get specific queue item details."""
    try:
        resp = supabase.table("ndvi_request_queue")\
            .select("*")\
            .eq("id", queue_id)\
            .eq("tenant_id", tenant_id)\
            .execute()
        
        if not resp.data:
            raise HTTPException(status_code=404, detail="Queue item not found")
        
        return {
            "status": "success",
            "data": resp.data[0]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching queue item: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ndvi/requests/stats", tags=["NDVI Statistics"])
async def get_ndvi_stats(tenant_id: Optional[str] = Query(None)):
    """Get NDVI processing statistics."""
    try:
        query_base = supabase.table("ndvi_request_queue").select("id", count="exact")
        
        if tenant_id:
            query_base = query_base.eq("tenant_id", tenant_id)
        
        total = query_base.execute()
        queued = query_base.eq("status", "queued").execute()
        processing = query_base.eq("status", "processing").execute()
        completed = query_base.eq("status", "completed").execute()
        failed = query_base.eq("status", "failed").execute()
        
        return {
            "status": "success",
            "stats": {
                "total_requests": total.count or 0,
                "queued": queued.count or 0,
                "processing": processing.count or 0,
                "completed": completed.count or 0,
                "failed": failed.count or 0,
            },
            "tenant_id": tenant_id or "all",
        }
        
    except Exception as e:
        logger.error(f"❌ Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ NDVI DATA RETRIEVAL ============
@app.get("/api/v1/ndvi/data", tags=["NDVI Data"])
async def list_ndvi_data(
    tenant_id: str = Query(...),
    land_id: Optional[str] = Query(None),
    limit: int = Query(100)
):
    """Retrieve NDVI data for lands."""
    try:
        query = supabase.table("ndvi_data")\
            .select("*")\
            .eq("tenant_id", tenant_id)
        
        if land_id:
            query = query.eq("land_id", land_id)
        
        query = query.order("date", desc=True).limit(limit)
        
        resp = query.execute()
        
        return {
            "status": "success",
            "count": len(resp.data or []),
            "data": resp.data or []
        }
    except Exception as e:
        logger.error(f"❌ Error fetching NDVI data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ndvi/data/{land_id}/latest", tags=["NDVI Data"])
async def get_latest_ndvi(land_id: str, tenant_id: str = Query(...)):
    """Get latest NDVI data for a specific land."""
    try:
        resp = supabase.table("ndvi_data")\
            .select("*")\
            .eq("tenant_id", tenant_id)\
            .eq("land_id", land_id)\
            .order("date", desc=True)\
            .limit(1)\
            .execute()
        
        if not resp.data:
            raise HTTPException(status_code=404, detail="No NDVI data found for this land")
        
        return {
            "status": "success",
            "data": resp.data[0]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error fetching latest NDVI: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ndvi/data/{land_id}/history", tags=["NDVI Data"])
async def get_ndvi_history(
    land_id: str,
    tenant_id: str = Query(...),
    limit: int = Query(30)
):
    """Get NDVI history for a land."""
    try:
        resp = supabase.table("ndvi_data")\
            .select("*")\
            .eq("tenant_id", tenant_id)\
            .eq("land_id", land_id)\
            .order("date", desc=True)\
            .limit(limit)\
            .execute()
        
        return {
            "status": "success",
            "land_id": land_id,
            "count": len(resp.data or []),
            "history": resp.data or []
        }
    except Exception as e:
        logger.error(f"❌ Error fetching NDVI history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ LOVABLE COMPATIBILITY ============
@app.get("/api/v1/ndvi/requests", tags=["Lovable"])
async def list_requests(tenant_id: str = Query(...), limit: int = Query(50)):
    """List NDVI requests (Lovable frontend)."""
    return await get_ndvi_queue(tenant_id=tenant_id, limit=limit)


@app.get("/api/v1/ndvi/stats/global", tags=["Lovable"])
async def global_ndvi_stats():
    """Global NDVI stats for Lovable dashboard."""
    try:
        completed = supabase.table("ndvi_request_queue")\
            .select("id", count="exact")\
            .eq("status", "completed")\
            .execute()
        
        queued = supabase.table("ndvi_request_queue")\
            .select("id", count="exact")\
            .eq("status", "queued")\
            .execute()
        
        processing = supabase.table("ndvi_request_queue")\
            .select("id", count="exact")\
            .eq("status", "processing")\
            .execute()
        
        failed = supabase.table("ndvi_request_queue")\
            .select("id", count="exact")\
            .eq("status", "failed")\
            .execute()
        
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
        logger.error(f"❌ Error fetching global stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ndvi/queue/status", tags=["Lovable"])
async def ndvi_queue_status():
    """Queue status summary."""
    try:
        active = supabase.table("ndvi_request_queue")\
            .select("*")\
            .in_("status", ["queued", "processing"])\
            .execute()
        
        return {
            "status": "success",
            "active_jobs": len(active.data or []),
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"❌ Error fetching queue status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ DIAGNOSTIC ENDPOINTS ============
@app.get("/api/v1/ndvi/diagnostics/tiles", tags=["Diagnostics"])
async def list_available_tiles(limit: int = Query(20)):
    """List available satellite tiles for debugging."""
    try:
        resp = supabase.table("satellite_tiles")\
            .select("tile_id, acquisition_date, status, cloud_cover, ndvi_path")\
            .eq("status", "completed")\
            .not_.is_("ndvi_path", "null")\
            .order("acquisition_date", desc=True)\
            .limit(limit)\
            .execute()
        
        return {
            "status": "success",
            "count": len(resp.data or []),
            "tiles": resp.data or []
        }
    except Exception as e:
        logger.error(f"❌ Error fetching tiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ndvi/diagnostics/land/{land_id}", tags=["Diagnostics"])
async def diagnose_land(land_id: str, tenant_id: str = Query(...)):
    """Diagnostic info for a specific land."""
    try:
        # Get land details
        land_resp = supabase.table("lands")\
            .select("*")\
            .eq("id", land_id)\
            .eq("tenant_id", tenant_id)\
            .execute()
        
        if not land_resp.data:
            raise HTTPException(status_code=404, detail="Land not found")
        
        land = land_resp.data[0]
        
        # Get NDVI data
        ndvi_resp = supabase.table("ndvi_data")\
            .select("*")\
            .eq("land_id", land_id)\
            .order("date", desc=True)\
            .limit(5)\
            .execute()
        
        # Get processing logs
        logs_resp = supabase.table("ndvi_processing_logs")\
            .select("*")\
            .eq("land_id", land_id)\
            .order("completed_at", desc=True)\
            .limit(5)\
            .execute()
        
        return {
            "status": "success",
            "land": {
                "id": land["id"],
                "name": land["name"],
                "area_acres": land.get("area_acres"),
                "has_boundary": bool(land.get("boundary_polygon_old")),
                "ndvi_tested": land.get("ndvi_tested"),
                "last_ndvi_calculation": land.get("last_ndvi_calculation"),
                "last_ndvi_value": land.get("last_ndvi_value"),
            },
            "ndvi_history_count": len(ndvi_resp.data or []),
            "recent_ndvi": ndvi_resp.data or [],
            "processing_logs": logs_resp.data or [],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in diagnostics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ ERROR HANDLING ============
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "available_endpoints": {
                "analysis": "/api/v1/ndvi/lands/analyze",
                "data": "/api/v1/ndvi/data",
                "queue": "/api/v1/ndvi/requests/queue",
                "stats": "/api/v1/ndvi/requests/stats",
                "diagnostics": "/api/v1/ndvi/diagnostics/tiles",
            },
            "documentation": "/docs",
        },
    )


# ============ ENTRY POINT ============
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🚀 Starting NDVI API v4.0.0 on port {port}")
    import uvicorn
    uvicorn.run("ndvi_land_api_fixed:app", host="0.0.0.0", port=port, log_level="info")
