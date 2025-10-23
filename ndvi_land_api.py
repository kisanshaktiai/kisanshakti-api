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
        "timestamp": datetime.datetime.utcnow
