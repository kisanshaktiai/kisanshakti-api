"""
NDVI Land API (v3.6)
--------------------
FastAPI backend for managing NDVI processing requests and returning results.
Designed for multi-tenant SaaS platforms using Supabase and shared B2 satellite tiles.

Updates from v3.5:
✅ Removed unnecessary fields: cloud_coverage, date_from, date_to
✅ Simplified /requests payload for preprocessed tiles
✅ Fully compatible with v3.7-secure NDVI worker
"""

import os
import datetime
import logging
from typing import Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
import ndvi_land_worker as worker

# === CONFIG ===
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ Missing Supabase configuration environment variables.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# === FASTAPI APP SETUP ===
app = FastAPI(
    title="NDVI Land Processor API",
    version="3.6.0",
    description="Handles NDVI requests, tile-based processing, and per-land result storage."
)

# Enable full cross-origin access (important for Lovable frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your domain list for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ndvi-api")

# ---------------- Health ----------------
@app.get("/health")
def health():
    """Basic health check and Supabase connectivity test."""
    try:
        supabase.table("lands").select("id").limit(1).execute()
        return {
            "service": "NDVI Land Processor API",
            "status": "healthy",
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return {
            "service": "NDVI Land Processor API",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }

# ---------------- Create NDVI Request ----------------
@app.post("/requests")
def create_request(payload: dict):
    """
    Queue a new NDVI request for a tenant and its lands.
    Simplified for preprocessed B2 + MPC workflow — no cloud/date filtering.
    """
    try:
        tenant_id = payload.get("tenant_id")
        land_ids = payload.get("land_ids", [])

        if not tenant_id:
            raise HTTPException(status_code=400, detail="tenant_id is required")

        # If land_ids not provided, fetch all active lands for tenant
        if not land_ids:
            lands = (
                supabase.table("lands")
                .select("id")
                .eq("tenant_id", tenant_id)
                .eq("is_active", True)
                .execute()
                .data
            )
            if not lands:
                raise HTTPException(status_code=404, detail="No lands found for tenant")
            land_ids = [l["id"] for l in lands]

        # Get latest available satellite tile
        tile_res = (
            supabase.table("satellite_tiles")
            .select("tile_id, acquisition_date")
            .order("acquisition_date", desc=True)
            .limit(1)
            .execute()
        )
        if not tile_res.data:
            raise HTTPException(status_code=404, detail="No satellite tiles found")
        tile_id = tile_res.data[0]["tile_id"]
        acquisition_date = tile_res.data[0].get("acquisition_date")

        # Insert NDVI job into queue
        result = (
            supabase.table("ndvi_request_queue")
            .insert({
                "tenant_id": tenant_id,
                "land_ids": land_ids,
                "tile_id": tile_id,
                "status": "queued",
                "batch_size": len(land_ids),
                "created_at": datetime.datetime.utcnow().isoformat(),
            })
            .execute()
        )

        req_id = result.data[0]["id"]
        logger.info(f"📋 NDVI request queued: {req_id} | Tenant: {tenant_id} | Tile: {tile_id}")

        return {
            "request_id": req_id,
            "tenant_id": tenant_id,
            "tile_id": tile_id,
            "acquisition_date": acquisition_date,
            "status": "queued",
            "land_count": len(land_ids),
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"❌ Request creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- Run Worker ----------------
@app.post("/run")
def run_worker(background: BackgroundTasks, limit: int = 10):
    """Trigger background NDVI worker."""
    try:
        background.add_task(worker.main, limit)
        logger.info(f"🚀 NDVI worker started in background (limit={limit})")
        return {
            "status": "started",
            "limit": limit,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"❌ Failed to start worker: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- Get NDVI Data ----------------
@app.get("/lands/{land_id}/ndvi")
def get_land_ndvi(land_id: str, tenant_id: Optional[str] = None, limit: int = 30):
    """Fetch NDVI data for a specific land (optionally tenant-scoped)."""
    try:
        q = (
            supabase.table("ndvi_data")
            .select("*")
            .eq("land_id", land_id)
            .order("date", desc=True)
        )
        if tenant_id:
            q = q.eq("tenant_id", tenant_id)

        res = q.limit(limit).execute()
        return {
            "land_id": land_id,
            "count": len(res.data),
            "data": res.data,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"❌ Failed to fetch NDVI data for land {land_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- Queue Stats ----------------
@app.get("/queue")
def get_queue_status(tenant_id: Optional[str] = None):
    """Return summary of NDVI processing queue."""
    try:
        query = (
            supabase.table("ndvi_request_queue")
            .select("*")
            .order("created_at", desc=True)
        )
        if tenant_id:
            query = query.eq("tenant_id", tenant_id)
        res = query.limit(25).execute()
        return {
            "count": len(res.data),
            "requests": res.data,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"❌ Queue fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- Stats ----------------
@app.get("/stats")
def stats():
    """Get overall NDVI job statistics (total requests, completed, failed)."""
    try:
        queued = supabase.table("ndvi_request_queue").select("id", count="exact").eq("status", "queued").execute()
        processing = supabase.table("ndvi_request_queue").select("id", count="exact").eq("status", "processing").execute()
        completed = supabase.table("ndvi_request_queue").select("id", count="exact").eq("status", "completed").execute()
        failed = supabase.table("ndvi_request_queue").select("id", count="exact").eq("status", "failed").execute()

        return {
            "queued": queued.count or 0,
            "processing": processing.count or 0,
            "completed": completed.count or 0,
            "failed": failed.count or 0,
            "total_requests": (queued.count or 0)
            + (processing.count or 0)
            + (completed.count or 0)
            + (failed.count or 0),
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"❌ Stats endpoint failed: {e}")
        # Always return JSON with CORS-safe response
        return {
            "error": str(e),
            "queued": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
            "total_requests": 0,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }
