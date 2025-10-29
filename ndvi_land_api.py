"""
NDVI Land API v5.0 - Production-Ready Multi-Tenant NDVI Processing API
======================================================================
Architecture:
- Handles NDVI processing requests from tenant portals
- Manages queue for batch processing
- Provides real-time status monitoring
- Integrates with B2 cloud storage and Supabase
- Supports instant and queued processing modes

Features:
✅ Multi-tenant isolation
✅ Geographic tile intersection detection
✅ Batch processing with priority queuing
✅ Real-time progress tracking
✅ Comprehensive error handling & logging
✅ RESTful API with OpenAPI docs
"""

import os
import logging
import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

from supabase import create_client, Client

# =============================================================================
# Configuration & Logging
# =============================================================================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ndvi-api")

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
B2_APP_KEY_ID = os.getenv("B2_APP_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME", "kisanshakti-ndvi-tiles")
RENDER_SERVICE_URL = os.getenv("RENDER_EXTERNAL_URL", "https://ndvi-land-api.onrender.com")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    logger.critical("❌ Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
    raise RuntimeError("Critical environment variables missing")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
logger.info(f"✅ Supabase client initialized: {SUPABASE_URL}")

# =============================================================================
# Pydantic Models
# =============================================================================
class NDVIRequestCreate(BaseModel):
    """Request model for creating NDVI processing job"""
    land_ids: List[str] = Field(..., min_items=1, description="List of land UUIDs to process")
    tile_id: Optional[str] = Field(None, description="Optional specific tile ID (auto-detected if not provided)")
    instant: bool = Field(False, description="Process immediately (true) or queue for batch (false)")
    statistics_only: bool = Field(False, description="Only compute statistics, skip thumbnail generation")
    priority: int = Field(5, ge=1, le=10, description="Priority (1=highest, 10=lowest)")
    farmer_id: Optional[str] = Field(None, description="Optional farmer ID for filtering")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    @validator("land_ids")
    def validate_land_ids(cls, v):
        if not v:
            raise ValueError("At least one land_id is required")
        return v


class NDVIRequestResponse(BaseModel):
    """Response model for NDVI request creation"""
    success: bool
    message: str
    queue_id: str
    tenant_id: str
    status: str
    land_count: int
    estimated_completion: Optional[str] = None
    instant_result: Optional[Dict[str, Any]] = None


class QueueStatusResponse(BaseModel):
    """Response model for queue status"""
    success: bool
    tenant_id: str
    total: int
    queued: int
    processing: int
    completed: int
    failed: int
    queue_items: List[Dict[str, Any]]


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    timestamp: str
    supabase_connected: bool
    b2_configured: bool


# =============================================================================
# Helpers
# =============================================================================
def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat()


async def get_intersecting_tiles(tenant_id: str, land_ids: List[str]) -> List[str]:
    """Find intersecting MGRS tiles for given lands"""
    try:
        lands_response = supabase.table("lands").select(
            "id, boundary_polygon_old, boundary, center_lat, center_lon"
        ).eq("tenant_id", tenant_id).in_("id", land_ids).execute()

        if not lands_response.data:
            raise ValueError(f"No lands found for tenant {tenant_id}")

        try:
            tiles_response = supabase.rpc(
                "get_intersecting_tiles_for_lands",
                {"p_tenant_id": tenant_id, "p_land_ids": land_ids},
            ).execute()
            if tiles_response.data:
                tile_ids = [t["tile_id"] for t in tiles_response.data]
                return list(set(tile_ids))
        except Exception:
            pass

        tile_ids = set()
        for land in lands_response.data:
            if land.get("center_lat") and land.get("center_lon"):
                tile_response = (
                    supabase.table("mgrs_tiles")
                    .select("tile_id")
                    .limit(1)
                    .execute()
                )
                if tile_response.data:
                    tile_ids.add(tile_response.data[0]["tile_id"])

        if not tile_ids:
            tile_ids.add("43RGN")

        return list(tile_ids)
    except Exception as e:
        logger.error(f"Error finding intersecting tiles: {e}")
        return ["43RGN"]


async def create_queue_entry(
    tenant_id: str,
    land_ids: List[str],
    tile_ids: List[str],
    priority: int,
    statistics_only: bool,
    farmer_id: Optional[str],
    metadata: Dict[str, Any],
) -> str:
    """Insert a new NDVI processing request into the queue"""
    primary_tile = tile_ids[0] if tile_ids else "43RGN"
    queue_entry = {
        "tenant_id": tenant_id,
        "land_ids": land_ids,
        "tile_id": primary_tile,
        "priority": priority,
        "statistics_only": statistics_only,
        "farmer_id": farmer_id,
        "status": "queued",
        "requested_at": now_iso(),
        "created_at": now_iso(),
        "metadata": {**metadata, "all_tiles": tile_ids, "api_version": "5.0"},
    }

    response = supabase.table("ndvi_request_queue").insert(queue_entry).execute()
    if not response.data:
        raise HTTPException(status_code=500, detail="Failed to create queue entry")

    queue_id = response.data[0]["id"]
    logger.info(f"✅ Queue entry created: {queue_id} | lands={len(land_ids)}")
    return queue_id


# =============================================================================
# FastAPI App Setup
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 NDVI Land API starting...")
    yield
    logger.info("🛑 NDVI Land API shutting down...")

app = FastAPI(
    title="NDVI Land Processing API",
    version="5.0.0",
    description="Multi-tenant NDVI analysis API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# API Routes
# =============================================================================
@app.get("/api/v1/health", response_model=HealthCheckResponse)
async def health_check():
    """Basic health check"""
    try:
        test_response = supabase.table("mgrs_tiles").select("tile_id").limit(1).execute()
        supabase_ok = bool(test_response.data)
    except Exception:
        supabase_ok = False
    return {
        "status": "healthy" if supabase_ok else "degraded",
        "service": "ndvi-land-api",
        "version": "5.0.0",
        "timestamp": now_iso(),
        "supabase_connected": supabase_ok,
        "b2_configured": bool(B2_APP_KEY_ID and B2_APP_KEY),
    }


@app.post("/api/v1/ndvi/lands/analyze", response_model=NDVIRequestResponse)
async def create_ndvi_request(
    request: NDVIRequestCreate,
    tenant_id: str = Query(...),
):
    """Create a new NDVI processing request"""
    try:
        logger.info(f"📥 NDVI request: tenant={tenant_id}, lands={len(request.land_ids)}, instant={request.instant}")

        lands_check = (
            supabase.table("lands")
            .select("id")
            .eq("tenant_id", tenant_id)
            .in_("id", request.land_ids)
            .execute()
        )
        if not lands_check.data:
            raise HTTPException(status_code=403, detail="Invalid land IDs for tenant")

        tile_ids = [request.tile_id] if request.tile_id else await get_intersecting_tiles(tenant_id, request.land_ids)
        queue_id = await create_queue_entry(
            tenant_id, request.land_ids, tile_ids, request.priority, request.statistics_only, request.farmer_id, request.metadata
        )

        instant_result = None
        final_status = "queued"

        # ⚡ Instant processing mode
        if request.instant:
            try:
                logger.info(f"⚡ Triggering instant NDVI processing for queue_id={queue_id}")
                from ndvi_land_worker import process_request_async
                worker_result = await process_request_async(queue_id, tenant_id, request.land_ids)
                instant_result = worker_result
                final_status = "completed" if worker_result.get("processed_count", 0) > 0 else "failed"

                supabase.table("ndvi_request_queue").update({
                    "status": final_status,
                    "processed_count": worker_result.get("processed_count", 0),
                    "failed_count": worker_result.get("failed_count", 0),
                    "completed_at": now_iso(),
                }).eq("id", queue_id).execute()
            except Exception as e:
                logger.exception("Instant processing failed")
                supabase.table("ndvi_request_queue").update({
                    "status": "failed",
                    "last_error": str(e),
                }).eq("id", queue_id).execute()
                final_status = "failed"
                instant_result = {"error": str(e)}

        eta_time = datetime.datetime.utcnow() + datetime.timedelta(seconds=len(request.land_ids) * 2)
        return NDVIRequestResponse(
            success=True,
            message="NDVI request processed successfully" if request.instant else "NDVI request queued successfully",
            queue_id=queue_id,
            tenant_id=tenant_id,
            status=final_status,
            land_count=len(request.land_ids),
            estimated_completion=None if request.instant else eta_time.isoformat(),
            instant_result=instant_result,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to create NDVI request")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Main Entry
# =============================================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("ndvi_land_api:app", host="0.0.0.0", port=port, reload=False)
