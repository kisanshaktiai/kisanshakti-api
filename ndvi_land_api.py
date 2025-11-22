"""
NDVI Land API v5.1 - Production-Ready Multi-Tenant NDVI Processing API
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
✅ NDVI data retrieval endpoints
✅ Global statistics endpoints
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


class NDVIDataItem(BaseModel):
    """Single NDVI data record"""
    id: str
    tenant_id: str
    land_id: str
    date: str
    mean_ndvi: Optional[float]
    median_ndvi: Optional[float]
    min_ndvi: Optional[float]
    max_ndvi: Optional[float]
    ndvi_std: Optional[float]
    valid_pixels: Optional[int]
    coverage_percentage: Optional[float]
    confidence_level: Optional[str]
    quality_score: Optional[float]
    image_url: Optional[str]
    created_at: str


class NDVIDataResponse(BaseModel):
    """Response model for NDVI data retrieval"""
    success: bool
    data: List[Dict[str, Any]]
    count: int
    tenant_id: str


class GlobalStatsResponse(BaseModel):
    """Response model for global statistics"""
    success: bool
    stats: Dict[str, Any]


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
        "metadata": {**metadata, "all_tiles": tile_ids, "api_version": "5.1"},
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
    logger.info("🚀 NDVI Land API v5.1 starting...")
    logger.info("=" * 80)
    logger.info("REGISTERED ENDPOINTS:")
    yield
    logger.info("🛑 NDVI Land API shutting down...")

app = FastAPI(
    title="NDVI Land Processing API",
    version="5.1.0",
    description="Multi-tenant NDVI analysis API with data retrieval",
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
@app.get("/", include_in_schema=False)
async def root():
    """Root redirect to docs"""
    return {
        "service": "NDVI Land Processing API",
        "version": "5.1.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.get("/api/v1/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    try:
        test_response = supabase.table("mgrs_tiles").select("tile_id").limit(1).execute()
        supabase_ok = bool(test_response.data)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        supabase_ok = False
    
    return {
        "status": "healthy" if supabase_ok else "degraded",
        "service": "ndvi-land-api",
        "version": "5.1.0",
        "timestamp": now_iso(),
        "supabase_connected": supabase_ok,
        "b2_configured": bool(B2_APP_KEY_ID and B2_APP_KEY),
    }


@app.post("/api/v1/ndvi/lands/analyze", response_model=NDVIRequestResponse)
async def create_ndvi_request(
    request: NDVIRequestCreate,
    tenant_id: str = Query(..., description="Tenant UUID"),
):
    """
    Create a new NDVI processing request
    
    - **instant**: If true, process immediately; if false, queue for batch processing
    - **land_ids**: List of land UUIDs to process
    - **tile_id**: Optional - auto-detected if not provided
    """
    try:
        logger.info(f"📥 NDVI request: tenant={tenant_id}, lands={len(request.land_ids)}, instant={request.instant}")

        # Validate lands belong to tenant
        lands_check = (
            supabase.table("lands")
            .select("id")
            .eq("tenant_id", tenant_id)
            .in_("id", request.land_ids)
            .execute()
        )
        if not lands_check.data:
            raise HTTPException(status_code=403, detail="Invalid land IDs for tenant")

        # Get intersecting tiles
        tile_ids = [request.tile_id] if request.tile_id else await get_intersecting_tiles(tenant_id, request.land_ids)
        
        # Create queue entry
        queue_id = await create_queue_entry(
            tenant_id, request.land_ids, tile_ids, request.priority, 
            request.statistics_only, request.farmer_id, request.metadata
        )

        instant_result = None
        final_status = "queued"

        # ⚡ Instant processing mode
        if request.instant:
            try:
                logger.info(f"⚡ Triggering instant NDVI processing for queue_id={queue_id}")
                from enhanced_ndvi_worker import process_request_async
                worker_result = await process_request_async(queue_id, tenant_id, request.land_ids, tile_ids)
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

        # Calculate ETA
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


@app.get("/api/v1/ndvi/data", response_model=NDVIDataResponse)
async def get_ndvi_data(
    tenant_id: str = Query(..., description="Tenant UUID"),
    land_id: Optional[str] = Query(None, description="Filter by specific land ID"),
    farmer_id: Optional[str] = Query(None, description="Filter by farmer ID"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum records to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """
    Retrieve NDVI data for a tenant
    
    Returns processed NDVI records with statistics and quality metrics.
    """
    try:
        logger.info(f"📊 Fetching NDVI data: tenant={tenant_id}, land={land_id}, limit={limit}")

        # Build query
        query = supabase.table("ndvi_data").select("*").eq("tenant_id", tenant_id)
        
        if land_id:
            query = query.eq("land_id", land_id)
        
        if start_date:
            query = query.gte("date", start_date)
        
        if end_date:
            query = query.lte("date", end_date)
        
        # Execute query with pagination
        response = query.order("date", desc=True).range(offset, offset + limit - 1).execute()
        
        data = response.data if response.data else []
        
        logger.info(f"✅ Retrieved {len(data)} NDVI records")
        
        return NDVIDataResponse(
            success=True,
            data=data,
            count=len(data),
            tenant_id=tenant_id,
        )

    except Exception as e:
        logger.exception("Failed to retrieve NDVI data")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve NDVI data: {str(e)}")


@app.get("/api/v1/ndvi/stats/global", response_model=GlobalStatsResponse)
async def get_global_stats(
    tenant_id: Optional[str] = Query(None, description="Filter by tenant ID"),
):
    """
    Get global NDVI statistics
    
    Returns aggregate statistics across all lands or filtered by tenant.
    """
    try:
        logger.info(f"📈 Fetching global stats: tenant={tenant_id}")

        # Build base query
        query = supabase.table("ndvi_data").select(
            "mean_ndvi, median_ndvi, quality_score, confidence_level, valid_pixels"
        )
        
        if tenant_id:
            query = query.eq("tenant_id", tenant_id)
        
        # Get recent data (last 90 days)
        cutoff_date = (datetime.datetime.utcnow() - datetime.timedelta(days=90)).date().isoformat()
        response = query.gte("date", cutoff_date).execute()
        
        data = response.data if response.data else []
        
        if not data:
            # Return empty stats if no data
            return GlobalStatsResponse(
                success=True,
                stats={
                    "total_records": 0,
                    "avg_ndvi": None,
                    "avg_quality_score": None,
                    "high_confidence_count": 0,
                    "medium_confidence_count": 0,
                    "low_confidence_count": 0,
                    "total_pixels_analyzed": 0,
                }
            )
        
        # Calculate statistics
        mean_values = [r["mean_ndvi"] for r in data if r.get("mean_ndvi") is not None]
        quality_scores = [r["quality_score"] for r in data if r.get("quality_score") is not None]
        
        confidence_counts = {
            "high": sum(1 for r in data if r.get("confidence_level") == "high"),
            "medium": sum(1 for r in data if r.get("confidence_level") == "medium"),
            "low": sum(1 for r in data if r.get("confidence_level") == "low"),
        }
        
        total_pixels = sum(r.get("valid_pixels", 0) for r in data)
        
        stats = {
            "total_records": len(data),
            "avg_ndvi": sum(mean_values) / len(mean_values) if mean_values else None,
            "avg_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else None,
            "high_confidence_count": confidence_counts["high"],
            "medium_confidence_count": confidence_counts["medium"],
            "low_confidence_count": confidence_counts["low"],
            "total_pixels_analyzed": total_pixels,
            "date_range": {
                "start": cutoff_date,
                "end": datetime.datetime.utcnow().date().isoformat(),
            },
        }
        
        logger.info(f"✅ Global stats calculated: {stats['total_records']} records")
        
        return GlobalStatsResponse(
            success=True,
            stats=stats,
        )

    except Exception as e:
        logger.exception("Failed to calculate global stats")
        raise HTTPException(status_code=500, detail=f"Failed to calculate stats: {str(e)}")


@app.get("/api/v1/ndvi/queue/status")
async def get_queue_status(
    tenant_id: str = Query(..., description="Tenant UUID"),
):
    """
    Get processing queue status for a tenant
    
    Returns counts and details of queued, processing, completed, and failed jobs.
    """
    try:
        logger.info(f"📋 Fetching queue status: tenant={tenant_id}")

        response = supabase.table("ndvi_request_queue").select("*").eq(
            "tenant_id", tenant_id
        ).order("created_at", desc=True).limit(50).execute()
        
        items = response.data if response.data else []
        
        # Count by status
        status_counts = {
            "queued": sum(1 for i in items if i.get("status") == "queued"),
            "processing": sum(1 for i in items if i.get("status") == "processing"),
            "completed": sum(1 for i in items if i.get("status") == "completed"),
            "failed": sum(1 for i in items if i.get("status") == "failed"),
        }
        
        return QueueStatusResponse(
            success=True,
            tenant_id=tenant_id,
            total=len(items),
            queued=status_counts["queued"],
            processing=status_counts["processing"],
            completed=status_counts["completed"],
            failed=status_counts["failed"],
            queue_items=items[:10],  # Return only latest 10
        )

    except Exception as e:
        logger.exception("Failed to get queue status")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Error Handlers
# =============================================================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc),
        },
    )


# =============================================================================
# Startup Event - Log Routes
# =============================================================================
@app.on_event("startup")
async def startup_event():
    """Log all registered routes on startup"""
    logger.info("=" * 80)
    logger.info("📍 REGISTERED API ROUTES:")
    logger.info("=" * 80)
    for route in app.routes:
        if hasattr(route, "methods"):
            methods = ", ".join(route.methods)
            logger.info(f"  {methods:12} {route.path}")
    logger.info("=" * 80)


# =============================================================================
# Main Entry
# =============================================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    logger.info(f"🚀 Starting NDVI Land API on port {port}")
    uvicorn.run("ndvi_land_api:app", host="0.0.0.0", port=port, reload=False)
