"""
NDVI Land API v5.0 - Production-Ready Multi-Tenant NDVI Processing API
=======================================================================
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
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ndvi-api")

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
B2_APP_KEY_ID = os.getenv("B2_APP_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME", "kisanshakti-ndvi-tiles")
RENDER_SERVICE_URL = os.getenv("RENDER_EXTERNAL_URL", "https://ndvi-land-api.onrender.com")

# Validate critical environment variables
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
    land_ids: List[str] = Field(..., min_items=1, max_items=1000, description="List of land UUIDs to process")
    tile_id: Optional[str] = Field(None, description="Optional specific tile ID (auto-detected if not provided)")
    instant: bool = Field(False, description="Process immediately (true) or queue for batch (false)")
    statistics_only: bool = Field(False, description="Only compute statistics, skip thumbnail generation")
    priority: int = Field(5, ge=1, le=10, description="Priority (1=highest, 10=lowest)")
    farmer_id: Optional[str] = Field(None, description="Optional farmer ID for filtering")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")

    @validator('land_ids')
    def validate_land_ids(cls, v):
        if not v or len(v) == 0:
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
# Helper Functions
# =============================================================================
def now_iso() -> str:
    """Return current UTC timestamp in ISO format"""
    return datetime.datetime.utcnow().isoformat()

async def get_intersecting_tiles(tenant_id: str, land_ids: List[str]) -> List[str]:
    """
    Find all MGRS tiles that intersect with given land parcels.
    Uses PostGIS spatial queries for efficient intersection detection.
    """
    try:
        # Fetch land boundaries
        lands_response = supabase.table("lands").select(
            "id, boundary_polygon_old, boundary, center_lat, center_lon"
        ).eq("tenant_id", tenant_id).in_("id", land_ids).execute()
        
        if not lands_response.data:
            raise ValueError(f"No lands found for tenant {tenant_id}")
        
        # Use RPC function if available, otherwise fallback to manual intersection
        try:
            tiles_response = supabase.rpc(
                "get_intersecting_tiles_for_lands",
                {"p_tenant_id": tenant_id, "p_land_ids": land_ids}
            ).execute()
            
            if tiles_response.data:
                tile_ids = [t["tile_id"] for t in tiles_response.data]
                logger.info(f"🗺️  Found {len(tile_ids)} intersecting tiles via RPC")
                return list(set(tile_ids))  # Remove duplicates
        except Exception as rpc_error:
            logger.warning(f"⚠️  RPC function not available, using fallback: {rpc_error}")
        
        # Fallback: Use center point to find containing tile
        tile_ids = set()
        for land in lands_response.data:
            if land.get("center_lat") and land.get("center_lon"):
                # Find tile containing this point
                tile_response = supabase.table("mgrs_tiles").select("tile_id").filter(
                    "geometry", "cs", f"POINT({land['center_lon']} {land['center_lat']})"
                ).limit(1).execute()
                
                if tile_response.data:
                    tile_ids.add(tile_response.data[0]["tile_id"])
        
        if not tile_ids:
            # Ultimate fallback: use default India tile
            logger.warning("⚠️  No tiles found via intersection, using default tile 43RGN")
            tile_ids.add("43RGN")
        
        logger.info(f"🗺️  Found {len(tile_ids)} intersecting tiles (fallback method)")
        return list(tile_ids)
        
    except Exception as e:
        logger.error(f"❌ Error finding intersecting tiles: {e}")
        # Return a default tile as last resort
        return ["43RGN"]

async def create_queue_entry(
    tenant_id: str,
    land_ids: List[str],
    tile_ids: List[str],
    priority: int,
    statistics_only: bool,
    farmer_id: Optional[str],
    metadata: Dict[str, Any]
) -> str:
    """Create an entry in ndvi_request_queue table"""
    
    # Use primary tile for tracking (first one)
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
        "metadata": {
            **metadata,
            "all_tiles": tile_ids,
            "api_version": "5.0",
            "source": "tenant_portal"
        }
    }
    
    response = supabase.table("ndvi_request_queue").insert(queue_entry).execute()
    
    if not response.data:
        raise HTTPException(status_code=500, detail="Failed to create queue entry")
    
    queue_id = response.data[0]["id"]
    logger.info(f"✅ Queue entry created: {queue_id} | lands={len(land_ids)} | tiles={len(tile_ids)}")
    
    return queue_id

# =============================================================================
# FastAPI Application
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    logger.info("🚀 NDVI Land API v5.0 starting up...")
    logger.info(f"📍 Service URL: {RENDER_SERVICE_URL}")
    logger.info(f"🗄️  Supabase: {SUPABASE_URL}")
    logger.info(f"☁️  B2 Bucket: {B2_BUCKET_NAME}")
    yield
    logger.info("🛑 NDVI Land API shutting down...")

app = FastAPI(
    title="NDVI Land Processing API",
    description="Multi-tenant NDVI analysis service for agricultural land monitoring",
    version="5.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# =============================================================================
# API Endpoints
# =============================================================================
@app.get("/", response_model=Dict[str, Any], tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "service": "NDVI Land Processing API",
        "version": "5.0.0",
        "status": "operational",
        "documentation": f"{RENDER_SERVICE_URL}/docs",
        "timestamp": now_iso()
    }

@app.get("/api/v1/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """Comprehensive health check"""
    supabase_ok = False
    try:
        # Test Supabase connection
        test_response = supabase.table("mgrs_tiles").select("tile_id").limit(1).execute()
        supabase_ok = test_response.data is not None
    except Exception as e:
        logger.error(f"❌ Supabase health check failed: {e}")
    
    b2_configured = bool(B2_APP_KEY_ID and B2_APP_KEY)
    
    return {
        "status": "healthy" if supabase_ok else "degraded",
        "service": "ndvi-land-api",
        "version": "5.0.0",
        "timestamp": now_iso(),
        "supabase_connected": supabase_ok,
        "b2_configured": b2_configured
    }

@app.post("/api/v1/ndvi/lands/analyze", response_model=NDVIRequestResponse, tags=["NDVI Processing"])
async def create_ndvi_request(
    request: NDVIRequestCreate,
    tenant_id: str = Query(..., description="Tenant UUID"),
    background_tasks: BackgroundTasks = None
):
    """
    Create NDVI processing request for specified lands.
    
    **Modes:**
    - `instant=false` (default): Queues request for batch processing by cron worker
    - `instant=true`: Processes immediately and returns results (may timeout for large batches)
    
    **Process:**
    1. Validates tenant and land ownership
    2. Detects intersecting MGRS tiles using PostGIS
    3. Creates queue entry in ndvi_request_queue
    4. If instant=true, triggers worker synchronously
    5. Returns queue ID for status tracking
    """
    
    try:
        logger.info(f"📥 NDVI request: tenant={tenant_id} | lands={len(request.land_ids)} | instant={request.instant}")
        
        # Validate lands belong to tenant
        lands_check = supabase.table("lands").select("id").eq(
            "tenant_id", tenant_id
        ).in_("id", request.land_ids).execute()
        
        if not lands_check.data or len(lands_check.data) != len(request.land_ids):
            raise HTTPException(
                status_code=403,
                detail=f"Some lands not found or not owned by tenant {tenant_id}"
            )
        
        # Auto-detect intersecting tiles if not specified
        tile_ids = [request.tile_id] if request.tile_id else await get_intersecting_tiles(tenant_id, request.land_ids)
        
        # Create queue entry
        queue_id = await create_queue_entry(
            tenant_id=tenant_id,
            land_ids=request.land_ids,
            tile_ids=tile_ids,
            priority=request.priority,
            statistics_only=request.statistics_only,
            farmer_id=request.farmer_id,
            metadata=request.metadata
        )
        
        instant_result = None
        final_status = "queued"
        
        # Instant processing mode
        if request.instant:
            try:
                logger.info(f"⚡ Triggering instant processing for queue_id={queue_id}")
                
                # Import worker and process synchronously
                from ndvi_land_worker import process_request_async
                
                # Update status to processing
                supabase.table("ndvi_request_queue").update({
                    "status": "processing",
                    "started_at": now_iso()
                }).eq("id", queue_id).execute()
                
                # Execute processing
                worker_result = process_request_async(
                    queue_id=queue_id,
                    tenant_id=tenant_id,
                    land_ids=request.land_ids,
                    tile_ids=tile_ids
                )
                
                instant_result = worker_result
                final_status = "completed" if worker_result.get("processed_count", 0) > 0 else "failed"
                
                # Update queue with final status
                supabase.table("ndvi_request_queue").update({
                    "status": final_status,
                    "processed_count": worker_result.get("processed_count", 0),
                    "completed_at": now_iso(),
                    "last_error": worker_result.get("error")
                }).eq("id", queue_id).execute()
                
                logger.info(f"✅ Instant processing completed: {final_status}")
                
            except Exception as e:
                logger.exception(f"❌ Instant processing failed for queue_id={queue_id}")
                
                # Update queue with failure
                supabase.table("ndvi_request_queue").update({
                    "status": "failed",
                    "last_error": str(e)[:500],
                    "completed_at": now_iso()
                }).eq("id", queue_id).execute()
                
                final_status = "failed"
                instant_result = {"error": str(e)}
        
        # Calculate estimated completion for queued requests
        estimated_completion = None
        if not request.instant:
            # Rough estimate: 2 seconds per land
            eta_seconds = len(request.land_ids) * 2
            eta_time = datetime.datetime.utcnow() + datetime.timedelta(seconds=eta_seconds)
            estimated_completion = eta_time.isoformat()
        
        return NDVIRequestResponse(
            success=True,
            message=f"NDVI request {'processed' if request.instant else 'queued'} successfully",
            queue_id=queue_id,
            tenant_id=tenant_id,
            status=final_status,
            land_count=len(request.land_ids),
            estimated_completion=estimated_completion,
            instant_result=instant_result
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Failed to create NDVI request")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/ndvi/requests/queue", response_model=QueueStatusResponse, tags=["Queue Management"])
async def get_queue_status(
    tenant_id: str = Query(..., description="Tenant UUID"),
    limit: int = Query(50, ge=1, le=500, description="Maximum items to return")
):
    """
    Get NDVI processing queue status for a tenant.
    Returns recent queue entries with status breakdown.
    """
    try:
        # Fetch queue entries
        response = supabase.table("ndvi_request_queue").select("*").eq(
            "tenant_id", tenant_id
        ).order("created_at", desc=True).limit(limit).execute()
        
        queue_items = response.data or []
        
        # Calculate status counts
        queued = sum(1 for item in queue_items if item.get("status") == "queued")
        processing = sum(1 for item in queue_items if item.get("status") == "processing")
        completed = sum(1 for item in queue_items if item.get("status") == "completed")
        failed = sum(1 for item in queue_items if item.get("status") == "failed")
        
        logger.info(f"📊 Queue status: tenant={tenant_id} | total={len(queue_items)} | queued={queued}")
        
        return QueueStatusResponse(
            success=True,
            tenant_id=tenant_id,
            total=len(queue_items),
            queued=queued,
            processing=processing,
            completed=completed,
            failed=failed,
            queue_items=queue_items
        )
        
    except Exception as e:
        logger.exception("❌ Failed to fetch queue status")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/ndvi/data", tags=["NDVI Data"])
async def get_ndvi_data(
    tenant_id: str = Query(..., description="Tenant UUID"),
    land_id: Optional[str] = Query(None, description="Optional land ID filter"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum items to return")
):
    """
    Retrieve processed NDVI data for a tenant.
    Returns recent NDVI calculations with statistics and thumbnail URLs.
    """
    try:
        query = supabase.table("ndvi_data").select("*").eq("tenant_id", tenant_id)
        
        if land_id:
            query = query.eq("land_id", land_id)
        
        response = query.order("date", desc=True).limit(limit).execute()
        
        data_items = response.data or []
        
        logger.info(f"📊 NDVI data: tenant={tenant_id} | land_id={land_id} | count={len(data_items)}")
        
        return {
            "success": True,
            "tenant_id": tenant_id,
            "land_id": land_id,
            "count": len(data_items),
            "data": data_items
        }
        
    except Exception as e:
        logger.exception("❌ Failed to fetch NDVI data")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/ndvi/stats/global", tags=["Statistics"])
async def get_global_stats():
    """
    Get global NDVI processing statistics across all tenants.
    Useful for monitoring system health and usage.
    """
    try:
        # Count by status
        completed_resp = supabase.table("ndvi_request_queue").select("id", count="exact").eq("status", "completed").execute()
        queued_resp = supabase.table("ndvi_request_queue").select("id", count="exact").eq("status", "queued").execute()
        processing_resp = supabase.table("ndvi_request_queue").select("id", count="exact").eq("status", "processing").execute()
        failed_resp = supabase.table("ndvi_request_queue").select("id", count="exact").eq("status", "failed").execute()
        
        stats = {
            "completed": completed_resp.count or 0,
            "queued": queued_resp.count or 0,
            "processing": processing_resp.count or 0,
            "failed": failed_resp.count or 0,
            "total": (completed_resp.count or 0) + (queued_resp.count or 0) + (processing_resp.count or 0) + (failed_resp.count or 0)
        }
        
        logger.info(f"📈 Global stats: {stats}")
        
        return {
            "success": True,
            "stats": stats,
            "timestamp": now_iso()
        }
        
    except Exception as e:
        logger.exception("❌ Failed to fetch global stats")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# Error Handlers
# =============================================================================
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "path": str(request.url.path),
            "documentation": f"{RENDER_SERVICE_URL}/docs"
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle 500 errors"""
    logger.exception("Internal server error")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": now_iso()
        }
    )

# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info(f"🚀 Starting NDVI Land API v5.0 on {host}:{port}")
    
    uvicorn.run(
        "ndvi_land_api:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True,
        reload=False  # Set to True for development
    )
