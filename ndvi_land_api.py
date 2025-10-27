"""
NDVI Land Processing API
========================
REST API endpoint for processing NDVI data for agricultural lands.
Handles multi-tenant authentication, validation, and async task queuing.

Author: KisanShakti Team
Version: 1.0.0
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import jwt
from supabase import create_client, Client
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ndvi_land_worker import NDVILandWorker

# ============================================================================
# Configuration & Logging
# ============================================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
JWT_SECRET = os.getenv("JWT_SECRET", "your-jwt-secret-change-in-production")
MAX_WORKERS = int(os.getenv("LAND_WORKER_CONCURRENCY", "8"))

# Initialize FastAPI app
app = FastAPI(
    title="NDVI Land Processing API",
    description="Multi-tenant NDVI processing service for agricultural lands",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for background tasks
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# Security
security = HTTPBearer()

# ============================================================================
# Request/Response Models
# ============================================================================

class NDVIProcessRequest(BaseModel):
    """Request model for NDVI processing."""
    land_id: str = Field(..., description="UUID of the land to process")
    lookback_days: int = Field(
        default=15,
        ge=1,
        le=90,
        description="Number of days to look back for satellite imagery"
    )
    max_cloud_cover: float = Field(
        default=20.0,
        ge=0.0,
        le=100.0,
        description="Maximum acceptable cloud cover percentage"
    )
    force_refresh: bool = Field(
        default=False,
        description="Force re-processing even if recent data exists"
    )

    @validator('land_id')
    def validate_uuid(cls, v):
        try:
            UUID(v)
        except ValueError:
            raise ValueError('land_id must be a valid UUID')
        return v


class NDVIStatistics(BaseModel):
    """NDVI statistics for a land."""
    min_ndvi: Optional[float] = Field(None, ge=-1.0, le=1.0)
    max_ndvi: Optional[float] = Field(None, ge=-1.0, le=1.0)
    mean_ndvi: Optional[float] = Field(None, ge=-1.0, le=1.0)
    median_ndvi: Optional[float] = Field(None, ge=-1.0, le=1.0)
    std_ndvi: Optional[float] = Field(None, ge=0.0)
    valid_pixels: Optional[int] = Field(None, ge=0)
    total_pixels: Optional[int] = Field(None, ge=0)
    coverage_percentage: Optional[float] = Field(None, ge=0.0, le=100.0)


class MicroTileInfo(BaseModel):
    """Information about a generated micro-tile."""
    tile_id: str
    acquisition_date: str
    thumbnail_url: Optional[str]
    overlay_url: Optional[str]
    bbox: Dict[str, Any]
    ndvi_stats: NDVIStatistics


class NDVIProcessResponse(BaseModel):
    """Response model for NDVI processing."""
    success: bool
    message: str
    land_id: str
    tenant_id: str
    processed_at: str
    overall_stats: Optional[NDVIStatistics]
    micro_tiles: list[MicroTileInfo]
    overlay_geojson: Optional[Dict[str, Any]]
    processing_time_seconds: Optional[float]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str


# ============================================================================
# Dependencies
# ============================================================================

def get_supabase_client() -> Client:
    """Get Supabase client instance."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supabase configuration missing"
        )
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """
    Verify JWT token and extract tenant/user information.
    
    Expected JWT payload:
    {
        "tenant_id": "uuid",
        "user_id": "uuid",
        "role": "admin|user",
        "exp": timestamp
    }
    """
    token = credentials.credentials
    
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        
        if "tenant_id" not in payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing tenant_id"
            )
        
        # Check expiration
        if datetime.utcnow().timestamp() > payload.get("exp", 0):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        
        return payload
    
    except jwt.InvalidTokenError as e:
        logger.error(f"Token validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )


async def validate_land_access(
    land_id: str,
    tenant_id: str,
    supabase: Client
) -> Dict[str, Any]:
    """
    Validate that the land exists and belongs to the requesting tenant.
    
    Args:
        land_id: UUID of the land
        tenant_id: UUID of the tenant making the request
        supabase: Supabase client
    
    Returns:
        Land record from database
    
    Raises:
        HTTPException: If land not found or access denied
    """
    try:
        response = supabase.table("lands").select(
            "id, tenant_id, farmer_id, name, area_acres, boundary, "
            "center_lat, center_lon, current_crop, is_active"
        ).eq("id", land_id).eq("tenant_id", tenant_id).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Land {land_id} not found or access denied"
            )
        
        land = response.data[0]
        
        # Check if land is active
        if not land.get("is_active", True):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Land is not active"
            )
        
        # Validate boundary exists
        if not land.get("boundary"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Land boundary not defined. Please add boundary coordinates first."
            )
        
        return land
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        logger.error(f"Error validating land access: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate land access"
        )


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0"
    )


@app.post("/api/v1/ndvi/process", response_model=NDVIProcessResponse)
async def process_land_ndvi(
    request: NDVIProcessRequest,
    background_tasks: BackgroundTasks,
    auth_payload: Dict[str, Any] = Depends(verify_token),
    supabase: Client = Depends(get_supabase_client)
):
    """
    Process NDVI data for a specific land parcel.
    
    This endpoint:
    1. Validates tenant access to the land
    2. Discovers/downloads satellite imagery
    3. Computes NDVI for the land polygon
    4. Generates colorized overlay and thumbnails
    5. Stores results in database and storage
    
    Args:
        request: NDVI processing parameters
        auth_payload: Authenticated user/tenant info
        supabase: Supabase client
    
    Returns:
        NDVIProcessResponse with statistics and overlay URLs
    """
    start_time = datetime.utcnow()
    tenant_id = auth_payload["tenant_id"]
    user_id = auth_payload.get("user_id")
    
    logger.info(
        f"NDVI processing request - Land: {request.land_id}, "
        f"Tenant: {tenant_id}, User: {user_id}"
    )
    
    try:
        # Validate land access
        land = await validate_land_access(request.land_id, tenant_id, supabase)
        
        # Check if recent processing exists and force_refresh is False
        if not request.force_refresh:
            recent_data = supabase.table("ndvi_data").select(
                "date, ndvi_value, mean_ndvi, min_ndvi, max_ndvi"
            ).eq("land_id", request.land_id).order(
                "date", desc=True
            ).limit(1).execute()
            
            if recent_data.data:
                latest = recent_data.data[0]
                latest_date = datetime.fromisoformat(latest["date"])
                days_old = (datetime.utcnow() - latest_date).days
                
                if days_old < 7:  # Data less than 7 days old
                    logger.info(
                        f"Recent NDVI data exists ({days_old} days old). "
                        "Use force_refresh=true to reprocess."
                    )
                    # Return existing data
                    return await _fetch_existing_ndvi_data(
                        request.land_id, tenant_id, supabase
                    )
        
        # Initialize worker
        worker = NDVILandWorker(
            supabase_url=SUPABASE_URL,
            supabase_key=SUPABASE_SERVICE_ROLE_KEY,
            b2_key_id=os.getenv("B2_KEY_ID"),
            b2_app_key=os.getenv("B2_APP_KEY"),
            b2_bucket=os.getenv("B2_BUCKET_RAW"),
            mpc_stac_url=os.getenv("MPC_STAC_BASE")
        )
        
        # Process NDVI (synchronous operation)
        logger.info(f"Starting NDVI processing for land {request.land_id}")
        result = await asyncio.get_event_loop().run_in_executor(
            executor,
            worker.process_land_ndvi,
            request.land_id,
            tenant_id,
            request.lookback_days,
            request.max_cloud_cover
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        logger.info(
            f"NDVI processing completed for land {request.land_id} "
            f"in {processing_time:.2f}s"
        )
        
        # Build response
        response = NDVIProcessResponse(
            success=True,
            message="NDVI processing completed successfully",
            land_id=request.land_id,
            tenant_id=tenant_id,
            processed_at=datetime.utcnow().isoformat(),
            overall_stats=NDVIStatistics(**result["overall_stats"]),
            micro_tiles=[
                MicroTileInfo(
                    tile_id=mt["tile_id"],
                    acquisition_date=mt["acquisition_date"],
                    thumbnail_url=mt.get("thumbnail_url"),
                    overlay_url=mt.get("overlay_url"),
                    bbox=mt["bbox"],
                    ndvi_stats=NDVIStatistics(**mt["ndvi_stats"])
                )
                for mt in result["micro_tiles"]
            ],
            overlay_geojson=result.get("overlay_geojson"),
            processing_time_seconds=processing_time
        )
        
        return response
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"NDVI processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"NDVI processing failed: {str(e)}"
        )


@app.get("/api/v1/ndvi/status/{land_id}", response_model=NDVIProcessResponse)
async def get_ndvi_status(
    land_id: str,
    auth_payload: Dict[str, Any] = Depends(verify_token),
    supabase: Client = Depends(get_supabase_client)
):
    """
    Get the latest NDVI processing status and data for a land.
    
    Args:
        land_id: UUID of the land
        auth_payload: Authenticated user/tenant info
        supabase: Supabase client
    
    Returns:
        Latest NDVI data for the land
    """
    tenant_id = auth_payload["tenant_id"]
    
    try:
        # Validate access
        await validate_land_access(land_id, tenant_id, supabase)
        
        # Fetch existing data
        return await _fetch_existing_ndvi_data(land_id, tenant_id, supabase)
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Failed to fetch NDVI status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch NDVI status: {str(e)}"
        )


# ============================================================================
# Helper Functions
# ============================================================================

async def _fetch_existing_ndvi_data(
    land_id: str,
    tenant_id: str,
    supabase: Client
) -> NDVIProcessResponse:
    """Fetch existing NDVI data from database."""
    
    # Get latest NDVI data
    ndvi_response = supabase.table("ndvi_data").select("*").eq(
        "land_id", land_id
    ).order("date", desc=True).limit(10).execute()
    
    # Get micro tiles
    tiles_response = supabase.table("ndvi_micro_tiles").select("*").eq(
        "land_id", land_id
    ).order("acquisition_date", desc=True).limit(10).execute()
    
    if not ndvi_response.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No NDVI data found for this land"
        )
    
    # Calculate overall stats from latest data
    latest = ndvi_response.data[0]
    overall_stats = NDVIStatistics(
        min_ndvi=latest.get("min_ndvi") or latest.get("ndvi_min"),
        max_ndvi=latest.get("max_ndvi") or latest.get("ndvi_max"),
        mean_ndvi=latest.get("mean_ndvi"),
        median_ndvi=None,  # Not stored in this schema
        std_ndvi=latest.get("ndvi_std"),
        valid_pixels=latest.get("valid_pixels"),
        total_pixels=latest.get("total_pixels"),
        coverage_percentage=latest.get("coverage_percentage")
    )
    
    # Build micro tiles list
    micro_tiles = []
    for tile in tiles_response.data:
        micro_tiles.append(MicroTileInfo(
            tile_id=tile.get("id"),
            acquisition_date=tile["acquisition_date"],
            thumbnail_url=tile.get("ndvi_thumbnail_url"),
            overlay_url=tile.get("ndvi_thumbnail_url"),  # Same as thumbnail for now
            bbox=tile.get("bbox", {}),
            ndvi_stats=NDVIStatistics(
                min_ndvi=tile.get("ndvi_min"),
                max_ndvi=tile.get("ndvi_max"),
                mean_ndvi=tile.get("ndvi_mean"),
                median_ndvi=None,
                std_ndvi=tile.get("ndvi_std_dev"),
                valid_pixels=None,
                total_pixels=None,
                coverage_percentage=None
            )
        ))
    
    return NDVIProcessResponse(
        success=True,
        message="Existing NDVI data retrieved",
        land_id=land_id,
        tenant_id=tenant_id,
        processed_at=latest.get("computed_at") or latest.get("created_at"),
        overall_stats=overall_stats,
        micro_tiles=micro_tiles,
        overlay_geojson=None,
        processing_time_seconds=None
    )


# ============================================================================
# Application Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting NDVI Land Processing API...")
    logger.info(f"Max worker threads: {MAX_WORKERS}")
    
    # Verify Supabase connection
    try:
        supabase = get_supabase_client()
        logger.info("✓ Supabase connection verified")
    except Exception as e:
        logger.error(f"✗ Failed to connect to Supabase: {e}")
    
    logger.info("API ready to accept requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down NDVI Land Processing API...")
    executor.shutdown(wait=True)
    logger.info("Shutdown complete")


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    
    uvicorn.run(
        "ndvi_land_api:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENV") == "development",
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
