# main.py

from fastapi import FastAPI, BackgroundTasks, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
import datetime
from supabase import create_client
import os
import farmer_land_ndvi_worker as ndvi_worker

# Initialize
app = FastAPI(
    title="NDVI Land Processing API",
    description="World-class NDVI processing system for agricultural land monitoring",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# === Pydantic Models ===
class NDVIRequest(BaseModel):
    land_ids: List[str]
    tile_id: str
    date_from: str
    date_to: str
    tenant_id: Optional[str] = None
    cloud_coverage: int = 20
    statistics_only: bool = False
    priority: int = 5


class ProcessingStatus(BaseModel):
    request_id: str
    status: str
    processed_count: int
    total_count: int
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]


# === Endpoints ===
@app.get("/")
def root():
    return {
        "service": "NDVI Land Processor",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "health": "GET /health",
            "run_worker": "POST /run",
            "create_request": "POST /requests",
            "get_status": "GET /requests/{request_id}",
            "get_land_ndvi": "GET /lands/{land_id}/ndvi",
            "get_farmer_ndvi": "GET /farmers/{farmer_id}/ndvi",
            "stats": "GET /stats"
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        # Test Supabase connection
        supabase.table("lands").select("id").limit(1).execute()
        return {
            "status": "healthy",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "supabase": "connected"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.datetime.utcnow().isoformat()
        }


@app.post("/run")
def run_worker(
    background_tasks: BackgroundTasks,
    limit: int = Query(None, description="Limit number of requests to process"),
    use_queue: bool = Query(True, description="Use ndvi_request_queue")
):
    """
    Run NDVI worker in background
    """
    try:
        background_tasks.add_task(ndvi_worker.main, limit, use_queue)
        logging.info(f"🚀 NDVI worker started (limit={limit}, use_queue={use_queue})")
        return {
            "status": "started",
            "limit": limit,
            "use_queue": use_queue,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
    except Exception as e:
        logging.error(f"❌ Failed to start NDVI worker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/requests", response_model=dict)
def create_ndvi_request(request: NDVIRequest):
    """
    Create a new NDVI processing request
    """
    try:
        # Validate lands exist
        lands = supabase.table("lands").select("id").in_("id", request.land_ids).execute()
        if not lands.data:
            raise HTTPException(status_code=404, detail="No valid lands found")
        
        # Create request
        result = supabase.table("ndvi_request_queue").insert({
            "tenant_id": request.tenant_id,
            "land_ids": request.land_ids,
            "tile_id": request.tile_id,
            "date_from": request.date_from,
            "date_to": request.date_to,
            "cloud_coverage": request.cloud_coverage,
            "statistics_only": request.statistics_only,
            "priority": request.priority,
            "status": "queued",
            "batch_size": len(request.land_ids)
        }).execute()
        
        request_id = result.data[0]["id"]
        
        logging.info(f"📋 Created NDVI request {request_id} for {len(request.land_ids)} lands")
        
        return {
            "request_id": request_id,
            "status": "queued",
            "land_count": len(request.land_ids),
            "message": "Request created successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to create request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/requests/{request_id}", response_model=ProcessingStatus)
def get_request_status(request_id: str):
    """
    Get status of an NDVI processing request
    """
    try:
        result = supabase.table("ndvi_request_queue")\
            .select("*")\
            .eq("id", request_id)\
            .single()\
            .execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Request not found")
        
        data = result.data
        return ProcessingStatus(
            request_id=data["id"],
            status=data["status"],
            processed_count=data.get("processed_count", 0),
            total_count=len(data["land_ids"]),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error_message=data.get("error_message")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to get request status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/lands/{land_id}/ndvi")
def get_land_ndvi(
    land_id: str,
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    limit: int = Query(30, le=365)
):
    """
    Get NDVI history for a specific land
    """
    try:
        query = supabase.table("ndvi_data")\
            .select("*")\
            .eq("land_id", land_id)\
            .order("date", desc=True)
        
        if start_date:
            query = query.gte("date", start_date)
        if end_date:
            query = query.lte("date", end_date)
        
        result = query.limit(limit).execute()
        
        return {
            "land_id": land_id,
            "count": len(result.data),
            "data": result.data
        }
    
    except Exception as e:
        logging.error(f"Failed to get land NDVI: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/lands/{land_id}/ndvi/latest")
def get_land_latest_ndvi(land_id: str):
    """
    Get latest NDVI data for a land
    """
    try:
        # Get from ndvi_micro_tiles (most recent)
        result = supabase.table("ndvi_micro_tiles")\
            .select("*")\
            .eq("land_id", land_id)\
            .order("acquisition_date", desc=True)\
            .limit(1)\
            .execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="No NDVI data found for this land")
        
        return result.data[0]
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to get latest NDVI: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/farmers/{farmer_id}/ndvi")
def get_farmer_ndvi(
    farmer_id: str,
    date: Optional[str] = Query(None, description="Specific date (YYYY-MM-DD)"),
    start_date: Optional[str] = Query(None, description="Start date for range (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date for range (YYYY-MM-DD)"),
    limit: int = Query(100, le=1000)
):
    """
    Get NDVI data for all lands owned by a farmer
    """
    try:
        # Get all lands for farmer
        lands = supabase.table("lands")\
            .select("id")\
            .eq("farmer_id", farmer_id)\
            .execute()
        
        if not lands.data:
            raise HTTPException(status_code=404, detail="No lands found for this farmer")
        
        land_ids = [land["id"] for land in lands.data]
        
        # Get NDVI data for all lands
        query = supabase.table("ndvi_micro_tiles")\
            .select("*")\
            .in_("land_id", land_ids)\
            .order("acquisition_date", desc=True)
        
        if date:
            query = query.eq("acquisition_date", date)
        else:
            if start_date:
                query = query.gte("acquisition_date", start_date)
            if end_date:
                query = query.lte("acquisition_date", end_date)
        
        result = query.limit(limit).execute()
        
        return {
            "farmer_id": farmer_id,
            "land_count": len(land_ids),
            "ndvi_records": len(result.data),
            "data": result.data
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to get farmer NDVI: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def get_stats():
    """
    Get processing statistics
    """
    try:
        # Total requests
        total_requests = supabase.table("ndvi_request_queue")\
            .select("id", count="exact")\
            .execute()
        
        # Queued requests
        queued = supabase.table("ndvi_request_queue")\
            .select("id", count="exact")\
            .eq("status", "queued")\
            .execute()
        
        # Processing
        processing = supabase.table("ndvi_request_queue")\
            .select("id", count="exact")\
            .eq("status", "processing")\
            .execute()
        
        # Completed
        completed = supabase.table("ndvi_request_queue")\
            .select("id", count="exact")\
            .eq("status", "completed")\
            .execute()
        
        # Failed
        failed = supabase.table("ndvi_request_queue")\
            .select("id", count="exact")\
            .eq("status", "failed")\
            .execute()
        
        # Total NDVI records
        ndvi_records = supabase.table("ndvi_micro_tiles")\
            .select("id", count="exact")\
            .execute()
        
        return {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "requests": {
                "total": total_requests.count,
                "queued": queued.count,
                "processing": processing.count,
                "completed": completed.count,
                "failed": failed.count
            },
            "ndvi_records": ndvi_records.count
        }
    
    except Exception as e:
        logging.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
