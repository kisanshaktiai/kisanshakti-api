"""
NDVI Land API
-------------
FastAPI backend for managing NDVI requests and fetching results.
Multi-tenant safe. Uses global satellite_tiles.
"""

import os
import datetime
import logging
from typing import List, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
import ndvi_land_worker as worker

# === CONFIG ===
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="NDVI Land Processor API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

# ---------------- Health ----------------
@app.get("/health")
def health():
    try:
        supabase.table("lands").select("id").limit(1).execute()
        return {"status": "healthy", "timestamp": datetime.datetime.utcnow().isoformat()}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# ---------------- Create NDVI Request ----------------
@app.post("/requests")
def create_request(payload: dict):
    try:
        tenant_id = payload.get("tenant_id")
        land_ids = payload.get("land_ids", [])
        cloud_coverage = payload.get("cloud_coverage", 20)
        date_to = payload.get("date_to") or datetime.date.today().isoformat()
        date_from = payload.get("date_from") or (datetime.date.today() - datetime.timedelta(days=30)).isoformat()

        if not tenant_id:
            raise HTTPException(status_code=400, detail="tenant_id is required")

        if not land_ids:
            lands = supabase.table("lands").select("id").eq("tenant_id", tenant_id).execute().data
            land_ids = [l["id"] for l in lands]

        tile_res = supabase.table("satellite_tiles").select("tile_id").order("acquisition_date", desc=True).limit(1).execute()
        if not tile_res.data:
            raise HTTPException(status_code=404, detail="No tiles available")
        tile_id = tile_res.data[0]["tile_id"]

        result = supabase.table("ndvi_request_queue").insert({
            "tenant_id": tenant_id,
            "land_ids": land_ids,
            "tile_id": tile_id,
            "date_from": date_from,
            "date_to": date_to,
            "cloud_coverage": cloud_coverage,
            "status": "queued",
            "batch_size": len(land_ids)
        }).execute()

        req_id = result.data[0]["id"]
        logging.info(f"📋 Queued NDVI request {req_id} for tenant {tenant_id}")

        return {"request_id": req_id, "tile_id": tile_id, "status": "queued"}
    except Exception as e:
        logging.error(f"Request creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- Run Worker ----------------
@app.post("/run")
def run_worker(background: BackgroundTasks, limit: int = 10):
    background.add_task(worker.main, limit)
    return {"status": "started", "limit": limit, "timestamp": datetime.datetime.utcnow().isoformat()}

# ---------------- Get NDVI Data ----------------
@app.get("/lands/{land_id}/ndvi")
def get_land_ndvi(land_id: str, tenant_id: Optional[str] = None, limit: int = 30):
    q = supabase.table("ndvi_data").select("*").eq("land_id", land_id).order("date", desc=True)
    if tenant_id:
        q = q.eq("tenant_id", tenant_id)
    res = q.limit(limit).execute()
    return {"count": len(res.data), "data": res.data}

# ---------------- Stats ----------------
@app.get("/stats")
def stats():
    rq = supabase.table("ndvi_request_queue").select("id", count="exact").execute()
    data = {
        "total_requests": rq.count,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }
    return data
