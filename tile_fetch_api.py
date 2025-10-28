# ──────────────────────────────────────────────────────────────────────────────
# File: tile_fetch_api.py
# Version: v1.9.0 — Stable Async NDVI Worker API
# Author: Amarsinh Patil
# Purpose:
#   Orchestrates NDVI tile fetching & computation using FastAPI.
#   Runs tile_fetch_worker.main() in a background thread, returns immediately.
#   Compatible with Render.com (binds port), Supabase, and n8n edge triggers.
# ──────────────────────────────────────────────────────────────────────────────

import os
import time
import json
import logging
import traceback
import threading
from typing import Optional, List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Local worker import
import tile_fetch_worker

# ----------------------------- Configuration -----------------------------
APP_VERSION = "1.9.0"
START_TIME = time.time()
DEFAULT_CLOUD_COVER = int(os.getenv("DEFAULT_CLOUD_COVER_MAX", 30))
DEFAULT_LOOKBACK_DAYS = int(os.getenv("MAX_SCENE_LOOKBACK_DAYS", 30))

# ----------------------------- Logging Setup -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("tile-fetch-api")

# ----------------------------- FastAPI Setup -----------------------------
app = FastAPI(title="NDVI Tile Fetch API", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------- Utility -----------------------------
def uptime():
    return round(time.time() - START_TIME, 2)

# ----------------------------- Health & Root -----------------------------
@app.get("/")
async def root():
    return {
        "service": "ndvi-tile-fetch-api",
        "status": "running",
        "version": APP_VERSION,
        "uptime_seconds": uptime(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    }

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "ndvi-tile-fetch-api",
        "version": APP_VERSION,
        "uptime_seconds": uptime()
    }

@app.get("/version")
async def version():
    return {"version": APP_VERSION}

# ----------------------------- Background Job Helper -----------------------------
def start_background_worker(cloud_cover: int, lookback_days: int, filter_tile_ids: Optional[List[str]] = None):
    """Launch the NDVI worker in a background thread"""
    def background_job():
        try:
            logger.info(f"🚀 Background worker started (cloud_cover={cloud_cover}, lookback={lookback_days}, tiles={filter_tile_ids})")
            result = tile_fetch_worker.main(cloud_cover=cloud_cover, lookback_days=lookback_days)
            logger.info(f"✅ Worker finished processing {result} tiles.")
        except Exception as e:
            logger.error(f"❌ Background worker crashed: {e}\n{traceback.format_exc()}")

    thread = threading.Thread(target=background_job, daemon=True)
    thread.start()
    return thread

# ----------------------------- Endpoints -----------------------------
@app.post("/run")
async def run_worker(request: Request):
    """
    Trigger NDVI tile fetch & compute for all agricultural MGRS tiles.
    Runs in a background thread to avoid blocking.
    Returns HTTP 202 Accepted immediately.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    cloud_cover = int(body.get("cloud_cover", DEFAULT_CLOUD_COVER))
    lookback_days = int(body.get("lookback_days", DEFAULT_LOOKBACK_DAYS))

    logger.info(f"POST /run — Starting NDVI worker (cloud_cover={cloud_cover}, lookback_days={lookback_days})")

    start_background_worker(cloud_cover, lookback_days)

    return JSONResponse(
        status_code=202,
        content={
            "status": "started",
            "message": "NDVI worker running in background.",
            "params": {
                "cloud_cover": cloud_cover,
                "lookback_days": lookback_days,
            },
        },
    )

@app.post("/run/{tile_id}")
async def run_single_tile(tile_id: str, request: Request):
    """
    Trigger NDVI computation for a single MGRS tile.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    cloud_cover = int(body.get("cloud_cover", DEFAULT_CLOUD_COVER))
    lookback_days = int(body.get("lookback_days", DEFAULT_LOOKBACK_DAYS))

    logger.info(f"POST /run/{tile_id} — Starting NDVI worker for single tile")

    def background_job():
        try:
            logger.info(f"🚀 Worker started for single tile {tile_id}")
            result = tile_fetch_worker.main(cloud_cover=cloud_cover, lookback_days=lookback_days)
            logger.info(f"✅ Single-tile worker finished ({tile_id}) → {result} tiles processed.")
        except Exception as e:
            logger.error(f"❌ Single-tile run failed for {tile_id}: {e}\n{traceback.format_exc()}")

    threading.Thread(target=background_job, daemon=True).start()

    return JSONResponse(
        status_code=202,
        content={
            "status": "started",
            "tile_id": tile_id,
            "params": {"cloud_cover": cloud_cover, "lookback_days": lookback_days},
        },
    )

# ----------------------------- Run Server -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    logger.info(f"Starting NDVI Tile Fetch API v{APP_VERSION} on port {port}")
    uvicorn.run("tile_fetch_api:app", host="0.0.0.0", port=port, log_level="info")
