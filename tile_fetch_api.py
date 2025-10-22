# ──────────────────────────────────────────────────────────────────────────────
# File: tile_fetch_api.py  (FastAPI service)
# Version: 1.8.2
# Runtime: Python 3.8+ recommended (3.8/3.9/3.11)
# Purpose: Orchestrates NDVI tile processing in background threads
# Updates in v1.8.2:
# - Added root route ("/") to prevent 404 "No server available" errors
# - Enhanced /health endpoint with uptime and app info
# - Improved structured logging at startup
# - Minor response consistency improvements
# ──────────────────────────────────────────────────────────────────────────────

import os
import json
import time
import logging
import traceback
import threading
from typing import List, Optional, Dict

from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Local worker module
import tile_fetch_worker as tile_worker

APP_VERSION = "1.8.2"
START_TIME = time.time()

# --------------------------- FastAPI App Setup -------------------------------
app = FastAPI(title="Tile Fetch Worker API", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("worker_api")
logger.info(f"🚀 Starting Tile Fetch Worker API v{APP_VERSION}")


# ------------------------------ Root Route -----------------------------------
@app.get("/")
def root():
    """Root endpoint to confirm API is running."""
    uptime = round(time.time() - START_TIME, 2)
    return {
        "status": "running",
        "version": APP_VERSION,
        "uptime_sec": uptime,
        "message": "Tile Fetch Worker API is active and ready.",
        "endpoints": ["/health", "/run", "/run/{tile_id}"],
    }


# ------------------------------ Health Check ---------------------------------
@app.get("/health")
def health_check():
    """Simple health check for uptime monitoring."""
    uptime = round(time.time() - START_TIME, 2)
    return {"status": "ok", "version": APP_VERSION, "uptime_sec": uptime}


# --------------------------- NDVI Worker Trigger -----------------------------
@app.post("/run")
async def run_worker(
    request: Request,
    cloud_cover: Optional[int] = Query(default=None, ge=0, le=100),
    lookback_days: Optional[int] = Query(default=None, ge=1, le=365),
    max_tiles: Optional[int] = Query(default=None, ge=1, le=10000),
):
    """Kick off the NDVI worker in a background thread."""
    try:
        body = {}
        try:
            body = await request.json()
        except Exception:
            pass

        cc = (
            cloud_cover
            if cloud_cover is not None
            else body.get("cloud_cover", int(os.getenv("DEFAULT_CLOUD_COVER_MAX", "30")))
        )
        lb = (
            lookback_days
            if lookback_days is not None
            else body.get("lookback_days", int(os.getenv("MAX_SCENE_LOOKBACK_DAYS", "30")))
        )
        mt = (
            max_tiles
            if max_tiles is not None
            else body.get("max_tiles")
        )
        tile_ids = body.get("tile_ids") or None
        force = bool(body.get("force", False))

        logger.info(
            "🛰️ Starting worker: cloud_cover=%s, lookback_days=%s, max_tiles=%s, force=%s, tiles=%s",
            cc,
            lb,
            mt,
            force,
            (tile_ids[:5] if isinstance(tile_ids, list) else tile_ids),
        )

        def background_job():
            try:
                stats = tile_worker.main(
                    cloud_cover=cc,
                    lookback_days=lb,
                    filter_tile_ids=tile_ids,
                    max_tiles=mt,
                    force=force,
                )
                logger.info("✅ Worker finished: %s", json.dumps(stats))
            except Exception:
                logger.error("❌ Background worker crashed: %s", traceback.format_exc())

        threading.Thread(target=background_job, daemon=True).start()

        return JSONResponse(
            status_code=200,
            content={
                "status": "started",
                "message": "Worker started in background.",
                "params": {
                    "cloud_cover": cc,
                    "lookback_days": lb,
                    "max_tiles": mt,
                    "force": force,
                    "tile_ids": tile_ids,
                },
            },
        )

    except Exception as e:
        logger.error("❌ Trigger failed: %s\n%s", e, traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


# --------------------------- Single Tile Trigger -----------------------------
@app.post("/run/{tile_id}")
async def run_single_tile(tile_id: str, request: Request):
    """Process a single MGRS tile id on-demand."""
    try:
        try:
            body = await request.json()
        except Exception:
            body = {}

        cc = int(body.get("cloud_cover", os.getenv("DEFAULT_CLOUD_COVER_MAX", 30)))
        lb = int(body.get("lookback_days", os.getenv("MAX_SCENE_LOOKBACK_DAYS", 30)))
        force = bool(body.get("force", False))

        def background_job():
            try:
                stats = tile_worker.main(
                    cloud_cover=cc,
                    lookback_days=lb,
                    filter_tile_ids=[tile_id],
                    max_tiles=1,
                    force=force,
                )
                logger.info("✅ Single-tile run finished: %s", json.dumps(stats))
            except Exception:
                logger.error("❌ Single-tile run crashed: %s", traceback.format_exc())

        threading.Thread(target=background_job, daemon=True).start()
        return JSONResponse(
            status_code=200,
            content={
                "status": "started",
                "tile_id": tile_id,
                "cloud_cover": cc,
                "lookback_days": lb,
                "force": force,
            },
        )

    except Exception as e:
        logger.error("❌ Trigger failed: %s\n%s", e, traceback.format_exc())
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
