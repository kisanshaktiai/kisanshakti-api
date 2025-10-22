# ──────────────────────────────────────────────────────────────────────────────
# File: tile_fetch_api.py  (FastAPI service)
# Version: 1.8.1
# Runtime: Python 3.8+ recommended (3.8/3.9/3.11)
# Purpose: Orchestrates NDVI tile processing in background threads
# Changes in v1.8.1:
# - Removed `from __future__ import annotations` to avoid SyntaxError on older runtimes
# - Improved logging and supabase response checking so failures are visible
# - Added safety around B2 existence checks and upload return values
# - Added a simple unit test file for core utilities
# - Added more defensive handling to ensure records are upserted even when NDVI already exists
# ──────────────────────────────────────────────────────────────────────────────

import os
import json
import logging
import traceback
import threading
from typing import List, Optional, Dict

from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Local worker module
import tile_fetch_worker as tile_worker

APP_VERSION = "1.8.1"

# --------------------------- FastAPI App Setup -------------------------------
app = FastAPI(title="Tile Fetch Worker API", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("worker_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@app.get("/health")
def health_check():
    """Simple health check for uptime monitoring."""
    return {"status": "ok", "version": APP_VERSION}


@app.post("/run")
async def run_worker(
    request: Request,
    cloud_cover: Optional[int] = Query(default=None, ge=0, le=100),
    lookback_days: Optional[int] = Query(default=None, ge=1, le=365),
    max_tiles: Optional[int] = Query(default=None, ge=1, le=10000),
):
    """Kick off the NDVI worker in a background thread.

    Query params override JSON body if provided.
    Body schema (all optional): {
      "cloud_cover": 30,
      "lookback_days": 5,
      "tile_ids": ["43QEG", "43QFG"],
      "max_tiles": 100,
      "force": false
    }
    """
    try:
        try:
            body = await request.json()
        except Exception:
            body = {}

        # Resolve parameters with precedence: query > body > env > defaults
        cc = (
            cloud_cover
            if cloud_cover is not None
            else body.get("cloud_cover")
            if body.get("cloud_cover") is not None
            else int(os.getenv("DEFAULT_CLOUD_COVER_MAX", "80"))
        )
        lb = (
            lookback_days
            if lookback_days is not None
            else body.get("lookback_days")
            if body.get("lookback_days") is not None
            else int(os.getenv("MAX_SCENE_LOOKBACK_DAYS", "90"))
        )
        mt = (
            max_tiles
            if max_tiles is not None
            else body.get("max_tiles")
            if body.get("max_tiles") is not None
            else None
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

        # Background job to avoid platform timeouts
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

        thread = threading.Thread(target=background_job, daemon=True)
        thread.start()

        return JSONResponse(
            status_code=200,
            content={
                "status": "started",
                "message": "Worker started in background.",
                "params": {"cloud_cover": cc, "lookback_days": lb, "max_tiles": mt, "force": force},
            },
        )
    except Exception as e:
        logger.error("❌ Trigger failed: %s\n%s", e, traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


@app.post("/run/{tile_id}")
async def run_single_tile(tile_id: str, request: Request):
    """Process a single MGRS tile id on-demand."""
    try:
        try:
            body = await request.json()
        except Exception:
            body = {}
        cc = int(body.get("cloud_cover", os.getenv("DEFAULT_CLOUD_COVER_MAX", 80)))
        lb = int(body.get("lookback_days", os.getenv("MAX_SCENE_LOOKBACK_DAYS", 90)))
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
        return {"status": "started", "tile_id": tile_id, "cloud_cover": cc, "lookback_days": lb}
    except Exception as e:
        logger.error("❌ Trigger failed: %s\n%s", e, traceback.format_exc())
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
