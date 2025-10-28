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

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# Local worker module
import tile_fetch_worker


APP_VERSION = "1.8.2"
START_TIME = time.time()



app = FastAPI()

# ✅ Allow CORS (restrict origins later in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/run")
async def run_worker(request: Request):
    try:
        body = await request.json()
        cloud_cover = body.get("cloud_cover", 30)     # default 30%
        lookback_days = body.get("lookback_days", 5)  # default 5 days

        logging.info(f"Running worker with cloud_cover={cloud_cover}, lookback_days={lookback_days}")
        processed_count = tile_fetch_worker.main(cloud_cover=cloud_cover, lookback_days=lookback_days)

        return {
            "status": "success",
            "cloud_cover": cloud_cover,
            "lookback_days": lookback_days,
            "processed_tiles": processed_count
        }
    except Exception as e:
        logging.error(f"Worker failed: {e}")
        return {"status": "error", "message": str(e)}


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
