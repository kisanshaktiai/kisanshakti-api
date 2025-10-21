from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import traceback
import threading

# Import your main NDVI worker script
import tile_fetch_worker_v1_7_1_fixed as tile_fetch_worker

# ------------------------------------------------------
# ✅ FastAPI App Setup
# ------------------------------------------------------
app = FastAPI(title="Tile Fetch Worker API", version="1.7.1")

# Enable CORS (wide open — you can restrict origins later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("worker_api")

# ------------------------------------------------------
# ✅ Health Endpoint (for Render monitoring)
# ------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}


# ------------------------------------------------------
# ✅ Main Worker Endpoint (POST /run)
#    Runs NDVI worker — async background by default
# ------------------------------------------------------
@app.post("/run")
async def run_worker(request: Request):
    """
    Trigger NDVI tile processing.
    The worker runs in a background thread to avoid 300s timeout issues
    on Supabase Edge Functions.
    """
    try:
        body = await request.json()
        cloud_cover = int(body.get("cloud_cover", 30))
        lookback_days = int(body.get("lookback_days", 5))

        logger.info(f"🛰️ Starting worker: cloud_cover={cloud_cover}, lookback_days={lookback_days}")

        # ---- Run background thread so Edge Function doesn't timeout ----
        def background_job():
            try:
                processed = tile_fetch_worker.main(
                    cloud_cover=cloud_cover,
                    lookback_days=lookback_days
                )
                logger.info(f"✅ Worker completed successfully: {processed} tiles processed")
            except Exception as e:
                logger.error(f"❌ Background worker failed: {e}\n{traceback.format_exc()}")

        thread = threading.Thread(target=background_job, daemon=True)
        thread.start()

        # Return immediately so Supabase Edge Function doesn’t hit timeout
        return JSONResponse(
            status_code=200,
            content={
                "status": "started",
                "message": "Worker started successfully in background.",
                "cloud_cover": cloud_cover,
                "lookback_days": lookback_days,
            },
        )

    except Exception as e:
        logger.error(f"❌ Worker trigger failed: {e}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "details": traceback.format_exc(),
            },
        )
