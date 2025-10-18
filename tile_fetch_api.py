from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import logging
import tile_fetch_worker

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
