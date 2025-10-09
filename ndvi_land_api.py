from fastapi import FastAPI, BackgroundTasks, Query
import logging
import ndvi_land_worker
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NDVI Land API")

# Allow all origins/methods (safe since this is a backend API, not public browser JS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in prod you can restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

@app.get("/")
def root():
    return {
        "service": "NDVI Land Processor",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "run": "POST /run?limit=50&use_queue=true"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/run")
def run_worker(
    background_tasks: BackgroundTasks,
    limit: int = Query(None, description="Limit number of lands to process"),
    use_queue: bool = Query(True, description="Use ndvi_request_queue instead of direct processing")
):
    """
    Run NDVI land worker in background.
    - limit: restrict number of lands
    - use_queue: process queued requests if True, otherwise scan lands directly
    """
    try:
        background_tasks.add_task(ndvi_land_worker.main, limit, use_queue)
        logging.info(f"🚀 NDVI worker started (limit={limit}, use_queue={use_queue})")
        return {
            "status": "started",
            "limit": limit,
            "use_queue": use_queue
        }
    except Exception as e:
        logging.error(f"❌ Failed to start NDVI worker: {e}")
        return {"status": "error", "message": str(e)}
