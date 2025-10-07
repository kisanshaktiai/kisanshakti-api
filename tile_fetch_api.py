from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import tile_fetch_worker  # or ndvi_land_worker for the other API

app = FastAPI()

# ✅ Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production, restrict to your Supabase/Lovable domains
    allow_credentials=True,
    allow_methods=["*"],   # includes OPTIONS
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/run")
def run_worker():
    try:
        tile_fetch_worker.main()
        return {"status": "success"}
    except Exception as e:
        logging.error(f"Worker failed: {e}")
        return {"status": "error", "message": str(e)}
