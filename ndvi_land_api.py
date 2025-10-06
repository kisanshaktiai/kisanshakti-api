from fastapi import FastAPI
import logging
import ndvi_land_worker

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/run")
def run_worker():
    try:
        ndvi_land_worker.main()
        return {"status": "success"}
    except Exception as e:
        logging.error(f"NDVI worker failed: {e}")
        return {"status": "error", "message": str(e)}
