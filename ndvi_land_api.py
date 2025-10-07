from fastapi import FastAPI, BackgroundTasks
import logging
import ndvi_land_worker

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/run")
def run_worker(background_tasks: BackgroundTasks):
    try:
        background_tasks.add_task(ndvi_land_worker.main)
        logging.info("NDVI worker started in background")
        return {"status": "started"}
    except Exception as e:
        logging.error(f"NDVI worker failed to start: {e}")
        return {"status": "error", "message": str(e)}
