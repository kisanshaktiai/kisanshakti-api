from fastapi import FastAPI
import logging
import tile_fetch_worker

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/run")
def run_worker():
    try:
        tile_fetch_worker.main()
        return {"status": "success"}
    except Exception as e:
        logging.error(f"Tile fetch worker failed: {e}")
        return {"status": "error", "message": str(e)}
