"""
NDVI Land Worker v5.1 — B2-Backed NDVI Processor
---------------------------------------------------------------
✅ Uses only pre-downloaded Sentinel-2 tiles from Backblaze B2
✅ No Planetary Computer or external API calls
✅ Derives MGRS tile_id automatically from land boundary if missing
✅ Logs all actions and errors in Supabase
✅ Inserts NDVI data into ndvi_data and ndvi_micro_tiles tables
"""

import os
import io
import json
import time
import logging
import datetime
import traceback
from typing import List, Optional

import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import shape, mapping
import mgrs
from PIL import Image
import matplotlib.cm as cm
from supabase import create_client
from b2sdk.v2 import InMemoryAccountInfo, B2Api, DownloadDestBytes

# ---------------------------------------------------------------
# Logging
# ---------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("NDVI_WORKER_LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("ndvi-worker-v5.1")

# ---------------------------------------------------------------
# Environment
# ---------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
B2_APP_KEY_ID = os.getenv("B2_APP_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME", "sentinel-tiles")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("❌ Missing Supabase environment variables")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize B2
b2_info = InMemoryAccountInfo()
b2_api = B2Api(b2_info)
if B2_APP_KEY_ID and B2_APP_KEY:
    b2_api.authorize_account("production", B2_APP_KEY_ID, B2_APP_KEY)
    b2_bucket = b2_api.get_bucket_by_name(B2_BUCKET_NAME)
else:
    logger.warning("⚠️ B2 credentials not set, worker will fail for B2 access")

# ---------------------------------------------------------------
# Utils
# ---------------------------------------------------------------
def now_iso():
    return datetime.datetime.utcnow().isoformat()

def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    np.seterr(divide="ignore", invalid="ignore")
    ndvi = (nir - red) / (nir + red)
    return np.clip(ndvi, -1, 1)

def calculate_statistics(arr: np.ndarray) -> dict:
    valid = arr[~np.isnan(arr)]
    total = int(arr.size)
    if valid.size == 0:
        return {
            "mean": None, "min": None, "max": None, "std": None,
            "valid_pixels": 0, "total_pixels": total, "coverage": 0.0
        }
    return {
        "mean": float(np.mean(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "std": float(np.std(valid)),
        "valid_pixels": int(valid.size),
        "total_pixels": total,
        "coverage": float(valid.size / total * 100)
    }

def create_colorized_ndvi_png(ndvi_array: np.ndarray) -> bytes:
    cmap = cm.get_cmap("RdYlGn")
    norm = np.clip((ndvi_array + 1) / 2, 0, 1)
    rgba = (cmap(norm) * 255).astype(np.uint8)
    rgba[..., 3][np.isnan(ndvi_array)] = 0
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.getvalue()

def upload_thumbnail_to_supabase(land_id: str, date: str, png_bytes: bytes) -> Optional[str]:
    path = f"{land_id}/{date}/ndvi_color.png"
    try:
        res = supabase.storage.from_("ndvi-thumbnails").upload(
            path, io.BytesIO(png_bytes), {"content-type": "image/png", "upsert": True}
        )
        if isinstance(res, dict) and res.get("error"):
            logger.error(f"Upload failed: {res.get('error')}")
            return None
        return f"{SUPABASE_URL}/storage/v1/object/public/ndvi-thumbnails/{path}"
    except Exception:
        logger.exception("Thumbnail upload failed")
        return None

# ---------------------------------------------------------------
# B2 Access
# ---------------------------------------------------------------
def download_b2_file(tile_id: str, filename: str) -> Optional[bytes]:
    """Download NDVI or band raster from B2."""
    try:
        b2_path = f"tiles/{tile_id}/{filename}"
        dest = DownloadDestBytes()
        b2_bucket.download_file_by_name(b2_path, dest)
        return dest.get_bytes_written()
    except Exception as e:
        logger.warning(f"⚠️ B2 file missing for {b2_path}: {e}")
        return None

# ---------------------------------------------------------------
# Core NDVI Logic
# ---------------------------------------------------------------
def derive_tile_id_from_geometry(geometry: dict) -> str:
    """Convert land centroid to MGRS 5-char tile ID."""
    geom = shape(geometry)
    lat, lon = geom.centroid.y, geom.centroid.x
    mgrs_code = mgrs.MGRS().toMGRS(lat, lon)
    return mgrs_code[:5]

def process_single_land(land: dict, tile: Optional[dict] = None) -> dict:
    land_id = land.get("id")
    tenant_id = land.get("tenant_id")
    geom_raw = land.get("boundary_polygon_old") or land.get("boundary")
    date = datetime.date.today().isoformat()

    result = {"success": False, "land_id": land_id, "error": None}
    try:
        if not geom_raw:
            raise ValueError("Missing boundary geometry")

        geometry = json.loads(geom_raw) if isinstance(geom_raw, str) else geom_raw
        tile_id = tile["tile_id"] if tile else derive_tile_id_from_geometry(geometry)
        logger.info(f"🛰️ Using tile {tile_id} for land {land_id}")

        # Try NDVI.tif from B2 first
        ndvi_bytes = download_b2_file(tile_id, "ndvi.tif")
        if ndvi_bytes:
            logger.info(f"✅ Found precomputed NDVI.tif for {tile_id}")
            with rasterio.MemoryFile(ndvi_bytes) as memfile:
                with memfile.open() as src:
                    ndvi_crop, _ = mask(src, [geometry], crop=True)
                    ndvi = ndvi_crop[0]
        else:
            # Try raw bands
            red_bytes = download_b2_file(tile_id, "B04.tif")
            nir_bytes = download_b2_file(tile_id, "B08.tif")
            if not (red_bytes and nir_bytes):
                raise RuntimeError(f"Missing B04/B08 or NDVI.tif for {tile_id}")
            with rasterio.MemoryFile(red_bytes) as red_mem, rasterio.MemoryFile(nir_bytes) as nir_mem:
                with red_mem.open() as red_ds, nir_mem.open() as nir_ds:
                    red_clip, _ = mask(red_ds, [geometry], crop=True)
                    nir_clip, _ = mask(nir_ds, [geometry], crop=True)
                    ndvi = calculate_ndvi(red_clip[0], nir_clip[0])

        stats = calculate_statistics(ndvi)
        if stats["valid_pixels"] == 0:
            raise RuntimeError("No valid NDVI pixels found")

        png_bytes = create_colorized_ndvi_png(ndvi)
        image_url = upload_thumbnail_to_supabase(land_id, date, png_bytes)
        now = now_iso()

        ndvi_record = {
            "tenant_id": tenant_id,
            "land_id": land_id,
            "date": date,
            "ndvi_value": stats["mean"],
            "coverage": stats["coverage"],
            "image_url": image_url,
            "created_at": now,
        }
        micro_record = {
            "tenant_id": tenant_id,
            "land_id": land_id,
            "acquisition_date": date,
            "ndvi_mean": stats["mean"],
            "ndvi_min": stats["min"],
            "ndvi_max": stats["max"],
            "ndvi_std_dev": stats["std"],
            "ndvi_thumbnail_url": image_url,
            "created_at": now,
        }

        supabase.table("ndvi_data").upsert(ndvi_record).execute()
        supabase.table("ndvi_micro_tiles").upsert(micro_record).execute()
        supabase.table("lands").update({
            "last_ndvi_value": stats["mean"],
            "last_ndvi_calculation": date,
            "ndvi_thumbnail_url": image_url,
            "updated_at": now,
        }).eq("id", land_id).execute()

        logger.info(f"✅ NDVI processed for {land_id} | mean={stats['mean']:.3f}")
        result.update({"success": True, "stats": stats, "image_url": image_url})
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"❌ Failed land {land_id}: {e}")
        try:
            supabase.table("ndvi_processing_logs").insert({
                "tenant_id": tenant_id,
                "land_id": land_id,
                "processing_step": "ndvi_calculation",
                "step_status": "failed",
                "error_message": str(e)[:500],
                "error_details": {"traceback": tb[:1200]},
                "created_at": now_iso(),
            }).execute()
        except Exception:
            logger.exception("Failed to write error log")
    return result

# ---------------------------------------------------------------
# Batch Processor
# ---------------------------------------------------------------
def process_request_sync(queue_id: str, tenant_id: str, land_ids: List[str], tile_id: Optional[str] = None) -> dict:
    logger.info(f"🚀 Processing NDVI request {queue_id} for tenant {tenant_id}")
    start = time.time()

    lands = supabase.table("lands").select("*").eq("tenant_id", tenant_id).in_("id", land_ids).execute().data or []
    if not lands:
        return {"processed_count": 0, "error": "No lands found"}

    tile = None
    if tile_id:
        res = supabase.table("satellite_tiles").select("*").eq("tile_id", tile_id).limit(1).execute()
        tile = res.data[0] if res.data else None
        if not tile:
            logger.warning(f"⚠️ Tile {tile_id} not found — will derive from land")

    processed, failed = 0, []
    for land in lands:
        res = process_single_land(land, tile)
        if res["success"]:
            processed += 1
        else:
            failed.append(land.get("id"))

    supabase.table("ndvi_request_queue").update({
        "status": "completed" if processed > 0 else "failed",
        "processed_count": processed,
        "completed_at": now_iso(),
    }).eq("id", queue_id).execute()

    logger.info(f"🏁 NDVI job {queue_id} done | processed={processed}, failed={len(failed)} | duration={int(time.time()-start)}s")
    return {"queue_id": queue_id, "processed": processed, "failed": failed}

# ---------------------------------------------------------------
# Entry Point (for Render Cron or API Instant Trigger)
# ---------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    logger.info(f"Starting NDVI Land Worker v5.1 — limit={args.limit}")
    try:
        jobs = supabase.table("ndvi_request_queue").select("*").eq("status", "queued").limit(args.limit).execute()
        rows = jobs.data or []
        if not rows:
            logger.info("No queued NDVI jobs.")
        for job in rows:
            queue_id = job["id"]
            tenant_id = job["tenant_id"]
            land_ids = job["land_ids"]
            tile_id = job.get("tile_id")
            supabase.table("ndvi_request_queue").update({"status": "processing"}).eq("id", queue_id).execute()
            process_request_sync(queue_id, tenant_id, land_ids, tile_id)
    except Exception:
        logger.exception("Fatal NDVI worker error")

    logger.info("NDVI Land Worker v5.1 finished.")
