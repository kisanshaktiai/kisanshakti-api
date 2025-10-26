"""
NDVI Land Worker v6.1 — Multi-Tile B2 NDVI Engine with CRS Handling
---------------------------------------------------------------
✅ Computes NDVI using already-downloaded Sentinel tiles (B2)
✅ Finds all intersecting tiles for each land boundary
✅ Clips and merges NDVI rasters per land polygon
✅ Supports multi-tenant architecture
✅ Handles CRS reprojection automatically
"""

import os
import io
import json
import time
import logging
import datetime
import traceback
from typing import List, Optional, Dict

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
from shapely.geometry import shape, mapping
from PIL import Image
import matplotlib.cm as cm
from supabase import create_client

# ✅ FIXED: Correct imports for b2sdk v2.x
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from b2sdk.v2.exception import NonExistentBucket

# ---------------------------------------------------------------
# Logging
# ---------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("NDVI_WORKER_LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("ndvi-worker-v6")

# ---------------------------------------------------------------
# Environment
# ---------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
B2_APP_KEY_ID = os.getenv("B2_APP_KEY_ID") or os.getenv("B2_KEY_ID")  # Support both variable names
B2_APP_KEY = os.getenv("B2_APP_KEY")
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME") or os.getenv("B2_BUCKET_RAW", "kisanshakti-ndvi-tiles")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("❌ Missing Supabase environment variables")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize B2 API
b2_info = InMemoryAccountInfo()
b2_api = B2Api(b2_info)
b2_bucket = None

if B2_APP_KEY_ID and B2_APP_KEY:
    try:
        b2_api.authorize_account("production", B2_APP_KEY_ID, B2_APP_KEY)
        b2_bucket = b2_api.get_bucket_by_name(B2_BUCKET_NAME)
        logger.info(f"✅ B2 initialized successfully: bucket={B2_BUCKET_NAME}")
    except Exception as e:
        logger.warning(f"⚠️ B2 initialization failed: {e}")
else:
    logger.warning("⚠️ B2 credentials not set — NDVI worker cannot access B2 data")

# ---------------------------------------------------------------
# Utility functions
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
# B2 Access - Using BytesIO instead of DownloadDestBytes
# ---------------------------------------------------------------
def download_b2_file(tile_id: str, subdir: str, filename: str) -> Optional[bytes]:
    """Download a raster (ndvi.tif, B04.tif, or B08.tif) from B2."""
    if not b2_bucket:
        logger.error("❌ B2 bucket not initialized")
        return None
    
    try:
        b2_path = f"tiles/{subdir}/{tile_id}/{filename}"
        
        # ✅ Use BytesIO buffer instead of DownloadDestBytes
        download_buffer = io.BytesIO()
        b2_bucket.download_file_by_name(b2_path).save(download_buffer)
        download_buffer.seek(0)
        return download_buffer.read()
        
    except Exception as e:
        logger.warning(f"⚠️ Missing B2 file {b2_path}: {e}")
        return None

# ---------------------------------------------------------------
# Tile and Raster Logic
# ---------------------------------------------------------------
def get_intersecting_tiles(geometry: dict) -> List[dict]:
    """Return all tiles that intersect a given land boundary with buffer."""
    try:
        resp = supabase.rpc("get_intersecting_tiles", {"land_geom": json.dumps(geometry)}).execute()
        return resp.data or []
    except Exception as e:
        logger.warning(f"⚠️ Fallback: searching tiles by land centroid")
        # Better fallback: use land centroid to find the correct MGRS tile
        land_shape = shape(geometry)
        centroid = land_shape.centroid
        lon, lat = centroid.x, centroid.y
        
        # Get tiles that might contain this point (with generous buffer)
        tiles = supabase.table("satellite_tiles").select("*").execute().data or []
        
        # Filter by centroid proximity first (much more accurate)
        candidates = []
        for tile in tiles:
            bbox_data = tile.get("bbox")
            if not bbox_data:
                continue
            
            # Parse bbox
            if isinstance(bbox_data, str):
                bbox_data = json.loads(bbox_data)
            
            if bbox_data.get("type") == "Polygon":
                coords = bbox_data["coordinates"][0]
                min_lon = min(c[0] for c in coords)
                max_lon = max(c[0] for c in coords)
                min_lat = min(c[1] for c in coords)
                max_lat = max(c[1] for c in coords)
                
                # Check if land centroid is within tile bounds (with small buffer)
                buffer = 0.01  # ~1km buffer
                if (min_lon - buffer <= lon <= max_lon + buffer and
                    min_lat - buffer <= lat <= max_lat + buffer):
                    candidates.append(tile)
                    logger.info(f"🎯 Found candidate tile: {tile['tile_id']} for point ({lon:.5f}, {lat:.5f})")
        
        return candidates if candidates else tiles[:2]  # Fallback to first 2 tiles if nothing found

def load_ndvi_from_tiles(tile_list: List[dict], land_geom: dict) -> np.ndarray:
    """Load and merge NDVI rasters for all intersecting tiles with CRS handling."""
    ndvi_crops = []
    
    # Debug: Log land geometry bounds
    land_shape = shape(land_geom)
    logger.info(f"📍 Land bounds (EPSG:4326): {land_shape.bounds}")

    for tile in tile_list:
        tile_id = tile["tile_id"]
        acq_date = tile.get("acquisition_date") or datetime.date.today().isoformat()
        ndvi_bytes = download_b2_file(tile_id, "ndvi", f"{acq_date}/ndvi.tif")

        if not ndvi_bytes:
            # Try raw bands
            red_bytes = download_b2_file(tile_id, "raw", f"{acq_date}/B04.tif")
            nir_bytes = download_b2_file(tile_id, "raw", f"{acq_date}/B08.tif")
            if not (red_bytes and nir_bytes):
                logger.warning(f"Tile {tile_id} missing raw bands or ndvi.tif")
                continue
            
            try:
                with rasterio.MemoryFile(red_bytes) as red_mem, rasterio.MemoryFile(nir_bytes) as nir_mem:
                    with red_mem.open() as red_ds, nir_mem.open() as nir_ds:
                        # Debug logging
                        logger.info(f"🗺️ Tile {tile_id} CRS: {red_ds.crs}, Bounds: {red_ds.bounds}")
                        
                        # Reproject geometry to match raster CRS
                        reprojected_geom = transform_geom(
                            "EPSG:4326", red_ds.crs, land_geom
                        )
                        logger.info(f"🔄 Reprojected land bounds (to {red_ds.crs}): {shape(reprojected_geom).bounds}")
                        
                        red_clip, _ = mask(red_ds, [reprojected_geom], crop=True, all_touched=True)
                        nir_clip, _ = mask(nir_ds, [reprojected_geom], crop=True, all_touched=True)
                        ndvi_crops.append(calculate_ndvi(red_clip[0], nir_clip[0]))
                        logger.info(f"✅ Processed tile {tile_id} from raw bands")
            except ValueError as e:
                logger.warning(f"Skipping tile {tile_id} (raw): {e}")
                continue
        else:
            # Use pre-computed NDVI
            try:
                with rasterio.MemoryFile(ndvi_bytes) as mem:
                    with mem.open() as src:
                        # Debug logging
                        logger.info(f"🗺️ Tile {tile_id} CRS: {src.crs}, Bounds: {src.bounds}")
                        
                        # Reproject geometry to match raster CRS
                        reprojected_geom = transform_geom(
                            "EPSG:4326", src.crs, land_geom
                        )
                        logger.info(f"🔄 Reprojected land bounds (to {src.crs}): {shape(reprojected_geom).bounds}")
                        
                        ndvi_clip, _ = mask(src, [reprojected_geom], crop=True, all_touched=True)
                        ndvi_crops.append(ndvi_clip[0])
                        logger.info(f"✅ Processed tile {tile_id} from NDVI")
            except ValueError as e:
                logger.warning(f"Skipping tile {tile_id} (ndvi): {e}")
                continue

    if not ndvi_crops:
        raise RuntimeError("No NDVI data found in intersecting tiles")
    
    if len(ndvi_crops) == 1:
        return ndvi_crops[0]
    
    # Merge multiple NDVI arrays using nanmean
    logger.info(f"🔗 Merging {len(ndvi_crops)} NDVI crops")
    return np.nanmean(ndvi_crops, axis=0)

# ---------------------------------------------------------------
# Core Processing
# ---------------------------------------------------------------
def process_single_land(land: dict) -> dict:
    land_id = land["id"]
    tenant_id = land["tenant_id"]
    geom_raw = land.get("boundary_polygon_old") or land.get("boundary")
    if not geom_raw:
        raise ValueError("Missing boundary geometry")

    geometry = json.loads(geom_raw) if isinstance(geom_raw, str) else geom_raw
    date = datetime.date.today().isoformat()
    result = {"land_id": land_id, "success": False}

    try:
        tiles = get_intersecting_tiles(geometry)
        logger.info(f"🌍 Land {land_id} intersects {len(tiles)} tile(s)")

        ndvi_array = load_ndvi_from_tiles(tiles, geometry)
        stats = calculate_statistics(ndvi_array)
        if stats["valid_pixels"] == 0:
            raise RuntimeError("No valid NDVI pixels")

        png_bytes = create_colorized_ndvi_png(ndvi_array)
        image_url = upload_thumbnail_to_supabase(land_id, date, png_bytes)
        now = now_iso()

        # Write results
        ndvi_record = {
            "tenant_id": tenant_id, "land_id": land_id, "date": date,
            "ndvi_value": stats["mean"], "coverage": stats["coverage"],
            "image_url": image_url, "created_at": now
        }
        micro_record = {
            "tenant_id": tenant_id, "land_id": land_id, "acquisition_date": date,
            "ndvi_mean": stats["mean"], "ndvi_min": stats["min"],
            "ndvi_max": stats["max"], "ndvi_std_dev": stats["std"],
            "ndvi_thumbnail_url": image_url, "created_at": now
        }

        supabase.table("ndvi_data").upsert(ndvi_record).execute()
        supabase.table("ndvi_micro_tiles").upsert(micro_record).execute()
        supabase.table("lands").update({
            "last_ndvi_value": stats["mean"],
            "last_ndvi_calculation": date,
            "ndvi_thumbnail_url": image_url,
            "updated_at": now
        }).eq("id", land_id).execute()

        result.update({"success": True, "stats": stats, "image_url": image_url})
        logger.info(f"✅ NDVI processed for {land_id} | mean={stats['mean']:.3f}")
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"❌ Failed {land_id}: {e}")
        supabase.table("ndvi_processing_logs").insert({
            "tenant_id": tenant_id, "land_id": land_id,
            "processing_step": "ndvi_calculation",
            "step_status": "failed", "error_message": str(e)[:500],
            "error_details": {"traceback": tb[:1000]},
            "created_at": now_iso()
        }).execute()
    return result

# ---------------------------------------------------------------
# Batch Runner
# ---------------------------------------------------------------
def process_request_sync(queue_id: str, tenant_id: str, land_ids: List[str], tile_id: Optional[str] = None) -> dict:
    logger.info(f"🚀 Processing NDVI request {queue_id} for tenant {tenant_id}")
    start = time.time()

    lands = supabase.table("lands").select("*").eq("tenant_id", tenant_id).in_("id", land_ids).execute().data or []
    if not lands:
        return {"processed_count": 0, "error": "No lands found"}

    processed, failed = 0, []
    for land in lands:
        res = process_single_land(land)
        if res.get("success"):
            processed += 1
        else:
            failed.append(land["id"])

    supabase.table("ndvi_request_queue").update({
        "status": "completed" if processed > 0 else "failed",
        "processed_count": processed,
        "completed_at": now_iso(),
    }).eq("id", queue_id).execute()

    logger.info(f"🏁 NDVI job {queue_id} done | processed={processed}, failed={len(failed)} | duration={int(time.time()-start)}s")
    return {"queue_id": queue_id, "processed_count": processed, "failed": failed}

# ---------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    logger.info(f"Starting NDVI Land Worker v6.1 — limit={args.limit}")
    try:
        jobs = supabase.table("ndvi_request_queue").select("*").eq("status", "queued").limit(args.limit).execute()
        for job in jobs.data or []:
            queue_id, tenant_id, land_ids = job["id"], job["tenant_id"], job["land_ids"]
            supabase.table("ndvi_request_queue").update({"status": "processing"}).eq("id", queue_id).execute()
            process_request_sync(queue_id, tenant_id, land_ids)
    except Exception:
        logger.exception("Fatal NDVI worker error")
    logger.info("NDVI Land Worker v6.1 finished.")
