"""
NDVI Land Worker v7.0 — COG Streaming Edition
----------------------------------------------
Processes NDVI from Sentinel-2 tiles stored in Backblaze B2 (COG format).

✅ Streams directly via HTTP range requests — no full downloads
✅ Falls back to B04/B08 calculation if NDVI COG missing
✅ Multi-tile support
✅ Supabase-integrated for lands, ndvi_data, ndvi_micro_tiles
✅ Thumbnail and statistics generation
"""

# =============================================================================
# Imports
# =============================================================================
import os
import io
import json
import time
import logging
import datetime
import traceback
import argparse
from typing import List, Optional, Dict, Any

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.errors import RasterioIOError
from rasterio.merge import merge
from rasterio.io import MemoryFile
from shapely.geometry import shape
from PIL import Image
import matplotlib.cm as cm
from supabase import create_client, Client
from b2sdk.v2 import InMemoryAccountInfo, B2Api

# =============================================================================
# Logging & Config
# =============================================================================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ndvi-worker")

# Environment
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
B2_APP_KEY_ID = os.getenv("B2_APP_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME", "kisanshakti-ndvi-tiles")
SUPABASE_NDVI_BUCKET = os.getenv("SUPABASE_NDVI_BUCKET", "ndvi-thumbnails")

# Validate environment
if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("❌ Missing Supabase credentials")
if not B2_APP_KEY_ID or not B2_APP_KEY:
    raise RuntimeError("❌ Missing B2 credentials")

# Initialize Supabase + B2 clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
b2_info = InMemoryAccountInfo()
b2_api = B2Api(b2_info)
b2_api.authorize_account("production", B2_APP_KEY_ID, B2_APP_KEY)
b2_bucket = b2_api.get_bucket_by_name(B2_BUCKET_NAME)

logger.info(f"✅ Initialized Supabase + B2 (bucket={B2_BUCKET_NAME})")

# =============================================================================
# Utility Functions
# =============================================================================
def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat()

def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    np.seterr(divide='ignore', invalid='ignore')
    ndvi = (nir - red) / (nir + red)
    ndvi = np.clip(ndvi, -1, 1)
    ndvi[np.isnan(ndvi)] = -1
    return ndvi

def calculate_statistics(ndvi_array: np.ndarray) -> Dict[str, Any]:
    valid_mask = (ndvi_array >= -1) & (ndvi_array <= 1) & ~np.isnan(ndvi_array)
    valid_pixels = ndvi_array[valid_mask]
    total_pixels = int(ndvi_array.size)
    valid_count = int(valid_pixels.size)
    if valid_count == 0:
        return {"mean": None, "min": None, "max": None, "std": None,
                "valid_pixels": 0, "total_pixels": total_pixels,
                "coverage": 0.0, "health_category": "no_data"}
    mean_ndvi = float(np.mean(valid_pixels))
    if mean_ndvi < 0.2:
        health = "poor"
    elif mean_ndvi < 0.4:
        health = "fair"
    elif mean_ndvi < 0.6:
        health = "good"
    else:
        health = "excellent"
    return {
        "mean": float(mean_ndvi),
        "min": float(np.min(valid_pixels)),
        "max": float(np.max(valid_pixels)),
        "std": float(np.std(valid_pixels)),
        "valid_pixels": valid_count,
        "total_pixels": total_pixels,
        "coverage": float(valid_count / total_pixels * 100),
        "health_category": health
    }

def create_colorized_thumbnail(ndvi_array: np.ndarray, max_size: int = 512) -> bytes:
    normalized = np.clip((ndvi_array + 1) / 2, 0, 1)
    cmap = cm.get_cmap("RdYlGn")
    rgba = (cmap(normalized) * 255).astype(np.uint8)
    rgba[..., 3][ndvi_array == -1] = 0
    img = Image.fromarray(rgba, mode="RGBA")
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)
    return buffer.getvalue()

def upload_thumbnail_to_supabase(land_id: str, date: str, png_bytes: bytes) -> Optional[str]:
    try:
        path = f"{land_id}/{date}/ndvi_colorized.png"
        supabase.storage.from_(SUPABASE_NDVI_BUCKET).upload(
            path=path, file=png_bytes,
            file_options={"content-type": "image/png", "upsert": "true"}
        )
        public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_NDVI_BUCKET}/{path}"
        return public_url
    except Exception as e:
        logger.error(f"❌ Thumbnail upload failed for {land_id}: {e}")
        return None

# =============================================================================
# STREAMING NDVI LOGIC (no download)
# =============================================================================
def stream_ndvi_from_b2(tile_id: str, acq_date: str, land_geom: dict) -> Optional[np.ndarray]:
    """
    Streams NDVI or computes from B04/B08 directly from Backblaze B2 (COG-based).
    """
    base_url = f"https://f000.backblazeb2.com/file/{B2_BUCKET_NAME}/tiles"
    ndvi_url = f"{base_url}/ndvi/{tile_id}/{acq_date}/ndvi.tif"
    red_url  = f"{base_url}/raw/{tile_id}/{acq_date}/B04.tif"
    nir_url  = f"{base_url}/raw/{tile_id}/{acq_date}/B08.tif"

    # Try precomputed NDVI
    try:
        with rasterio.Env():
            with rasterio.open(ndvi_url) as src:
                ndvi_clip, _ = mask(src, [land_geom], crop=True, all_touched=True)
                if ndvi_clip.size == 0:
                    raise ValueError("Empty NDVI clip")
                logger.debug(f"✅ Streamed NDVI COG: {ndvi_url}")
                return ndvi_clip[0]
    except RasterioIOError:
        logger.warning(f"⚠️ NDVI COG not found for {tile_id}. Recomputing.")
    except Exception as e:
        logger.warning(f"⚠️ NDVI COG read failed: {e}")

    # Fallback compute from Red + NIR
    try:
        with rasterio.Env():
            with rasterio.open(red_url) as red_src, rasterio.open(nir_url) as nir_src:
                red_clip, _ = mask(red_src, [land_geom], crop=True, all_touched=True)
                nir_clip, _ = mask(nir_src, [land_geom], crop=True, all_touched=True)
                if red_clip.size == 0 or nir_clip.size == 0:
                    raise ValueError(f"No overlap in {tile_id}")
                red = red_clip[0].astype(np.float32)
                nir = nir_clip[0].astype(np.float32)
                ndvi = calculate_ndvi(red, nir)
                return ndvi
    except Exception as e:
        logger.error(f"❌ Failed to compute NDVI from raw bands: {e}")
        return None

# =============================================================================
# Core Processing
# =============================================================================
def process_single_land(land: Dict[str, Any], tile_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    land_id = land["id"]
    tenant_id = land["tenant_id"]
    result = {"land_id": land_id, "success": False, "error": None, "stats": None}

    try:
        geom_raw = land.get("boundary_polygon_old") or land.get("boundary_geom")
        if not geom_raw:
            raise ValueError("Land has no geometry")
        geometry = json.loads(geom_raw) if isinstance(geom_raw, str) else geom_raw

        # Get intersecting tiles
        tiles = tile_ids or [t["tile_id"] for t in supabase.table("mgrs_tiles").select("tile_id").execute().data]
        ndvi_clips = []
        acquisition_date = datetime.date.today().isoformat()

        for tile_id in tiles:
            ndvi_clip = stream_ndvi_from_b2(tile_id, acquisition_date, geometry)
            if ndvi_clip is not None:
                ndvi_clips.append(ndvi_clip)

        if not ndvi_clips:
            raise ValueError("No NDVI data available")

        final_ndvi = merge(ndvi_clips)[0] if len(ndvi_clips) > 1 else ndvi_clips[0]
        stats = calculate_statistics(final_ndvi)
        if stats["valid_pixels"] == 0:
            raise ValueError("No valid NDVI pixels")

        thumbnail_bytes = create_colorized_thumbnail(final_ndvi)
        thumbnail_url = upload_thumbnail_to_supabase(land_id, acquisition_date, thumbnail_bytes)

        ndvi_data = {
            "tenant_id": tenant_id,
            "land_id": land_id,
            "date": acquisition_date,
            "mean_ndvi": stats["mean"],
            "min_ndvi": stats["min"],
            "max_ndvi": stats["max"],
            "ndvi_std": stats["std"],
            "coverage_percentage": stats["coverage"],
            "image_url": thumbnail_url,
            "created_at": now_iso()
        }
        supabase.table("ndvi_data").upsert(ndvi_data, on_conflict="land_id,date").execute()

        supabase.table("lands").update({
            "last_ndvi_value": stats["mean"],
            "ndvi_thumbnail_url": thumbnail_url,
            "last_ndvi_calculation": acquisition_date,
            "updated_at": now_iso()
        }).eq("id", land_id).execute()

        result["success"] = True
        result["stats"] = stats
        result["thumbnail_url"] = thumbnail_url
        logger.info(f"✅ Land {land_id} processed successfully")

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"❌ Land {land_id} failed: {e}")
        supabase.table("ndvi_processing_logs").insert({
            "tenant_id": tenant_id, "land_id": land_id,
            "processing_step": "ndvi", "step_status": "failed",
            "error_message": str(e), "created_at": now_iso()
        }).execute()

    return result

# =============================================================================
# Queue Processing
# =============================================================================
def process_request_sync(queue_id: str, tenant_id: str, land_ids: List[str], tile_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    logger.info(f"🚀 Starting NDVI request {queue_id} for tenant {tenant_id}")
    start = time.time()

    lands_resp = supabase.table("lands").select("*").eq("tenant_id", tenant_id).in_("id", land_ids).execute()
    lands = lands_resp.data or []
    if not lands:
        return {"queue_id": queue_id, "processed_count": 0, "failed_count": len(land_ids)}

    processed, failed = 0, []
    for land in lands:
        res = process_single_land(land, tile_ids)
        if res["success"]:
            processed += 1
        else:
            failed.append(res)

    dur = int((time.time() - start) * 1000)
    logger.info(f"🏁 Queue {queue_id} complete: {processed}/{len(lands)} processed in {dur}ms")

    return {"queue_id": queue_id, "processed_count": processed, "failed_count": len(failed), "duration_ms": dur}

# =============================================================================
# Main Entrypoint
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NDVI Land Worker v7.0 — Streaming Edition")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--mode", choices=["single", "cron"], default="single")
    parser.add_argument("--queue-id", type=str)
    args = parser.parse_args()

    logger.info(f"🛰️ Starting NDVI Worker v7.0 mode={args.mode}")

    if args.mode == "single" and args.queue_id:
        queue = supabase.table("ndvi_request_queue").select("*").eq("id", args.queue_id).single().execute()
        if queue.data:
            process_request_sync(queue.data["id"], queue.data["tenant_id"], queue.data["land_ids"])
    else:
        logger.info("Run with --mode single --queue-id <id> for manual testing")

    logger.info("✅ Worker finished")
