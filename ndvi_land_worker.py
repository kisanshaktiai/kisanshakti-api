"""
ndvi_land_worker_v7_async.py
NDVI Land Worker v7.1 — Streaming + Async Orchestration (Threadpool)
- Streams COGs from Backblaze B2 (no full downloads)
- Falls back to compute NDVI from B04/B08
- Parallel processing of lands per queue using asyncio + ThreadPoolExecutor
- Writes to ndvi_data, ndvi_micro_tiles, lands, ndvi_processing_logs
"""

import os
import io
import json
import time
import logging
import datetime
import traceback
import argparse
from typing import List, Optional, Dict, Any, Tuple

import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.errors import RasterioIOError
from rasterio.merge import merge as rio_merge
from rasterio.io import MemoryFile
from shapely.geometry import shape
from PIL import Image
import matplotlib.cm as cm

from supabase import create_client, Client
from b2sdk.v2 import InMemoryAccountInfo, B2Api

# ----------------------------
# Logging & Basic Config
# ----------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ndvi-worker-async")

# ----------------------------
# Environment / Config
# ----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
B2_APP_KEY_ID = os.getenv("B2_APP_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME", "kisanshakti-ndvi-tiles")
SUPABASE_NDVI_BUCKET = os.getenv("SUPABASE_NDVI_BUCKET", "ndvi-thumbnails")

# Concurrency tuning
MAX_CONCURRENT_LANDS = int(os.getenv("MAX_CONCURRENT_LANDS", "8"))  # concurrent lands processed
THREAD_POOL_WORKERS = int(os.getenv("THREAD_POOL_WORKERS", "12"))  # threadpool for rasterio/db calls

# Validate env
if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
if not B2_APP_KEY_ID or not B2_APP_KEY:
    raise RuntimeError("Missing B2_APP_KEY_ID or B2_APP_KEY")

# ----------------------------
# Clients init (sync) - will be used in threadpool
# ----------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

b2_info = InMemoryAccountInfo()
b2_api = B2Api(b2_info)
b2_api.authorize_account("production", B2_APP_KEY_ID, B2_APP_KEY)
try:
    b2_bucket = b2_api.get_bucket_by_name(B2_BUCKET_NAME)
    logger.info(f"✅ B2 bucket accessible: {B2_BUCKET_NAME}")
except Exception as e:
    logger.error(f"❌ Could not access B2 bucket '{B2_BUCKET_NAME}': {e}")
    b2_bucket = None

# ----------------------------
# Utilities
# ----------------------------
def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

def calculate_ndvi_from_bands(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    np.seterr(divide='ignore', invalid='ignore')
    red_f = red.astype(np.float32)
    nir_f = nir.astype(np.float32)
    ndvi = (nir_f - red_f) / (nir_f + red_f)
    ndvi = np.clip(ndvi, -1, 1)
    ndvi[np.isnan(ndvi)] = -1
    return ndvi

def calculate_statistics(ndvi: np.ndarray) -> Dict[str, Any]:
    valid_mask = (ndvi >= -1) & (ndvi <= 1) & ~np.isnan(ndvi)
    valid_pixels = ndvi[valid_mask]
    total_pixels = int(ndvi.size)
    valid_count = int(valid_pixels.size)
    if valid_count == 0:
        return {"mean": None, "min": None, "max": None, "std": None,
                "valid_pixels": 0, "total_pixels": total_pixels, "coverage": 0.0}
    mean_val = float(np.mean(valid_pixels))
    return {
        "mean": mean_val,
        "min": float(np.min(valid_pixels)),
        "max": float(np.max(valid_pixels)),
        "std": float(np.std(valid_pixels)),
        "valid_pixels": valid_count,
        "total_pixels": total_pixels,
        "coverage": float(valid_count / total_pixels * 100)
    }

def create_colorized_thumbnail(ndvi_array: np.ndarray, max_size: int = 512) -> bytes:
    norm = np.clip((ndvi_array + 1) / 2, 0, 1)
    cmap = cm.get_cmap("RdYlGn")
    rgba = (cmap(norm) * 255).astype(np.uint8)
    rgba[..., 3][ndvi_array == -1] = 0
    img = Image.fromarray(rgba, mode="RGBA")
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.getvalue()

def upload_thumbnail_to_supabase_sync(land_id: str, date: str, png_bytes: bytes) -> Optional[str]:
    """Blocking upload to supabase storage (run in threadpool)."""
    try:
        path = f"{land_id}/{date}/ndvi_colorized.png"
        # Supabase client expects file-like or bytes; method signature varies by client version.
        # Using .upload(path, file, options) style.
        supabase.storage.from_(SUPABASE_NDVI_BUCKET).upload(
            path=path,
            file=png_bytes,
            file_options={"content-type": "image/png", "upsert": "true"}
        )
        public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_NDVI_BUCKET}/{path}"
        logger.debug(f"Uploaded thumbnail for {land_id} -> {public_url}")
        return public_url
    except Exception as e:
        logger.error(f"Thumbnail upload failed (land={land_id}): {e}")
        return None

# ----------------------------
# Streaming NDVI (blocking functions run in threadpool)
# ----------------------------
def _make_b2_public_base_url() -> str:
    # Backblaze public file URL pattern; adjust if you use a CDN or private signed urls
    # Example: https://f000.backblazeb2.com/file/<bucket>/<path>
    return f"https://f000.backblazeb2.com/file/{B2_BUCKET_NAME}/tiles"

def stream_ndvi_blocking(tile_id: str, acq_date: str, land_geom: dict) -> Optional[np.ndarray]:
    """
    Blocking: Try to read precomputed NDVI COG via HTTP range reads. If missing, read B04/B08 and compute.
    Returns ndvi numpy array cropped to land_geom or None.
    """
    base_url = _make_b2_public_base_url()
    ndvi_url = f"{base_url}/ndvi/{tile_id}/{acq_date}/ndvi.tif"
    red_url = f"{base_url}/raw/{tile_id}/{acq_date}/B04.tif"
    nir_url = f"{base_url}/raw/{tile_id}/{acq_date}/B08.tif"

    # Try NDVI directly
    try:
        with rasterio.Env():
            with rasterio.open(ndvi_url) as src:
                ndvi_clip, _ = mask(src, [land_geom], crop=True, all_touched=True)
                if ndvi_clip.size == 0:
                    raise ValueError("Empty NDVI clip")
                return ndvi_clip[0]
    except RasterioIOError:
        logger.debug(f"NDVI COG not found: {ndvi_url}")
    except Exception as e:
        logger.warning(f"NDVI COG read error ({ndvi_url}): {e}")

    # Fallback to B04/B08
    try:
        with rasterio.Env():
            with rasterio.open(red_url) as red_src, rasterio.open(nir_url) as nir_src:
                red_clip, _ = mask(red_src, [land_geom], crop=True, all_touched=True)
                nir_clip, _ = mask(nir_src, [land_geom], crop=True, all_touched=True)
                if red_clip.size == 0 or nir_clip.size == 0:
                    logger.debug(f"No overlap for tile {tile_id} / date {acq_date}")
                    return None
                red = red_clip[0].astype(np.float32)
                nir = nir_clip[0].astype(np.float32)
                ndvi = calculate_ndvi_from_bands(red, nir)
                return ndvi
    except RasterioIOError as e:
        logger.debug(f"Red/NIR COG not found for {tile_id}: {e}")
    except Exception as e:
        logger.error(f"NDVI compute (B04/B08) failed for {tile_id}: {e}")

    return None

# ----------------------------
# DB helpers (blocking) to run in threadpool
# ----------------------------
def get_latest_tile_date_sync(tile_id: str) -> Optional[str]:
    """Query satellite_tiles for latest completed acquisition_date for tile"""
    try:
        resp = supabase.table("satellite_tiles").select("acquisition_date").eq("tile_id", tile_id).eq("status", "ready").order("acquisition_date", desc=True).limit(1).execute()
        if resp.data:
            return resp.data[0]["acquisition_date"]
    except Exception as e:
        logger.debug(f"get_latest_tile_date_sync failed for {tile_id}: {e}")
    return None

def upsert_ndvi_data_sync(record: Dict[str, Any]) -> None:
    try:
        # upsert - on_conflict depends on client; using generic upsert
        supabase.table("ndvi_data").upsert(record).execute()
    except Exception as e:
        logger.error(f"ndvi_data upsert failed: {e}")

def upsert_micro_tile_sync(record: Dict[str, Any]) -> None:
    try:
        supabase.table("ndvi_micro_tiles").upsert(record).execute()
    except Exception as e:
        logger.error(f"ndvi_micro_tiles upsert failed: {e}")

def update_land_sync(land_id: str, update_payload: Dict[str, Any]) -> None:
    try:
        supabase.table("lands").update(update_payload).eq("id", land_id).execute()
    except Exception as e:
        logger.error(f"lands.update failed for {land_id}: {e}")

def insert_processing_log_sync(record: Dict[str, Any]) -> None:
    try:
        supabase.table("ndvi_processing_logs").insert(record).execute()
    except Exception as e:
        logger.error(f"ndvi_processing_logs insert failed: {e}")

# ----------------------------
# Single land processing (async wrapper)
# ----------------------------
async def process_single_land_async(
    land: Dict[str, Any],
    tile_ids: Optional[List[str]],
    acquisition_date_override: Optional[str],
    executor: ThreadPoolExecutor
) -> Dict[str, Any]:
    """
    Orchestrates processing for a single land record:
    - finds tile dates
    - streams NDVI / computes fallback
    - computes stats, generates thumbnail
    - uploads thumbnail & writes DB records
    """
    loop = asyncio.get_running_loop()
    land_id = land["id"]
    tenant_id = land["tenant_id"]
    result: Dict[str, Any] = {"land_id": land_id, "success": False, "error": None, "stats": None}

    try:
        geom_raw = land.get("boundary_polygon_old") or land.get("boundary")
        if not geom_raw:
            raise ValueError("Missing geometry")
        geometry = json.loads(geom_raw) if isinstance(geom_raw, str) else geom_raw

        # Determine tiles to process
        if tile_ids:
            tiles_to_try = tile_ids
        else:
            # call RPC or fallback: query satellite_tiles for tiles overlapping bbox? For simplicity, use mgrs_tiles intersection RPC if available.
            # Here we attempt to call a RPC 'get_intersecting_tiles' — if not available, fallback to mgrs_tiles intersection by bbox.
            try:
                resp = supabase.rpc("get_intersecting_tiles", {"land_geom": json.dumps(geometry)}).execute()
                tiles_to_try = [t["tile_id"] for t in (resp.data or [])] if getattr(resp, "data", None) else []
            except Exception:
                # fallback get candidate tiles from mgrs_tiles (simple bbox overlap)
                candidate_resp = supabase.table("mgrs_tiles").select("tile_id, bbox").execute()
                candidate_tiles = candidate_resp.data or []
                tiles_to_try = []
                land_shape = shape(geometry)
                for t in candidate_tiles:
                    try:
                        t_bbox = t.get("bbox")
                        if not t_bbox:
                            continue
                        t_geom = t_bbox if isinstance(t_bbox, dict) else json.loads(t_bbox)
                        tile_shape = shape(t_geom)
                        if land_shape.intersects(tile_shape):
                            tiles_to_try.append(t["tile_id"])
                    except Exception:
                        continue

        if not tiles_to_try:
            raise ValueError("No intersecting tiles found")

        ndvi_clips: List[np.ndarray] = []
        # For each tile, find latest date (or use override) and stream NDVI
        for tile_id in tiles_to_try:
            acq_date = acquisition_date_override or await loop.run_in_executor(executor, get_latest_tile_date_sync, tile_id)
            if not acq_date:
                logger.debug(f"No acquisition date found for tile {tile_id}, skipping")
                continue

            ndvi = await loop.run_in_executor(executor, stream_ndvi_blocking, tile_id, acq_date, geometry)
            if ndvi is None:
                logger.debug(f"No NDVI for tile {tile_id}/{acq_date}")
                continue
            ndvi_clips.append(ndvi)

        if not ndvi_clips:
            raise ValueError("No NDVI data extracted from intersecting tiles")

        # Merge arrays if more than one clip — assume same resolution/CRS for simplicity
        if len(ndvi_clips) == 1:
            final_ndvi = ndvi_clips[0]
        else:
            try:
                # use rasterio.merge (expects datasets) - create MemoryFile datasets around arrays
                datasets = []
                for arr in ndvi_clips:
                    # Build minimal dataset metadata using MemoryFile (no geotransform provided - merging may be coarse)
                    mem = MemoryFile()
                    ds = mem.open(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=arr.dtype)
                    ds.write(arr, 1)
                    datasets.append(ds)
                mosaic, out_trans = rio_merge(datasets)
                for ds in datasets:
                    ds.close()
                final_ndvi = mosaic[0]
            except Exception:
                # fallback to averaged stacking (simpler)
                final_ndvi = np.nanmean(np.stack(ndvi_clips), axis=0)

        # Stats
        stats = calculate_statistics(final_ndvi)
        if stats["valid_pixels"] == 0:
            raise ValueError("No valid NDVI pixels after processing")

        # Thumbnail
        acq_date_for_record = acquisition_date_override or datetime.date.today().isoformat()
        thumbnail_bytes = await loop.run_in_executor(executor, create_colorized_thumbnail, final_ndvi)
        # Upload thumbnail (blocking)
        thumbnail_url = await loop.run_in_executor(executor, upload_thumbnail_to_supabase_sync, land_id, acq_date_for_record, thumbnail_bytes)

        # Prepare DB records
        ndvi_data_record = {
            "tenant_id": tenant_id,
            "land_id": land_id,
            "date": acq_date_for_record,
            "ndvi_value": stats["mean"],
            "mean_ndvi": stats["mean"],
            "min_ndvi": stats["min"],
            "max_ndvi": stats["max"],
            "ndvi_std": stats["std"],
            "valid_pixels": stats["valid_pixels"],
            "total_pixels": stats["total_pixels"],
            "coverage_percentage": stats["coverage"],
            "image_url": thumbnail_url,
            "satellite_source": "SENTINEL-2",
            "processing_level": "L2A",
            "created_at": now_iso(),
            "computed_at": now_iso()
        }

        micro_tile_record = {
            "tenant_id": tenant_id,
            "land_id": land_id,
            "farmer_id": land.get("farmer_id"),
            "acquisition_date": acq_date_for_record,
            "ndvi_mean": stats["mean"],
            "ndvi_min": stats["min"],
            "ndvi_max": stats["max"],
            "ndvi_std_dev": stats["std"],
            "ndvi_thumbnail_url": thumbnail_url,
            "bbox": geometry if isinstance(geometry, dict) else None,
            "cloud_cover": 0,
            "created_at": now_iso()
        }

        # Write DB records (blocking - run in executor)
        await loop.run_in_executor(executor, upsert_ndvi_data_sync, ndvi_data_record)
        await loop.run_in_executor(executor, upsert_micro_tile_sync, micro_tile_record)
        await loop.run_in_executor(executor, update_land_sync, land_id, {
            "last_ndvi_value": stats["mean"],
            "last_ndvi_calculation": acq_date_for_record,
            "ndvi_thumbnail_url": thumbnail_url,
            "updated_at": now_iso()
        })

        result["success"] = True
        result["stats"] = stats
        result["thumbnail_url"] = thumbnail_url
        logger.info(f"✅ Processed land {land_id} (mean={stats['mean']:.3f}, coverage={stats['coverage']:.1f}%)")

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"❌ Failed land {land.get('id')}: {e}\n{tb}")
        result["error"] = str(e)
        # log into ndvi_processing_logs (best-effort)
        try:
            log_record = {
                "tenant_id": land.get("tenant_id"),
                "land_id": land.get("id"),
                "processing_step": "ndvi_async",
                "step_status": "failed",
                "error_message": str(e)[:500],
                "error_details": {"traceback": tb[:1000]},
                "created_at": now_iso()
            }
            await asyncio.get_running_loop().run_in_executor(executor, insert_processing_log_sync, log_record)
        except Exception:
            logger.debug("Failed to write processing log")

    return result

# ----------------------------
# Orchestrator for a queue item
# ----------------------------
async def process_request_async(queue_id: str, tenant_id: str, land_ids: List[str], tile_ids: Optional[List[str]] = None):
    logger.info(f"🚀 Starting async processing: queue={queue_id} tenant={tenant_id} lands={len(land_ids)}")
    start_ts = time.time()

    # Fetch lands records
    try:
        resp = supabase.table("lands").select("*").eq("tenant_id", tenant_id).in_("id", land_ids).execute()
        lands = resp.data or []
    except Exception as e:
        logger.error(f"Failed to fetch lands for tenant {tenant_id}: {e}")
        lands = []

    if not lands:
        logger.warning("No lands found to process")
        return {"queue_id": queue_id, "processed_count": 0, "failed_count": len(land_ids)}

    # ThreadPool for blocking ops
    executor = ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS)
    sem = asyncio.Semaphore(MAX_CONCURRENT_LANDS)

    async def _process_with_semaphore(land):
        async with asyncio.Lock():
            # small pause to prevent burst storm
            pass
        async with sem:  # limit concurrent land processing
            return await process_single_land_async(land, tile_ids, None, executor)

    tasks = [asyncio.create_task(_process_with_semaphore(land)) for land in lands]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    processed = sum(1 for r in results if r.get("success"))
    failed = [r for r in results if not r.get("success")]

    duration_ms = int((time.time() - start_ts) * 1000)
    logger.info(f"🏁 Queue {queue_id} finished: processed={processed}/{len(lands)} duration={duration_ms}ms failed={len(failed)}")

    # Update queue status (blocking) - mark completed or failed
    final_status = "completed" if processed > 0 else "failed"
    try:
        supabase.table("ndvi_request_queue").update({
            "status": final_status,
            "processed_count": processed,
            "failed_count": len(failed),
            "processing_duration_ms": duration_ms,
            "completed_at": now_iso()
        }).eq("id", queue_id).execute()
    except Exception as e:
        logger.error(f"Failed to update queue status for {queue_id}: {e}")

    return {
        "queue_id": queue_id,
        "processed_count": processed,
        "failed_count": len(failed),
        "failed": failed,
        "duration_ms": duration_ms
    }

# ----------------------------
# Cron style runner (sync entry)
# ----------------------------
def run_cron(limit: int = 10, max_retries: int = 3):
    """
    Synchronous entrypoint to pick queued requests and run async processing for each.
    """
    logger.info("🔄 NDVI async worker starting (cron)")
    # fetch queued items
    try:
        queue_resp = supabase.table("ndvi_request_queue").select("*").eq("status", "queued").order("created_at", desc=False).limit(limit).execute()
        items = queue_resp.data or []
    except Exception as e:
        logger.error(f"Failed to fetch queue items: {e}")
        items = []

    async def _handle_item(item):
        queue_id = item["id"]
        tenant_id = item["tenant_id"]
        land_ids = item.get("land_ids", [])
        tile_id = item.get("tile_id")
        retry_count = item.get("retry_count", 0)

        if retry_count >= max_retries:
            logger.warning(f"Max retries reached for {queue_id}, marking failed")
            supabase.table("ndvi_request_queue").update({
                "status": "failed",
                "last_error": f"Max retries ({max_retries}) exceeded",
                "completed_at": now_iso()
            }).eq("id", queue_id).execute()
            return

        # mark processing
        supabase.table("ndvi_request_queue").update({
            "status": "processing",
            "started_at": now_iso(),
            "retry_count": retry_count + 1
        }).eq("id", queue_id).execute()

        # run async processing
        try:
            result = await process_request_async(queue_id, tenant_id, land_ids, [tile_id] if tile_id else None)
            logger.info(f"Queue {queue_id} result: {result}")
        except Exception as e:
            logger.exception(f"Failed processing queue {queue_id}: {e}")
            # re-queue
            supabase.table("ndvi_request_queue").update({
                "status": "queued",
                "last_error": str(e)[:500]
            }).eq("id", queue_id).execute()

    # Run the asyncio loop for all items sequentially (could be parallelized if desired)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [ _handle_item(item) for item in items ]
    loop.run_until_complete(asyncio.gather(*tasks))
    loop.close()
    logger.info("✅ Cron run finished")

# ----------------------------
# Main entrypoint (CLI)
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NDVI Land Worker v7 Async")
    parser.add_argument("--mode", choices=["cron", "single"], default="cron")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--queue-id", type=str, help="Process a single queue id (use with --mode single)")
    args = parser.parse_args()

    logger.info(f"Starting NDVI Land Worker v7 async mode={args.mode}")

    if args.mode == "single" and args.queue_id:
        # process a single queue item synchronously using the async orchestrator
        try:
            queue_item = supabase.table("ndvi_request_queue").select("*").eq("id", args.queue_id).single().execute()
            if not queue_item.data:
                logger.error(f"Queue id {args.queue_id} not found")
            else:
                item = queue_item.data
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(process_request_async(item["id"], item["tenant_id"], item.get("land_ids", []), [item.get("tile_id")] if item.get("tile_id") else None))
                loop.close()
        except Exception as e:
            logger.exception(f"Error processing single queue id: {e}")
    else:
        run_cron(limit=args.limit)

    logger.info("Worker finished")
