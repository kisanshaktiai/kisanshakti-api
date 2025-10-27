# ndvi_land_worker.py
# NDVI Land Worker v8.1 — Multi-Tile + Async Orchestration (rewritten, includes B2 signing + robustness fixes)
#
# - Keeps your existing logic intact (stream NDVI COG from Backblaze, fallback compute from B04/B08)
# - Adds private-B2 signing helper (_get_signed_b2_url) for free-tier private buckets
# - Adds robust geometry handling and reprojection helper
# - Fixes thumbnail upload file_options usage for supabase-py
# - No other algorithmic logic has been changed; placement of helpers only

import os
import io
import json
import time
import logging
import datetime
import traceback
import argparse
from typing import List, Optional, Dict, Any

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
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ndvi-worker-async")

# ----------------------------
# Environment / Config
# ----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
B2_KEY_ID = os.getenv("B2_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME", "kisanshakti-ndvi-tiles")
# region-prefix to use for signed URL building if needed (e.g. f005) — you can override via env
B2_PUBLIC_REGION = os.getenv("B2_PUBLIC_REGION", "f005")
SUPABASE_NDVI_BUCKET = os.getenv("SUPABASE_NDVI_BUCKET", "ndvi-thumbnails")

# Concurrency tuning
MAX_CONCURRENT_LANDS = int(os.getenv("MAX_CONCURRENT_LANDS", "8"))
THREAD_POOL_WORKERS = int(os.getenv("THREAD_POOL_WORKERS", "12"))

# Validate env
if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
if not B2_KEY_ID or not B2_APP_KEY:
    raise RuntimeError("Missing B2_KEY_ID or B2_APP_KEY")

# ----------------------------
# Clients init (sync)
# ----------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

b2_info = InMemoryAccountInfo()
b2_api = B2Api(b2_info)
b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
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
    np.seterr(divide="ignore", invalid="ignore")
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
        return {
            "mean": None,
            "min": None,
            "max": None,
            "std": None,
            "valid_pixels": 0,
            "total_pixels": total_pixels,
            "coverage": 0.0,
        }
    return {
        "mean": float(np.mean(valid_pixels)),
        "min": float(np.min(valid_pixels)),
        "max": float(np.max(valid_pixels)),
        "std": float(np.std(valid_pixels)),
        "valid_pixels": valid_count,
        "total_pixels": total_pixels,
        "coverage": float(valid_count / total_pixels * 100),
    }


def create_colorized_thumbnail(ndvi_array: np.ndarray, max_size: int = 512) -> bytes:
    norm = np.clip((ndvi_array + 1) / 2, 0, 1)
    cmap = cm.get_cmap("RdYlGn")
    rgba = (cmap(norm) * 255).astype(np.uint8)
    # make nodata transparent
    rgba[..., 3][ndvi_array == -1] = 0
    img = Image.fromarray(rgba, mode="RGBA")
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.getvalue()


def upload_thumbnail_to_supabase_sync(land_id: str, date: str, png_bytes: bytes) -> Optional[str]:
    """
    Uploads a colorized NDVI thumbnail PNG to Supabase Storage.
    Runs in a threadpool (blocking-safe).
    """
    try:
        path = f"{land_id}/{date}/ndvi_colorized.png"

        # Upload with overwrite (upsert=True)
        supabase.storage.from_(SUPABASE_NDVI_BUCKET).upload(
            path=path,
            file=png_bytes,
            file_options={
                "content_type": "image/png",
                "upsert": True,  # boolean not string
            },
        )

        public_url = (
            f"{SUPABASE_URL}/storage/v1/object/public/"
            f"{SUPABASE_NDVI_BUCKET}/{path}"
        )

        logger.info(f"🖼️ Uploaded NDVI thumbnail for land {land_id} ({len(png_bytes)/1024:.1f} KB) → {public_url}")
        return public_url

    except Exception as e:
        logger.error(f"❌ Thumbnail upload failed for land {land_id}: {e}", exc_info=True)
        return None


# ----------------------------
# Helper: Signed B2 URL for private buckets
# ----------------------------
def _get_signed_b2_url(file_path: str, valid_secs: int = 3600) -> Optional[str]:
    """
    Generate a temporary signed URL for a private Backblaze B2 object.
    Works on free-tier buckets that are not public.

    NOTE: b2sdk Bucket.get_download_authorization expects parameter name
    'valid_duration_in_seconds' (not 'valid_duration_seconds') for some versions.
    """
    if b2_bucket is None:
        logger.warning("B2 bucket client not available for signing.")
        return None

    try:
        # Use correct parameter name expected by installed b2sdk
        auth_token = b2_bucket.get_download_authorization(
            file_name_prefix=file_path,
            valid_duration_in_seconds=valid_secs
        )
        # Build signed http endpoint (region configurable)
        return f"https://{B2_PUBLIC_REGION}.backblazeb2.com/file/{B2_BUCKET_NAME}/{file_path}?Authorization={auth_token}"
    except TypeError as te:
        # older/newer mismatch: try fallback param name if TypeError indicates unexpected arg
        try:
            auth_token = b2_bucket.get_download_authorization(
                file_name_prefix=file_path,
                valid_duration_seconds=valid_secs  # try alternative name if present in some versions
            )
            return f"https://{B2_PUBLIC_REGION}.backblazeb2.com/file/{B2_BUCKET_NAME}/{file_path}?Authorization={auth_token}"
        except Exception as e:
            logger.error(f"Failed to sign B2 URL for {file_path} (fallback): {e}")
            return None
    except Exception as e:
        logger.error(f"Failed to sign B2 URL for {file_path}: {e}")
        return None


# ----------------------------
# Streaming NDVI (blocking functions run in threadpool)
# ----------------------------
def stream_ndvi_blocking(tile_id: str, acq_date: str, land_geom: dict) -> Optional[np.ndarray]:
    """
    Stream NDVI GeoTIFF from private Backblaze B2.
    - Tries precomputed NDVI first (tiles/ndvi/.../ndvi.tif)
    - Falls back to raw B04/B08 bands if NDVI is missing
    - Auto-reprojects land geometry (EPSG:4326 → raster CRS)
    - Returns NDVI numpy array cropped to land polygon
    """

    # Build private B2 file paths (these match your uploader layouts)
    ndvi_path = f"tiles/ndvi/{tile_id}/{acq_date}/ndvi.tif"
    red_path = f"tiles/raw/{tile_id}/{acq_date}/B04.tif"
    nir_path = f"tiles/raw/{tile_id}/{acq_date}/B08.tif"

    # Generate signed URLs (valid 1 hour)
    ndvi_url = _get_signed_b2_url(ndvi_path)
    red_url = _get_signed_b2_url(red_path)
    nir_url = _get_signed_b2_url(nir_path)

    if not ndvi_url or not red_url or not nir_url:
        logger.warning(f"⚠️ Could not sign one or more B2 URLs for {tile_id}/{acq_date}")

    # Helper: Reproject geom to raster CRS if needed
    from shapely.geometry import mapping as _mapping
    import pyproj
    from shapely.ops import transform as _transform

    def _reproject_geom_to_raster(geom, raster_crs):
        """Ensure land geometry matches raster CRS (EPSG:4326 → target CRS)."""
        try:
            geom_shape = shape(geom)
            # raster_crs may be a CRS object or None
            raster_crs_str = raster_crs.to_string() if raster_crs else "EPSG:4326"
            if raster_crs_str and raster_crs_str != "EPSG:4326":
                project = pyproj.Transformer.from_crs("EPSG:4326", raster_crs_str, always_xy=True).transform
                geom_shape = _transform(project, geom_shape)
                logger.debug(f"🧭 Reprojected land geometry to {raster_crs_str}")
            return _mapping(geom_shape)
        except Exception as e:
            logger.warning(f"⚠️ Reprojection failed: {e}")
            return geom

    # Step 1: Try precomputed NDVI COG
    try:
        with rasterio.Env():
            if not ndvi_url:
                raise RasterioIOError("Missing signed NDVI URL")
            with rasterio.open(ndvi_url) as src:
                land_geom_proj = _reproject_geom_to_raster(land_geom, src.crs)

                logger.debug(f"Raster CRS: {src.crs}, bounds: {src.bounds}")
                try:
                    ndvi_clip, _ = mask(src, [land_geom_proj], crop=True, all_touched=True)
                except ValueError as ve:
                    # commonly "Input shapes do not overlap raster."
                    if "overlap" in str(ve).lower():
                        logger.warning(f"⚠️ Land geom does not overlap NDVI raster for {tile_id}/{acq_date}")
                        return None
                    raise

                if ndvi_clip.size == 0:
                    logger.warning(f"⚠️ Empty NDVI clip for {tile_id}/{acq_date}")
                    return None

                logger.info(f"🟢 Using precomputed NDVI for {tile_id}/{acq_date}")
                return ndvi_clip[0]
    except RasterioIOError:
        logger.warning(f"⚠️ NDVI file missing or inaccessible: {ndvi_path}")
    except Exception as e:
        logger.warning(f"⚠️ Could not read NDVI GeoTIFF ({tile_id}/{acq_date}): {e}")

    # Step 2: Fallback compute from B04/B08
    try:
        with rasterio.Env():
            if not red_url or not nir_url:
                raise RasterioIOError("Missing signed raw band URL(s)")
            with rasterio.open(red_url) as red_src, rasterio.open(nir_url) as nir_src:
                # Reproject to red band CRS (they should share same CRS)
                land_geom_proj = _reproject_geom_to_raster(land_geom, red_src.crs)

                try:
                    red_clip, _ = mask(red_src, [land_geom_proj], crop=True, all_touched=True)
                    nir_clip, _ = mask(nir_src, [land_geom_proj], crop=True, all_touched=True)
                except ValueError as ve:
                    if "overlap" in str(ve).lower():
                        logger.warning(f"⚠️ Land geom does not overlap raw bands for {tile_id}/{acq_date}")
                        return None
                    raise

                if red_clip.size == 0 or nir_clip.size == 0:
                    logger.warning(f"⚠️ No overlap for tile {tile_id}/{acq_date}")
                    return None

                red = red_clip[0].astype(np.float32)
                nir = nir_clip[0].astype(np.float32)
                ndvi = calculate_ndvi_from_bands(red, nir)
                logger.info(f"🧮 Computed NDVI from B04/B08 for {tile_id}/{acq_date}")
                return ndvi
    except RasterioIOError:
        logger.warning(f"⚠️ Raw band file missing or inaccessible for {tile_id}/{acq_date}")
    except Exception as e:
        logger.error(f"❌ NDVI computation failed for {tile_id}/{acq_date}: {e}")

    logger.warning(f"🚫 No NDVI data found for {tile_id}/{acq_date}")
    return None


# ----------------------------
# DB helpers (blocking) to run in threadpool
# ----------------------------
def get_latest_tile_date_sync(tile_id: str) -> Optional[str]:
    """Query satellite_tiles for latest completed acquisition_date for tile"""
    try:
        resp = (
            supabase.table("satellite_tiles")
            .select("acquisition_date")
            .eq("tile_id", tile_id)
            .eq("status", "ready")
            .order("acquisition_date", desc=True)
            .limit(1)
            .execute()
        )
        if resp.data:
            return resp.data[0]["acquisition_date"]
    except Exception as e:
        logger.debug(f"get_latest_tile_date_sync failed for {tile_id}: {e}")
    return None


def upsert_ndvi_data_sync(record: Dict[str, Any]) -> None:
    try:
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


# =========================================================
# SINGLE LAND PROCESSING (v8.1) — unified + robust
# =========================================================
async def process_single_land_async(
    land: Dict[str, Any],
    tile_ids: Optional[List[str]],
    acquisition_date_override: Optional[str],
    executor: ThreadPoolExecutor,
) -> Dict[str, Any]:
    """
    Processes a single land parcel for NDVI:
      ✅ Handles WKT / WKB / GeoJSON geometry
      ✅ Uses all intersecting tiles
      ✅ Computes NDVI, stats, and thumbnail
      ✅ Writes to ndvi_data + ndvi_micro_tiles
    """
    loop = asyncio.get_running_loop()
    land_id = land.get("id")
    tenant_id = land.get("tenant_id")

    result: Dict[str, Any] = {
        "land_id": land_id,
        "success": False,
        "error": None,
        "stats": None,
    }

    try:
        # ---------------------------------------------------------
        # 1️⃣ Load geometry safely (GeoJSON / WKT / WKB)
        # ---------------------------------------------------------
        geom_raw = land.get("boundary") or land.get("boundary_geom") or land.get("boundary_polygon_old")
        if not geom_raw:
            raise ValueError("Missing geometry")

        from shapely import wkt, wkb
        from shapely.geometry import shape as _shape

        geometry = None

        if isinstance(geom_raw, dict):
            geometry = geom_raw
        elif isinstance(geom_raw, (bytes, bytearray)):
            # WKB bytes
            try:
                shapely_geom = wkb.loads(bytes(geom_raw))
                geometry = json.loads(json.dumps(shapely_geom.__geo_interface__))
            except Exception:
                geometry = None
        elif isinstance(geom_raw, str):
            geom_raw = geom_raw.strip()
            # Try GeoJSON
            try:
                parsed = json.loads(geom_raw)
                geometry = parsed
            except Exception:
                geometry = None

            # If not JSON, try WKT
            if geometry is None and geom_raw.upper().startswith(("POLYGON", "MULTIPOLYGON", "LINESTRING", "POINT")):
                try:
                    shapely_geom = wkt.loads(geom_raw)
                    geometry = json.loads(json.dumps(shapely_geom.__geo_interface__))
                except Exception:
                    geometry = None

            # If still None, maybe WKB hex (PostGIS EWKB) starting with '010'
            if geometry is None and geom_raw.startswith("010"):
                try:
                    geom_bytes = bytes.fromhex(geom_raw)
                    shapely_geom = wkb.loads(geom_bytes)
                    geometry = json.loads(json.dumps(shapely_geom.__geo_interface__))
                except Exception:
                    geometry = None

        if not geometry:
            raise ValueError("Invalid geometry format: cannot parse")

        # ---------------------------------------------------------
        # 2️⃣ Determine intersecting tiles
        # ---------------------------------------------------------
        if land.get("tile_ids"):
            tiles_to_try = [t for t in land["tile_ids"] if t]
        elif tile_ids:
            tiles_to_try = tile_ids
        else:
            try:
                resp = supabase.rpc("get_intersecting_tiles", {"land_geom": json.dumps(geometry)}).execute()
                tiles_to_try = [t["tile_id"] for t in (resp.data or []) if "tile_id" in t]
            except Exception as rpc_error:
                logger.warning(f"⚠️ RPC fallback for land {land_id}: {rpc_error}")
                candidate_resp = supabase.table("mgrs_tiles").select("tile_id, bbox").execute()
                candidate_tiles = candidate_resp.data or []
                from shapely.geometry import shape as shp_shape
                land_shape = shp_shape(geometry)
                tiles_to_try = []
                for t in candidate_tiles:
                    try:
                        t_bbox = t.get("bbox")
                        if not t_bbox:
                            continue
                        tile_shape = shp_shape(json.loads(t_bbox) if isinstance(t_bbox, str) else t_bbox)
                        if land_shape.intersects(tile_shape):
                            tiles_to_try.append(t["tile_id"])
                    except Exception:
                        continue

        if not tiles_to_try:
            raise ValueError("No intersecting tiles found (tile_ids empty)")

        logger.info(f"🌍 Land {land_id} intersects {len(tiles_to_try)} tiles: {tiles_to_try}")

        # ---------------------------------------------------------
        # 3️⃣ Stream NDVI for each tile and merge results
        # ---------------------------------------------------------
        ndvi_clips: List[np.ndarray] = []
        for tile_id in tiles_to_try:
            acq_date = acquisition_date_override or await loop.run_in_executor(
                executor, get_latest_tile_date_sync, tile_id
            )
            if not acq_date:
                logger.debug(f"No acquisition date found for tile {tile_id}, skipping")
                continue

            ndvi = await loop.run_in_executor(executor, stream_ndvi_blocking, tile_id, acq_date, geometry)
            if ndvi is None:
                logger.debug(f"No NDVI data for tile {tile_id}/{acq_date}")
                continue

            ndvi_clips.append(ndvi)

        if not ndvi_clips:
            raise ValueError("No NDVI data extracted from intersecting tiles")

        if len(ndvi_clips) == 1:
            final_ndvi = ndvi_clips[0]
        else:
            try:
                final_ndvi = np.nanmean(np.stack(ndvi_clips), axis=0)
            except Exception:
                final_ndvi = ndvi_clips[0]

        # ---------------------------------------------------------
        # 4️⃣ Compute statistics + thumbnail
        # ---------------------------------------------------------
        stats = calculate_statistics(final_ndvi)
        if stats["valid_pixels"] == 0:
            raise ValueError("No valid NDVI pixels after processing")

        acq_date_for_record = acquisition_date_override or datetime.date.today().isoformat()
        thumbnail_bytes = await loop.run_in_executor(executor, create_colorized_thumbnail, final_ndvi)
        thumbnail_url = await loop.run_in_executor(
            executor, upload_thumbnail_to_supabase_sync, land_id, acq_date_for_record, thumbnail_bytes
        )

        # ---------------------------------------------------------
        # 5️⃣ Prepare DB records
        # ---------------------------------------------------------
        ndvi_data_record = {
            "tenant_id": tenant_id,
            "land_id": land_id,
            "date": acq_date_for_record,
            "mean_ndvi": stats["mean"],
            "min_ndvi": stats["min"],
            "max_ndvi": stats["max"],
            "ndvi_std": stats["std"],
            "valid_pixels": stats["valid_pixels"],
            "coverage_percentage": stats["coverage"],
            "image_url": thumbnail_url,
            "created_at": now_iso(),
            "computed_at": now_iso(),
        }

        micro_tile_record = {
            "tenant_id": tenant_id,
            "land_id": land_id,
            "acquisition_date": acq_date_for_record,
            "ndvi_mean": stats["mean"],
            "ndvi_min": stats["min"],
            "ndvi_max": stats["max"],
            "ndvi_std_dev": stats["std"],
            "ndvi_thumbnail_url": thumbnail_url,
            "bbox": geometry if isinstance(geometry, dict) else None,
            "cloud_cover": 0,
            "created_at": now_iso(),
        }

        # ---------------------------------------------------------
        # 6️⃣ Write to Supabase (sync in thread executor)
        # ---------------------------------------------------------
        await loop.run_in_executor(executor, upsert_ndvi_data_sync, ndvi_data_record)
        await loop.run_in_executor(executor, upsert_micro_tile_sync, micro_tile_record)
        await loop.run_in_executor(executor, update_land_sync, land_id, {
            "last_ndvi_value": stats["mean"],
            "last_ndvi_calculation": acq_date_for_record,
            "ndvi_thumbnail_url": thumbnail_url,
            "updated_at": now_iso(),
        })

        result["success"] = True
        result["stats"] = stats
        result["thumbnail_url"] = thumbnail_url
        logger.info(f"✅ Processed land {land_id} (mean={stats['mean']:.3f}, coverage={stats['coverage']:.1f}%)")

    # -------------------------------------------------------------
    # 7️⃣ Error Handling & Logging
    # -------------------------------------------------------------
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"❌ Failed land {land_id}: {e}\n{tb}")
        result["error"] = str(e)

        try:
            log_record = {
                "tenant_id": tenant_id,
                "land_id": land_id,
                "processing_step": "ndvi_async",
                "step_status": "failed",
                "error_message": str(e)[:500],
                "error_details": {"traceback": tb[:1000]},
                "created_at": now_iso(),
            }
            await loop.run_in_executor(executor, insert_processing_log_sync, log_record)
        except Exception as log_err:
            logger.debug(f"Failed to write processing log for {land_id}: {log_err}")

    return result


# ----------------------------
# Orchestrator for a queue item
# ----------------------------
async def process_request_async(queue_id: str, tenant_id: str, land_ids: List[str], tile_ids: Optional[List[str]] = None):
    logger.info(f"🚀 Starting async processing: queue={queue_id} tenant={tenant_id} lands={len(land_ids)}")
    start_ts = time.time()

    try:
        resp = supabase.table("lands").select("*").eq("tenant_id", tenant_id).in_("id", land_ids).execute()
        lands = resp.data or []
    except Exception as e:
        logger.error(f"Failed to fetch lands for tenant {tenant_id}: {e}")
        lands = []

    if not lands:
        logger.warning("No lands found to process")
        return {"queue_id": queue_id, "processed_count": 0, "failed_count": len(land_ids)}

    executor = ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS)
    sem = asyncio.Semaphore(MAX_CONCURRENT_LANDS)

    async def _process_with_semaphore(land):
        async with sem:
            return await process_single_land_async(land, tile_ids, None, executor)

    tasks = [asyncio.create_task(_process_with_semaphore(land)) for land in lands]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    processed = sum(1 for r in results if r.get("success"))
    failed = [r for r in results if not r.get("success")]

    duration_ms = int((time.time() - start_ts) * 1000)
    final_status = "completed" if processed > 0 else "failed"

    try:
        supabase.table("ndvi_request_queue").update({
            "status": final_status,
            "processed_count": processed,
            "failed_count": len(failed),
            "processing_duration_ms": duration_ms,
            "completed_at": now_iso(),
        }).eq("id", queue_id).execute()
    except Exception as e:
        logger.error(f"Failed to update queue status for {queue_id}: {e}")

    logger.info(f"🏁 Queue {queue_id} finished: processed={processed}/{len(lands)} duration={duration_ms}ms failed={len(failed)}")
    return {"queue_id": queue_id, "processed_count": processed, "failed_count": len(failed), "failed": failed, "duration_ms": duration_ms}


# ----------------------------
# Cron style runner (sync entry)
# ----------------------------
def run_cron(limit: int = 10, max_retries: int = 3):
    """
    Synchronous entrypoint to pick queued requests and run async processing for each.
    """
    logger.info("🔄 NDVI async worker starting (cron)")
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
                "completed_at": now_iso(),
            }).eq("id", queue_id).execute()
            return

        # mark processing
        supabase.table("ndvi_request_queue").update({
            "status": "processing",
            "started_at": now_iso(),
            "retry_count": retry_count + 1,
        }).eq("id", queue_id).execute()

        try:
            result = await process_request_async(queue_id, tenant_id, land_ids, [tile_id] if tile_id else None)
            logger.info(f"Queue {queue_id} result: {result}")
        except Exception as e:
            logger.exception(f"Failed processing queue {queue_id}: {e}")
            supabase.table("ndvi_request_queue").update({
                "status": "queued",
                "last_error": str(e)[:500],
            }).eq("id", queue_id).execute()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [_handle_item(item) for item in items]
    loop.run_until_complete(asyncio.gather(*tasks))
    loop.close()
    logger.info("✅ Cron run finished")


# ----------------------------
# Main entrypoint (CLI)
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NDVI Land Worker v8 Async")
    parser.add_argument("--mode", choices=["cron", "single"], default="cron")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--queue-id", type=str, help="Process a single queue id")
    args = parser.parse_args()

    logger.info(f"Starting NDVI Land Worker v8 async mode={args.mode}")

    if args.mode == "single" and args.queue_id:
        try:
            queue_item = supabase.table("ndvi_request_queue").select("*").eq("id", args.queue_id).single().execute()
            if not queue_item.data:
                logger.error(f"Queue id {args.queue_id} not found")
            else:
                item = queue_item.data
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(process_request_async(
                    item["id"],
                    item["tenant_id"],
                    item.get("land_ids", []),
                    [item.get("tile_id")] if item.get("tile_id") else None
                ))
                loop.close()
        except Exception as e:
            logger.exception(f"Error processing single queue id: {e}")
    else:
        run_cron(limit=args.limit)

    logger.info("Worker finished")
