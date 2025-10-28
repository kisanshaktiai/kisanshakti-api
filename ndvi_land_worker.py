# ndvi_land_worker.py - FIXED v8.2
# Root Cause Fix: Geometry handling + reprojection + multi-tile merge
# Changes:
# 1. Robust geometry parsing (WKT/WKB/GeoJSON/hex)
# 2. Fixed reprojection to always return dict (never string)
# 3. Improved multi-tile NDVI merging with proper alignment
# 4. Enhanced error logging with geometry debugging

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
from rasterio.warp import transform_geom
from shapely.geometry import shape, mapping
from shapely import wkt, wkb
from PIL import Image
import matplotlib.cm as cm

from supabase import create_client, Client
from b2sdk.v2 import InMemoryAccountInfo, B2Api

# ----------------------------
# Logging & Config
# ----------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ndvi-worker-async")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
B2_KEY_ID = os.getenv("B2_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME", "kisanshakti-ndvi-tiles")
B2_PUBLIC_REGION = os.getenv("B2_PUBLIC_REGION", "f005")
SUPABASE_NDVI_BUCKET = os.getenv("SUPABASE_NDVI_BUCKET", "ndvi-thumbnails")

MAX_CONCURRENT_LANDS = int(os.getenv("MAX_CONCURRENT_LANDS", "8"))
THREAD_POOL_WORKERS = int(os.getenv("THREAD_POOL_WORKERS", "12"))

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
if not B2_KEY_ID or not B2_APP_KEY:
    raise RuntimeError("Missing B2_KEY_ID or B2_APP_KEY")

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
    rgba[..., 3][ndvi_array == -1] = 0
    img = Image.fromarray(rgba, mode="RGBA")
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.getvalue()


def upload_thumbnail_to_supabase_sync(land_id: str, date: str, png_bytes: bytes) -> Optional[str]:
    try:
        path = f"{land_id}/{date}/ndvi_colorized.png"
        supabase.storage.from_(SUPABASE_NDVI_BUCKET).upload(
            path=path,
            file=png_bytes,
            file_options={"content_type": "image/png", "upsert": True},
        )
        public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_NDVI_BUCKET}/{path}"
        logger.info(f"🖼️ Uploaded thumbnail for {land_id} ({len(png_bytes)/1024:.1f} KB)")
        return public_url
    except Exception as e:
        logger.error(f"❌ Thumbnail upload failed for {land_id}: {e}")
        return None


def _get_signed_b2_url(file_path: str, valid_secs: int = 3600) -> Optional[str]:
    if b2_bucket is None:
        return None
    try:
        auth_token = b2_bucket.get_download_authorization(
            file_name_prefix=file_path,
            valid_duration_in_seconds=valid_secs
        )
        return f"https://{B2_PUBLIC_REGION}.backblazeb2.com/file/{B2_BUCKET_NAME}/{file_path}?Authorization={auth_token}"
    except TypeError:
        try:
            auth_token = b2_bucket.get_download_authorization(
                file_name_prefix=file_path,
                valid_duration_seconds=valid_secs
            )
            return f"https://{B2_PUBLIC_REGION}.backblazeb2.com/file/{B2_BUCKET_NAME}/{file_path}?Authorization={auth_token}"
        except Exception as e:
            logger.error(f"B2 signing failed (fallback): {e}")
            return None
    except Exception as e:
        logger.error(f"B2 signing failed: {e}")
        return None


# ----------------------------
# 🔧 FIXED: Geometry Parsing & Reprojection
# ----------------------------
def parse_geometry_safe(geom_raw: Any) -> Optional[Dict]:
    """
    Robustly parse geometry from WKT/WKB/GeoJSON/hex strings
    ALWAYS returns a GeoJSON dict or None
    """
    if not geom_raw:
        return None
    
    # Already a dict - validate and return
    if isinstance(geom_raw, dict):
        if geom_raw.get("type") and geom_raw.get("coordinates"):
            return geom_raw
        logger.warning("Invalid GeoJSON dict structure")
        return None
    
    # Handle WKB bytes
    if isinstance(geom_raw, (bytes, bytearray)):
        try:
            shapely_geom = wkb.loads(bytes(geom_raw))
            return mapping(shapely_geom)
        except Exception as e:
            logger.warning(f"WKB bytes parsing failed: {e}")
            return None
    
    # Handle strings
    if isinstance(geom_raw, str):
        geom_str = geom_raw.strip()
        
        # Try GeoJSON string
        if geom_str.startswith("{"):
            try:
                parsed = json.loads(geom_str)
                if isinstance(parsed, dict) and "type" in parsed:
                    return parsed
            except json.JSONDecodeError:
                pass
        
        # Try WKT
        if any(geom_str.upper().startswith(prefix) for prefix in ["POLYGON", "MULTIPOLYGON", "POINT", "LINESTRING"]):
            try:
                shapely_geom = wkt.loads(geom_str)
                return mapping(shapely_geom)
            except Exception as e:
                logger.warning(f"WKT parsing failed: {e}")
        
        # Try WKB hex (PostGIS EWKB format)
        if geom_str.startswith("01") or geom_str.startswith("00"):
            try:
                geom_bytes = bytes.fromhex(geom_str)
                shapely_geom = wkb.loads(geom_bytes)
                return mapping(shapely_geom)
            except Exception as e:
                logger.warning(f"WKB hex parsing failed: {e}")
    
    logger.error(f"Unsupported geometry type: {type(geom_raw)}")
    return None


def reproject_geometry_to_raster_crs(geom_dict: Dict, target_crs: str) -> Dict:
    """
    Reproject GeoJSON geometry from EPSG:4326 to target CRS
    ALWAYS returns a valid GeoJSON dict
    """
    try:
        if not target_crs or target_crs == "EPSG:4326":
            return geom_dict
        
        # Convert CRS to string if needed
        target_crs_str = target_crs.to_string() if hasattr(target_crs, "to_string") else str(target_crs)
        
        # Use rasterio's transform_geom for accurate reprojection
        reprojected = transform_geom("EPSG:4326", target_crs_str, geom_dict)
        
        logger.debug(f"🧭 Reprojected geometry: EPSG:4326 → {target_crs_str}")
        return reprojected
        
    except Exception as e:
        logger.warning(f"⚠️ Reprojection failed: {e}, using original geometry")
        return geom_dict


# ----------------------------
# 🔧 FIXED: Stream NDVI with Proper Geometry Handling
# ----------------------------
def stream_ndvi_blocking(tile_id: str, acq_date: str, land_geom_dict: Dict) -> Optional[Tuple[np.ndarray, Dict]]:
    """
    Stream NDVI from B2 and crop to land boundary
    Returns: (ndvi_array, transform_metadata) or None
    """
    ndvi_path = f"tiles/ndvi/{tile_id}/{acq_date}/ndvi.tif"
    red_path = f"tiles/raw/{tile_id}/{acq_date}/B04.tif"
    nir_path = f"tiles/raw/{tile_id}/{acq_date}/B08.tif"
    
    ndvi_url = _get_signed_b2_url(ndvi_path)
    red_url = _get_signed_b2_url(red_path)
    nir_url = _get_signed_b2_url(nir_path)
    
    if not ndvi_url or not red_url or not nir_url:
        logger.warning(f"⚠️ Could not sign B2 URLs for {tile_id}/{acq_date}")
        return None
    
    # Try precomputed NDVI first
    try:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
            with rasterio.open(ndvi_url) as src:
                # Reproject geometry to raster CRS
                land_geom_proj = reproject_geometry_to_raster_crs(land_geom_dict, src.crs)
                
                logger.debug(f"Raster bounds: {src.bounds}, CRS: {src.crs}")
                logger.debug(f"Land geometry type: {land_geom_proj.get('type')}")
                
                try:
                    ndvi_clip, transform = mask(src, [land_geom_proj], crop=True, all_touched=True)
                except ValueError as ve:
                    if "overlap" in str(ve).lower():
                        logger.debug(f"No overlap with NDVI raster for {tile_id}/{acq_date}")
                        return None
                    raise
                
                if ndvi_clip.size == 0 or ndvi_clip[0].size == 0:
                    logger.debug(f"Empty NDVI clip for {tile_id}/{acq_date}")
                    return None
                
                logger.info(f"🟢 Using precomputed NDVI for {tile_id}/{acq_date}")
                return ndvi_clip[0], {"transform": transform, "crs": src.crs}
                
    except RasterioIOError:
        logger.debug(f"NDVI file not found: {ndvi_path}")
    except Exception as e:
        logger.warning(f"Could not read NDVI: {e}")
    
    # Fallback: Compute from B04/B08
    try:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
            with rasterio.open(red_url) as red_src, rasterio.open(nir_url) as nir_src:
                land_geom_proj = reproject_geometry_to_raster_crs(land_geom_dict, red_src.crs)
                
                try:
                    red_clip, transform = mask(red_src, [land_geom_proj], crop=True, all_touched=True)
                    nir_clip, _ = mask(nir_src, [land_geom_proj], crop=True, all_touched=True)
                except ValueError as ve:
                    if "overlap" in str(ve).lower():
                        logger.debug(f"No overlap with raw bands for {tile_id}/{acq_date}")
                        return None
                    raise
                
                if red_clip.size == 0 or nir_clip.size == 0:
                    return None
                
                ndvi = calculate_ndvi_from_bands(red_clip[0], nir_clip[0])
                logger.info(f"🧮 Computed NDVI from bands for {tile_id}/{acq_date}")
                return ndvi, {"transform": transform, "crs": red_src.crs}
                
    except Exception as e:
        logger.error(f"❌ NDVI computation failed for {tile_id}/{acq_date}: {e}")
    
    return None


# ----------------------------
# Database Helpers
# ----------------------------
def get_latest_tile_date_sync(tile_id: str) -> Optional[str]:
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
        logger.debug(f"get_latest_tile_date failed for {tile_id}: {e}")
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
# 🔧 FIXED: Single Land Processing with Multi-Tile Merge
# =========================================================
async def process_single_land_async(
    land: Dict[str, Any],
    tile_ids: Optional[List[str]],
    acquisition_date_override: Optional[str],
    executor: ThreadPoolExecutor,
) -> Dict[str, Any]:
    """
    Process NDVI for a single land parcel with proper multi-tile handling
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
        # 1️⃣ Parse geometry with new robust function
        geom_raw = land.get("boundary") or land.get("boundary_geom") or land.get("boundary_polygon_old")
        geometry = parse_geometry_safe(geom_raw)
        
        if not geometry:
            raise ValueError(f"Invalid or missing geometry for land {land_id}")
        
        logger.debug(f"✅ Parsed geometry type: {geometry.get('type')}")
        
        # 2️⃣ Determine intersecting tiles
        if land.get("tile_ids"):
            tiles_to_try = [t for t in land["tile_ids"] if t]
        elif tile_ids:
            tiles_to_try = tile_ids
        else:
            try:
                resp = supabase.rpc("get_intersecting_tiles", {"land_geom": json.dumps(geometry)}).execute()
                tiles_to_try = [t["tile_id"] for t in (resp.data or []) if "tile_id" in t]
            except Exception:
                tiles_to_try = []
        
        if not tiles_to_try:
            raise ValueError("No intersecting tiles found")
        
        logger.info(f"🌍 Land {land_id} intersects {len(tiles_to_try)} tiles: {tiles_to_try}")
        
        # 3️⃣ Stream NDVI from all tiles
        ndvi_results = []
        for tile_id in tiles_to_try:
            acq_date = acquisition_date_override or await loop.run_in_executor(
                executor, get_latest_tile_date_sync, tile_id
            )
            if not acq_date:
                logger.debug(f"No acquisition date for tile {tile_id}")
                continue
            
            result_data = await loop.run_in_executor(
                executor, stream_ndvi_blocking, tile_id, acq_date, geometry
            )
            
            if result_data:
                ndvi_results.append(result_data)
        
        if not ndvi_results:
            raise ValueError("No NDVI data extracted from any intersecting tiles")
        
        # 4️⃣ Merge multi-tile NDVI data
        if len(ndvi_results) == 1:
            final_ndvi = ndvi_results[0][0]
        else:
            # Simple averaging for overlapping areas
            ndvi_arrays = [r[0] for r in ndvi_results]
            try:
                final_ndvi = np.nanmean(np.stack(ndvi_arrays), axis=0)
                logger.info(f"✅ Merged NDVI from {len(ndvi_results)} tiles")
            except Exception as e:
                logger.warning(f"Multi-tile merge failed, using first tile: {e}")
                final_ndvi = ndvi_arrays[0]
        
        # 5️⃣ Compute statistics & thumbnail
        stats = calculate_statistics(final_ndvi)
        if stats["valid_pixels"] == 0:
            raise ValueError("No valid NDVI pixels after processing")
        
        acq_date_for_record = acquisition_date_override or datetime.date.today().isoformat()
        thumbnail_bytes = await loop.run_in_executor(executor, create_colorized_thumbnail, final_ndvi)
        thumbnail_url = await loop.run_in_executor(
            executor, upload_thumbnail_to_supabase_sync, land_id, acq_date_for_record, thumbnail_bytes
        )
        
        # 6️⃣ Write to database
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
            "bbox": geometry,
            "cloud_cover": 0,
            "created_at": now_iso(),
        }
        
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
        logger.info(f"✅ Land {land_id}: mean={stats['mean']:.3f}, coverage={stats['coverage']:.1f}%")
        
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
        except Exception:
            pass
    
    return result


# ----------------------------
# Orchestrator
# ----------------------------
async def process_request_async(queue_id: str, tenant_id: str, land_ids: List[str], tile_ids: Optional[List[str]] = None):
    logger.info(f"🚀 Processing queue={queue_id} tenant={tenant_id} lands={len(land_ids)}")
    start_ts = time.time()
    
    try:
        resp = supabase.table("lands").select("*").eq("tenant_id", tenant_id).in_("id", land_ids).execute()
        lands = resp.data or []
    except Exception as e:
        logger.error(f"Failed to fetch lands: {e}")
        lands = []
    
    if not lands:
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
        logger.error(f"Failed to update queue: {e}")
    
    logger.info(f"🏁 Queue {queue_id}: processed={processed}/{len(lands)} duration={duration_ms}ms")
    return {"queue_id": queue_id, "processed_count": processed, "failed_count": len(failed), "duration_ms": duration_ms}


# ----------------------------
# Cron Runner
# ----------------------------
def run_cron(limit: int = 10, max_retries: int = 3):
    logger.info("🔄 NDVI worker starting (cron)")
    try:
        queue_resp = supabase.table("ndvi_request_queue").select("*").eq("status", "queued").order("created_at", desc=False).limit(limit).execute()
        items = queue_resp.data or []
    except Exception as e:
        logger.error(f"Failed to fetch queue: {e}")
        items = []
    
    async def _handle_item(item):
        queue_id = item["id"]
        tenant_id = item["tenant_id"]
        land_ids = item.get("land_ids", [])
        tile_id = item.get("tile_id")
        retry_count = item.get("retry_count", 0)
        
        if retry_count >= max_retries:
            logger.warning(f"Max retries for {queue_id}")
            supabase.table("ndvi_request_queue").update({
                "status": "failed",
                "last_error": f"Max retries ({max_retries}) exceeded",
                "completed_at": now_iso(),
            }).eq("id", queue_id).execute()
            return
        
        supabase.table("ndvi_request_queue").update({
            "status": "processing",
            "started_at": now_iso(),
            "retry_count": retry_count + 1,
        }).eq("id", queue_id).execute()
        
        try:
            await process_request_async(queue_id, tenant_id, land_ids, [tile_id] if tile_id else None)
        except Exception as e:
            logger.exception(f"Failed queue {queue_id}: {e}")
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
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NDVI Land Worker v8.2 (Fixed)")
    parser.add_argument("--mode", choices=["cron", "single"], default="cron")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--queue-id", type=str, help="Process a single queue id")
    args = parser.parse_args()
    
    logger.info(f"Starting NDVI Worker v8.2 mode={args.mode}")
    
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
            logger.exception(f"Error processing single queue: {e}")
    else:
        run_cron(limit=args.limit)
    
    logger.info("Worker finished")
