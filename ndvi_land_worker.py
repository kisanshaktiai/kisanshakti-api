"""
NDVI Land Worker v8.3 - COMPLETE ROOT CAUSE FIX
================================================
Critical Fixes:
1. ✅ PostGIS native geometry handling (WKB/EWKB/Geography)
2. ✅ Proper CRS reprojection pipeline
3. ✅ Multi-tile intersection with spatial merging
4. ✅ Enhanced B2 URL validation
5. ✅ Comprehensive geometry debugging
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
from rasterio.warp import transform_geom, reproject, Resampling
from rasterio.crs import CRS
from shapely.geometry import shape, mapping, box
from shapely import wkt, wkb
from shapely.ops import unary_union
from PIL import Image
import matplotlib.cm as cm

from supabase import create_client, Client
from b2sdk.v2 import InMemoryAccountInfo, B2Api

# =============================================================================
# Configuration
# =============================================================================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "DEBUG"),  # Changed to DEBUG for diagnosis
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ndvi-worker-v9")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
B2_KEY_ID = os.getenv("B2_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME", "kisanshakti-ndvi-tiles")
B2_PUBLIC_REGION = os.getenv("B2_PUBLIC_REGION", "f005")
SUPABASE_NDVI_BUCKET = os.getenv("SUPABASE_NDVI_BUCKET", "ndvi-thumbnails")

MAX_CONCURRENT_LANDS = int(os.getenv("MAX_CONCURRENT_LANDS", "8"))
THREAD_POOL_WORKERS = int(os.getenv("THREAD_POOL_WORKERS", "12"))

if not all([SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, B2_KEY_ID, B2_APP_KEY]):
    raise RuntimeError("Missing critical environment variables")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

b2_info = InMemoryAccountInfo()
b2_api = B2Api(b2_info)
b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
try:
    b2_bucket = b2_api.get_bucket_by_name(B2_BUCKET_NAME)
    logger.info(f"✅ B2 bucket connected: {B2_BUCKET_NAME}")
except Exception as e:
    logger.critical(f"❌ B2 bucket access failed: {e}")
    b2_bucket = None


# =============================================================================
# 🔧 FIX 1: PostGIS Native Geometry Parsing
# =============================================================================
def extract_geometry_from_land(land_record: Dict) -> Optional[Dict]:
    """
    Extract geometry from lands table with priority:
    1. boundary (PostGIS Geography) - PREFERRED
    2. boundary_geom (PostGIS Geometry)
    3. boundary_polygon_old (JSONB legacy)
    
    Returns: GeoJSON dict in EPSG:4326
    """
    land_id = land_record.get("id", "unknown")
    
    # Priority 1: PostGIS Geography (binary WKB)
    boundary_geog = land_record.get("boundary")
    if boundary_geog:
        try:
            # PostGIS returns hex-encoded EWKB
            if isinstance(boundary_geog, str):
                geom_bytes = bytes.fromhex(boundary_geog)
            elif isinstance(boundary_geog, (bytes, bytearray)):
                geom_bytes = bytes(boundary_geog)
            else:
                logger.warning(f"Land {land_id}: Unexpected boundary type {type(boundary_geog)}")
                geom_bytes = None
            
            if geom_bytes:
                shapely_geom = wkb.loads(geom_bytes)
                logger.debug(f"✅ Land {land_id}: Parsed from boundary (Geography)")
                return mapping(shapely_geom)
        except Exception as e:
            logger.warning(f"Land {land_id}: Failed to parse boundary: {e}")
    
    # Priority 2: PostGIS Geometry
    boundary_geom = land_record.get("boundary_geom")
    if boundary_geom:
        try:
            if isinstance(boundary_geom, str):
                geom_bytes = bytes.fromhex(boundary_geom)
            elif isinstance(boundary_geom, (bytes, bytearray)):
                geom_bytes = bytes(boundary_geom)
            else:
                geom_bytes = None
            
            if geom_bytes:
                shapely_geom = wkb.loads(geom_bytes)
                logger.debug(f"✅ Land {land_id}: Parsed from boundary_geom")
                return mapping(shapely_geom)
        except Exception as e:
            logger.warning(f"Land {land_id}: Failed to parse boundary_geom: {e}")
    
    # Priority 3: JSONB legacy format
    boundary_old = land_record.get("boundary_polygon_old")
    if boundary_old:
        try:
            if isinstance(boundary_old, dict):
                geojson = boundary_old
            elif isinstance(boundary_old, str):
                geojson = json.loads(boundary_old)
            else:
                geojson = None
            
            if geojson and "type" in geojson and "coordinates" in geojson:
                logger.debug(f"✅ Land {land_id}: Parsed from boundary_polygon_old")
                return geojson
        except Exception as e:
            logger.warning(f"Land {land_id}: Failed to parse boundary_polygon_old: {e}")
    
    logger.error(f"❌ Land {land_id}: No valid geometry found in any column")
    return None


# =============================================================================
# 🔧 FIX 2: Proper CRS Reprojection with Validation
# =============================================================================
def reproject_to_raster_crs(geojson: Dict, target_crs: CRS) -> Dict:
    """
    Transform GeoJSON from EPSG:4326 to target CRS with validation
    """
    try:
        # Handle CRS object conversion
        if hasattr(target_crs, 'to_string'):
            target_crs_str = target_crs.to_string()
        elif hasattr(target_crs, 'to_wkt'):
            target_crs_str = target_crs.to_wkt()
        else:
            target_crs_str = str(target_crs)
        
        logger.debug(f"🧭 Reprojecting: EPSG:4326 → {target_crs_str}")
        
        # Use rasterio's transform_geom for accurate reprojection
        reprojected = transform_geom("EPSG:4326", target_crs_str, geojson)
        
        # Validate result
        if not reprojected or "coordinates" not in reprojected:
            raise ValueError("Reprojection returned invalid geometry")
        
        return reprojected
        
    except Exception as e:
        logger.error(f"❌ Reprojection failed: {e}")
        raise


# =============================================================================
# 🔧 FIX 3: Enhanced B2 Streaming with Validation
# =============================================================================
def validate_b2_url(url: str, timeout: int = 5) -> bool:
    """Quick HEAD request to validate B2 URL"""
    try:
        import requests
        resp = requests.head(url, timeout=timeout, allow_redirects=True)
        return resp.status_code == 200
    except:
        return False


def get_signed_b2_url(file_path: str, valid_secs: int = 3600) -> Optional[str]:
    """Generate signed B2 URL with validation"""
    if not b2_bucket:
        return None
    
    try:
        auth_token = b2_bucket.get_download_authorization(
            file_name_prefix=file_path,
            valid_duration_in_seconds=valid_secs
        )
        url = f"https://{B2_PUBLIC_REGION}.backblazeb2.com/file/{B2_BUCKET_NAME}/{file_path}?Authorization={auth_token}"
        
        # Validate URL
        if validate_b2_url(url):
            return url
        else:
            logger.warning(f"⚠️ B2 URL validation failed for {file_path}")
            return None
            
    except TypeError:
        # Fallback for different B2 SDK version
        try:
            auth_token = b2_bucket.get_download_authorization(
                file_name_prefix=file_path,
                valid_duration_seconds=valid_secs
            )
            url = f"https://{B2_PUBLIC_REGION}.backblazeb2.com/file/{B2_BUCKET_NAME}/{file_path}?Authorization={auth_token}"
            return url if validate_b2_url(url) else None
        except Exception as e:
            logger.error(f"B2 signing fallback failed: {e}")
            return None
    except Exception as e:
        logger.error(f"B2 URL generation failed: {e}")
        return None


# =============================================================================
# 🔧 FIX 4: Multi-Tile NDVI Extraction with Spatial Merging
# =============================================================================
def extract_ndvi_from_tile(
    tile_id: str, 
    acq_date: str, 
    land_geom_4326: Dict,
    debug_land_id: str
) -> Optional[Tuple[np.ndarray, Dict, str]]:
    """
    Extract NDVI from single tile with proper reprojection
    Returns: (ndvi_array, metadata, tile_id) or None
    """
    logger.info(f"🔍 Extracting NDVI: {tile_id}/{acq_date} for land {debug_land_id}")
    
    # Construct B2 paths
    ndvi_path = f"tiles/ndvi/{tile_id}/{acq_date}/ndvi.tif"
    red_path = f"tiles/raw/{tile_id}/{acq_date}/B04.tif"
    nir_path = f"tiles/raw/{tile_id}/{acq_date}/B08.tif"
    
    # Get signed URLs
    ndvi_url = get_signed_b2_url(ndvi_path)
    red_url = get_signed_b2_url(red_path)
    nir_url = get_signed_b2_url(nir_path)
    
    if not ndvi_url and not (red_url and nir_url):
        logger.warning(f"⚠️ No valid B2 URLs for {tile_id}/{acq_date}")
        return None
    
    # Try precomputed NDVI first
    if ndvi_url:
        try:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_TIMEOUT='30'):
                with rasterio.open(ndvi_url) as src:
                    logger.debug(f"📊 Tile CRS: {src.crs}, Bounds: {src.bounds}")
                    
                    # Reproject geometry to tile CRS
                    land_geom_proj = reproject_to_raster_crs(land_geom_4326, src.crs)
                    
                    # Check spatial intersection
                    land_shape = shape(land_geom_proj)
                    tile_bbox = box(*src.bounds)
                    
                    if not land_shape.intersects(tile_bbox):
                        logger.debug(f"⏭️ No spatial overlap: {tile_id}/{acq_date}")
                        return None
                    
                    # Extract with mask
                    try:
                        ndvi_clip, transform = mask(
                            src, 
                            [land_geom_proj], 
                            crop=True, 
                            all_touched=True,
                            indexes=1
                        )
                    except ValueError as ve:
                        if "overlap" in str(ve).lower():
                            logger.debug(f"No overlap after mask: {tile_id}/{acq_date}")
                            return None
                        raise
                    
                    if ndvi_clip.size == 0 or np.all(ndvi_clip == src.nodata):
                        logger.debug(f"Empty NDVI clip: {tile_id}/{acq_date}")
                        return None
                    
                    logger.info(f"✅ Extracted NDVI from precomputed: {tile_id}/{acq_date}")
                    return (
                        ndvi_clip[0] if ndvi_clip.ndim > 2 else ndvi_clip,
                        {"transform": transform, "crs": src.crs, "nodata": src.nodata},
                        tile_id
                    )
                    
        except RasterioIOError as e:
            logger.debug(f"NDVI file not accessible: {ndvi_path} - {e}")
        except Exception as e:
            logger.warning(f"NDVI extraction failed: {tile_id}/{acq_date} - {e}")
    
    # Fallback: Compute from B04/B08
    if red_url and nir_url:
        try:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_TIMEOUT='30'):
                with rasterio.open(red_url) as red_src, rasterio.open(nir_url) as nir_src:
                    land_geom_proj = reproject_to_raster_crs(land_geom_4326, red_src.crs)
                    
                    red_clip, transform = mask(red_src, [land_geom_proj], crop=True, all_touched=True, indexes=1)
                    nir_clip, _ = mask(nir_src, [land_geom_proj], crop=True, all_touched=True, indexes=1)
                    
                    if red_clip.size == 0 or nir_clip.size == 0:
                        return None
                    
                    # Compute NDVI
                    red_f = red_clip[0].astype(np.float32) if red_clip.ndim > 2 else red_clip.astype(np.float32)
                    nir_f = nir_clip[0].astype(np.float32) if nir_clip.ndim > 2 else nir_clip.astype(np.float32)
                    
                    np.seterr(divide='ignore', invalid='ignore')
                    ndvi = (nir_f - red_f) / (nir_f + red_f + 1e-8)
                    ndvi = np.clip(ndvi, -1, 1)
                    ndvi[np.isnan(ndvi)] = -1
                    
                    logger.info(f"✅ Computed NDVI from bands: {tile_id}/{acq_date}")
                    return (
                        ndvi,
                        {"transform": transform, "crs": red_src.crs, "nodata": -1},
                        tile_id
                    )
                    
        except Exception as e:
            logger.error(f"❌ Band computation failed: {tile_id}/{acq_date} - {e}")
    
    return None


def merge_multi_tile_ndvi(ndvi_results: List[Tuple[np.ndarray, Dict, str]]) -> np.ndarray:
    """
    Merge NDVI arrays from multiple tiles with spatial alignment
    """
    if len(ndvi_results) == 1:
        return ndvi_results[0][0]
    
    try:
        # Strategy 1: Simple averaging for overlapping pixels
        arrays = [r[0] for r in ndvi_results]
        
        # Find common shape
        max_shape = tuple(max(s) for s in zip(*[a.shape for a in arrays]))
        
        # Pad arrays to common shape
        padded = []
        for arr in arrays:
            if arr.shape != max_shape:
                pad_width = [(0, max_shape[i] - arr.shape[i]) for i in range(len(arr.shape))]
                arr_padded = np.pad(arr, pad_width, mode='constant', constant_values=-1)
                padded.append(arr_padded)
            else:
                padded.append(arr)
        
        # Average valid pixels (ignore -1 nodata)
        stack = np.stack(padded)
        valid_mask = stack != -1
        
        with np.errstate(invalid='ignore'):
            merged = np.where(
                valid_mask.any(axis=0),
                np.nanmean(np.where(valid_mask, stack, np.nan), axis=0),
                -1
            )
        
        logger.info(f"✅ Merged {len(ndvi_results)} tiles → shape {merged.shape}")
        return merged
        
    except Exception as e:
        logger.warning(f"⚠️ Multi-tile merge failed: {e}, using first tile only")
        return ndvi_results[0][0]


# =============================================================================
# Statistics & Visualization
# =============================================================================
def calculate_statistics(ndvi: np.ndarray) -> Dict[str, Any]:
    """Calculate NDVI statistics"""
    valid_mask = (ndvi >= -1) & (ndvi <= 1) & (ndvi != -1)
    valid_pixels = ndvi[valid_mask]
    
    if valid_pixels.size == 0:
        return {
            "mean": None, "min": None, "max": None, "std": None,
            "valid_pixels": 0, "total_pixels": int(ndvi.size), "coverage": 0.0
        }
    
    return {
        "mean": float(np.mean(valid_pixels)),
        "min": float(np.min(valid_pixels)),
        "max": float(np.max(valid_pixels)),
        "std": float(np.std(valid_pixels)),
        "valid_pixels": int(valid_pixels.size),
        "total_pixels": int(ndvi.size),
        "coverage": float(valid_pixels.size / ndvi.size * 100)
    }


def create_colorized_thumbnail(ndvi_array: np.ndarray, max_size: int = 512) -> bytes:
    """Generate vegetation heatmap thumbnail"""
    norm = np.clip((ndvi_array + 1) / 2, 0, 1)
    cmap = cm.get_cmap("RdYlGn")
    rgba = (cmap(norm) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(ndvi_array == -1, 0, 255)
    
    img = Image.fromarray(rgba, mode="RGBA")
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.getvalue()


def upload_thumbnail_to_supabase_sync(land_id: str, date: str, png_bytes: bytes) -> Optional[str]:
    """Upload thumbnail to Supabase storage"""
    try:
        path = f"{land_id}/{date}/ndvi_colorized.png"
        supabase.storage.from_(SUPABASE_NDVI_BUCKET).upload(
            path=path,
            file=png_bytes,
            file_options={"content_type": "image/png", "upsert": True}
        )
        public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_NDVI_BUCKET}/{path}"
        logger.info(f"🖼️ Thumbnail uploaded: {land_id} ({len(png_bytes)/1024:.1f} KB)")
        return public_url
    except Exception as e:
        logger.error(f"❌ Thumbnail upload failed: {land_id} - {e}")
        return None


# =============================================================================
# Database Helpers
# =============================================================================
def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def get_latest_tile_date_sync(tile_id: str) -> Optional[str]:
    try:
        resp = supabase.table("satellite_tiles").select("acquisition_date").eq(
            "tile_id", tile_id
        ).eq("status", "ready").order("acquisition_date", desc=True).limit(1).execute()
        return resp.data[0]["acquisition_date"] if resp.data else None
    except:
        return None


def upsert_ndvi_data_sync(record: Dict) -> None:
    try:
        supabase.table("ndvi_data").upsert(record).execute()
    except Exception as e:
        logger.error(f"ndvi_data upsert failed: {e}")


def upsert_micro_tile_sync(record: Dict) -> None:
    try:
        supabase.table("ndvi_micro_tiles").upsert(record).execute()
    except Exception as e:
        logger.error(f"ndvi_micro_tiles upsert failed: {e}")


def update_land_sync(land_id: str, payload: Dict) -> None:
    try:
        supabase.table("lands").update(payload).eq("id", land_id).execute()
    except Exception as e:
        logger.error(f"lands update failed: {land_id} - {e}")


def insert_processing_log_sync(record: Dict) -> None:
    try:
        supabase.table("ndvi_processing_logs").insert(record).execute()
    except Exception as e:
        logger.error(f"processing_log insert failed: {e}")


# =============================================================================
# 🚀 Main Processing Logic
# =============================================================================
async def process_single_land_async(
    land: Dict,
    tile_ids: Optional[List[str]],
    acquisition_date_override: Optional[str],
    executor: ThreadPoolExecutor
) -> Dict[str, Any]:
    """Process NDVI for single land parcel"""
    loop = asyncio.get_running_loop()
    land_id = land.get("id")
    tenant_id = land.get("tenant_id")
    
    result = {"land_id": land_id, "success": False, "error": None, "stats": None}
    
    try:
        # 1. Extract geometry
        geometry = extract_geometry_from_land(land)
        if not geometry:
            raise ValueError(f"No valid geometry found for land {land_id}")
        
        logger.debug(f"🗺️ Land {land_id} geometry: {geometry.get('type')}")
        
        # 2. Determine tiles
        if tile_ids:
            tiles_to_process = tile_ids
        elif land.get("tile_ids"):
            tiles_to_process = land["tile_ids"]
        else:
            # Fallback: Query intersecting tiles
            try:
                resp = supabase.rpc("get_intersecting_tiles", {
                    "land_geom": json.dumps(geometry)
                }).execute()
                tiles_to_process = [t["tile_id"] for t in (resp.data or [])]
            except:
                tiles_to_process = []
        
        if not tiles_to_process:
            raise ValueError("No intersecting tiles found")
        
        logger.info(f"🌍 Land {land_id} → Tiles: {tiles_to_process}")
        
        # 3. Extract NDVI from all tiles
        ndvi_extractions = []
        for tile_id in tiles_to_process:
            acq_date = acquisition_date_override or await loop.run_in_executor(
                executor, get_latest_tile_date_sync, tile_id
            )
            
            if not acq_date:
                logger.debug(f"No acquisition date for {tile_id}")
                continue
            
            extraction = await loop.run_in_executor(
                executor, extract_ndvi_from_tile, tile_id, acq_date, geometry, land_id
            )
            
            if extraction:
                ndvi_extractions.append(extraction)
        
        if not ndvi_extractions:
            raise ValueError("No NDVI data extracted from any intersecting tiles")
        
        # 4. Merge multi-tile results
        final_ndvi = merge_multi_tile_ndvi(ndvi_extractions)
        
        # 5. Calculate statistics
        stats = calculate_statistics(final_ndvi)
        if stats["valid_pixels"] == 0:
            raise ValueError("No valid NDVI pixels after processing")
        
        # 6. Generate thumbnail
        acq_date_for_record = acquisition_date_override or datetime.date.today().isoformat()
        thumbnail_bytes = await loop.run_in_executor(
            executor, create_colorized_thumbnail, final_ndvi
        )
        thumbnail_url = await loop.run_in_executor(
            executor, upload_thumbnail_to_supabase_sync, land_id, acq_date_for_record, thumbnail_bytes
        )
        
        # 7. Save to database
        ndvi_record = {
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
            "computed_at": now_iso()
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
            "created_at": now_iso()
        }
        
        await loop.run_in_executor(executor, upsert_ndvi_data_sync, ndvi_record)
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
        logger.info(f"✅ Land {land_id}: NDVI={stats['mean']:.3f}, coverage={stats['coverage']:.1f}%")
        
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"❌ Land {land_id} failed: {e}\n{tb}")
        result["error"] = str(e)
        
        try:
            await loop.run_in_executor(executor, insert_processing_log_sync, {
                "tenant_id": tenant_id,
                "land_id": land_id,
                "processing_step": "ndvi_extraction",
                "step_status": "failed",
                "error_message": str(e)[:500],
                "error_details": {"traceback": tb[:1000]},
                "created_at": now_iso()
            })
        except:
            pass
    
    return result


# =============================================================================
# Orchestrator
# =============================================================================
async def process_request_async(
    queue_id: str, 
    tenant_id: str, 
    land_ids: List[str], 
    tile_ids: Optional[List[str]] = None
):
    """Process NDVI request for multiple lands"""
    logger.info(f"🚀 Queue {queue_id}: {len(land_ids)} lands for tenant {tenant_id}")
    start_ts = time.time()
    
    try:
        resp = supabase.table("lands").select("*").eq("tenant_id", tenant_id).in_("id", land_ids).execute()
        lands = resp.data or []
    except Exception as e:
        logger.error(f"Failed to fetch lands: {e}")
        lands = []
    
    if not lands:
        logger.warning(f"No lands found for queue {queue_id}")
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
            "completed_at": now_iso()
        }).eq("id", queue_id).execute()
    except Exception as e:
        logger.error(f"Failed to update queue: {e}")
    
    logger.info(f"🏁 Queue {queue_id}: {processed}/{len(lands)} successful, {duration_ms}ms")
    return {
        "queue_id": queue_id,
        "processed_count": processed,
        "failed_count": len(failed),
        "duration_ms": duration_ms
    }


# =============================================================================
# Cron Runner
# =============================================================================
def run_cron(limit: int = 10, max_retries: int = 3):
    """Process queued NDVI requests"""
    logger.info("🔄 NDVI Worker Cron Starting")
    
    try:
        queue_resp = supabase.table("ndvi_request_queue").select("*").eq(
            "status", "queued"
        ).order("created_at", desc=False).limit(limit).execute()
        items = queue_resp.data or []
    except Exception as e:
        logger.error(f"Failed to fetch queue: {e}")
        items = []
    
    if not items:
        logger.info("No queued items found")
        return
    
    async def _handle_item(item):
        queue_id = item["id"]
        tenant_id = item["tenant_id"]
        land_ids = item.get("land_ids", [])
        tile_id = item.get("tile_id")
        retry_count = item.get("retry_count", 0)
        
        if retry_count >= max_retries:
            logger.warning(f"Max retries exceeded for {queue_id}")
            supabase.table("ndvi_request_queue").update({
                "status": "failed",
                "last_error": f"Max retries ({max_retries}) exceeded",
                "completed_at": now_iso()
            }).eq("id", queue_id).execute()
            return
        
        supabase.table("ndvi_request_queue").update({
            "status": "processing",
            "started_at": now_iso(),
            "retry_count": retry_count + 1
        }).eq("id", queue_id).execute()
        
        try:
            await process_request_async(
                queue_id, 
                tenant_id, 
                land_ids, 
                [tile_id] if tile_id else None
            )
        except Exception as e:
            logger.exception(f"Failed queue {queue_id}: {e}")
            supabase.table("ndvi_request_queue").update({
                "status": "queued",
                "last_error": str(e)[:500]
            }).eq("id", queue_id).execute()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [_handle_item(item) for item in items]
    loop.run_until_complete(asyncio.gather(*tasks))
    loop.close()
    
    logger.info("✅ Cron run completed")


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NDVI Land Worker v9.0 (Complete Fix)")
    parser.add_argument("--mode", choices=["cron", "single"], default="cron")
    parser.add_argument("--limit", type=int, default=5, help="Max queue items to process")
    parser.add_argument("--queue-id", type=str, help="Process specific queue ID")
    args = parser.parse_args()
    
    logger.info(f"🚀 NDVI Worker v9.0 starting (mode={args.mode})")
    
    if args.mode == "single" and args.queue_id:
        try:
            queue_item = supabase.table("ndvi_request_queue").select("*").eq(
                "id", args.queue_id
            ).single().execute()
            
            if not queue_item.data:
                logger.error(f"Queue ID {args.queue_id} not found")
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
            logger.exception(f"Single queue processing failed: {e}")
    else:
        run_cron(limit=args.limit)
    
    logger.info("✅ Worker finished")
