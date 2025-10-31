#!/usr/bin/env python3
# ndvi_land_worker_v9.2.py — corrected / hardened
# v9.2 — Signed B2 URLs + CRS-safe reprojection + structured returns
# (Adapted / fixed: upload, signed URL validation, rasterio open robustness, better logging)

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
from rasterio.warp import transform_geom
from rasterio.crs import CRS
from shapely.geometry import shape, mapping, box
from shapely import wkb
from PIL import Image
import matplotlib.cm as cm

from supabase import create_client, Client
from b2sdk.v2 import InMemoryAccountInfo, B2Api

# ---------------- Logging ----------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ndvi-worker-v9.2")

# ---------------- Configuration / Env ----------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
B2_KEY_ID = os.getenv("B2_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME", "kisanshakti-ndvi-tiles")
B2_PUBLIC_REGION = os.getenv("B2_PUBLIC_REGION", "f005")  # backblaze region host prefix (if applicable)
SUPABASE_NDVI_BUCKET = os.getenv("SUPABASE_NDVI_BUCKET", "ndvi-thumbnails")

MAX_CONCURRENT_LANDS = int(os.getenv("MAX_CONCURRENT_LANDS", "6"))
THREAD_POOL_WORKERS = int(os.getenv("THREAD_POOL_WORKERS", "8"))

# Tune chunk size if memory constrained
NDVI_CHUNK_ROWS = int(os.getenv("NDVI_CHUNK_ROWS", "1024"))

# Validate critical env
if not all([SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, B2_KEY_ID, B2_APP_KEY]):
    logger.critical("Missing required environment variables: SUPABASE_* or B2_*")
    raise RuntimeError("Missing required environment variables")

# ---------------- Clients ----------------
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

# ---------------- Utilities ----------------
def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

def safe_json_load(s: Any) -> Optional[Dict]:
    try:
        if s is None:
            return None
        if isinstance(s, dict):
            return s
        if isinstance(s, str):
            return json.loads(s)
        return None
    except Exception:
        return None

def get_file_size(filepath: str) -> Optional[int]:
    try:
        return os.path.getsize(filepath)
    except Exception:
        return None

# ---------------- Signed B2 URL logic ----------------
def validate_http_url(url: str, timeout: int = 6) -> bool:
    """
    HEAD may be blocked on some endpoints; fall back to a small GET streaming request.
    """
    try:
        import requests
        try:
            r = requests.head(url, timeout=timeout, allow_redirects=True)
            if 200 <= r.status_code < 400:
                return True
        except Exception as e_head:
            logger.debug(f"HEAD failed for validation: {e_head} — trying GET stream")
            try:
                r = requests.get(url, stream=True, timeout=timeout, allow_redirects=True)
                if 200 <= r.status_code < 400:
                    return True
            except Exception as e_get:
                logger.debug(f"GET stream also failed during validation: {e_get}")
                return False
        return False
    except Exception as e:
        logger.debug(f"requests not available or validation failed: {e}")
        return False

def get_signed_b2_url(file_path: str, valid_secs: int = 3600) -> Optional[str]:
    """
    Generate a signed B2 URL that rasterio/GDAL can open.
    Returns None on failure.
    """
    if not b2_bucket:
        logger.error("B2 bucket client not available")
        return None

    try:
        # try different kwarg name variants across b2sdk versions
        auth_token = None
        for kwargs in (
            {"file_name_prefix": file_path, "valid_duration_in_seconds": valid_secs},
            {"file_name_prefix": file_path, "valid_duration_seconds": valid_secs},
            {"file_name_prefix": file_path, "valid_durationInSeconds": valid_secs},
        ):
            try:
                auth_token = b2_bucket.get_download_authorization(**kwargs)
                break
            except TypeError:
                continue
            except Exception as e:
                logger.debug(f"get_download_authorization attempt failed: {e}")
                continue

        if not auth_token:
            logger.warning(f"No auth token obtained for {file_path}")
            # Try public file path test (maybe it's public)
            url_try = f"https://{B2_PUBLIC_REGION}.backblazeb2.com/file/{B2_BUCKET_NAME}/{file_path}"
            if validate_http_url(url_try):
                logger.debug(f"Public URL works for {file_path}")
                return url_try
            return None

        # Build URL (Backblaze allows Authorization query param for download auth)
        url = f"https://{B2_PUBLIC_REGION}.backblazeb2.com/file/{B2_BUCKET_NAME}/{file_path}?Authorization={auth_token}"
        if not validate_http_url(url):
            logger.warning(f"Signed B2 URL validation failed for {file_path}")
            return None
        return url

    except Exception as e:
        logger.exception(f"Failed to generate signed B2 URL for {file_path}: {e}")
        return None

# ---------------- Geometry / reprojection helpers ----------------
def extract_geometry_from_land(land_record: Dict) -> Optional[Dict]:
    """
    Extract geometry GeoJSON in EPSG:4326 from a land record.
    Tries different fields and formats robustly.
    """
    land_id = land_record.get("id", "<unknown>")
    # 1: boundary (EWKB hex or bytes)
    for key in ("boundary", "boundary_geom"):
        val = land_record.get(key)
        if not val:
            continue
        try:
            if isinstance(val, str):
                s = val.strip()
                # JSON-like?
                if s.startswith("{") or s.startswith("["):
                    return json.loads(s)
                # hex WKB? small heuristic: hex must be even length and contain hex chars
                if all(c in "0123456789abcdefABCDEF" for c in s.replace("0x", "")) and (len(s) % 2 == 0):
                    geom_bytes = bytes.fromhex(s.replace("0x", ""))
                else:
                    # unknown string form — try JSON load defensively
                    try:
                        return json.loads(s)
                    except Exception:
                        raise ValueError("String geometry not JSON nor hex WKB")
            elif isinstance(val, (bytes, bytearray)):
                geom_bytes = bytes(val)
            elif isinstance(val, dict):
                return val
            else:
                continue
            shapely_geom = wkb.loads(geom_bytes)
            geojson = mapping(shapely_geom)
            return geojson
        except Exception as e:
            logger.debug(f"land {land_id}: failed parse {key}: {e}")

    # 3: legacy JSON column
    legacy = land_record.get("boundary_polygon_old")
    if legacy:
        try:
            if isinstance(legacy, dict):
                return legacy
            if isinstance(legacy, str):
                return json.loads(legacy)
        except Exception:
            logger.debug(f"land {land_id}: legacy polygon parse failed")

    logger.error(f"❌ Land {land_id}: no valid geometry found")
    return None

def reproject_to_raster_crs(geojson: Dict, target_crs: CRS) -> Dict:
    """
    Reproject a GeoJSON geometry (assumed EPSG:4326) to target_crs (CRS object/string).
    Returns the reprojected geojson dict.
    """
    if geojson is None:
        raise ValueError("geojson is None")
    if isinstance(target_crs, CRS):
        target_crs_str = target_crs.to_string()
    else:
        target_crs_str = str(target_crs)
    try:
        reprojected = transform_geom("EPSG:4326", target_crs_str, geojson)
        if not reprojected or "coordinates" not in reprojected:
            raise ValueError("Invalid reprojection result")
        return reprojected
    except Exception as e:
        logger.error(f"Reprojection failed: {e}")
        raise

# ---------------- NDVI extraction helpers ----------------
def _rasterio_open_with_retries(url: str, mode_kwargs: Dict = None, retries: int = 2, timeout: int = 30):
    """
    Small helper: rasterio.open on HTTP can fail transiently; do a couple retries.
    mode_kwargs: passed to rasterio.open as kwargs with context manager compatibility.
    """
    attempt = 0
    mode_kwargs = mode_kwargs or {}
    while attempt <= retries:
        try:
            env_kwargs = {
                "GDAL_DISABLE_READDIR_ON_OPEN": "TRUE",
                "GDAL_HTTP_TIMEOUT": str(timeout),
            }
            # Use rasterio.Env for consistent GDAL options
            env = rasterio.Env(**env_kwargs)
            env.__enter__()
            src = rasterio.open(url, **mode_kwargs)
            # We'll return src but ensure the Env remains usable for caller's context
            return src, env
        except RasterioIOError as e:
            attempt += 1
            logger.debug(f"rasterio open attempt {attempt} failed for {url}: {e}")
            if attempt > retries:
                logger.exception(f"Rasterio open final failure for {url}")
                raise
            time.sleep(1 + attempt)
        except Exception as e:
            attempt += 1
            logger.debug(f"rasterio.open unexpected error (attempt {attempt}) for {url}: {e}")
            if attempt > retries:
                raise
            time.sleep(0.5)
    raise RuntimeError("rasterio open retries exhausted")

def extract_ndvi_from_tile(
    tile_id: str,
    acq_date: str,
    land_geom_4326: Dict,
    debug_land_id: str
) -> Optional[Tuple[np.ndarray, Dict[str, Any], str]]:
    """
    Attempt to obtain NDVI for the given land from a tile.
    Priority:
      1) Try precomputed NDVI COG in B2
      2) If not available or fails, compute NDVI from B04/B08 on-the-fly
    Returns tuple (ndvi_array, metadata, tile_id) or None on failure.
    metadata includes transform, crs, nodata
    """
    logger.info(f"🔍 Extracting NDVI: {tile_id}/{acq_date} for land {debug_land_id}")

    ndvi_rel = f"tiles/ndvi/{tile_id}/{acq_date}/ndvi.tif"
    b04_rel = f"tiles/raw/{tile_id}/{acq_date}/B04.tif"
    b08_rel = f"tiles/raw/{tile_id}/{acq_date}/B08.tif"

    ndvi_url = get_signed_b2_url(ndvi_rel)
    b04_url = get_signed_b2_url(b04_rel)
    b08_url = get_signed_b2_url(b08_rel)

    # 1) Try precomputed NDVI COG
    if ndvi_url:
        try:
            src, env = _rasterio_open_with_retries(ndvi_url, {}, retries=2)
            try:
                logger.debug(f"NDVI tile opened: CRS={src.crs}, bounds={src.bounds}, nodata={src.nodata}")
                land_proj = reproject_to_raster_crs(land_geom_4326, src.crs)
                land_shape = shape(land_proj)
                tile_bbox = box(*src.bounds)
                if not land_shape.intersects(tile_bbox):
                    logger.debug(f"No spatial overlap for {tile_id}/{acq_date} (precomputed)")
                    src.close()
                    env.__exit__(None, None, None)
                    return None
                try:
                    clip, transform = mask(src, [land_proj], crop=True, all_touched=True, indexes=1)
                except ValueError as ve:
                    logger.debug(f"mask(): {ve}")
                    src.close()
                    env.__exit__(None, None, None)
                    return None
                if clip.size == 0:
                    logger.debug("NDVI clip is empty")
                    src.close()
                    env.__exit__(None, None, None)
                    return None
                arr = clip[0] if clip.ndim > 2 else clip
                nod = src.nodata
                if nod is not None:
                    arr = np.where(arr == nod, -1.0, arr).astype(np.float32)
                else:
                    arr = arr.astype(np.float32)
                src.close()
                env.__exit__(None, None, None)
                return arr, {"transform": transform, "crs": src.crs, "nodata": -1.0}, tile_id
            except Exception:
                # ensure closing env/src
                try:
                    src.close()
                except Exception:
                    pass
                env.__exit__(None, None, None)
                raise
        except RasterioIOError as e:
            logger.debug(f"Rasterio cannot open precomputed NDVI {ndvi_rel}: {e}")
        except Exception as e:
            logger.warning(f"Failed extraction from precomputed NDVI {ndvi_rel}: {e}")

    # 2) Fallback compute from B04/B08
    if b04_url and b08_url:
        try:
            red_src, red_env = _rasterio_open_with_retries(b04_url, {}, retries=2)
            nir_src, nir_env = _rasterio_open_with_retries(b08_url, {}, retries=2)
            try:
                if str(red_src.crs) != str(nir_src.crs):
                    logger.debug("Band CRSs differ; continuing with red CRS and reprojecting if necessary")
                land_proj = reproject_to_raster_crs(land_geom_4326, red_src.crs)
                red_clip, transform = mask(red_src, [land_proj], crop=True, all_touched=True, indexes=1)
                nir_clip, _ = mask(nir_src, [land_proj], crop=True, all_touched=True, indexes=1)
                if red_clip.size == 0 or nir_clip.size == 0:
                    logger.debug("Band clips empty")
                    red_src.close(); nir_src.close()
                    red_env.__exit__(None, None, None); nir_env.__exit__(None, None, None)
                    return None
                red = red_clip[0].astype(np.float32)
                nir = nir_clip[0].astype(np.float32)
                np.seterr(divide="ignore", invalid="ignore")
                denom = nir + red
                ndvi = np.where(denom != 0, (nir - red) / denom, np.nan)
                ndvi = np.clip(ndvi, -1.0, 1.0)
                ndvi = np.where(np.isnan(ndvi), -1.0, ndvi).astype(np.float32)
                red_src.close(); nir_src.close()
                red_env.__exit__(None, None, None); nir_env.__exit__(None, None, None)
                return ndvi, {"transform": transform, "crs": red_src.crs, "nodata": -1.0}, tile_id
            except Exception:
                try:
                    red_src.close(); nir_src.close()
                except Exception:
                    pass
                red_env.__exit__(None, None, None); nir_env.__exit__(None, None, None)
                raise
        except Exception as e:
            logger.error(f"Failed to compute NDVI from bands for {tile_id}/{acq_date}: {e}")

    logger.debug(f"No NDVI available for {tile_id}/{acq_date}")
    return None

def merge_multi_tile_ndvi(ndvi_results: List[Tuple[np.ndarray, Dict, str]]) -> np.ndarray:
    """
    Merge multiple NDVI results into a single array by aligning shapes and averaging valid pixels.
    """
    if not ndvi_results:
        raise ValueError("No ndvi_results to merge")
    arrays = [r[0] for r in ndvi_results]
    if len(arrays) == 1:
        return arrays[0].astype(np.float32)
    # Determine max shape
    max_rows = max(a.shape[0] for a in arrays)
    max_cols = max(a.shape[1] for a in arrays)
    padded = []
    for a in arrays:
        rows, cols = a.shape
        pad_rows = max_rows - rows
        pad_cols = max_cols - cols
        pad = ((0, pad_rows), (0, pad_cols))
        padded.append(np.pad(a, pad, mode="constant", constant_values=-1.0))
    stack = np.stack(padded, axis=0).astype(np.float32)  # shape (n, r, c)
    valid_mask = stack != -1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        numerator = np.where(valid_mask, stack, np.nan).sum(axis=0)
        counts = valid_mask.sum(axis=0)
        mean = np.where(counts > 0, numerator / counts, -1.0)
    return mean.astype(np.float32)

def calculate_statistics(ndvi: np.ndarray) -> Dict[str, Any]:
    valid_mask = (ndvi != -1.0) & (ndvi >= -1.0) & (ndvi <= 1.0)
    valid_pixels = ndvi[valid_mask]
    total_pixels = int(ndvi.size)
    if valid_pixels.size == 0:
        return {
            "mean": None, "min": None, "max": None, "std": None,
            "valid_pixels": 0, "total_pixels": total_pixels, "coverage": 0.0
        }
    return {
        "mean": float(np.mean(valid_pixels)),
        "min": float(np.min(valid_pixels)),
        "max": float(np.max(valid_pixels)),
        "std": float(np.std(valid_pixels)),
        "valid_pixels": int(valid_pixels.size),
        "total_pixels": total_pixels,
        "coverage": float(valid_pixels.size / total_pixels * 100.0)
    }

def create_colorized_thumbnail(ndvi_array: np.ndarray, max_size: int = 512) -> bytes:
    norm = np.clip((ndvi_array + 1.0) / 2.0, 0.0, 1.0)
    cmap = cm.get_cmap("RdYlGn")
    rgba = (cmap(norm) * 255).astype(np.uint8)
    alpha = np.where(ndvi_array == -1.0, 0, 255).astype(np.uint8)
    rgba[..., 3] = alpha
    img = Image.fromarray(rgba, mode="RGBA")
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.getvalue()

def upload_thumbnail_to_supabase_sync(land_id: str, date: str, png_bytes: bytes) -> Optional[str]:
    """
    Upload thumbnail to Supabase storage. Wrap bytes in BytesIO and check response.
    Returns public URL or None.
    """
    try:
        path = f"{land_id}/{date}/ndvi_colorized.png"
        file_obj = io.BytesIO(png_bytes)
        file_obj.seek(0)
        # supabase-py storage API: .from_(bucket).upload(path, file, file_options=...)
        res = supabase.storage.from_(SUPABASE_NDVI_BUCKET).upload(
            path=path,
            file=file_obj,
            file_options={"content_type": "image/png", "upsert": True}
        )
        # supabase-py may return { "error": None, "data": ... } or raise — handle both
        if isinstance(res, dict) and res.get("error"):
            logger.error(f"Supabase upload error: {res.get('error')}")
            return None
        # Construct public URL (assumes public bucket or published)
        public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_NDVI_BUCKET}/{path}"
        logger.info(f"Uploaded thumbnail for {land_id} {date} -> {public_url}")
        return public_url
    except Exception as e:
        logger.error(f"Failed to upload thumbnail: {e}")
        return None

# ---------------- DB helpers ----------------
def get_latest_tile_date_sync(tile_id: str) -> Optional[str]:
    try:
        resp = supabase.table("satellite_tiles").select("acquisition_date").eq(
            "tile_id", tile_id
        ).eq("status", "ready").order("acquisition_date", desc=True).limit(1).execute()
        # handle different response shapes
        data = getattr(resp, "data", None) or (resp.get("data") if isinstance(resp, dict) else None)
        if data:
            rec = data[0]
            return rec.get("acquisition_date")
        return None
    except Exception as e:
        logger.debug(f"get_latest_tile_date_sync failed for {tile_id}: {e}")
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

# ---------------- Main single land processing (async-friendly) ----------------
async def process_single_land_async(
    land: Dict,
    tile_ids: Optional[List[str]],
    acquisition_date_override: Optional[str],
    executor: ThreadPoolExecutor
) -> Dict[str, Any]:
    loop = asyncio.get_running_loop()
    land_id = land.get("id")
    tenant_id = land.get("tenant_id")
    out = {"land_id": land_id, "success": False, "error": None, "stats": None, "thumbnail_url": None}
    try:
        geometry = extract_geometry_from_land(land)
        if not geometry:
            raise ValueError("No valid geometry for land")

        # determine tiles
        if tile_ids:
            tiles_to_process = tile_ids
        elif land.get("tile_ids"):
            tiles_to_process = land["tile_ids"]
        else:
            try:
                resp = supabase.rpc("get_intersecting_tiles", {"land_geom": json.dumps(geometry)}).execute()
                data = getattr(resp, "data", None) or (resp.get("data") if isinstance(resp, dict) else None)
                tiles_to_process = [t["tile_id"] for t in (data or [])]
            except Exception as e:
                logger.debug(f"RPC get_intersecting_tiles failed: {e}")
                tiles_to_process = []

        if not tiles_to_process:
            raise ValueError("No intersecting tiles found")

        # iterate tiles and extract NDVI
        ndvi_extractions = []
        for tile_id in tiles_to_process:
            acq_date = acquisition_date_override or await loop.run_in_executor(executor, get_latest_tile_date_sync, tile_id)
            if not acq_date:
                logger.debug(f"No acquisition date for {tile_id}")
                continue
            extraction = await loop.run_in_executor(executor, extract_ndvi_from_tile, tile_id, acq_date, geometry, land_id)
            if extraction:
                ndvi_extractions.append(extraction)

        if not ndvi_extractions:
            raise ValueError("No NDVI data extracted from any intersecting tiles")

        # merge
        merged_ndvi = await loop.run_in_executor(executor, merge_multi_tile_ndvi, ndvi_extractions)
        stats = await loop.run_in_executor(executor, calculate_statistics, merged_ndvi)
        if stats["valid_pixels"] == 0:
            raise ValueError("No valid NDVI pixels after merging")

        # thumbnail
        date_for_record = acquisition_date_override or datetime.date.today().isoformat()
        thumbnail_bytes = await loop.run_in_executor(executor, create_colorized_thumbnail, merged_ndvi)
        thumbnail_url = await loop.run_in_executor(executor, upload_thumbnail_to_supabase_sync, land_id, date_for_record, thumbnail_bytes)

        ndvi_record = {
            "tenant_id": tenant_id,
            "land_id": land_id,
            "date": date_for_record,
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
            "acquisition_date": date_for_record,
            "ndvi_mean": stats["mean"],
            "ndvi_min": stats["min"],
            "ndvi_max": stats["max"],
            "ndvi_std_dev": stats["std"],
            "ndvi_thumbnail_url": thumbnail_url,
            "bbox": geometry,
            "cloud_cover": 0,
            "created_at": now_iso()
        }

        # upsert DB rows in executor
        await loop.run_in_executor(executor, upsert_ndvi_data_sync, ndvi_record)
        await loop.run_in_executor(executor, upsert_micro_tile_sync, micro_tile_record)
        await loop.run_in_executor(executor, update_land_sync, land_id, {
            "last_ndvi_value": stats["mean"],
            "last_ndvi_calculation": date_for_record,
            "ndvi_thumbnail_url": thumbnail_url,
            "updated_at": now_iso()
        })

        out.update({"success": True, "stats": stats, "thumbnail_url": thumbnail_url})
        logger.info(f"✅ Land {land_id} processed: mean={stats['mean']:.3f}, coverage={stats['coverage']:.1f}%")

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"❌ Land {land_id} failed: {e}\n{tb}")
        out["error"] = str(e)
        # insert processing log
        try:
            await loop.run_in_executor(executor, insert_processing_log_sync, {
                "tenant_id": land.get("tenant_id"),
                "land_id": land_id,
                "processing_step": "ndvi_extraction",
                "step_status": "failed",
                "error_message": str(e)[:500],
                "error_details": {"traceback": tb[:1000]},
                "created_at": now_iso()
            })
        except Exception:
            pass
    return out

# ---------------- Orchestrator (async) ----------------
async def process_request_async(
    queue_id: str,
    tenant_id: str,
    land_ids: List[str],
    tile_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    logger.info(f"🚀 Queue {queue_id}: processing {len(land_ids)} lands for tenant {tenant_id}")
    start_ts = time.time()
    try:
        resp = supabase.table("lands").select("*").eq("tenant_id", tenant_id).in_("id", land_ids).execute()
        lands = getattr(resp, "data", None) or (resp.get("data") if isinstance(resp, dict) else None) or []
    except Exception as e:
        logger.error(f"Failed to fetch lands: {e}")
        lands = []

    if not lands:
        logger.warning("No lands found for processing")
        return {"queue_id": queue_id, "processed_count": 0, "failed_count": len(land_ids), "results": [], "duration_ms": 0}

    executor = ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS)
    sem = asyncio.Semaphore(MAX_CONCURRENT_LANDS)

    async def _proc(land):
        async with sem:
            return await process_single_land_async(land, tile_ids, None, executor)

    tasks = [asyncio.create_task(_proc(land)) for land in lands]
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
        logger.error(f"Failed to update ndvi_request_queue: {e}")

    return {
        "queue_id": queue_id,
        "processed_count": processed,
        "failed_count": len(failed),
        "results": results,
        "duration_ms": duration_ms
    }

# ---------------- Cron runner ----------------
def run_cron(limit: int = 10, max_retries: int = 3):
    logger.info("🔄 NDVI Worker Cron Start")
    try:
        queue_resp = supabase.table("ndvi_request_queue").select("*").eq("status", "queued").order("created_at", desc=False).limit(limit).execute()
        items = getattr(queue_resp, "data", None) or (queue_resp.get("data") if isinstance(queue_resp, dict) else None) or []
    except Exception as e:
        logger.error(f"Failed to fetch queue items: {e}")
        items = []

    if not items:
        logger.info("No queued items")
        return

    async def _handle_item(item):
        queue_id = item["id"]
        tenant_id = item["tenant_id"]
        land_ids = item.get("land_ids", [])
        tile_id_single = item.get("tile_id")
        retry_count = item.get("retry_count", 0)
        if retry_count >= max_retries:
            logger.warning(f"Max retries for queue {queue_id}")
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
            await process_request_async(queue_id, tenant_id, land_ids, [tile_id_single] if tile_id_single else None)
        except Exception as e:
            logger.exception(f"Queue {queue_id} failed: {e}")
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

# ---------------- CLI / Entrypoint ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NDVI Land Worker v9.2")
    parser.add_argument("--mode", choices=["cron", "single"], default="cron")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--queue-id", type=str, help="process specific queue id (single mode)")
    args = parser.parse_args()

    logger.info(f"Starting NDVI Worker v9.2 (mode={args.mode})")
    if args.mode == "single" and args.queue_id:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            queue_item = supabase.table("ndvi_request_queue").select("*").eq("id", args.queue_id).single().execute()
            item = getattr(queue_item, "data", None) or (queue_item.get("data") if isinstance(queue_item, dict) else None)
            if not item:
                logger.error("Queue id not found")
            else:
                res = loop.run_until_complete(process_request_async(item["id"], item["tenant_id"], item.get("land_ids", []), [item.get("tile_id")] if item.get("tile_id") else None))
                logger.info(f"Single run result: {res}")
            loop.close()
        except Exception as e:
            logger.exception(f"Single-run failed: {e}")
    else:
        run_cron(limit=args.limit)

    logger.info("Worker finished")
