#!/usr/bin/env python3
"""
tile_fetch_worker_v1.7.0_reliant_streamfix.py

- Reliable MPC -> B2 -> Supabase worker
- Streamed recompression and NDVI calculation (low memory)
- Always upserts satellite_tiles (marks failed where appropriate)
- Uses geojson_geometry for MPC queries (geometry still preserved in DB)
"""

import os
import json
import time
import math
import tempfile
import datetime
import logging
import traceback
from typing import Optional, Tuple, Dict

import requests
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely import wkb, wkt
from shapely.geometry import mapping, shape
import planetary_computer as pc
from supabase import create_client
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from requests.adapters import HTTPAdapter, Retry

# ---- Tunables (via env) ----
TIFF_COMPRESSION = os.getenv("TIFF_COMPRESSION", "LZW")   # LZW recommended
BLOCK_ROWS = int(os.getenv("BLOCK_ROWS", "1024"))         # rows processed per block
B2_UPLOAD_RETRIES = int(os.getenv("B2_UPLOAD_RETRIES", "3"))
B2_LARGE_THRESHOLD = int(os.getenv("B2_LARGE_THRESHOLD_BYTES", str(100 * 1024 * 1024)))  # 100MB
DOWNSAMPLE_FACTOR = int(os.getenv("DOWNSAMPLE_FACTOR", "1"))  # >1 reduces resolution to save memory

# ---- Config (required envs) ----
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
B2_KEY_ID = os.getenv("B2_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
B2_BUCKET_NAME = os.getenv("B2_BUCKET_RAW", "kisanshakti-ndvi-tiles")
B2_PREFIX = os.getenv("B2_PREFIX", "tiles/")
MPC_COLLECTION = os.getenv("MPC_COLLECTION", "sentinel-2-l2a")
CLOUD_COVER_DEFAULT = int(os.getenv("DEFAULT_CLOUD_COVER_MAX", "100"))
LOOKBACK_DAYS_DEFAULT = int(os.getenv("MAX_SCENE_LOOKBACK_DAYS", "90"))

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tile_worker_v1.7.0")

# ---- Validate envs ----
if not (SUPABASE_URL and SUPABASE_KEY and B2_KEY_ID and B2_APP_KEY):
    raise RuntimeError("Missing required env vars: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, B2_KEY_ID, B2_APP_KEY")

# ---- Clients ----
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

info = InMemoryAccountInfo()
b2_api = B2Api(info)
b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
bucket = b2_api.get_bucket_by_name(B2_BUCKET_NAME)

session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

# ---- Small helpers ----
def safe_float(value, decimals: Optional[int] = None):
    """Return native python float or None; round if decimals provided."""
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            value = value.item()
        f = float(value)
        return round(f, decimals) if decimals is not None else f
    except Exception:
        return None


def to_int(value):
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            value = value.item()
        return int(value)
    except Exception:
        return None


def decode_geom_to_geojson(tile_row: dict):
    """Prefer geojson_geometry for MPC queries; fallback to geometry."""
    geom = tile_row.get("geojson_geometry") or tile_row.get("geometry")
    if geom is None:
        return None
    try:
        if isinstance(geom, dict) and "type" in geom:
            return geom
        if isinstance(geom, (bytes, bytearray)):
            return mapping(wkb.loads(geom))
        if isinstance(geom, str):
            s = geom.strip()
            if s.startswith("{") and s.endswith("}"):
                return json.loads(s)
            # try WKT
            try:
                return mapping(wkt.loads(s))
            except Exception:
                pass
            # try hex WKB
            try:
                return mapping(wkb.loads(bytes.fromhex(s)))
            except Exception:
                pass
    except Exception as e:
        logger.warning("decode_geom_to_geojson failed: %s", e)
    return None


def extract_bbox(geom_json):
    if not geom_json:
        return None
    g = shape(geom_json)
    minx, miny, maxx, maxy = g.bounds
    return {"type": "Polygon", "coordinates": [[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]]}


def get_b2_key(tile_id: str, acq_date: str, subdir: str, filename: str) -> str:
    return f"{B2_PREFIX}{subdir}/{tile_id}/{acq_date}/{filename}"


def b2_file_exists(b2_name: str) -> Tuple[bool, Optional[int]]:
    try:
        info = bucket.get_file_info_by_name(b2_name)
        size = getattr(info, "size", None)
        return True, int(size) if size is not None else None
    except Exception:
        return False, None


def upload_local_file_with_retries(local_path: str, b2_name: str, max_attempts: int = B2_UPLOAD_RETRIES) -> Optional[int]:
    """Upload a local file to B2, returns size bytes on success."""
    for attempt in range(1, max_attempts + 1):
        try:
            if not os.path.exists(local_path):
                logger.error("Upload failed: local file missing: %s", local_path)
                return None
            size = os.path.getsize(local_path)
            # prefer upload_local_file (streams from disk)
            bucket.upload_local_file(local_file=local_path, file_name=b2_name)
            logger.info("✅ Uploaded %s (%.2fMB)", b2_name, size / 1024.0 / 1024.0)
            return int(size)
        except Exception as e:
            logger.warning("Upload attempt %d/%d failed for %s: %s", attempt, max_attempts, b2_name, str(e))
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
            else:
                logger.error("Upload failed after %d attempts: %s", max_attempts, b2_name)
                return None
    return None


def recompress_tif_streamed(input_path: str, compression: str = TIFF_COMPRESSION) -> Optional[str]:
    """
    Recompress GeoTIFF to a new temporary file (LZW tiled) using block streaming.
    Returns path to compressed file or None.
    """
    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    out_tmp.close()
    try:
        with rasterio.open(input_path) as src:
            meta = src.meta.copy()
            # optionally downsample by integer factor
            if DOWNSAMPLE_FACTOR > 1:
                # We'll write full-size meta but we'll resample when reading blocks below.
                # Simpler: create smaller output by adjusting height/width/transform
                new_h = src.height // DOWNSAMPLE_FACTOR
                new_w = src.width // DOWNSAMPLE_FACTOR
                transform = src.transform * src.transform.scale(
                    (src.width / new_w),
                    (src.height / new_h)
                )
                meta.update(height=new_h, width=new_w, transform=transform)
            # ensure tiled + compression
            meta.update(tiled=True, compress=compression, blockxsize=256, blockysize=256)

            with rasterio.open(out_tmp.name, "w", **meta) as dst:
                height = src.height
                width = src.width
                band_count = src.count
                # iterate windows; if downsampling, read window and resample using rasterio's read(..., out_shape=...)
                for row_start in range(0, height, BLOCK_ROWS):
                    nrows = min(BLOCK_ROWS, height - row_start)
                    window = Window(0, row_start, width, nrows)
                    if DOWNSAMPLE_FACTOR > 1:
                        # compute out_shape corresponding to the destination (scaled rows)
                        out_rows = max(1, nrows // DOWNSAMPLE_FACTOR)
                        out_shape = (band_count, out_rows, max(1, width // DOWNSAMPLE_FACTOR))
                        data = src.read(window=window, out_shape=out_shape, resampling=rasterio.enums.Resampling.average)
                        # compute target window in dst
                        dst_row_start = row_start // DOWNSAMPLE_FACTOR
                        dst_window = Window(0, dst_row_start, out_shape[2], out_shape[1])
                        dst.write(data, window=dst_window)
                    else:
                        data = src.read(window=window)
                        dst.write(data, window=window)
        return out_tmp.name
    except Exception as e:
        logger.error("recompress_tif_streamed failed: %s", e)
        try:
            if os.path.exists(out_tmp.name):
                os.unlink(out_tmp.name)
        except Exception:
            pass
        return None


def compute_ndvi_streamed_and_write(red_path: str, nir_path: str, out_profile: dict) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Compute NDVI streaming from red/nir and write to a compressed tif.
    Returns (ndvi_local_path, stats) or (None, None) on failure.
    """
    ndvi_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    ndvi_tmp.close()
    try:
        profile = out_profile.copy()
        profile.update(dtype=rasterio.float32, count=1, compress=TIFF_COMPRESSION, tiled=True, blockxsize=256, blockysize=256)

        total_pixels = 0
        valid_pixels = 0
        sum_val = 0.0
        sum_sq = 0.0
        minv = float("inf")
        maxv = float("-inf")
        veg_pixels = 0

        with rasterio.open(red_path) as rsrc, rasterio.open(nir_path) as nsrc:
            height = rsrc.height
            width = rsrc.width

            with rasterio.open(ndvi_tmp.name, "w", **profile) as dst:
                for row_start in range(0, height, BLOCK_ROWS):
                    nrows = min(BLOCK_ROWS, height - row_start)
                    window = Window(0, row_start, width, nrows)
                    red = rsrc.read(1, window=window).astype("float32")
                    nir = nsrc.read(1, window=window).astype("float32")

                    with np.errstate(divide="ignore", invalid="ignore"):
                        denom = nir + red
                        ndvi = np.where(denom != 0, (nir - red) / denom, np.nan)

                    dst.write(ndvi.astype(np.float32), 1, window=window)

                    mask_valid = np.isfinite(ndvi)
                    count_valid = int(mask_valid.sum())
                    total_pixels += int(nrows * width)
                    if count_valid > 0:
                        vals = ndvi[mask_valid]
                        valid_pixels += count_valid
                        sum_val += float(vals.sum())
                        sum_sq += float((vals ** 2).sum())
                        block_min = float(vals.min())
                        block_max = float(vals.max())
                        if block_min < minv:
                            minv = block_min
                        if block_max > maxv:
                            maxv = block_max
                        veg_pixels += int((vals > 0.3).sum())

        if valid_pixels == 0:
            stats = {
                "ndvi_min": None, "ndvi_max": None, "ndvi_mean": None, "ndvi_std_dev": None,
                "vegetation_coverage_percent": 0.0, "data_completeness_percent": 0.0,
                "pixel_count": total_pixels, "valid_pixel_count": 0, "vegetation_health_score": None
            }
        else:
            mean = sum_val / valid_pixels
            variance = (sum_sq / valid_pixels) - (mean ** 2)
            std = math.sqrt(max(0.0, variance))
            veg_cov = veg_pixels / valid_pixels * 100.0
            data_comp = valid_pixels / total_pixels * 100.0
            health = ((mean + 1.0) / 2.0 * 100.0) * 0.5 + veg_cov * 0.3 + data_comp * 0.2

            stats = {
                "ndvi_min": safe_float(minv, 3),
                "ndvi_max": safe_float(maxv, 3),
                "ndvi_mean": safe_float(mean, 3),
                "ndvi_std_dev": safe_float(std, 3),
                "vegetation_coverage_percent": safe_float(veg_cov, 2),
                "data_completeness_percent": safe_float(data_comp, 2),
                "pixel_count": to_int(total_pixels),
                "valid_pixel_count": to_int(valid_pixels),
                "vegetation_health_score": safe_float(health, 2),
            }

        return ndvi_tmp.name, stats
    except Exception as e:
        logger.error("compute_ndvi_streamed_and_write failed: %s\n%s", e, traceback.format_exc())
        try:
            if os.path.exists(ndvi_tmp.name):
                os.unlink(ndvi_tmp.name)
        except Exception:
            pass
        return None, None


# ---- MPC Query helpers ----
def query_mpc(tile_geom, start_date, end_date, cloud_limit=CLOUD_COVER_DEFAULT):
    url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    body = {
        "collections": [MPC_COLLECTION],
        "intersects": tile_geom,
        "datetime": f"{start_date}/{end_date}",
        "query": {"eo:cloud_cover": {"lt": cloud_limit}}
    }
    try:
        logger.info("🛰️ Querying MPC: %s %s->%s cloud<%s", MPC_COLLECTION, start_date, end_date, cloud_limit)
        r = session.post(url, json=body, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data.get("features", [])
    except Exception as e:
        logger.error("query_mpc error: %s", e)
        return []


def pick_best_scene(scenes):
    if not scenes:
        return None
    try:
        return sorted(
            scenes,
            key=lambda s: (
                s.get("properties", {}).get("eo:cloud_cover", 100),
                -datetime.datetime.fromisoformat(s.get("properties", {}).get("datetime", "").replace("Z", "+00:00")).timestamp()
            )
        )[0]
    except Exception:
        return scenes[0]


def _record_exists_in_db(tile_id: str, acq_date: str, collection: Optional[str] = None):
    try:
        q = supabase.table("satellite_tiles").select("id, created_at").eq("tile_id", tile_id).eq("acquisition_date", acq_date)
        if collection:
            q = q.eq("collection", collection)
        resp = q.limit(1).execute()
        rows = getattr(resp, "data", None) or []
        return (len(rows) > 0, rows[0] if rows else None)
    except Exception as e:
        logger.warning("DB check failed: %s", e)
        return False, None


# ---- Core processing ----
def process_tile(tile_row: dict, cloud_cover: int = CLOUD_COVER_DEFAULT, lookback_days: int = LOOKBACK_DAYS_DEFAULT) -> bool:
    red_tmp = nir_tmp = red_cmp = nir_cmp = ndvi_file = None
    red_size = nir_size = ndvi_size = None
    stats = None
    acq_date = None

    try:
        tile_id = tile_row.get("tile_id")
        if not tile_id:
            logger.error("Tile row missing tile_id")
            return False

        geom_json = decode_geom_to_geojson(tile_row)
        if not geom_json:
            logger.error("No usable geojson for tile %s", tile_id)
            return False

        bbox = extract_bbox(geom_json)
        today = datetime.date.today()
        start_date = (today - datetime.timedelta(days=lookback_days)).isoformat()
        end_date = today.isoformat()

        scenes = query_mpc(geom_json, start_date, end_date, cloud_cover)
        if not scenes:
            logger.info("No scenes for %s in %s..%s", tile_id, start_date, end_date)
            return False

        scene = pick_best_scene(scenes)
        if not scene:
            logger.info("No best scene for %s", tile_id)
            return False

        acq_date = scene["properties"]["datetime"].split("T")[0]
        cloud_v = scene["properties"].get("eo:cloud_cover")

        assets = scene.get("assets", {})
        red_href = assets.get("red", assets.get("B04", {})).get("href")
        nir_href = assets.get("nir", assets.get("B08", {})).get("href")
        if not red_href or not nir_href:
            logger.error("Missing red/nir asset href for tile %s scene %s", tile_id, scene.get("id"))
            return False

        # sign urls (SAS) if possible
        try:
            red_url = pc.sign(red_href)
        except Exception:
            red_url = red_href
        try:
            nir_url = pc.sign(nir_href)
        except Exception:
            nir_url = nir_href

        red_b2_key = get_b2_key(tile_id, acq_date, "raw", "B04.tif")
        nir_b2_key = get_b2_key(tile_id, acq_date, "raw", "B08.tif")
        ndvi_b2_key = get_b2_key(tile_id, acq_date, "ndvi", "ndvi.tif")

        red_exists, red_size_b2 = b2_file_exists(red_b2_key)
        nir_exists, nir_size_b2 = b2_file_exists(nir_b2_key)
        ndvi_exists, ndvi_size_b2 = b2_file_exists(ndvi_b2_key)
        logger.info("Existing in B2: red=%s nir=%s ndvi=%s", red_exists, nir_exists, ndvi_exists)

        db_exists, db_row = _record_exists_in_db(tile_id, acq_date, MPC_COLLECTION.upper())
        original_created_at = db_row.get("created_at") if db_row else None

        # If all exist and DB exists, update timestamp and skip
        if red_exists and nir_exists and ndvi_exists and db_exists:
            logger.info("All files exist for %s %s; updating timestamp only", tile_id, acq_date)
            try:
                now = datetime.datetime.utcnow().isoformat() + "Z"
                supabase.table("satellite_tiles").upsert({
                    "tile_id": tile_id,
                    "acquisition_date": acq_date,
                    "collection": MPC_COLLECTION.upper(),
                    "updated_at": now
                }, on_conflict="tile_id,acquisition_date,collection").execute()
                return True
            except Exception as e:
                logger.warning("DB timestamp update failed: %s", e)
                # continue to attempt processing if DB update fails

        # === Download bands if missing locally ===
        if not red_exists:
            red_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
            red_tmp.close()
            logger.info("Downloading RED for %s -> %s", tile_id, red_tmp.name)
            with session.get(red_url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(red_tmp.name, "wb") as f:
                    for chunk in r.iter_content(chunk_size=512 * 1024):
                        if chunk:
                            f.write(chunk)
        else:
            logger.info("RED exists in B2; will download from B2 if needed for NDVI computation")

        if not nir_exists:
            nir_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
            nir_tmp.close()
            logger.info("Downloading NIR for %s -> %s", tile_id, nir_tmp.name)
            with session.get(nir_url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(nir_tmp.name, "wb") as f:
                    for chunk in r.iter_content(chunk_size=512 * 1024):
                        if chunk:
                            f.write(chunk)
        else:
            logger.info("NIR exists in B2; will download from B2 if needed for NDVI computation")

        # If existed in B2 and we didn't download, fetch locally for computation
        if red_exists and not red_tmp:
            red_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
            red_tmp.close()
            logger.info("Downloading RED from B2 for computation: %s", red_b2_key)
            bucket.download_file_by_name(red_b2_key, red_tmp.name)

        if nir_exists and not nir_tmp:
            nir_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
            nir_tmp.close()
            logger.info("Downloading NIR from B2 for computation: %s", nir_b2_key)
            bucket.download_file_by_name(nir_b2_key, nir_tmp.name)

        # === Recompress & Upload raw bands if they weren't in B2 ===
        if not red_exists:
            logger.info("Recompressing RED for %s", tile_id)
            red_cmp = recompress_tif_streamed(red_tmp.name, compression=TIFF_COMPRESSION)
            if not red_cmp:
                logger.error("Failed recompress RED for %s", tile_id)
                # make sure to attempt a DB upsert marking failure below
                raise Exception("Failed recompress RED")
            red_size = upload_local_file_with_retries(red_cmp, red_b2_key)
            if not red_size:
                raise Exception("Failed upload RED to B2")
        else:
            red_size = red_size_b2

        if not nir_exists:
            logger.info("Recompressing NIR for %s", tile_id)
            nir_cmp = recompress_tif_streamed(nir_tmp.name, compression=TIFF_COMPRESSION)
            if not nir_cmp:
                raise Exception("Failed recompress NIR")
            nir_size = upload_local_file_with_retries(nir_cmp, nir_b2_key)
            if not nir_size:
                raise Exception("Failed upload NIR to B2")
        else:
            nir_size = nir_size_b2

        # === NDVI computation (streamed) ===
        logger.info("Computing NDVI for %s", tile_id)
        # pick a profile source for NDVI output (prefer compressed versions)
        profile_src_path = red_cmp or red_tmp or nir_cmp or nir_tmp
        if not profile_src_path:
            raise Exception("No band available to build profile")
        with rasterio.open(profile_src_path) as psrc:
            profile = psrc.profile.copy()

        ndvi_file, stats = compute_ndvi_streamed_and_write(red_tmp.name, nir_tmp.name, profile)
        if not ndvi_file:
            raise Exception("NDVI computation failed")
        # upload NDVI if absent in B2
        if not ndvi_exists:
            ndvi_size = upload_local_file_with_retries(ndvi_file, ndvi_b2_key)
            if not ndvi_size:
                raise Exception("Failed upload NDVI")
        else:
            ndvi_size = ndvi_size_b2

        # === Build DB payload ===
        now = datetime.datetime.utcnow().isoformat() + "Z"
        total_bytes = (red_size or 0) + (nir_size or 0) + (ndvi_size or 0)
        total_mb = safe_float(total_bytes / (1024.0 * 1024.0), 2) if total_bytes else None

        payload = {
            "tile_id": tile_id,
            "acquisition_date": acq_date,
            "collection": MPC_COLLECTION.upper(),
            "cloud_cover": safe_float(cloud_v, 2),
            "processing_level": "L2A",
            "file_size_mb": total_mb,
            "red_band_path": f"b2://{B2_BUCKET_NAME}/{red_b2_key}",
            "nir_band_path": f"b2://{B2_BUCKET_NAME}/{nir_b2_key}",
            "ndvi_path": f"b2://{B2_BUCKET_NAME}/{ndvi_b2_key}",
            "red_band_size_bytes": to_int(red_size),
            "nir_band_size_bytes": to_int(nir_size),
            "ndvi_size_bytes": to_int(ndvi_size),
            "resolution": "R10m" if DOWNSAMPLE_FACTOR == 1 else f"R{10 * DOWNSAMPLE_FACTOR}m",
            "status": "ready",
            "processing_method": "cog_streaming",
            "api_source": "planetary_computer",
            "actual_download_status": "downloaded",
            "processing_stage": "completed",
            "updated_at": now,
            "processing_completed_at": now,
            "ndvi_calculation_timestamp": now,
            "bbox": json.dumps(bbox) if bbox else None,
            "country_id": tile_row.get("country_id"),
            "mgrs_tile_id": tile_row.get("id"),
        }

        if original_created_at:
            payload["created_at"] = original_created_at
        else:
            payload["created_at"] = now

        if stats:
            # stats already rounded in compute step
            payload.update({
                "ndvi_min": safe_float(stats.get("ndvi_min"), 3),
                "ndvi_max": safe_float(stats.get("ndvi_max"), 3),
                "ndvi_mean": safe_float(stats.get("ndvi_mean"), 3),
                "ndvi_std_dev": safe_float(stats.get("ndvi_std_dev"), 3),
                "vegetation_coverage_percent": safe_float(stats.get("vegetation_coverage_percent"), 2),
                "data_completeness_percent": safe_float(stats.get("data_completeness_percent"), 2),
                "pixel_count": to_int(stats.get("pixel_count")),
                "valid_pixel_count": to_int(stats.get("valid_pixel_count")),
                "vegetation_health_score": safe_float(stats.get("vegetation_health_score"), 2),
            })

        # Upsert into DB
        try:
            resp = supabase.table("satellite_tiles").upsert(payload, on_conflict="tile_id,acquisition_date,collection").execute()
            logger.info("✅ DB upsert complete for %s %s", tile_id, acq_date)
        except Exception as e:
            logger.error("DB upsert failed for %s: %s\n%s", tile_id, e, traceback.format_exc())
            # In the unlikely case supabase client fails, still return failure to trigger manual review
            return False

        return True

    except Exception as e:
        logger.error("process_tile failed for %s: %s\n%s", tile_row.get("tile_id"), e, traceback.format_exc())

        # Ensure we persist a db record marking failure with available metadata
        try:
            now = datetime.datetime.utcnow().isoformat() + "Z"
            partial = {
                "tile_id": tile_row.get("tile_id"),
                "acquisition_date": acq_date or (datetime.date.today().isoformat()),
                "collection": MPC_COLLECTION.upper(),
                "status": "failed",
                "error_message": str(e),
                "updated_at": now,
                "processing_completed_at": now,
                "ndvi_calculation_timestamp": now,
                "bbox": json.dumps(extract_bbox(decode_geom_to_geojson(tile_row))) if decode_geom_to_geojson(tile_row) else None,
                "country_id": tile_row.get("country_id"),
                "mgrs_tile_id": tile_row.get("id"),
            }
            # include sizes where available
            if red_size:
                partial["red_band_size_bytes"] = to_int(red_size)
            if nir_size:
                partial["nir_band_size_bytes"] = to_int(nir_size)
            if ndvi_size:
                partial["ndvi_size_bytes"] = to_int(ndvi_size)

            # ensure created_at set if possible
            try:
                db_exists, db_row = _record_exists_in_db(tile_row.get("tile_id"), partial["acquisition_date"], MPC_COLLECTION.upper())
                if db_exists and db_row and db_row.get("created_at"):
                    partial["created_at"] = db_row.get("created_at")
                else:
                    partial["created_at"] = now
            except Exception:
                partial["created_at"] = now

            supabase.table("satellite_tiles").upsert(partial, on_conflict="tile_id,acquisition_date,collection").execute()
            logger.info("Inserted failure record for %s", tile_row.get("tile_id"))
        except Exception as db_e:
            logger.error("Failed to insert failure record: %s\n%s", db_e, traceback.format_exc())

        return False

    finally:
        # cleanup all temporary files (check names)
        for p in (red_tmp and getattr(red_tmp, "name", None),
                  nir_tmp and getattr(nir_tmp, "name", None),
                  red_cmp, nir_cmp, ndvi_file):
            if p and isinstance(p, str):
                try:
                    if os.path.exists(p):
                        os.unlink(p)
                        logger.debug("Cleaned up temp: %s", p)
                except Exception:
                    pass


def fetch_agri_tiles():
    """Fetch active MGRS tiles; prefer geojson_geometry for MPC queries."""
    try:
        resp = supabase.table("mgrs_tiles").select("id, tile_id, geometry, geojson_geometry, country_id") \
            .eq("is_agri", True).eq("is_land_contain", True).execute()
        tiles = getattr(resp, "data", None) or []
        logger.info("Fetched %d agri tiles.", len(tiles))
        return tiles
    except Exception as e:
        logger.error("Failed to fetch mgrs_tiles: %s", e)
        return []


def main(cloud_cover: int = CLOUD_COVER_DEFAULT, lookback_days: int = LOOKBACK_DAYS_DEFAULT):
    logger.info("🚀 Starting tile worker v1.7.0 (cloud<%s, lookback=%s days)", cloud_cover, lookback_days)
    tiles = fetch_agri_tiles()
    if not tiles:
        logger.info("No tiles to process.")
        return 0

    processed = 0
    for i, t in enumerate(tiles, start=1):
        tile_id = t.get("tile_id")
        logger.info("🔄 [%d/%d] Processing %s", i, len(tiles), tile_id)
        try:
            ok = process_tile(t, cloud_cover, lookback_days)
            if ok:
                processed += 1
                logger.info("✅ [%d/%d] Success: %s", i, len(tiles), tile_id)
            else:
                logger.info("⏭️  [%d/%d] Skipped/Failed: %s", i, len(tiles), tile_id)
        except Exception as e:
            logger.error("Top-level error while processing %s: %s", tile_id, e)

    logger.info("✨ Finished: processed %d/%d tiles successfully", processed, len(tiles))
    return processed


if __name__ == "__main__":
    cc = int(os.getenv("RUN_CLOUD_COVER", str(CLOUD_COVER_DEFAULT)))
    lb = int(os.getenv("RUN_LOOKBACK_DAYS", str(LOOKBACK_DAYS_DEFAULT)))
    main(cc, lb)
