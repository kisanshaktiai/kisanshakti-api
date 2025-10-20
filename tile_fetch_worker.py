#!/usr/bin/env python3
"""
tile_fetch_worker_v1.6.8_compress_opt.py
---------------------------------------
Streaming & compressing worker:
- Recompress Sentinel-2 bands (B04/B08) to LZW (tiled) before upload
- Stream NDVI calculation & write (no full-image arrays in RAM)
- Skip upload if file already exists in B2
- Upload with bucket.upload_local_file (no large memory buffering)
- JSON-safe numeric outputs for Supabase (ndvi_min/mean/std -> numeric(5,3))
"""

import os
import json
import datetime
import tempfile
import logging
import traceback
import time
import math
from typing import Tuple, Optional, Dict

import requests
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import mapping, shape
from shapely import wkb, wkt
from supabase import create_client
import planetary_computer as pc
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from requests.adapters import HTTPAdapter, Retry

# ---- Tunables ----
COMPRESSION = os.getenv("TIFF_COMPRESSION", "LZW")  # LZW recommended
BLOCK_ROWS = int(os.getenv("BLOCK_ROWS", "1024"))   # rows per block for streaming
B2_UPLOAD_RETRIES = int(os.getenv("B2_UPLOAD_RETRIES", "3"))
B2_LARGE_THRESHOLD = int(os.getenv("B2_LARGE_THRESHOLD_BYTES", str(100 * 1024 * 1024)))  # 100MB

# ---- Config from environment ----
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
B2_KEY_ID = os.getenv("B2_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
B2_BUCKET_NAME = os.getenv("B2_BUCKET_RAW", "kisanshakti-ndvi-tiles")
B2_PREFIX = os.getenv("B2_PREFIX", "tiles/")
MPC_COLLECTION = os.getenv("MPC_COLLECTION", "sentinel-2-l2a")
CLOUD_COVER_DEFAULT = int(os.getenv("DEFAULT_CLOUD_COVER_MAX", "100"))
LOOKBACK_DAYS_DEFAULT = int(os.getenv("MAX_SCENE_LOOKBACK_DAYS", "90"))
DOWNSAMPLE_FACTOR = int(os.getenv("DOWNSAMPLE_FACTOR", "1"))  # keep 1 for full-res

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tile_worker_v1.6.8")

# ---- Suppress B2SDK closed-file flush noise (benign) ----
try:
    import b2sdk._internal.stream.wrapper as b2wrap

    def _safe_flush(self, *a, **kw):
        try:
            if hasattr(self.stream, "flush"):
                self.stream.flush()
        except ValueError:
            # ignore "I/O on closed file"
            return

    b2wrap.ReadingStreamWithProgress.flush = _safe_flush
except Exception:
    # best-effort; not fatal
    pass

# ---- Clients ----
if not (SUPABASE_URL and SUPABASE_KEY and B2_KEY_ID and B2_APP_KEY):
    raise RuntimeError("Missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY / B2_KEY_ID / B2_APP_KEY env vars")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
info = InMemoryAccountInfo()
b2_api = B2Api(info)
b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
bucket = b2_api.get_bucket_by_name(B2_BUCKET_NAME)

session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))


# ---- Helpers ----
def safe_float(v, decimals: Optional[int] = None):
    if v is None:
        return None
    # convert numpy types to native Python
    try:
        if hasattr(v, "item"):
            v = v.item()
        f = float(v)
        if decimals is not None:
            return round(f, decimals)
        return f
    except Exception:
        return None


def decode_geom_to_geojson(tile_row: dict):
    """Prefer geojson_geometry for MPC, fallback to geometry (string / wkb / wkt)."""
    geom = tile_row.get("geojson_geometry") or tile_row.get("geometry")
    if geom is None:
        return None
    if isinstance(geom, dict) and "type" in geom:
        return geom
    if isinstance(geom, (bytes, bytearray)):
        return mapping(wkb.loads(geom))
    if isinstance(geom, str):
        s = geom.strip()
        # JSON string
        if s.startswith("{") and s.endswith("}"):
            try:
                return json.loads(s)
            except Exception:
                pass
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
    return None


def extract_bbox(geom_json):
    if not geom_json:
        return None
    g = shape(geom_json)
    minx, miny, maxx, maxy = g.bounds
    return {"type": "Polygon", "coordinates": [[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]] ]}


def get_b2_key(tile_id: str, acq_date: str, subdir: str, filename: str) -> str:
    return f"{B2_PREFIX}{subdir}/{tile_id}/{acq_date}/{filename}"


def b2_file_exists(b2_name: str) -> Tuple[bool, Optional[int]]:
    """Return (exists, size_bytes)"""
    try:
        info = bucket.get_file_info_by_name(b2_name)
        size = getattr(info, "size", None)
        return True, int(size) if size is not None else None
    except Exception:
        return False, None


def upload_local_file_with_retries(local_path: str, b2_name: str, max_attempts: int = B2_UPLOAD_RETRIES) -> Optional[int]:
    """Upload local file to B2 using upload_local_file (no huge memory). Returns size bytes or None."""
    for attempt in range(1, max_attempts + 1):
        try:
            if not os.path.exists(local_path):
                logger.error("Upload failed: local file missing: %s", local_path)
                return None
            size = os.path.getsize(local_path)
            # use upload_local_file which streams from disk into B2 SDK
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


def recompress_tif_streamed(input_path: str, compression: str = COMPRESSION) -> Optional[str]:
    """
    Recompress a GeoTIFF using block streaming to a new temp file (LZW by default).
    Returns path to compressed file or None on failure.
    """
    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    out_tmp.close()
    try:
        with rasterio.open(input_path) as src:
            meta = src.meta.copy()
            # Ensure single-band or multi-band preserved
            meta.update(tiled=True, compress=compression, blockxsize=256, blockysize=256)
            # For large images, adjust profile for writing
            with rasterio.open(out_tmp.name, "w", **meta) as dst:
                # stream block by block
                height = src.height
                width = src.width
                band_count = src.count
                for row_start in range(0, height, BLOCK_ROWS):
                    nrows = min(BLOCK_ROWS, height - row_start)
                    window = Window(0, row_start, width, nrows)
                    # read all bands in window
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


def compute_ndvi_streamed(red_path: str, nir_path: str, out_profile: dict) -> Tuple[Optional[str], Optional[dict]]:
    """
    Compute NDVI in blocks streaming from red/nir and writing compressed NDVI file.
    out_profile: template rasterio profile from red band (will be updated)
    Returns (ndvi_local_path, stats) or (None, None) on failure.
    """
    ndvi_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    ndvi_tmp.close()
    try:
        # prepare output profile
        profile = out_profile.copy()
        profile.update(dtype=rasterio.float32, count=1, compress=COMPRESSION, tiled=True, blockxsize=256, blockysize=256)

        # statistics accumulators
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
                # iterate rows
                for row_start in range(0, height, BLOCK_ROWS):
                    nrows = min(BLOCK_ROWS, height - row_start)
                    window = Window(0, row_start, width, nrows)
                    red = rsrc.read(1, window=window).astype("float32")
                    nir = nsrc.read(1, window=window).astype("float32")

                    # NDVI with safe division
                    with np.errstate(divide="ignore", invalid="ignore"):
                        denom = (nir + red)
                        ndvi = np.where(denom != 0, (nir - red) / denom, np.nan)

                    # write block to dst
                    dst.write(ndvi.astype(np.float32), 1, window=window)

                    # accumulate stats
                    mask_valid = np.isfinite(ndvi)
                    count_valid = int(mask_valid.sum())
                    total_pixels += nrows * width
                    if count_valid > 0:
                        block_vals = ndvi[mask_valid]
                        valid_pixels += count_valid
                        sum_val += float(block_vals.sum())
                        sum_sq += float((block_vals ** 2).sum())
                        block_min = float(block_vals.min())
                        block_max = float(block_vals.max())
                        if block_min < minv:
                            minv = block_min
                        if block_max > maxv:
                            maxv = block_max
                        veg_pixels += int((block_vals > 0.3).sum())

        # finalize stats
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
                "pixel_count": int(total_pixels),
                "valid_pixel_count": int(valid_pixels),
                "vegetation_health_score": safe_float(health, 2),
            }

        return ndvi_tmp.name, stats

    except Exception as e:
        logger.error("compute_ndvi_streamed failed: %s\n%s", e, traceback.format_exc())
        try:
            if os.path.exists(ndvi_tmp.name):
                os.unlink(ndvi_tmp.name)
        except Exception:
            pass
        return None, None


# ---- MPC & DB helpers ----
def query_mpc(tile_geom, start_date, end_date, cloud_limit=CLOUD_COVER_DEFAULT):
    try:
        url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
        body = {
            "collections": [MPC_COLLECTION],
            "intersects": tile_geom,
            "datetime": f"{start_date}/{end_date}",
            "query": {"eo:cloud_cover": {"lt": cloud_limit}}
        }
        logger.info("🛰️ Querying MPC: %s %s->%s, cloud<%s", MPC_COLLECTION, start_date, end_date, cloud_limit)
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
                s["properties"].get("eo:cloud_cover", 100),
                -datetime.datetime.fromisoformat(s["properties"]["datetime"].replace("Z", "+00:00")).timestamp()
            )
        )[0]
    except Exception:
        return scenes[0]


def _record_exists_in_db(tile_id: str, acq_date: str, collection: str = None):
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
    """Process one tile: get scene, fetch assets, compress bands, compute NDVI, upload to B2, upsert DB."""
    red_tmp = nir_tmp = red_cmp = nir_cmp = ndvi_file = None
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
        # get hrefs, fallback keys
        red_href = assets.get("red", assets.get("B04", {})).get("href")
        nir_href = assets.get("nir", assets.get("B08", {})).get("href")
        if not red_href or not nir_href:
            logger.error("Missing red/nir asset href for tile %s scene %s", tile_id, scene.get("id"))
            return False

        # sign urls using planetary_computer (this adds SAS token)
        try:
            red_url = pc.sign(red_href)
            nir_url = pc.sign(nir_href)
        except Exception:
            # pc.sign can accept item/asset or plain URL; fallback to use href directly
            red_url = red_href
            nir_url = nir_href

        # determine B2 target keys
        red_b2_key = get_b2_key(tile_id, acq_date, "raw", "B04.tif")
        nir_b2_key = get_b2_key(tile_id, acq_date, "raw", "B08.tif")
        ndvi_b2_key = get_b2_key(tile_id, acq_date, "ndvi", "ndvi.tif")

        # check existing files in B2
        red_exists, red_size = b2_file_exists(red_b2_key)
        nir_exists, nir_size = b2_file_exists(nir_b2_key)
        ndvi_exists, ndvi_size = b2_file_exists(ndvi_b2_key)
        logger.info("Existing in B2: red=%s nir=%s ndvi=%s", red_exists, nir_exists, ndvi_exists)

        # If both raw and ndvi exist and DB has record, update timestamp and skip heavy work
        db_exists, db_row = _record_exists_in_db(tile_id, acq_date, MPC_COLLECTION.upper())
        if red_exists and nir_exists and ndvi_exists and db_exists:
            logger.info("All files already exist for %s %s; updating timestamp only", tile_id, acq_date)
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
                logger.warning("DB upsert timestamp failed: %s", e)
                # continue to attempt processing if DB update fails

        # === Download raw bands (if needed) ===
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

        # If files exist in B2 but we didn't download them, fetch them to local temp for processing
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

        # === Recompress raw bands if not exist in B2 (or if you prefer reupload even if exists) ===
        if not red_exists:
            logger.info("Recompressing RED (streamed) for %s", tile_id)
            red_cmp = recompress_tif_streamed(red_tmp.name, compression=COMPRESSION)
            if not red_cmp:
                logger.error("Failed to recompress RED for %s", tile_id)
                return False
            red_size = upload_local_file_with_retries(red_cmp, red_b2_key)
            if not red_size:
                return False
        else:
            logger.info("Skipping recompress/upload of RED (already in B2)")

        if not nir_exists:
            logger.info("Recompressing NIR (streamed) for %s", tile_id)
            nir_cmp = recompress_tif_streamed(nir_tmp.name, compression=COMPRESSION)
            if not nir_cmp:
                logger.error("Failed to recompress NIR for %s", tile_id)
                return False
            nir_size = upload_local_file_with_retries(nir_cmp, nir_b2_key)
            if not nir_size:
                return False
        else:
            logger.info("Skipping recompress/upload of NIR (already in B2)")

        # === Compute NDVI streaming (needs local red/nir files) ===
        logger.info("Computing NDVI (streamed) for %s", tile_id)
        # re-open one of the files to produce a template profile (use compressed versions if available)
        profile_src = None
        profile_path = red_cmp or red_tmp or nir_cmp or nir_tmp
        if not profile_path:
            logger.error("No band file available to build profile for NDVI")
            return False
        with rasterio.open(profile_path) as src_profile:
            profile_src = src_profile.profile

        ndvi_file, stats = compute_ndvi_streamed(red_tmp.name, nir_tmp.name, profile_src)
        if not ndvi_file:
            logger.error("NDVI compute failed for %s", tile_id)
            return False

        # Upload NDVI if not present
        if not ndvi_exists:
            ndvi_size = upload_local_file_with_retries(ndvi_file, ndvi_b2_key)
            if not ndvi_size:
                logger.error("Failed to upload NDVI for %s", tile_id)
                return False
        else:
            logger.info("NDVI already present in B2; skipping upload")

        # Build DB payload (JSON-safe numbers)
        now = datetime.datetime.utcnow().isoformat() + "Z"
        total_size_mb = None
        try:
            total_size_bytes = (red_size or 0) + (nir_size or 0) + (ndvi_size or 0)
            total_size_mb = safe_float(total_size_bytes / (1024.0 * 1024.0), 2) if total_size_bytes else None
        except Exception:
            total_size_mb = None

        payload = {
            "tile_id": tile_id,
            "acquisition_date": acq_date,
            "collection": MPC_COLLECTION.upper(),
            "cloud_cover": safe_float(cloud_v, 2),
            "processing_level": "L2A",
            "file_size_mb": total_size_mb,
            "red_band_path": f"b2://{B2_BUCKET_NAME}/{red_b2_key}",
            "nir_band_path": f"b2://{B2_BUCKET_NAME}/{nir_b2_key}",
            "ndvi_path": f"b2://{B2_BUCKET_NAME}/{ndvi_b2_key}",
            "red_band_size_bytes": red_size,
            "nir_band_size_bytes": nir_size,
            "ndvi_size_bytes": ndvi_size,
            "resolution": "R10m",
            "status": "ready",
            "processing_method": "cog_streaming",
            "api_source": "planetary_computer",
            "actual_download_status": "downloaded",
            "processing_stage": "completed",
            "updated_at": now,
            "processing_completed_at": now,
            "ndvi_calculation_timestamp": now,
            "bbox": bbox,
            "country_id": tile_row.get("country_id"),
            "mgrs_tile_id": tile_row.get("id"),
        }

        # Add NDVI stats (numeric 5,3 etc.)
        if stats:
            payload.update({
                "ndvi_min": safe_float(stats.get("ndvi_min"), 3),
                "ndvi_max": safe_float(stats.get("ndvi_max"), 3),
                "ndvi_mean": safe_float(stats.get("ndvi_mean"), 3),
                "ndvi_std_dev": safe_float(stats.get("ndvi_std_dev"), 3),
                "vegetation_coverage_percent": safe_float(stats.get("vegetation_coverage_percent"), 2),
                "data_completeness_percent": safe_float(stats.get("data_completeness_percent"), 2),
                "pixel_count": int(stats.get("pixel_count")) if stats.get("pixel_count") is not None else None,
                "valid_pixel_count": int(stats.get("valid_pixel_count")) if stats.get("valid_pixel_count") is not None else None,
                "vegetation_health_score": safe_float(stats.get("vegetation_health_score"), 2)
            })

        # Upsert into DB with ON CONFLICT tile_id,acquisition_date,collection
        try:
            resp = supabase.table("satellite_tiles").upsert(payload, on_conflict="tile_id,acquisition_date,collection").execute()
            logger.info("✅ DB upsert complete for %s %s (resp ok)", tile_id, acq_date)
        except Exception as e:
            logger.error("DB upsert failed for %s: %s\n%s", tile_id, e, traceback.format_exc())
            # do not treat DB failure as fatal for uploaded data
            return False

        return True

    except Exception as e:
        logger.error("process_tile failed for %s: %s\n%s", tile_row.get("tile_id"), e, traceback.format_exc())
        return False
    finally:
        # cleanup temporary files
        for path in (red_tmp and getattr(red_tmp, "name", None),
                     nir_tmp and getattr(nir_tmp, "name", None),
                     red_cmp, nir_cmp, ndvi_file):
            if path and isinstance(path, str):
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                        logger.debug("Cleaned up temp: %s", path)
                except Exception:
                    pass


def fetch_agri_tiles():
    """Fetch active MGRS tiles with geojson_geometry preferred."""
    try:
        resp = supabase.table("mgrs_tiles").select("id, tile_id, geometry, geojson_geometry, country_id")\
            .eq("is_agri", True).eq("is_land_contain", True).execute()
        tiles = resp.data or []
        logger.info("Fetched %d agri tiles.", len(tiles))
        return tiles
    except Exception as e:
        logger.error("Failed to fetch mgrs_tiles: %s", e)
        return []


def main(cloud_cover: int = CLOUD_COVER_DEFAULT, lookback_days: int = LOOKBACK_DAYS_DEFAULT):
    logger.info("🚀 Starting tile worker compress_opt (cloud<%s, lookback=%s days)", cloud_cover, lookback_days)
    tiles = fetch_agri_tiles()
    if not tiles:
        logger.info("No tiles to process")
        return 0

    processed = 0
    for i, t in enumerate(tiles, start=1):
        logger.info("🔄 [%d/%d] Processing %s", i, len(tiles), t.get("tile_id"))
        ok = process_tile(t, cloud_cover, lookback_days)
        if ok:
            processed += 1
            logger.info("✅ [%d/%d] Success: %s", i, len(tiles), t.get("tile_id"))
        else:
            logger.info("⏭️  [%d/%d] Skipped/Failed: %s", i, len(tiles), t.get("tile_id"))

    logger.info("✨ Finished: processed %d/%d tiles successfully", processed, len(tiles))
    return processed


if __name__ == "__main__":
    cc = int(os.getenv("RUN_CLOUD_COVER", str(CLOUD_COVER_DEFAULT)))
    lb = int(os.getenv("RUN_LOOKBACK_DAYS", str(LOOKBACK_DAYS_DEFAULT)))
    main(cc, lb)
