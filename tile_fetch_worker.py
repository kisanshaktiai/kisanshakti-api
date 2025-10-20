#!/usr/bin/env python3
"""
tile_fetch_worker.py (v1.5.0)
- Uses MPC pystac_client + planetary_computer.sign_inplace
- Prefers geojson_geometry (falls back to geometry)
- Sanitizes payloads to avoid numpy/int64 JSON errors
- Chunked NDVI compute and streaming write (COG-friendly)
- Uploads to Backblaze B2 and upserts to Supabase
"""

import os
import io
import json
import datetime
import tempfile
import logging
import traceback
import requests
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely import wkb, wkt
from shapely.geometry import mapping, shape
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from supabase import create_client
import pystac_client
import planetary_computer

# ---------------- CONFIG ----------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

B2_KEY_ID = os.getenv("B2_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
B2_BUCKET = os.getenv("B2_BUCKET_RAW", "kisanshakti-ndvi-tiles")
B2_PREFIX = os.getenv("B2_PREFIX", "tiles/")

MPC_COLLECTION = os.getenv("MPC_COLLECTION", "sentinel-2-l2a")
CLOUD_COVER = int(os.getenv("DEFAULT_CLOUD_COVER_MAX", "20"))
LOOKBACK_DAYS = int(os.getenv("MAX_SCENE_LOOKBACK_DAYS", "5"))
DOWNSAMPLE_FACTOR = int(os.getenv("DOWNSAMPLE_FACTOR", "4"))

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tile-fetch-worker")

# ---------------- CLIENTS ----------------
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

if not B2_KEY_ID or not B2_APP_KEY:
    raise RuntimeError("Missing B2_KEY_ID or B2_APP_KEY")

# B2 client
info = InMemoryAccountInfo()
b2_api = B2Api(info)
b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
bucket = b2_api.get_bucket_by_name(B2_BUCKET)

# HTTP session
session = requests.Session()
session.headers.update({"User-Agent": "tile-fetch-worker/1.6.3"})

# MPC client with in-place signing so assets' hrefs include SAS tokens
mpc_catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# ---------------- Utilities ----------------

def decode_geom_to_geojson(geom_value):
    """Return a GeoJSON dict from either geojson text, WKB bytes, or WKT string."""
    try:
        if geom_value is None:
            return None
        if isinstance(geom_value, dict) and "type" in geom_value:
            return geom_value
        if isinstance(geom_value, (bytes, bytearray)):
            return mapping(wkb.loads(geom_value))
        if isinstance(geom_value, str):
            s = geom_value.strip()
            if s.startswith("{"):
                return json.loads(s)
            # try WKT
            try:
                return mapping(wkt.loads(s))
            except Exception:
                # maybe hex WKB
                try:
                    return mapping(wkb.loads(bytes.fromhex(s)))
                except Exception:
                    return None
    except Exception as e:
        logger.error(f"decode_geom_to_geojson error: {e}")
    return None


def extract_bbox(geom_json):
    try:
        g = shape(geom_json)
        minx, miny, maxx, maxy = g.bounds
        return {"type": "Polygon", "coordinates": [[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]]}
    except Exception:
        return None


# JSON sanitizer to convert numpy scalars and other non-serializable types to Python builtins
def _to_python_native(value):
    import numbers
    # numpy integers/floats/bools
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    # bytes
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode()
        except Exception:
            return str(value)
    # datetimes
    if isinstance(value, (datetime.datetime, datetime.date)):
        if isinstance(value, datetime.datetime):
            return value.isoformat() + "Z" if value.tzinfo is None else value.isoformat()
        return value.isoformat()
    # lists/tuples
    if isinstance(value, (list, tuple)):
        return [_to_python_native(v) for v in value]
    # dicts
    if isinstance(value, dict):
        return sanitize_payload(value)
    # numpy arrays -> list (careful, only small)
    if isinstance(value, np.ndarray):
        return value.tolist()
    # numpy generic .item()
    if hasattr(value, "item"):
        try:
            return _to_python_native(value.item())
        except Exception:
            pass
    # primitives
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # fallback
    return str(value)

def sanitize_payload(d):
    """Recursively sanitize dictionary values to JSON-serializable Python types."""
    if not isinstance(d, dict):
        return _to_python_native(d)
    out = {}
    for k, v in d.items():
        try:
            out[k] = _to_python_native(v)
        except Exception:
            out[k] = str(v)
    return out

# ---------------- B2 helpers ----------------

def b2_path_for(tile_id, acq_date, band_or_ndvi):
    """
    band_or_ndvi: 'raw/B04', 'raw/B08', 'ndvi', etc.
    examples returned: 'tiles/raw/<tile_id>/<date>/B04.tif'
    """
    if band_or_ndvi.startswith("raw/"):
        _, fname = band_or_ndvi.split("/", 1)
        return f"{B2_PREFIX}raw/{tile_id}/{acq_date}/{fname}.tif"
    if band_or_ndvi == "ndvi":
        return f"{B2_PREFIX}ndvi/{tile_id}/{acq_date}/ndvi.tif"
    # generic
    return f"{B2_PREFIX}{band_or_ndvi}/{tile_id}/{acq_date}"

def check_b2_exists(path):
    """Return (exists: bool, size_bytes: int|None)"""
    try:
        info = bucket.get_file_info_by_name(path)
        size = getattr(info, "size", None)
        return True, int(size) if size is not None else None
    except Exception:
        return False, None

def upload_local_to_b2(local_path, b2_path):
    """Upload a local file to B2 and return b2:// URI and size (int)."""
    try:
        bucket.upload_local_file(local_file=local_path, file_name=b2_path)
        size = os.path.getsize(local_path)
        return f"b2://{B2_BUCKET}/{b2_path}", int(size)
    except Exception:
        raise

# ---------------- MPC Query ----------------

def query_mpc_item_collection(geom_json, start_date, end_date, cloud_cover=CLOUD_COVER, limit=5):
    """Return a list of pystac.Item objects (signed) from MPC for the supplied GeoJSON intersect."""
    try:
        if not geom_json:
            return []
        logger.info(f"🛰️ Querying MPC STAC: {MPC_COLLECTION} {start_date}->{end_date}, cloud<{cloud_cover}%")
        search = mpc_catalog.search(
            collections=[MPC_COLLECTION],
            intersects=geom_json,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": cloud_cover}},
            limit=limit,
        )
        # item_collection() is the recommended method instead of get_all_items()
        item_collection = search.item_collection()
        items = list(item_collection.get_all_items()) if hasattr(item_collection, "get_all_items") else list(item_collection)
        return items
    except Exception as e:
        logger.error(f"MPC query failed: {e}")
        logger.debug(traceback.format_exc())
        return []

def pick_best_scene(items):
    """Given a list of pystac.Item objects, pick the one with lowest cloud then latest datetime."""
    try:
        if not items:
            return None
        def key_fn(it):
            props = it.properties
            cc = props.get("eo:cloud_cover", 100)
            dt = props.get("datetime")
            # robust parse
            dt_ts = 0
            try:
                dt_ts = datetime.datetime.fromisoformat(dt.replace("Z", "+00:00")).timestamp()
            except Exception:
                dt_ts = 0
            return (float(cc), -dt_ts)
        items_sorted = sorted(items, key=key_fn)
        return items_sorted[0] if items_sorted else None
    except Exception as e:
        logger.error(f"pick_best_scene error: {e}")
        logger.debug(traceback.format_exc())
        return None

# ---------------- Download & compress band ----------------

def download_and_compress_band(href, b2_path, downsample_factor=DOWNSAMPLE_FACTOR):
    """
    Download a band (href should be signed so direct GET works), optionally downsample,
    compress to LZW tiled COG-like TIFF and upload to B2.
    Returns (local_compressed_path, b2_uri, file_size_bytes)
    """
    raw_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    compressed_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    raw_tmp.close()
    compressed_tmp.close()
    try:
        logger.info(f"📥 Downloading {href[:160]}")
        r = session.get(href, stream=True, timeout=180)
        if not r.ok:
            logger.error(f"Download failed: HTTP {r.status_code} for {href}")
            return None, None, None
        # stream to raw_tmp
        with open(raw_tmp.name, "wb") as f:
            for chunk in r.iter_content(chunk_size=512 * 1024):
                if chunk:
                    f.write(chunk)
        # compress / downsample
        with rasterio.open(raw_tmp.name) as src:
            meta = src.meta.copy()
            if downsample_factor and downsample_factor > 1:
                out_h = max(1, src.height // downsample_factor)
                out_w = max(1, src.width // downsample_factor)
                data = src.read(1, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.average)
                # scale transform
                scale_x = src.width / out_w if out_w else 1
                scale_y = src.height / out_h if out_h else 1
                new_transform = src.transform * src.transform.scale(scale_x, scale_y)
                meta.update({"height": out_h, "width": out_w, "transform": new_transform})
            else:
                data = src.read(1)
            # set COG-friendly/write options
            meta.update({
                "driver": "GTiff",
                "dtype": "float32",
                "count": 1,
                "compress": "LZW",
                "predictor": 2,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "BIGTIFF": "YES",
            })
            # write compressed TIFF
            with rasterio.open(compressed_tmp.name, "w", **meta) as dst:
                dst.write(data.astype("float32"), 1)
        # upload
        b2_uri, size = upload_local_to_b2(compressed_tmp.name, b2_path)
        logger.info(f"✅ Uploaded {b2_path} ({size/1024/1024:.2f} MB)")
        return compressed_tmp.name, b2_uri, int(size)
    except Exception as e:
        logger.error(f"download_and_compress_band failed: {e}")
        logger.debug(traceback.format_exc())
        # cleanup on error
        try:
            os.remove(raw_tmp.name)
        except Exception:
            pass
        try:
            os.remove(compressed_tmp.name)
        except Exception:
            pass
        return None, None, None
    finally:
        # safe attempt to remove raw input; compressed_tmp returned to caller for upload cleanup if needed
        try:
            if os.path.exists(raw_tmp.name):
                os.remove(raw_tmp.name)
        except Exception:
            pass

# ---------------- NDVI compute (streaming) ----------------

def compute_streaming_ndvi_and_upload(tile_id, acq_date, red_local, nir_local):
    """
    red_local / nir_local are local file paths (compressed COGs or TIFFs)
    Compute NDVI in chunks, stream-write to disk (COG-friendly), upload to B2, return (b2_uri, stats, size_bytes)
    Stats guaranteed to be Python native types.
    """
    ndvi_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    ndvi_tmp.close()
    try:
        with rasterio.open(red_local) as rsrc, rasterio.open(nir_local) as nsrc:
            meta = rsrc.meta.copy()
            height, width = rsrc.height, rsrc.width
            chunk_rows = 1024
            total_pixels = int(height) * int(width)

            # prepare output metadata
            meta.update({
                "driver": "GTiff",
                "dtype": "float32",
                "count": 1,
                "compress": "LZW",
                "predictor": 2,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "BIGTIFF": "YES",
            })

            # stats accumulators
            valid_pixels = 0
            sum_ndvi = 0.0
            sum_sq_ndvi = 0.0
            min_val = float("inf")
            max_val = float("-inf")
            veg_pixels = 0

            with rasterio.open(ndvi_tmp.name, "w", **meta) as dst:
                for y in range(0, height, chunk_rows):
                    rows = min(chunk_rows, height - y)
                    window = Window(0, y, width, rows)
                    red = rsrc.read(1, window=window).astype("float32")
                    nir = nsrc.read(1, window=window).astype("float32")
                    # NDVI
                    denom = nir + red
                    # avoid division warnings; use numpy where
                    ndvi_chunk = np.where(denom != 0, (nir - red) / denom, np.nan).astype("float32")
                    # write chunk
                    dst.write(ndvi_chunk, 1, window=window)
                    # update stats using masks
                    mask = ~np.isnan(ndvi_chunk)
                    if mask.any():
                        vals = ndvi_chunk[mask].astype("float64")
                        cnt = int(vals.size)
                        valid_pixels += cnt
                        sum_ndvi += float(vals.sum())
                        sum_sq_ndvi += float((vals ** 2).sum())
                        mn = float(vals.min()); mx = float(vals.max())
                        if mn < min_val: min_val = mn
                        if mx > max_val: max_val = mx
                        veg_pixels += int((vals > 0.3).sum())

            # finalize stats
            stats = None
            if valid_pixels > 0:
                mean = sum_ndvi / valid_pixels
                variance = max(0.0, (sum_sq_ndvi / valid_pixels) - (mean ** 2))
                std = float(np.sqrt(variance))
                veg_cover = float(veg_pixels) / float(valid_pixels) * 100.0
                completeness = float(valid_pixels) / float(total_pixels) * 100.0
                # vegetation health formula (same as before, normalized)
                ndvi_score = ((mean + 1.0) / 2.0) * 100.0
                health = ndvi_score * 0.5 + veg_cover * 0.3 + completeness * 0.2
                stats = {
                    "ndvi_min": float(min_val) if np.isfinite(min_val) else None,
                    "ndvi_max": float(max_val) if np.isfinite(max_val) else None,
                    "ndvi_mean": float(mean),
                    "ndvi_std_dev": float(std),
                    "vegetation_coverage_percent": float(round(veg_cover, 4)),
                    "data_completeness_percent": float(round(completeness, 4)),
                    "pixel_count": int(total_pixels),
                    "valid_pixel_count": int(valid_pixels),
                    "vegetation_health_score": float(round(health, 2)),
                }
            else:
                stats = {
                    "ndvi_min": None,
                    "ndvi_max": None,
                    "ndvi_mean": None,
                    "ndvi_std_dev": None,
                    "vegetation_coverage_percent": 0.0,
                    "data_completeness_percent": 0.0,
                    "pixel_count": int(total_pixels),
                    "valid_pixel_count": 0,
                    "vegetation_health_score": None,
                }

        # upload NDVI file
        ndvi_b2_path = b2_path_for(tile_id, acq_date, "ndvi")
        b2_uri, ndvi_size = upload_local_to_b2(ndvi_tmp.name, ndvi_b2_path)
        logger.info(f"✅ NDVI uploaded {ndvi_b2_path} ({ndvi_size/1024/1024:.2f} MB)")
        return b2_uri, stats, int(ndvi_size)
    except Exception as e:
        logger.error(f"compute_streaming_ndvi_and_upload failed: {e}")
        logger.debug(traceback.format_exc())
        return None, None, None
    finally:
        try:
            if os.path.exists(ndvi_tmp.name):
                os.remove(ndvi_tmp.name)
        except Exception:
            pass

# ---------------- Main tile process ----------------

def process_tile(tile):
    """
    tile: dict from mgrs_tiles table. We *prefer* geojson_geometry for MPC queries,
    fallback to geometry.
    """
    try:
        tile_id = tile.get("tile_id")
        mgrs_tile_id = tile.get("id")
        country_id = tile.get("country_id")

        # build geometry for MPC: prefer geojson_geometry (string or dict)
        geom_json = None
        if tile.get("geojson_geometry"):
            try:
                gj = tile.get("geojson_geometry")
                geom_json = gj if isinstance(gj, dict) else json.loads(gj)
            except Exception as e:
                logger.warning(f"geojson_geometry parse failed for {tile_id}: {e}")
                geom_json = decode_geom_to_geojson(tile.get("geometry"))
        else:
            geom_json = decode_geom_to_geojson(tile.get("geometry"))

        if not geom_json:
            logger.error(f"No usable geometry for tile {tile_id}, skipping.")
            return False

        today = datetime.date.today()
        start_date = (today - datetime.timedelta(days=LOOKBACK_DAYS)).isoformat()
        end_date = today.isoformat()

        items = query_mpc_item_collection(geom_json, start_date, end_date, cloud_cover=CLOUD_COVER)
        if not items:
            logger.info(f"No MPC scenes for {tile_id} in {start_date}..{end_date}")
            return False

        scene = pick_best_scene(items)
        if not scene:
            logger.info(f"No valid scene after sort for {tile_id}")
            return False

        # acquisition date and cloud cover
        acq_date = scene.properties.get("datetime", "").split("T")[0] or end_date
        cloud = scene.properties.get("eo:cloud_cover")

        # assets (should be signed by planetary_computer.sign_inplace)
        assets = scene.assets or {}
        # try several asset keys
        red_asset = assets.get("red") or assets.get("B04") or assets.get("B04_10m")
        nir_asset = assets.get("nir08") or assets.get("B08") or assets.get("B08_10m")

        if not red_asset or not nir_asset:
            logger.error(f"Missing red/nir assets for scene {scene.id} (tile {tile_id})")
            return False

        red_href = red_asset.href
        nir_href = nir_asset.href

        # b2 paths
        path_red = b2_path_for(tile_id, acq_date, "raw/B04")
        path_nir = b2_path_for(tile_id, acq_date, "raw/B08")
        path_ndvi = b2_path_for(tile_id, acq_date, "ndvi")

        # check existing in B2
        red_exists, red_size = check_b2_exists(path_red)
        nir_exists, nir_size = check_b2_exists(path_nir)
        ndvi_exists, ndvi_size = check_b2_exists(path_ndvi)

        # download & compress only missing bands
        red_local = None; nir_local = None
        red_b2_uri = f"b2://{B2_BUCKET}/{path_red}"
        nir_b2_uri = f"b2://{B2_BUCKET}/{path_nir}"

        try:
            if not red_exists:
                logger.info(f"Downloading & compressing RED for {tile_id} {acq_date}")
                red_local, red_b2_uri, red_size = download_and_compress_band(red_href, path_red)
            else:
                logger.info(f"RED already in B2, will download for NDVI computation as needed")
                # download to temp for NDVI computation
                temp_local_red = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
                temp_local_red.close()
                bucket.download_file_by_name(path_red, temp_local_red.name)
                red_local = temp_local_red.name
            if not nir_exists:
                logger.info(f"Downloading & compressing NIR for {tile_id} {acq_date}")
                nir_local, nir_b2_uri, nir_size = download_and_compress_band(nir_href, path_nir)
            else:
                logger.info(f"NIR already in B2, will download for NDVI computation as needed")
                temp_local_nir = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
                temp_local_nir.close()
                bucket.download_file_by_name(path_nir, temp_local_nir.name)
                nir_local = temp_local_nir.name
        except Exception as e:
            logger.error(f"Failed to obtain bands for {tile_id}: {e}")
            logger.debug(traceback.format_exc())
            # cleanup partial files
            try:
                if red_local and os.path.exists(red_local): os.remove(red_local)
            except: pass
            try:
                if nir_local and os.path.exists(nir_local): os.remove(nir_local)
            except: pass
            return False

        # compute NDVI if needed
        ndvi_b2_uri = f"b2://{B2_BUCKET}/{path_ndvi}"
        stats = None
        ndvi_size_bytes = None

        if not ndvi_exists:
            # require red_local and nir_local to exist at this point
            if not red_local or not nir_local:
                logger.error(f"Missing local band files for NDVI compute for {tile_id}")
                return False
            ndvi_b2_uri, stats, ndvi_size_bytes = compute_streaming_ndvi_and_upload(tile_id, acq_date, red_local, nir_local)
            # cleanup local band files we created for NDVI compute (if they are temp)
            try:
                if red_exists and red_local and os.path.exists(red_local): os.remove(red_local)
            except: pass
            try:
                if nir_exists and nir_local and os.path.exists(nir_local): os.remove(nir_local)
            except: pass
        else:
            logger.info(f"NDVI already exists in B2: {ndvi_b2_uri}")
            # stats unknown if ndvi_exists; leave stats None

        # compute total file sizes (int, MB)
        total_size_mb = None
        try:
            parts = []
            if red_size: parts.append(int(red_size))
            if nir_size: parts.append(int(nir_size))
            if ndvi_size_bytes: parts.append(int(ndvi_size_bytes))
            if parts:
                total_size_mb = round(sum(parts) / (1024 * 1024), 2)
        except Exception:
            total_size_mb = None

        # prepare payload; ensure Python native types
        bbox = extract_bbox(geom_json)
        now = datetime.datetime.utcnow().isoformat() + "Z"

        payload = {
            "tile_id": tile_id,
            "acquisition_date": acq_date,
            "collection": MPC_COLLECTION.upper(),
            "processing_level": "L2A",
            "cloud_cover": float(cloud) if cloud is not None else None,
            "red_band_path": red_b2_uri,
            "nir_band_path": nir_b2_uri,
            "ndvi_path": ndvi_b2_uri,
            "file_size_mb": total_size_mb,
            "red_band_size_bytes": int(red_size) if red_size is not None else None,
            "nir_band_size_bytes": int(nir_size) if nir_size is not None else None,
            "ndvi_size_bytes": int(ndvi_size_bytes) if ndvi_size_bytes is not None else None,
            "resolution": "10m",
            "status": "ready",
            "updated_at": now,
            "processing_completed_at": now,
            "ndvi_calculation_timestamp": now if stats else None,
            "api_source": "planetary_computer",
            "processing_method": "cog_streaming",
            "actual_download_status": "downloaded",
            "processing_stage": "completed",
            "country_id": country_id,
            "mgrs_tile_id": mgrs_tile_id,
            "bbox": json.dumps(bbox) if bbox else None,
            "created_at": now,
        }

        # attach statistics if available (ensuring types are native)
        if stats:
            payload.update({
                "ndvi_min": stats.get("ndvi_min"),
                "ndvi_max": stats.get("ndvi_max"),
                "ndvi_mean": stats.get("ndvi_mean"),
                "ndvi_std_dev": stats.get("ndvi_std_dev"),
                "vegetation_coverage_percent": stats.get("vegetation_coverage_percent"),
                "data_completeness_percent": stats.get("data_completeness_percent"),
                "pixel_count": stats.get("pixel_count"),
                "valid_pixel_count": stats.get("valid_pixel_count"),
                "vegetation_health_score": stats.get("vegetation_health_score"),
            })

        # sanitize payload
        payload_safe = sanitize_payload(payload)

        # debug: log types (can be removed later)
        logger.debug("Upserting satellite_tiles payload types: %s", {k: type(v).__name__ for k, v in payload_safe.items()})

        # upsert to Supabase
        try:
            resp = supabase.table("satellite_tiles").upsert(payload_safe, on_conflict="tile_id,acquisition_date,collection").execute()
            logger.info(f"✅ Upserted satellite_tiles for {tile_id} {acq_date}")
        except Exception as e:
            logger.error(f"DB upsert failed for {tile_id} {acq_date}: {e}")
            logger.debug(traceback.format_exc())
            return False

        return True

    except Exception as e:
        logger.error(f"process_tile failed for {tile.get('tile_id') if isinstance(tile, dict) else 'unknown'}: {e}")
        logger.debug(traceback.format_exc())
        return False

# ---------------- Fetching MGRS tiles ----------------

def fetch_agri_tiles():
    try:
        resp = supabase.table("mgrs_tiles").select("id,tile_id,geometry,geojson_geometry,country_id").eq("is_agri", True).eq("is_land_contain", True).execute()
        tiles = resp.data or []
        logger.info(f"Fetched {len(tiles)} agri tiles.")
        return tiles
    except Exception as e:
        logger.error(f"fetch_agri_tiles failed: {e}")
        logger.debug(traceback.format_exc())
        return []

# ---------------- Main entry ----------------

def main(cloud_cover=CLOUD_COVER, lookback_days=LOOKBACK_DAYS):
    global CLOUD_COVER, LOOKBACK_DAYS
    CLOUD_COVER = int(cloud_cover)
    LOOKBACK_DAYS = int(lookback_days)
    logger.info(f"🚀 Starting tile worker (cloud<{CLOUD_COVER}%, lookback={LOOKBACK_DAYS}d)")
    tiles = fetch_agri_tiles()
    if not tiles:
        logger.warning("No tiles to process.")
        return 0
    processed = 0
    for idx, t in enumerate(tiles, 1):
        logger.info(f"🔄 [{idx}/{len(tiles)}] Processing {t.get('tile_id')}")
        if process_tile(t):
            processed += 1
            logger.info(f"✅ [{idx}] Success: {t.get('tile_id')}")
        else:
            logger.info(f"⏭️  [{idx}] Skipped/Failed: {t.get('tile_id')}")
    logger.info(f"✨ Finished: processed {processed}/{len(tiles)} tiles successfully")
    return processed

if __name__ == "__main__":
    cc = int(os.environ.get("RUN_CLOUD_COVER", str(CLOUD_COVER)))
    lb = int(os.environ.get("RUN_LOOKBACK_DAYS", str(LOOKBACK_DAYS)))
    main(cc, lb)
