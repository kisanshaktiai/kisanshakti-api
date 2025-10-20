# Tiles Fetching & Processing Worker v1.3.0
# - Full pipeline: STAC -> download bands -> NDVI -> upload -> upsert satellite_tiles
# - Update: 20/10/2025
# Requirements: supabase, b2sdk, requests, rasterio, numpy, shapely

import os
import io
import json
import tempfile
import logging
import traceback
import datetime
from typing import Optional, Tuple

import requests
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.mask import mask
from shapely.geometry import shape, mapping
from supabase import create_client
from requests.adapters import HTTPAdapter, Retry
from b2sdk.v2 import InMemoryAccountInfo, B2Api

# --------------------------------------------------
# CONFIG / ENV
# --------------------------------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

B2_KEY_ID = os.environ.get("B2_KEY_ID")
B2_APP_KEY = os.environ.get("B2_APP_KEY")
B2_BUCKET_NAME = os.environ.get("B2_BUCKET_RAW", "kisanshakti-ndvi-tiles")
B2_PREFIX = os.environ.get("B2_PREFIX", "tiles/")

MPC_STAC = os.environ.get("MPC_STAC_BASE", "https://planetarycomputer.microsoft.com/api/stac/v1/search")
DEFAULT_COLLECTIONS = ["sentinel-2-l2a", "sentinel-2-l1c"]

DOWNSAMPLE_FACTOR = int(os.environ.get("DOWNSAMPLE_FACTOR", "4"))

# Defaults used by main
DEFAULT_CLOUD_COVER = int(os.environ.get("RUN_CLOUD_COVER", "40"))
DEFAULT_LOOKBACK = int(os.environ.get("RUN_LOOKBACK_DAYS", "30"))

# --------------------------------------------------
# LOGGING
# --------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --------------------------------------------------
# HTTP session with retries
# --------------------------------------------------
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries, pool_maxsize=10))

# --------------------------------------------------
# Clients
# --------------------------------------------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

if not B2_KEY_ID or not B2_APP_KEY:
    raise RuntimeError("Missing B2_KEY_ID or B2_APP_KEY")

b2_info = InMemoryAccountInfo()
b2_api = B2Api(b2_info)
b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
bucket = b2_api.get_bucket_by_name(B2_BUCKET_NAME)
if bucket is None:
    raise RuntimeError(f"B2 bucket '{B2_BUCKET_NAME}' not found or inaccessible")

# --------------------------------------------------
# Helpers: geometry + STAC
# --------------------------------------------------
def decode_geom_to_geojson(geom_value):
    if geom_value is None:
        return None
    if isinstance(geom_value, dict) and "type" in geom_value:
        return geom_value
    if isinstance(geom_value, str):
        s = geom_value.strip()
        try:
            return json.loads(s)
        except:
            # other formats not expected — return None
            return None
    return None

def fetch_agri_tiles():
    try:
        resp = supabase.table("mgrs_tiles").select("id,tile_id,geojson_geometry,country_id").eq("is_agri", True).eq("is_land_contain", True).execute()
        tiles = resp.data or []
        logging.info("Fetched %d active agricultural tiles containing lands.", len(tiles))
        return tiles
    except Exception as e:
        logging.error("fetch_agri_tiles failed: %s", e)
        return []

def extract_bbox_geojson(geom_json):
    try:
        shp = shape(geom_json)
        minx, miny, maxx, maxy = shp.bounds
        # return footprint polygon (useful for cropping)
        return {
            "type": "Polygon",
            "coordinates": [[
                [minx, miny], [maxx, miny],
                [maxx, maxy], [minx, maxy],
                [minx, miny]
            ]]
        }
    except Exception as e:
        logging.error("extract_bbox failed: %s", e)
        return None

def query_mpc(tile_geom, start_date, end_date, cloud_cover):
    """Query STAC (MPC) with fallback. Returns list of features."""
    if not tile_geom:
        return []
    for coll in DEFAULT_COLLECTIONS:
        try:
            body = {
                "collections": [coll],
                "intersects": tile_geom,
                "datetime": f"{start_date}/{end_date}",
                "query": {"eo:cloud_cover": {"lt": cloud_cover}},
                "limit": 10
            }
            logging.info("STAC query: %s, %s->%s, cloud<%s", coll, start_date, end_date, cloud_cover)
            r = session.post(MPC_STAC, json=body, timeout=60)
            if not r.ok:
                logging.warning("STAC %s returned %s", coll, r.status_code)
                continue
            feats = r.json().get("features", [])
            if feats:
                return feats
        except Exception as e:
            logging.warning("STAC query failed for %s: %s", coll, e)
            continue
    return []

def pick_best_scene(scenes):
    if not scenes:
        return None
    try:
        def keyf(s):
            cloud = s["properties"].get("eo:cloud_cover", 100)
            dt = s["properties"].get("datetime", "")
            # prefer lower cloud, then newer
            ts = datetime.datetime.fromisoformat(dt.replace("Z", "+00:00")).timestamp() if dt else 0
            return (cloud, -ts)
        return sorted(scenes, key=keyf)[0]
    except Exception as e:
        logging.error("pick_best_scene failed: %s", e)
        return None

# --------------------------------------------------
# Download helper: download asset URL to local file
# --------------------------------------------------
def download_asset_to_file(url: str, dest_path: str, bbox_geojson=None, resampling_factor:int=1) -> None:
    """
    Downloads a raster asset to dest_path.
    If the asset is a cloud-optimized GeoTIFF and supports range requests, we try streaming full file.
    If bbox_geojson provided we will mask later when processing.
    """
    logging.info("Downloading asset: %s -> %s", url, dest_path)
    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=4*1024*1024):
                if chunk:
                    f.write(chunk)
    logging.info("Downloaded asset to %s", dest_path)

# --------------------------------------------------
# NDVI compute helper (reads red and nir files, returns stats and writes NDVI tiff)
# --------------------------------------------------
def compute_ndvi_and_stats(red_path: str, nir_path: str, geom_mask_geojson=None) -> Tuple[str, dict]:
    """
    Reads red and nir rasters using rasterio, resamples to same grid if needed,
    computes NDVI raster, writes NDVI to a temporary file (GeoTIFF), returns path and statistics dict.
    Statistics include min,max,mean,std,pixel_count,valid_pixel_count,data_completeness_percent.
    """
    logging.info("Computing NDVI from %s and %s", red_path, nir_path)
    try:
        with rasterio.open(red_path) as rsrc_red, rasterio.open(nir_path) as rsrc_nir:
            # Align/resample to the smaller resolution / same transform if needed
            # Use red as base grid; if shapes differ, reproject nir to red
            if rsrc_red.crs != rsrc_nir.crs:
                raise RuntimeError("CRS mismatch between red and nir assets")

            # Read full arrays (careful on memory; these are usually 60m COGs — okay for single tile)
            red = rsrc_red.read(1, out_shape=(rsrc_red.count, rsrc_red.height, rsrc_red.width))  # shape: (h,w)
            nir = rsrc_nir.read(1, out_shape=(rsrc_nir.count, rsrc_nir.height, rsrc_nir.width))
            # If shapes differ, resample NIR to red's shape
            if red.shape != nir.shape:
                # resample nir to red's dimensions
                logging.info("Resampling NIR to match Red shape")
                nir = rsrc_nir.read(
                    1,
                    out_shape=(rsrc_red.height, rsrc_red.width),
                    resampling=Resampling.bilinear
                )

            # Handle no-data: read nodata values
            red_nodata = rsrc_red.nodata
            nir_nodata = rsrc_nir.nodata

            # Convert to float32 for NDVI calculation
            red_f = red.astype("float32")
            nir_f = nir.astype("float32")

            # mask invalid (nodata)
            valid_mask = np.ones_like(red_f, dtype=bool)
            if red_nodata is not None:
                valid_mask &= (red_f != red_nodata)
            if nir_nodata is not None:
                valid_mask &= (nir_f != nir_nodata)

            # Avoid division by zero
            denom = (nir_f + red_f)
            with np.errstate(divide="ignore", invalid="ignore"):
                ndvi = np.where(valid_mask, (nir_f - red_f) / denom, np.nan)

            # Mask out invalid where denom == 0
            ndvi = np.where(np.isfinite(ndvi), ndvi, np.nan)

            # If a geometry mask is provided, apply it (clip)
            if geom_mask_geojson:
                try:
                    out_image, out_transform = mask(rsrc_red, [geom_mask_geojson], crop=False, all_touched=False, filled=False)
                    # out_image is band-first — will be useful only to get mask; fallback to valid_mask
                except Exception as e:
                    logging.warning("Masking with geom failed: %s", e)

            # compute statistics ignoring nan
            valid_pixels = np.count_nonzero(~np.isnan(ndvi))
            total_pixels = ndvi.size
            valid_percent = round((valid_pixels / total_pixels) * 100, 2) if total_pixels else 0.0

            # compute min/max/mean/std with nan-aware functions
            ndvi_min = float(np.nanmin(ndvi)) if valid_pixels else None
            ndvi_max = float(np.nanmax(ndvi)) if valid_pixels else None
            ndvi_mean = float(np.nanmean(ndvi)) if valid_pixels else None
            ndvi_std = float(np.nanstd(ndvi)) if valid_pixels else None

            stats = {
                "pixel_count": int(total_pixels),
                "valid_pixel_count": int(valid_pixels),
                "data_completeness_percent": float(valid_percent),
                "ndvi_min": ndvi_min,
                "ndvi_max": ndvi_max,
                "ndvi_mean": ndvi_mean,
                "ndvi_std_dev": ndvi_std
            }

            # Write NDVI to temp GeoTIFF (float32, single band)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
            ndvi_out_path = tmp.name
            tmp.close()

            meta = rsrc_red.meta.copy()
            meta.update({
                "count": 1,
                "dtype": "float32",
                "driver": "GTiff",
                "compress": "deflate",
                "nodata": np.nan
            })

            # Write using rasterio
            with rasterio.open(ndvi_out_path, "w", **meta) as dst:
                dst.write(ndvi.astype("float32"), 1)

            # Optionally, you can build overviews / convert to COG externally if required
            logging.info("NDVI written to %s", ndvi_out_path)
            return ndvi_out_path, stats

    except Exception as e:
        logging.error("compute_ndvi_and_stats failed: %s", e)
        logging.error(traceback.format_exc())
        raise

# --------------------------------------------------
# B2 upload helper (returns file path and byte size)
# --------------------------------------------------
def upload_file_to_b2(local_path: str, dest_key: str) -> Tuple[str, int]:
    """
    Uploads local_path to B2 bucket under key dest_key (prefix applied).
    Returns (b2_path, size_bytes)
    """
    try:
        b2_key = os.path.join(B2_PREFIX, dest_key).replace("\\", "/")
        logging.info("Uploading %s to B2 as %s", local_path, b2_key)
        with open(local_path, "rb") as fh:
            size_bytes = os.path.getsize(local_path)
            file_info = bucket.upload_bytes(fh.read(), b2_key)
            # Using upload_bytes returns a file-like object; but to be safe, construct public URL path pattern or B2 file id
            # Backblaze public URL depends on settings; we store the path (bucket/key) for your use
            b2_path = f"b2://{B2_BUCKET_NAME}/{b2_key}"
            logging.info("Uploaded to B2: %s (%d bytes)", b2_path, size_bytes)
            return b2_path, size_bytes
    except Exception as e:
        logging.error("upload_file_to_b2 failed: %s", e)
        raise

# --------------------------------------------------
# Supabase insert / upsert function for satellite_tiles
# --------------------------------------------------
def upsert_satellite_tile_record(record: dict):
    """
    Upserts record into public.satellite_tiles using on_conflict on (tile_id, acquisition_date, collection).
    Must match your table schema keys.
    """
    try:
        resp = supabase.table("satellite_tiles").upsert(record, on_conflict=["tile_id", "acquisition_date", "collection"]).execute()
        logging.info("Upserted satellite_tiles row for tile=%s acquisition=%s", record.get("tile_id"), record.get("acquisition_date"))
        return resp
    except Exception as e:
        logging.error("upsert_satellite_tile_record failed: %s", e)
        logging.error(traceback.format_exc())
        raise

# --------------------------------------------------
# Main per-tile process function (end-to-end)
# --------------------------------------------------
def process_tile(tile_row, cloud_cover:int, lookback_days:int) -> bool:
    """
    tile_row: row from mgrs_tiles (expects tile_id and geojson_geometry)
    Performs STAC query, downloads bands, computes NDVI, uploads results, upserts satellite_tiles
    """
    tile_id = tile_row.get("tile_id")
    logging.info("Processing tile %s", tile_id)
    try:
        geom_json = tile_row.get("geojson_geometry") or decode_geom_to_geojson(tile_row.get("geometry"))
        if not geom_json:
            logging.warning("No geometry for tile %s", tile_id)
            return False

        today = datetime.date.today()
        start_date = (today - datetime.timedelta(days=lookback_days)).isoformat()
        end_date = today.isoformat()

        scenes = query_mpc(geom_json, start_date, end_date, cloud_cover)
        if not scenes:
            # retry extended
            logging.info("Retrying with extended window for tile %s", tile_id)
            retry_start = (today - datetime.timedelta(days=45)).isoformat()
            scenes = query_mpc(geom_json, retry_start, end_date, min(cloud_cover + 20, 95))

        if not scenes:
            logging.info("No scenes for tile %s in window %s..%s", tile_id, start_date, end_date)
            return False

        scene = pick_best_scene(scenes)
        if not scene:
            logging.info("No valid scene selected for tile %s", tile_id)
            return False

        scene_id = scene.get("id")
        acq_date = scene["properties"].get("datetime", "").split("T")[0]
        cc = scene["properties"].get("eo:cloud_cover")

        logging.info("Selected scene %s date=%s cloud=%s for tile %s", scene_id, acq_date, cc, tile_id)

        # Find red and nir asset URLs - common keys for Sentinel-2 on MPC:
        # L2A usually has B04 (red) and B08 (nir). Assets may be under "assets" dict with keys "B04" and "B08".
        assets = scene.get("assets", {})
        red_url = None
        nir_url = None

        # Common candidates
        for key in ["B04", "red", "band4", "B04.jp2", "B04.tif"]:
            if key in assets and assets[key].get("href"):
                red_url = assets[key]["href"]
                break
        # general fallback: try 'visual' or 'red' asset properties
        if not red_url:
            red_url = assets.get("B04", {}).get("href") or assets.get("red", {}).get("href") or assets.get("visual", {}).get("href")

        for key in ["B08", "nir", "band8", "B08.jp2", "B08.tif"]:
            if key in assets and assets[key].get("href"):
                nir_url = assets[key]["href"]
                break
        if not nir_url:
            nir_url = assets.get("B08", {}).get("href") or assets.get("nir", {}).get("href")

        if not red_url or not nir_url:
            logging.error("Could not find red/nir assets for scene %s", scene_id)
            # upsert row marking failed download
            record = {
                "tile_id": tile_id,
                "acquisition_date": acq_date or None,
                "collection": scene.get("collection", "SENTINEL-2"),
                "cloud_cover": cc,
                "status": "failed",
                "error_message": "missing_red_or_nir_asset",
                "processing_stage": "asset_discovery",
                "processing_completed_at": None,
                "created_at": datetime.datetime.utcnow().isoformat(),
                "updated_at": datetime.datetime.utcnow().isoformat(),
                "api_source": "planetary_computer",
                "mgrs_tile_id": tile_row.get("id"),
                "bbox": extract_bbox_geojson(geom_json)
            }
            try:
                upsert_satellite_tile_record(record)
            except:
                pass
            return False

        # Download assets to temp files
        red_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        nir_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        red_tmp.close(); nir_tmp.close()

        try:
            download_asset_to_file(red_url, red_tmp.name)
            download_asset_to_file(nir_url, nir_tmp.name)
        except Exception as e:
            logging.error("Asset download failed for tile %s scene %s: %s", tile_id, scene_id, e)
            # mark failure in table
            rec_fail = {
                "tile_id": tile_id,
                "acquisition_date": acq_date or None,
                "collection": scene.get("collection", "SENTINEL-2"),
                "status": "failed",
                "error_message": f"asset_download_failed: {e}",
                "processing_stage": "download",
                "updated_at": datetime.datetime.utcnow().isoformat(),
                "mgrs_tile_id": tile_row.get("id"),
                "bbox": extract_bbox_geojson(geom_json)
            }
            try:
                upsert_satellite_tile_record(rec_fail)
            except:
                pass
            return False

        # Compute NDVI and stats
        try:
            ndvi_path, stats = compute_ndvi_and_stats(red_tmp.name, nir_tmp.name, geom_mask_geojson=extract_bbox_geojson(geom_json))
        except Exception as e:
            logging.error("NDVI computation failed for tile %s: %s", tile_id, e)
            rec_fail = {
                "tile_id": tile_id,
                "acquisition_date": acq_date or None,
                "collection": scene.get("collection", "SENTINEL-2"),
                "status": "failed",
                "error_message": f"ndvi_compute_failed: {e}",
                "processing_stage": "ndvi_compute",
                "updated_at": datetime.datetime.utcnow().isoformat(),
                "mgrs_tile_id": tile_row.get("id")
            }
            try:
                upsert_satellite_tile_record(rec_fail)
            except:
                pass
            return False

        # Upload red/nir/ndvi to B2 and capture sizes
        try:
            red_key = f"{tile_id}/{acq_date}/{os.path.basename(red_tmp.name)}"
            nir_key = f"{tile_id}/{acq_date}/{os.path.basename(nir_tmp.name)}"
            ndvi_key = f"{tile_id}/{acq_date}/{os.path.basename(ndvi_path)}"

            red_b2_path, red_size = upload_file_to_b2(red_tmp.name, red_key)
            nir_b2_path, nir_size = upload_file_to_b2(nir_tmp.name, nir_key)
            ndvi_b2_path, ndvi_size = upload_file_to_b2(ndvi_path, ndvi_key)

        except Exception as e:
            logging.error("Upload to B2 failed for tile %s: %s", tile_id, e)
            rec_fail = {
                "tile_id": tile_id,
                "acquisition_date": acq_date or None,
                "collection": scene.get("collection", "SENTINEL-2"),
                "status": "failed",
                "error_message": f"b2_upload_failed: {e}",
                "processing_stage": "upload",
                "updated_at": datetime.datetime.utcnow().isoformat(),
                "mgrs_tile_id": tile_row.get("id")
            }
            try:
                upsert_satellite_tile_record(rec_fail)
            except:
                pass
            return False

        # Build record to upsert into satellite_tiles
        record = {
            "tile_id": tile_id,
            "acquisition_date": acq_date,
            "collection": scene.get("collection", "SENTINEL-2"),
            "cloud_cover": cc,
            "ndvi_path": ndvi_b2_path,
            "red_band_path": red_b2_path,
            "nir_band_path": nir_b2_path,
            "file_size_mb": round((red_size + nir_size + ndvi_size) / (1024*1024), 2),
            "processing_level": scene.get("properties", {}).get("sentinel:processing_level", "L2A"),
            "status": "processed",
            "error_message": None,
            "created_at": datetime.datetime.utcnow().isoformat(),
            "updated_at": datetime.datetime.utcnow().isoformat(),
            "processing_completed_at": datetime.datetime.utcnow().isoformat(),
            "actual_download_status": "downloaded",
            "processing_stage": "complete",
            "red_band_size_bytes": int(red_size),
            "nir_band_size_bytes": int(nir_size),
            "ndvi_size_bytes": int(ndvi_size),
            "resolution": "R60m",  # or deduce from scene props
            "vegetation_health_score": None,
            "vegetation_coverage_percent": stats.get("data_completeness_percent"),
            "ndvi_min": stats.get("ndvi_min"),
            "ndvi_max": stats.get("ndvi_max"),
            "ndvi_mean": stats.get("ndvi_mean"),
            "ndvi_std_dev": stats.get("ndvi_std_dev"),
            "ndvi_calculation_timestamp": datetime.datetime.utcnow().isoformat(),
            "pixel_count": stats.get("pixel_count"),
            "valid_pixel_count": stats.get("valid_pixel_count"),
            "data_completeness_percent": stats.get("data_completeness_percent"),
            "mgrs_tile_id": tile_row.get("id"),
            "bbox": extract_bbox_geojson(geom_json),
            "api_source": "planetary_computer"
        }

        # Upsert into Supabase table
        try:
            upsert_satellite_tile_record(record)
            logging.info("Tile %s processed and saved.", tile_id)
        except Exception as e:
            logging.error("Failed to upsert result for tile %s: %s", tile_id, e)
            return False

        return True

    except Exception as e:
        logging.error("Unexpected error processing tile %s: %s", tile_id, e)
        logging.error(traceback.format_exc())
        return False

# --------------------------------------------------
# MAIN ENTRY
# --------------------------------------------------
def main(cloud_cover:int = DEFAULT_CLOUD_COVER, lookback_days:int = DEFAULT_LOOKBACK):
    logging.info("Starting tile worker (cloud<%s, lookback=%s)", cloud_cover, lookback_days)
    tiles = fetch_agri_tiles()
    if not tiles:
        logging.warning("No tiles to process.")
        return 0

    processed = 0
    for i, t in enumerate(tiles, start=1):
        tile_id = t.get("tile_id")
        logging.info("[%d/%d] Processing %s", i, len(tiles), tile_id)
        ok = process_tile(t, cloud_cover, lookback_days)
        if ok:
            processed += 1
            logging.info("✅ [%d/%d] Success: %s", i, len(tiles), tile_id)
        else:
            logging.info("⏭️  [%d/%d] Skipped/Failed: %s", i, len(tiles), tile_id)
    logging.info("Finished: processed %d/%d tiles", processed, len(tiles))
    return processed

if __name__ == "__main__":
    cc = int(os.environ.get("RUN_CLOUD_COVER", str(DEFAULT_CLOUD_COVER)))
    lb = int(os.environ.get("RUN_LOOKBACK_DAYS", str(DEFAULT_LOOKBACK)))
    main(cc, lb)
