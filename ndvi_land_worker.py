# ndvi_land_worker_v4.py
"""
NDVI Land Worker v4 - Scalable, multi-tenant, log-rich worker
- Functions:
  - process_queue(limit): claim queue items and process them
  - process_request_sync(queue_id, tenant_id, land_ids, tile_id): single-request entry used by API
- Behavior:
  - Uses B2 (Backblaze) auth to download tiles
  - Prefers precomputed ndvi.tif; falls back to computing NDVI from B04/B08
  - Handles multi-tile overlap using rasterio.merge
  - Creates colorized PNG with transparency and uploads to Supabase storage
  - Inserts/upserts ndvi_data, ndvi_micro_tiles, updates lands
  - Writes detailed ndvi_processing_logs
"""
import os
import io
import json
import time
import base64
import datetime
import logging
import traceback
from typing import List, Optional

import numpy as np
import requests
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge as rio_merge
from rasterio.warp import transform_geom
from shapely.geometry import shape, mapping
from PIL import Image
import matplotlib.cm as cm

from supabase import create_client

# ----------------------
# Logging
# ----------------------
logging.basicConfig(level=os.getenv("NDVI_WORKER_LOG_LEVEL", "INFO"), format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ndvi-worker-v4")

# ----------------------
# Config (default to your values where helpful)
# ----------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://qfklkkzxemsbeniyugiz.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_NDVI_BUCKET = os.getenv("SUPABASE_NDVI_BUCKET", "ndvi-thumbnails")

B2_KEY_ID = os.getenv("B2_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
B2_BUCKET = os.getenv("B2_BUCKET", "kisanshakti-ndvi-tiles")
B2_PREFIX = os.getenv("B2_PREFIX", "tiles/")  # you specified 'tiles/'

if not SUPABASE_KEY:
    logger.error("Missing SUPABASE_SERVICE_ROLE_KEY")
    raise RuntimeError("Missing SUPABASE_SERVICE_ROLE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------------------
# Utilities
# ----------------------
def now_iso():
    return datetime.datetime.utcnow().isoformat()

def get_b2_auth() -> dict:
    if not all([B2_KEY_ID, B2_APP_KEY, B2_BUCKET]):
        raise RuntimeError("Missing B2 credentials (B2_KEY_ID, B2_APP_KEY, B2_BUCKET)")
    cred = base64.b64encode(f"{B2_KEY_ID}:{B2_APP_KEY}".encode()).decode()
    headers = {"Authorization": f"Basic {cred}"}
    resp = requests.get("https://api.backblazeb2.com/b2api/v2/b2_authorize_account", headers=headers, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"B2 authorize failed: {resp.status_code} {resp.text[:200]}")
    data = resp.json()
    return {
        "api_url": data.get("apiUrl"),
        "download_url": data.get("downloadUrl"),
        "auth_token": data.get("authorizationToken"),
        "bucket_name": B2_BUCKET,
    }

def build_b2_path(tile_id: str, date: str, subdir: str, filename: str) -> str:
    prefix = (B2_PREFIX or "").rstrip("/")
    return f"{prefix}/{subdir}/{tile_id}/{date}/{filename}"

def download_b2_file(tile_id: str, date: str, subdir: str, filename: str) -> Optional[io.BytesIO]:
    try:
        b2 = get_b2_auth()
        path = build_b2_path(tile_id, date, subdir, filename)
        url = f"{b2['download_url'].rstrip('/')}/file/{b2['bucket_name']}/{path}"
        headers = {"Authorization": b2["auth_token"]}
        logger.debug("B2 download request", extra={"url": url})
        resp = requests.get(url, headers=headers, timeout=120)
        if resp.status_code == 200:
            logger.info("B2 file downloaded", extra={"tile": tile_id, "date": date, "subdir": subdir, "filename": filename, "bytes": len(resp.content)})
            return io.BytesIO(resp.content)
        elif resp.status_code == 404:
            logger.debug("B2 file not found", extra={"path": path})
            return None
        else:
            logger.error("B2 download error", extra={"status": resp.status_code, "text": resp.text[:200]})
            return None
    except Exception as ex:
        logger.exception("Exception downloading from B2")
        return None

# ----------------------
# NDVI math & helpers
# ----------------------
def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    np.seterr(divide="ignore", invalid="ignore")
    red = red.astype(float)
    nir = nir.astype(float)
    denom = nir + red
    ndvi = np.where(denom == 0, np.nan, (nir - red) / denom)
    return np.clip(ndvi, -1.0, 1.0)

def calculate_statistics(arr: np.ndarray) -> dict:
    valid = arr[~np.isnan(arr)]
    total = int(arr.size)
    if valid.size == 0:
        return {"mean": None, "min": None, "max": None, "std": None, "valid_pixels": 0, "total_pixels": total, "coverage": 0.0}
    return {
        "mean": float(np.mean(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "std": float(np.std(valid)),
        "valid_pixels": int(valid.size),
        "total_pixels": total,
        "coverage": float(valid.size / total * 100),
    }

def create_colorized_ndvi_png(ndvi_array: np.ndarray, cmap_name: str = "RdYlGn") -> bytes:
    ndvi_normalized = np.clip((ndvi_array + 1.0) / 2.0, 0.0, 1.0)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(ndvi_normalized)
    nan_mask = np.isnan(ndvi_array)
    rgba_uint8 = (rgba * 255).astype(np.uint8)
    if nan_mask.any():
        rgba_uint8[..., 3][nan_mask] = 0
    else:
        rgba_uint8[..., 3] = 255
    img = Image.fromarray(rgba_uint8, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.getvalue()

def upload_thumbnail_to_supabase(land_id: str, date: str, png_bytes: bytes) -> Optional[str]:
    path = f"{land_id}/{date}/ndvi_color.png"
    try:
        file_obj = io.BytesIO(png_bytes)
        res = supabase.storage.from_(SUPABASE_NDVI_BUCKET).upload(path, file_obj, {"content-type": "image/png", "upsert": True})
        if isinstance(res, dict) and res.get("error"):
            logger.error("Supabase upload error", extra={"error": res.get("error")})
            return None
        public_url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/public/{SUPABASE_NDVI_BUCKET}/{path}"
        logger.info("Uploaded thumbnail to supabase", extra={"land_id": land_id, "url": public_url})
        return public_url
    except Exception:
        logger.exception("Exception uploading thumbnail to supabase")
        return None

# ----------------------
# Core per-land processing (production grade, returns dict with details)
# ----------------------
def process_single_land(land: dict, tile: dict) -> dict:
    """
    Process NDVI for single land. Returns result dict:
    {
      "success": bool,
      "land_id": ...,
      "stats": {...} or None,
      "image_url": str or None,
      "error": str or None
    }
    """
    land_id = land.get("id")
    tenant_id = land.get("tenant_id")
    tile_id = tile.get("tile_id")
    date = tile.get("acquisition_date") or datetime.date.today().isoformat()

    result = {"success": False, "land_id": land_id, "stats": None, "image_url": None, "error": None}
    start_ts = time.time()

    try:
        logger.info("Start processing land", extra={"land_id": land_id, "tile_id": tile_id, "tenant_id": tenant_id})

        # parse geometry
        geom_raw = land.get("boundary_polygon_old") or land.get("boundary") or land.get("boundary_polygon")
        if not geom_raw:
            raise ValueError("Missing boundary polygon")

        geom_json = json.loads(geom_raw) if isinstance(geom_raw, str) else geom_raw
        land_geom = shape(geom_json)
        if not land_geom.is_valid:
            logger.warning("Invalid geometry, attempting repair", extra={"land_id": land_id})
            land_geom = land_geom.buffer(0)
            if not land_geom.is_valid:
                raise ValueError("Invalid geometry and cannot repair")

        if land_geom.is_empty:
            raise ValueError("Geometry empty after parse/repair")

        # Determine overlapping tiles (simple approach: use satellite_tiles table)
        # We will fetch satellite_tiles rows for the given tile_id OR intersect (if your mgrs mapping exists you can expand)
        candidate_tiles = []
        # prefer provided tile_id first
        tile_rows = supabase.table("satellite_tiles").select("*").eq("tile_id", tile_id).execute()
        if tile_rows.data:
            candidate_tiles.append(tile_rows.data[0])
            logger.debug("Tile metadata loaded", extra={"tile_id": tile_id})
        else:
            # fallback: try to find recent tiles that intersect land bbox (if bbox exists)
            logger.debug("Tile metadata not found; trying search by bbox", extra={"tile_id": tile_id})

        # For each candidate tile, try to get precomputed ndvi.tif or raw bands
        ndvi_sources = []
        for t in candidate_tiles:
            t_tile_id = t.get("tile_id")
            t_date = t.get("acquisition_date") or date
            # try ndvi.tif first
            ndvi_buf = download_b2_file(t_tile_id, t_date, "ndvi", "ndvi.tif")
            if ndvi_buf:
                ndvi_sources.append(("ndvi", t_tile_id, t_date, ndvi_buf))
                continue
            # fallback to raw bands
            b04 = download_b2_file(t_tile_id, t_date, "raw", "B04.tif")
            b08 = download_b2_file(t_tile_id, t_date, "raw", "B08.tif")
            if b04 and b08:
                ndvi_sources.append(("raw", t_tile_id, t_date, (b04, b08)))
            else:
                logger.warning("Tile missing both ndvi and raw bands", extra={"tile": t_tile_id, "date": t_date})

        if not ndvi_sources:
            raise FileNotFoundError("No NDVI or raw band sources available for tiles")

        # If multiple sources exist, we will open rasters and mosaic them (prefer ndvi products)
        ndvi_arrays = []
        ndvi_meta = None
        memfiles = []

        # Helper to open rasterio dataset from BytesIO
        def open_dataset_from_bytes(b: io.BytesIO):
            mem = rasterio.io.MemoryFile(b.read())
            return mem

        # Collect raster datasets (single-band NDVI) or compute NDVI from raw bands and collect as in-memory MemoryFile datasets
        datasets = []
        for src_type, ttid, tdate, payload in ndvi_sources:
            try:
                if src_type == "ndvi":
                    mem = open_dataset_from_bytes(payload)
                    ds = mem.open()
                    datasets.append(ds)
                    memfiles.append(mem)
                else:
                    # raw bands -> compute NDVI and create in-memory single-band GeoTIFF memoryfile
                    red_buf, nir_buf = payload
                    red_mem = rasterio.io.MemoryFile(red_buf.read())
                    nir_mem = rasterio.io.MemoryFile(nir_buf.read())
                    red_ds = red_mem.open()
                    nir_ds = nir_mem.open()
                    # reproject/align is assumed simpler: same CRS/transform for sentinel raw bands (if not, more logic required)
                    # clip to land geometry in red_ds CRS
                    geom_trans = transform_geom("EPSG:4326", red_ds.crs.to_string(), mapping(land_geom))
                    red_clip, red_transform = mask(red_ds, [geom_trans], crop=True, all_touched=True, nodata=np.nan)
                    nir_clip, nir_transform = mask(nir_ds, [geom_trans], crop=True, all_touched=True, nodata=np.nan)
                    red_band = red_clip[0].astype(float)
                    nir_band = nir_clip[0].astype(float)
                    ndvi = calculate_ndvi(red_band, nir_band)
                    # Build a MemoryFile dataset from ndvi array using the red_ds.meta as template
                    meta = red_ds.meta.copy()
                    meta.update({"count": 1, "dtype": "float32", "driver": "GTiff", "nodata": np.nan})
                    mem_out = rasterio.io.MemoryFile()
                    with mem_out.open(**meta) as dst:
                        dst.write(ndvi.astype("float32"), 1)
                    # reopen mem_out as dataset for merging
                    ds = mem_out.open()
                    datasets.append(ds)
                    memfiles.extend([red_mem, nir_mem, mem_out])
            except Exception as e:
                logger.exception("Failed preparing dataset for tile", extra={"tile": ttid, "date": tdate, "error": str(e)})
                continue

        if not datasets:
            raise RuntimeError("No raster datasets could be prepared")

        # Merge datasets if more than one (handles overlap)
        if len(datasets) == 1:
            ds = datasets[0]
            # transform geometry to ds CRS
            geom_trans = transform_geom("EPSG:4326", ds.crs.to_string(), mapping(land_geom))
            clipped, out_transform = mask(ds, [geom_trans], crop=True, all_touched=True, nodata=np.nan)
            ndvi_array = clipped[0].astype(float)
            ndvi_meta = ds.meta
        else:
            # mosaic then mask to land geometry in mosaic CRS
            logger.info("Merging multiple tile rasters (overlap handling)", extra={"land_id": land_id, "count": len(datasets)})
            mosaic, out_transform = rio_merge(datasets)
            # mosaic has shape (bands, h, w) but NDVI is single band (first)
            ndvi_full = mosaic[0].astype(float)
            # create in-memory dataset from mosaic for mask call
            mosaic_meta = datasets[0].meta.copy()
            mosaic_meta.update({"height": ndvi_full.shape[0], "width": ndvi_full.shape[1], "transform": out_transform, "count": 1, "dtype": "float32"})
            mem_out = rasterio.io.MemoryFile()
            with mem_out.open(**mosaic_meta) as dst:
                dst.write(ndvi_full.astype("float32"), 1)
            with mem_out.open() as merged_ds:
                geom_trans = transform_geom("EPSG:4326", merged_ds.crs.to_string(), mapping(land_geom))
                clipped, _ = mask(merged_ds, [geom_trans], crop=True, all_touched=True, nodata=np.nan)
                ndvi_array = clipped[0].astype(float)
                ndvi_meta = merged_ds.meta
            memfiles.append(mem_out)

        # close datasets and memfiles (they will be closed in finally)
        # verify ndvi_array
        if ndvi_array is None or ndvi_array.size == 0:
            raise RuntimeError("NDVI array empty after clipping/mosaic")

        # Stats
        stats = calculate_statistics(ndvi_array)
        if stats["valid_pixels"] == 0:
            raise ValueError("No valid NDVI pixels after mask")

        # Create PNG thumbnail
        png_bytes = create_colorized_ndvi_png(ndvi_array)
        image_url = upload_thumbnail_to_supabase(land_id, date, png_bytes)

        # Prepare DB records with schema fields
        current_time = now_iso()
        ndvi_record = {
            "tenant_id": tenant_id,
            "land_id": land_id,
            "date": date,
            "ndvi_value": stats["mean"],
            "ndvi_min": stats["min"],
            "ndvi_max": stats["max"],
            "ndvi_std": stats["std"],
            "coverage": stats["coverage"],
            "image_url": image_url,
            "created_at": current_time,
            "metadata": stats,
            "tile_id": tile_id,
            "mean_ndvi": stats["mean"],
            "min_ndvi": stats["min"],
            "max_ndvi": stats["max"],
            "valid_pixels": stats["valid_pixels"],
            "total_pixels": stats["total_pixels"],
            "coverage_percentage": stats["coverage"],
            "computed_at": current_time,
        }

        micro_tile_record = {
            "tenant_id": tenant_id,
            "land_id": land_id,
            "farmer_id": land.get("farmer_id"),
            "bbox": land.get("boundary_polygon_old") or {},
            "acquisition_date": date,
            "cloud_cover": tile.get("cloud_cover"),
            "ndvi_mean": stats["mean"],
            "ndvi_min": stats["min"],
            "ndvi_max": stats["max"],
            "ndvi_std_dev": stats["std"],
            "ndvi_thumbnail_url": image_url,
            "thumbnail_size_kb": round(len(png_bytes)/1024, 2),
            "statistics_only": False,
            "created_at": current_time,
        }

        # Write to DB (upsert ndvi_data and ndvi_micro_tiles; update lands)
        try:
            ndvi_up = supabase.table("ndvi_data").upsert(ndvi_record, on_conflict="land_id,date").execute()
            micro_up = supabase.table("ndvi_micro_tiles").upsert(micro_tile_record, on_conflict="land_id,acquisition_date").execute()
            lands_up = supabase.table("lands").update({
                "last_ndvi_value": stats["mean"],
                "last_ndvi_calculation": date,
                "ndvi_thumbnail_url": image_url,
                "ndvi_tested": True,
                "last_processed_at": current_time,
                "updated_at": current_time,
            }).eq("id", land_id).execute()
            logger.info("DB writes done", extra={"land_id": land_id, "ndvi_up": bool(getattr(ndvi_up, "data", None)), "micro_up": bool(getattr(micro_up, "data", None))})
        except Exception as db_ex:
            logger.exception("DB write failed")
            raise db_ex

        # log processing success in ndvi_processing_logs
        duration_ms = int((time.time() - start_ts) * 1000)
        try:
            log_record = {
                "tenant_id": tenant_id,
                "land_id": land_id,
                "satellite_tile_id": tile.get("id"),
                "processing_step": "ndvi_calculation",
                "step_status": "completed",
                "started_at": current_time,
                "completed_at": now_iso(),
                "duration_ms": duration_ms,
                "error_message": None,
                "error_details": None,
                "metadata": {"tile_id": tile_id, "stats": stats},
                "created_at": now_iso(),
            }
            supabase.table("ndvi_processing_logs").insert(log_record).execute()
        except Exception:
            logger.exception("Failed to write processing log")

        result.update({"success": True, "stats": stats, "image_url": image_url})
        logger.info("Land processed successfully", extra={"land_id": land_id, "mean": stats["mean"], "coverage": stats["coverage"]})
        return result

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Failed processing land", extra={"land_id": land_id, "error": str(e)[:300], "trace": tb[:500]})
        # write failure log
        try:
            log_record = {
                "tenant_id": tenant_id,
                "land_id": land_id,
                "satellite_tile_id": tile.get("id"),
                "processing_step": "ndvi_calculation",
                "step_status": "failed",
                "started_at": now_iso(),
                "completed_at": now_iso(),
                "duration_ms": int((time.time() - start_ts) * 1000),
                "error_message": str(e)[:1000],
                "error_details": {"traceback": tb[:2000]},
                "metadata": {"tile_id": tile_id},
                "created_at": now_iso(),
            }
            supabase.table("ndvi_processing_logs").insert(log_record).execute()
        except Exception:
            logger.exception("Failed to write failure log")

        result["error"] = str(e)
        return result
    finally:
        # close any open MemFiles/datasets if present
        try:
            for ds in locals().get("datasets", []) or []:
                try:
                    ds.close()
                except Exception:
                    pass
            for mem in locals().get("memfiles", []) or []:
                try:
                    mem.close()
                except Exception:
                    pass
        except Exception:
            pass

# ----------------------
# Worker orchestration
# ----------------------
def process_request_sync(queue_id: str, tenant_id: str, land_ids: List[str], tile_id: str) -> dict:
    """Synchronous wrapper that processes specified lands for a queue id. Returns summary dict."""
    logger.info("process_request_sync start", extra={"queue_id": queue_id, "tenant_id": tenant_id, "tile_id": tile_id, "len": len(land_ids)})
    start_time = time.time()
    processed_count = 0
    failed_lands = []
    last_error = None

    # fetch lands
    try:
        lands_res = supabase.table("lands").select("*").eq("tenant_id", tenant_id).in_("id", land_ids).execute()
        lands = lands_res.data or []
        if not lands:
            raise RuntimeError("No lands found for provided ids")
    except Exception as e:
        logger.exception("Failed to fetch lands")
        return {"processed_count": 0, "failed_lands": land_ids, "last_error": str(e)}

    # fetch tile metadata
    tile_res = supabase.table("satellite_tiles").select("*").eq("tile_id", tile_id).limit(1).execute()
    if not tile_res.data:
        logger.error("Tile metadata missing", extra={"tile_id": tile_id})
        return {"processed_count": 0, "failed_lands": land_ids, "last_error": "Missing tile metadata"}

    tile = tile_res.data[0]

    for land in lands:
        res = process_single_land(land, tile)
        if res.get("success"):
            processed_count += 1
        else:
            failed_lands.append(land.get("id"))
            last_error = res.get("error")

    duration_s = int(time.time() - start_time)
    logger.info("process_request_sync finished", extra={"queue_id": queue_id, "processed_count": processed_count, "failed": len(failed_lands), "duration_s": duration_s})

    return {
        "queue_id": queue_id,
        "processed_count": processed_count,
        "total_lands": len(lands),
        "failed_lands": failed_lands,
        "last_error": last_error,
        "duration_s": duration_s,
    }

def process_queue(limit: int = 5):
    """
    Claim queued items and process them (intended for cron / container run).
    """
    logger.info("Worker scanning queue", extra={"limit": limit})
    try:
        rq = supabase.table("ndvi_request_queue").select("*").eq("status", "queued").order("created_at", desc=False).limit(limit).execute()
        requests = rq.data or []
    except Exception as e:
        logger.exception("Failed to read ndvi_request_queue")
        return

    if not requests:
        logger.info("No queued requests found")
        return

    for req in requests:
        req_id = req.get("id")
        tenant_id = req.get("tenant_id")
        tile_id = req.get("tile_id")
        logger.info("Claiming queue item", extra={"queue_id": req_id, "tenant_id": tenant_id})
        # claim
        supabase.table("ndvi_request_queue").update({"status": "processing", "started_at": now_iso()}).eq("id", req_id).execute()
        # process
        lands_ids = req.get("land_ids", [])
        result = process_request_sync(req_id, tenant_id, lands_ids, tile_id)
        final_status = "completed" if result.get("processed_count", 0) > 0 else "failed"
        supabase.table("ndvi_request_queue").update({
            "status": final_status,
            "processed_count": result.get("processed_count", 0),
            "completed_at": now_iso(),
            "last_error": result.get("last_error"),
        }).eq("id", req_id).execute()
        logger.info("Queue item processed", extra={"queue_id": req_id, "status": final_status})

# ----------------------
# CLI
# ----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NDVI Worker v4")
    parser.add_argument("--limit", type=int, default=int(os.getenv("NDVI_WORKER_LIMIT", "5")))
    args = parser.parse_args()
    logger.info(f"Starting NDVI Worker v4 with limit={args.limit}")
    process_queue(limit=args.limit)
    logger.info("NDVI Worker finished")
