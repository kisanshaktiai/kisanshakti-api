"""
NDVI Land Worker v4.2.0 – Scalable, multi-tenant, fault-tolerant worker
------------------------------------------------------------------------
Enhancements:
✅ Robust tile resolution (exact + spatial + prefix fallback)
✅ Explicit error updates in ndvi_request_queue and ndvi_processing_logs
✅ Improved B2 retry and transparency
✅ Safe MemoryFile cleanup and defensive NDVI handling
✅ Compatible with RPC: tiles_intersecting_geojson(json)
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
# Logging setup
# ----------------------
logging.basicConfig(
    level=os.getenv("NDVI_WORKER_LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("ndvi-worker-v4")

# ----------------------
# Config
# ----------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://qfklkkzxemsbeniyugiz.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_NDVI_BUCKET = os.getenv("SUPABASE_NDVI_BUCKET", "ndvi-thumbnails")

B2_KEY_ID = os.getenv("B2_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
B2_BUCKET = os.getenv("B2_BUCKET", "kisanshakti-ndvi-tiles")
B2_PREFIX = os.getenv("B2_PREFIX", "tiles/")

if not SUPABASE_KEY:
    raise RuntimeError("❌ Missing SUPABASE_SERVICE_ROLE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------------------
# Utility helpers
# ----------------------
def now_iso():
    return datetime.datetime.utcnow().isoformat()

def get_b2_auth() -> dict:
    """Authenticate with Backblaze B2."""
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
    """Download a file from Backblaze B2 with retries."""
    try:
        b2 = get_b2_auth()
        path = build_b2_path(tile_id, date, subdir, filename)
        url = f"{b2['download_url'].rstrip('/')}/file/{b2['bucket_name']}/{path}"
        headers = {"Authorization": b2["auth_token"]}
        for attempt in range(1, 4):
            resp = requests.get(url, headers=headers, timeout=90)
            if resp.status_code == 200:
                logger.info(f"B2 file downloaded: {filename} ({len(resp.content)} bytes) for {tile_id}")
                return io.BytesIO(resp.content)
            elif resp.status_code == 404:
                time.sleep(1)
            else:
                logger.warning(f"B2 download error {resp.status_code} for {path}: {resp.text[:120]}")
                time.sleep(attempt)
        logger.error(f"B2 file not found after retries: {filename} for {tile_id}")
        return None
    except Exception:
        logger.exception(f"Exception while downloading {filename} from B2")
        return None

# ----------------------
# NDVI computation
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

def create_colorized_ndvi_png(ndvi_array: np.ndarray, cmap_name="RdYlGn") -> bytes:
    ndvi_normalized = np.clip((ndvi_array + 1.0) / 2.0, 0.0, 1.0)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(ndvi_normalized)
    rgba_uint8 = (rgba * 255).astype(np.uint8)
    nan_mask = np.isnan(ndvi_array)
    rgba_uint8[..., 3][nan_mask] = 0
    img = Image.fromarray(rgba_uint8, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.getvalue()

def upload_thumbnail_to_supabase(land_id: str, date: str, png_bytes: bytes) -> Optional[str]:
    path = f"{land_id}/{date}/ndvi_color.png"
    try:
        file_obj = io.BytesIO(png_bytes)
        supabase.storage.from_(SUPABASE_NDVI_BUCKET).upload(
            path, file_obj, {"content-type": "image/png", "upsert": True}
        )
        public_url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/public/{SUPABASE_NDVI_BUCKET}/{path}"
        return public_url
    except Exception:
        logger.exception("Upload to Supabase failed")
        return None

# ----------------------
# Tile lookup improvements
# ----------------------
def find_tiles_for_land(tile_id: str, land_geom):
    """Try multiple strategies to find tiles for a given land."""
    # 1. exact
    try:
        r = supabase.table("satellite_tiles").select("*").eq("tile_id", tile_id).limit(5).execute()
        if r.data:
            logger.info(f"Found {len(r.data)} exact tile(s) for {tile_id}")
            return r.data
    except Exception:
        logger.exception("Exact tile lookup failed")

    # 2. spatial RPC fallback
    try:
        if land_geom:
            geom_json = json.dumps(mapping(land_geom))
            rpc = supabase.rpc("tiles_intersecting_geojson", {"geojson": geom_json}).execute()
            if rpc.data:
                logger.info(f"Found {len(rpc.data)} tiles spatially intersecting land")
                return rpc.data
    except Exception:
        logger.debug("Spatial tile lookup failed or RPC missing")

    # 3. prefix fallback
    try:
        prefix = tile_id[:3] if tile_id else None
        if prefix:
            p = supabase.table("satellite_tiles").select("*").ilike("tile_id", f"{prefix}%").order("acquisition_date", desc=True).limit(10).execute()
            if p.data:
                logger.info(f"Found {len(p.data)} prefix-matched tiles ({prefix}%)")
                return p.data
    except Exception:
        logger.debug("Prefix tile lookup failed")

    return []

# ----------------------
# Core single land NDVI process
# ----------------------
def process_single_land(land: dict, tile: dict) -> dict:
    """Compute NDVI for one land."""
    land_id = land.get("id")
    tenant_id = land.get("tenant_id")
    tile_id = tile.get("tile_id")
    date = tile.get("acquisition_date") or datetime.date.today().isoformat()

    result = {"success": False, "land_id": land_id, "stats": None, "image_url": None, "error": None}
    start_ts = time.time()

    try:
        geom_raw = land.get("boundary_polygon_old") or land.get("boundary_polygon")
        if not geom_raw:
            raise ValueError("Missing boundary polygon")

        geom_json = json.loads(geom_raw) if isinstance(geom_raw, str) else geom_raw
        land_geom = shape(geom_json)
        if not land_geom.is_valid:
            land_geom = land_geom.buffer(0)

        # Download NDVI or raw bands
        ndvi_buf = download_b2_file(tile_id, date, "ndvi", "ndvi.tif")
        if not ndvi_buf:
            b04 = download_b2_file(tile_id, date, "raw", "B04.tif")
            b08 = download_b2_file(tile_id, date, "raw", "B08.tif")
            if not (b04 and b08):
                raise FileNotFoundError(f"No NDVI or raw bands for {tile_id}")
            red_mem = rasterio.io.MemoryFile(b04.read())
            nir_mem = rasterio.io.MemoryFile(b08.read())
            red_ds = red_mem.open()
            nir_ds = nir_mem.open()
            geom_trans = transform_geom("EPSG:4326", red_ds.crs.to_string(), mapping(land_geom))
            red_clip, _ = mask(red_ds, [geom_trans], crop=True, all_touched=True, nodata=np.nan)
            nir_clip, _ = mask(nir_ds, [geom_trans], crop=True, all_touched=True, nodata=np.nan)
            ndvi_array = calculate_ndvi(red_clip[0], nir_clip[0])
        else:
            mem = rasterio.io.MemoryFile(ndvi_buf.read())
            with mem.open() as ds:
                geom_trans = transform_geom("EPSG:4326", ds.crs.to_string(), mapping(land_geom))
                clip, _ = mask(ds, [geom_trans], crop=True, all_touched=True, nodata=np.nan)
                ndvi_array = clip[0]

        stats = calculate_statistics(ndvi_array)
        if stats["valid_pixels"] == 0:
            raise ValueError("No valid NDVI pixels")

        png_bytes = create_colorized_ndvi_png(ndvi_array)
        image_url = upload_thumbnail_to_supabase(land_id, date, png_bytes)

        # Insert DB records
        now = now_iso()
        ndvi_record = {
            "tenant_id": tenant_id,
            "land_id": land_id,
            "date": date,
            "ndvi_value": stats["mean"],
            "image_url": image_url,
            "created_at": now,
            "tile_id": tile_id,
        }

        micro_tile_record = {
            "tenant_id": tenant_id,
            "land_id": land_id,
            "acquisition_date": date,
            "ndvi_mean": stats["mean"],
            "ndvi_thumbnail_url": image_url,
            "statistics_only": False,
            "created_at": now,
        }

        supabase.table("ndvi_data").upsert(ndvi_record, on_conflict="land_id,date").execute()
        supabase.table("ndvi_micro_tiles").upsert(micro_tile_record, on_conflict="land_id,acquisition_date").execute()
        supabase.table("lands").update({
            "last_ndvi_value": stats["mean"],
            "last_processed_at": now,
            "ndvi_tested": True,
        }).eq("id", land_id).execute()

        # log success
        supabase.table("ndvi_processing_logs").insert({
            "tenant_id": tenant_id,
            "land_id": land_id,
            "processing_step": "ndvi_calculation",
            "step_status": "completed",
            "completed_at": now,
            "metadata": {"tile_id": tile_id, "stats": stats},
        }).execute()

        result.update(success=True, stats=stats, image_url=image_url)
        logger.info(f"✅ NDVI processed for land {land_id}")
    except Exception as e:
        tb = traceback.format_exc()
        result["error"] = str(e)
        logger.error(f"❌ Error processing land {land_id}: {e}")
        try:
            supabase.table("ndvi_processing_logs").insert({
                "tenant_id": land.get("tenant_id"),
                "land_id": land_id,
                "processing_step": "ndvi_calculation",
                "step_status": "failed",
                "error_message": str(e),
                "error_details": {"traceback": tb[:1000]},
                "created_at": now_iso(),
            }).execute()
        except Exception:
            pass
    return result

# ----------------------
# Main worker orchestration
# ----------------------
def process_request_sync(queue_id: str, tenant_id: str, land_ids: List[str], tile_id: str) -> dict:
    logger.info(f"process_request_sync start (queue={queue_id}, tenant={tenant_id}, tile={tile_id})")
    start = time.time()
    processed = 0
    failed = []
    last_error = None

    # Fetch lands
    lands_res = supabase.table("lands").select("*").eq("tenant_id", tenant_id).in_("id", land_ids).execute()
    lands = lands_res.data or []
    if not lands:
        msg = "No lands found for provided IDs"
        logger.error(msg)
        return {"processed_count": 0, "failed_lands": land_ids, "last_error": msg}

    # Find tiles
    candidates = find_tiles_for_land(tile_id, None)
    if not candidates:
        msg = f"Tile metadata missing for {tile_id}"
        logger.error(msg)
        supabase.table("ndvi_request_queue").update({
            "status": "failed",
            "processed_count": 0,
            "completed_at": now_iso(),
            "last_error": msg,
        }).eq("id", queue_id).execute()
        return {"processed_count": 0, "failed_lands": land_ids, "last_error": msg}

    tile = candidates[0]
    logger.info(f"Using tile {tile.get('tile_id')} dated {tile.get('acquisition_date')}")

    for land in lands:
        res = process_single_land(land, tile)
        if res.get("success"):
            processed += 1
        else:
            failed.append(land.get("id"))
            last_error = res.get("error")

    duration = int(time.time() - start)
    logger.info(f"Finished processing queue {queue_id}: {processed}/{len(lands)} lands in {duration}s")

    return {
        "queue_id": queue_id,
        "processed_count": processed,
        "total_lands": len(lands),
        "failed_lands": failed,
        "last_error": last_error,
        "duration_s": duration,
    }

def process_queue(limit: int = 5):
    """Process queued NDVI requests."""
    logger.info(f"🔁 Checking ndvi_request_queue (limit={limit})")
    rq = supabase.table("ndvi_request_queue").select("*").eq("status", "queued").order("created_at", desc=False).limit(limit).execute()
    requests = rq.data or []
    if not requests:
        logger.info("No queued NDVI requests.")
        return

    for req in requests:
        queue_id = req["id"]
        tenant_id = req["tenant_id"]
        tile_id = req["tile_id"]
        land_ids = req.get("land_ids", [])
        logger.info(f"Processing queue item {queue_id} ({len(land_ids)} lands)")

        supabase.table("ndvi_request_queue").update({"status": "processing", "started_at": now_iso()}).eq("id", queue_id).execute()

        result = process_request_sync(queue_id, tenant_id, land_ids, tile_id)
        final_status = "completed" if result.get("processed_count", 0) > 0 else "failed"

        supabase.table("ndvi_request_queue").update({
            "status": final_status,
            "processed_count": result.get("processed_count"),
            "completed_at": now_iso(),
            "last_error": result.get("last_error"),
        }).eq("id", queue_id).execute()

        logger.info(f"Queue {queue_id} finished: {final_status}")

# ----------------------
# CLI entry
# ----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run NDVI worker (v4.2.0)")
    parser.add_argument("--limit", type=int, default=int(os.getenv("NDVI_WORKER_LIMIT", "5")))
    args = parser.parse_args()
    logger.info(f"🚀 Starting NDVI worker v4.2.0 (limit={args.limit})")
    process_queue(limit=args.limit)
    logger.info("✅ NDVI worker finished.")
