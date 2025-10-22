"""
NDVI Land Worker (v3.9.4-STABLE)
--------------------------------
🚀 Production-grade NDVI processor for KisanShaktiAI
- Auto-detects satellite tile intersections for each land
- Fixed Backblaze B2 path layout (tiles/raw & tiles/ndvi)
- Robust geometry parsing & validation (boundary_polygon_old)
- Multi-threaded NDVI queue processor
- Graceful error handling & detailed Supabase logging
- Fully compatible with NDVI Land API v3.9.1-FIXED

© 2025 KisanShaktiAI | Amarsinh Patil
"""

import io
import os
import json
import base64
import datetime
import logging
import traceback
from typing import Optional, Tuple
import numpy as np
import requests
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
from shapely.geometry import shape, mapping
from shapely.errors import WKTReadingError
from supabase import create_client
from PIL import Image
import matplotlib.cm as cm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# CONFIGURATION
# ============================================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

B2_KEY_ID = os.getenv("B2_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
B2_BUCKET = os.getenv("B2_BUCKET", "kisanshakti-ndvi-tiles")
SUPABASE_NDVI_BUCKET = os.getenv("SUPABASE_NDVI_BUCKET", "ndvi-thumbnails")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ Missing Supabase credentials (SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY)")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ndvi-worker-v3.9.4")

# ============================================================
# B2 HELPERS (Fixed for kisanshakti-ndvi-tiles layout)
# ============================================================
def get_b2_auth() -> dict:
    if not all([B2_KEY_ID, B2_APP_KEY, B2_BUCKET]):
        raise ValueError("❌ Missing B2 credentials")
    creds = base64.b64encode(f"{B2_KEY_ID}:{B2_APP_KEY}".encode()).decode()
    headers = {"Authorization": f"Basic {creds}"}
    resp = requests.get("https://api.backblazeb2.com/b2api/v2/b2_authorize_account", headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return {
        "download_url": data["downloadUrl"],
        "auth_token": data["authorizationToken"],
        "bucket_name": B2_BUCKET,
    }

def build_b2_path(tile_id: str, date: str, subdir: str, filename: str) -> str:
    return f"tiles/{subdir.strip('/')}/{tile_id}/{date}/{filename}"

def download_b2_file(tile_id: str, date: str, subdir: str, filename: str) -> Optional[io.BytesIO]:
    try:
        b2 = get_b2_auth()
        path = build_b2_path(tile_id, date, subdir, filename)
        url = f"{b2['download_url']}/file/{b2['bucket_name']}/{path}"
        headers = {"Authorization": b2["auth_token"]}
        resp = requests.get(url, headers=headers, timeout=120)
        if resp.status_code == 200:
            logger.info(f"✅ B2 file found: {path}")
            return io.BytesIO(resp.content)
        logger.warning(f"⚠️ B2 file missing: {path} ({resp.status_code})")
        return None
    except Exception as e:
        logger.error(f"❌ B2 download failed for {tile_id}/{filename}: {e}")
        return None

# ============================================================
# NDVI UTILITIES
# ============================================================
def calculate_ndvi(red, nir):
    np.seterr(divide="ignore", invalid="ignore")
    ndvi = (nir - red) / (nir + red)
    return np.clip(ndvi, -1, 1)

def calculate_statistics(arr):
    valid = arr[~np.isnan(arr)]
    total = arr.size
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

def create_colorized_ndvi_png(ndvi, cmap="RdYlGn") -> bytes:
    normalized = np.clip((ndvi + 1) / 2, 0, 1)
    rgba = (cm.get_cmap(cmap)(normalized) * 255).astype(np.uint8)
    rgba[..., 3][np.isnan(ndvi)] = 0
    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    buf.seek(0)
    return buf.getvalue()

def upload_thumbnail_to_supabase(land_id, date, png_bytes) -> Optional[str]:
    path = f"{land_id}/{date}/ndvi_color.png"
    try:
        supabase.storage.from_(SUPABASE_NDVI_BUCKET).upload(path, io.BytesIO(png_bytes),
            {"content-type": "image/png", "upsert": True})
        return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_NDVI_BUCKET}/{path}"
    except Exception as e:
        logger.warning(f"⚠️ Upload failed: {e}")
        return None

# ============================================================
# TILE DETECTION
# ============================================================
def find_intersecting_tile(land_geom, land):
    """
    Find satellite tile that intersects the given land polygon.
    ✅ Uses only spatial intersection, ignores state/district filters.
    ✅ Works with GeoJSON geometries in Supabase.
    """
    try:
        # Fetch a reasonable number of tiles to check intersection
        tiles_res = supabase.table("satellite_tiles").select("tile_id,geometry").limit(2000).execute()
        tiles = tiles_res.data or []

        if not tiles:
            logger.warning("⚠️ No satellite tiles found in database.")
            return None

        for t in tiles:
            try:
                geom_data = t.get("geometry")
                if not geom_data:
                    continue

                # Parse geometry safely
                if isinstance(geom_data, str):
                    geom = shape(json.loads(geom_data))
                elif isinstance(geom_data, dict):
                    geom = shape(geom_data)
                else:
                    continue

                # Spatial intersection test
                if geom.intersects(land_geom):
                    logger.info(f"🛰️ Found intersecting tile {t['tile_id']} for land {land.get('id')}")
                    return t

            except Exception as ge:
                logger.warning(f"⚠️ Invalid tile geometry skipped: {ge}")
                continue

        logger.warning(f"⚠️ No intersecting tile found for land {land.get('id')}")
        return None

    except Exception as e:
        logger.error(f"❌ Tile intersection search failed: {e}")
        logger.error(traceback.format_exc())
        return None

# ============================================================
# MAIN LAND PROCESSOR
# ============================================================
def process_farmer_land(land: dict, tile: Optional[dict] = None) -> bool:
    land_id = land.get("id")
    tenant_id = land.get("tenant_id")
    try:
        logger.info(f"🌿 Processing land {land_id}")

        raw_geom = land.get("boundary_polygon_old") or land.get("boundary_polygon") or land.get("boundary")
        if not raw_geom:
            raise ValueError("Missing boundary_polygon_old")

        # Geometry normalization
        if isinstance(raw_geom, str):
            geom_json = json.loads(raw_geom)
        elif isinstance(raw_geom, dict):
            geom_json = raw_geom
        else:
            raise ValueError(f"Invalid geometry type: {type(raw_geom)}")

        land_geom = shape(geom_json)
        if not land_geom.is_valid:
            land_geom = land_geom.buffer(0)

        # Find satellite tile
        if not tile or not tile.get("tile_id"):
            tile = find_intersecting_tile(land_geom, land)
            if not tile:
                raise ValueError("Unable to determine satellite tile for this land.")
        tile_id = tile["tile_id"]
        date = tile.get("acquisition_date") or datetime.date.today().isoformat()

        # Fetch or compute NDVI
        ndvi_buf = download_b2_file(tile_id, date, "ndvi", "ndvi.tif")
        ndvi_array = None
        if ndvi_buf:
            try:
                with rasterio.io.MemoryFile(ndvi_buf.read()) as mem:
                    with mem.open() as src:
                        geom_t = transform_geom("EPSG:4326", src.crs.to_string(), mapping(land_geom))
                        arr, _ = mask(src, [geom_t], crop=True, all_touched=True, nodata=np.nan)
                        ndvi_array = arr[0]
            except Exception as e:
                logger.warning(f"⚠️ Failed reading precomputed NDVI: {e}")

        if ndvi_array is None:
            b04 = download_b2_file(tile_id, date, "raw", "B04.tif")
            b08 = download_b2_file(tile_id, date, "raw", "B08.tif")
            if not b04 or not b08:
                raise FileNotFoundError(f"Missing raw bands for {tile_id}")
            with rasterio.io.MemoryFile(b04.read()) as red_mem, rasterio.io.MemoryFile(b08.read()) as nir_mem:
                with red_mem.open() as red, nir_mem.open() as nir:
                    geom_t = transform_geom("EPSG:4326", red.crs.to_string(), mapping(land_geom))
                    r_clip, _ = mask(red, [geom_t], crop=True, all_touched=True, nodata=np.nan)
                    n_clip, _ = mask(nir, [geom_t], crop=True, all_touched=True, nodata=np.nan)
                    ndvi_array = calculate_ndvi(r_clip[0], n_clip[0])

        stats = calculate_statistics(ndvi_array)
        if stats["valid_pixels"] == 0:
            raise ValueError("No valid NDVI pixels after masking")

        png_bytes = create_colorized_ndvi_png(ndvi_array)
        image_url = upload_thumbnail_to_supabase(land_id, date, png_bytes)
        now = datetime.datetime.utcnow().isoformat()

        # Write NDVI data
        supabase.table("ndvi_data").upsert({
            "tenant_id": tenant_id, "land_id": land_id, "tile_id": tile_id,
            "date": date, "ndvi_value": stats["mean"], "ndvi_min": stats["min"],
            "ndvi_max": stats["max"], "ndvi_std": stats["std"],
            "coverage_percentage": stats["coverage"], "image_url": image_url,
            "metadata": stats, "updated_at": now
        }, on_conflict="land_id,date").execute()

        supabase.table("ndvi_micro_tiles").upsert({
            "tenant_id": tenant_id, "land_id": land_id,
            "acquisition_date": date, "ndvi_mean": stats["mean"],
            "ndvi_min": stats["min"], "ndvi_max": stats["max"],
            "ndvi_std_dev": stats["std"], "ndvi_thumbnail_url": image_url,
            "cloud_cover": 0, "created_at": now
        }, on_conflict="land_id,acquisition_date").execute()

        supabase.table("lands").update({
            "last_ndvi_value": stats["mean"], "last_ndvi_calculation": date,
            "ndvi_thumbnail_url": image_url, "ndvi_tested": True,
            "last_processed_at": now
        }).eq("id", land_id).execute()

        logger.info(f"✅ Land {land_id} processed successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Land {land_id} failed: {e}")
        try:
            supabase.table("ndvi_processing_logs").insert({
                "tenant_id": tenant_id, "land_id": land_id,
                "processing_step": "ndvi_calculation", "step_status": "failed",
                "completed_at": datetime.datetime.utcnow().isoformat(),
                "error_message": str(e)[:400],
                "error_details": {"traceback": traceback.format_exc()[:1000]},
            }).execute()
        except Exception as log_err:
            logger.warning(f"⚠️ Log insert failed: {log_err}")
        return False

# ============================================================
# QUEUE PROCESSOR (Multi-threaded)
# ============================================================
def process_single_land(land, tile):
    try:
        ok = process_farmer_land(land, tile)
        return (land["id"], ok)
    except Exception as e:
        logger.error(f"Threaded land failure: {e}")
        return (land["id"], False)

def process_queue(limit=10, max_workers=4):
    logger.info("🧾 Checking NDVI request queue...")
    rq = supabase.table("ndvi_request_queue").select("*").eq("status", "queued").limit(limit).execute()
    requests = rq.data or []
    if not requests:
        logger.info("No queued NDVI requests.")
        return

    for req in requests:
        req_id = req["id"]
        tenant_id = req["tenant_id"]
        land_ids = req.get("land_ids", [])
        tile_id = req.get("tile_id")

        supabase.table("ndvi_request_queue").update({"status": "processing"}).eq("id", req_id).execute()
        lands_res = supabase.table("lands").select("*").eq("tenant_id", tenant_id).in_("id", land_ids).execute()
        lands = lands_res.data or []

        tile_res = supabase.table("satellite_tiles").select("*").eq("tile_id", tile_id).limit(1).execute()
        tile = tile_res.data[0] if tile_res.data else None

        processed, failed = 0, 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_land, land, tile) for land in lands]
            for future in as_completed(futures):
                lid, ok = future.result()
                if ok:
                    processed += 1
                else:
                    failed += 1

        supabase.table("ndvi_request_queue").update({
            "status": "completed" if failed == 0 else "failed",
            "processed_count": processed,
            "failed_count": failed,
            "completed_at": datetime.datetime.utcnow().isoformat(),
            "last_error": None if failed == 0 else f"{failed} lands failed"
        }).eq("id", req_id).execute()

        logger.info(f"🎯 Queue {req_id} done | OK={processed}, Failed={failed}")

# ============================================================
# ENTRY POINT
# ============================================================
def main():
    limit = int(os.getenv("NDVI_WORKER_LIMIT", 10))
    workers = int(os.getenv("NDVI_WORKER_THREADS", 4))
    logger.info(f"🚀 NDVI Worker v3.9.4 starting | limit={limit}, threads={workers}")
    try:
        process_queue(limit=limit, max_workers=workers)
    except Exception as e:
        logger.error(f"❌ Worker crashed: {e}")
        logger.error(traceback.format_exc())
    logger.info("🏁 NDVI Worker finished")

if __name__ == "__main__":
    main()
