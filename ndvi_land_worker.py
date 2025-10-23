"""
NDVI Land Worker (v4.1.0-PRODUCTION)
------------------------------------
🌿 KisanShaktiAI NDVI Processor - Schema-Aligned Version

✅ Computes NDVI dynamically from Sentinel-2 B04/B08 bands
✅ Generates vegetation map per land (boundary_polygon_old)
✅ Uploads vegetation PNG to Supabase bucket
✅ Writes results to ndvi_data + ndvi_micro_tiles
✅ Logs failures to ndvi_processing_logs
✅ Fully aligned with your database schema

Supabase Bucket: ndvi-thumbnails
Public URL Base: https://qfklkkzxemsbeniyugiz.supabase.co/storage/v1/object/public/ndvi-thumbnails
© 2025 KisanShaktiAI | Developed by Amarsinh Patil
"""

import io
import os
import json
import base64
import datetime
import logging
import traceback
from typing import Optional
import numpy as np
import requests
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
from shapely.geometry import shape, mapping
from supabase import create_client
from PIL import Image
import matplotlib.cm as cm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# CONFIGURATION
# ============================================================
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://qfklkkzxemsbeniyugiz.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
B2_KEY_ID = os.getenv("B2_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
B2_BUCKET = os.getenv("B2_BUCKET", "kisanshakti-ndvi-tiles")
SUPABASE_NDVI_BUCKET = os.getenv("SUPABASE_NDVI_BUCKET", "ndvi-thumbnails")

if not SUPABASE_KEY:
    raise ValueError("❌ Missing Supabase credentials!")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ndvi-worker-v4.1.0")

# ============================================================
# B2 HELPERS
# ============================================================
def get_b2_auth() -> dict:
    creds = base64.b64encode(f"{B2_KEY_ID}:{B2_APP_KEY}".encode()).decode()
    headers = {"Authorization": f"Basic {creds}"}
    resp = requests.get("https://api.backblazeb2.com/b2api/v2/b2_authorize_account", headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return {"download_url": data["downloadUrl"], "auth_token": data["authorizationToken"], "bucket_name": B2_BUCKET}

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

def upload_thumbnail_to_supabase(path: str, png_bytes: bytes) -> Optional[str]:
    try:
        supabase.storage.from_(SUPABASE_NDVI_BUCKET).upload(path, io.BytesIO(png_bytes),
            {"content-type": "image/png", "upsert": True})
        url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_NDVI_BUCKET}/{path}"
        logger.info(f"🖼️ Uploaded vegetation map: {url}")
        return url
    except Exception as e:
        logger.warning(f"⚠️ Upload failed: {e}")
        return None

# ============================================================
# TILE DETECTION
# ============================================================
def find_intersecting_tile(land_geom):
    try:
        tiles = supabase.table("satellite_tiles").select("tile_id,geometry").limit(2000).execute().data or []
        for t in tiles:
            geom = shape(t["geometry"] if isinstance(t["geometry"], dict) else json.loads(t["geometry"]))
            if geom.intersects(land_geom):
                return t
        return None
    except Exception as e:
        logger.error(f"Tile intersection failed: {e}")
        return None

# ============================================================
# MAIN LAND PROCESSOR
# ============================================================
def process_farmer_land(land: dict, tile: Optional[dict] = None) -> bool:
    land_id = land.get("id")
    tenant_id = land.get("tenant_id")
    try:
        logger.info(f"🌿 Processing land {land_id}")

        geom_raw = land.get("boundary_polygon_old") or land.get("boundary_polygon")
        if not geom_raw:
            raise ValueError("Missing boundary polygon")

        land_geom = shape(geom_raw if isinstance(geom_raw, dict) else json.loads(geom_raw))
        if not land_geom.is_valid:
            land_geom = land_geom.buffer(0)

        tile = tile or find_intersecting_tile(land_geom)
        if not tile:
            raise ValueError("No intersecting satellite tile found.")

        tile_id = tile["tile_id"]
        date = datetime.date.today().isoformat()

        # Download raw bands
        b04 = download_b2_file(tile_id, date, "raw", "B04.tif")
        b08 = download_b2_file(tile_id, date, "raw", "B08.tif")
        if not b04 or not b08:
            raise FileNotFoundError(f"Missing raw bands for {tile_id}")

        with rasterio.io.MemoryFile(b04.read()) as red_mem, rasterio.io.MemoryFile(b08.read()) as nir_mem:
            with red_mem.open() as red, nir_mem.open() as nir:
                geom_t = transform_geom("EPSG:4326", red.crs.to_string(), mapping(land_geom.buffer(0.0003)))
                arr_r, _ = mask(red, [geom_t], crop=True, all_touched=True, nodata=np.nan)
                arr_n, _ = mask(nir, [geom_t], crop=True, all_touched=True, nodata=np.nan)
                ndvi = calculate_ndvi(arr_r[0], arr_n[0])

        stats = calculate_statistics(ndvi)
        if stats["valid_pixels"] == 0:
            raise ValueError("No valid NDVI pixels after masking")

        png_bytes = create_colorized_ndvi_png(ndvi)
        path = f"{tenant_id}/{land_id}/{date}/vegetation_map.png"
        image_url = upload_thumbnail_to_supabase(path, png_bytes)
        now = datetime.datetime.utcnow().isoformat()

        # Insert NDVI summary
        supabase.table("ndvi_data").upsert({
            "tenant_id": tenant_id,
            "land_id": land_id,
            "tile_id": tile_id,
            "date": date,
            "ndvi_value": stats["mean"],
            "ndvi_min": stats["min"],
            "ndvi_max": stats["max"],
            "ndvi_std": stats["std"],
            "mean_ndvi": stats["mean"],
            "min_ndvi": stats["min"],
            "max_ndvi": stats["max"],
            "valid_pixels": stats["valid_pixels"],
            "total_pixels": stats["total_pixels"],
            "coverage_percentage": stats["coverage"],
            "coverage": stats["coverage"],
            "image_url": image_url,
            "updated_at": now
        }, on_conflict="land_id,date").execute()

        # Insert NDVI micro tile
        supabase.table("ndvi_micro_tiles").upsert({
            "tenant_id": tenant_id,
            "land_id": land_id,
            "acquisition_date": date,
            "cloud_cover": 0,
            "ndvi_mean": stats["mean"],
            "ndvi_min": stats["min"],
            "ndvi_max": stats["max"],
            "ndvi_std_dev": stats["std"],
            "ndvi_thumbnail_url": image_url,
            "resolution_meters": 10,
            "bbox": mapping(land_geom),
            "created_at": now
        }, on_conflict="land_id,acquisition_date").execute()

        logger.info(f"✅ NDVI saved for land {land_id} (mean={stats['mean']:.3f})")
        return True

    except Exception as e:
        logger.error(f"❌ Land {land_id} failed: {e}")
        supabase.table("ndvi_processing_logs").insert({
            "tenant_id": tenant_id,
            "land_id": land_id,
            "processing_step": "ndvi_calculation",
            "step_status": "failed",
            "completed_at": datetime.datetime.utcnow().isoformat(),
            "error_message": str(e)[:400],
            "error_details": {"traceback": traceback.format_exc()[:1000]},
        }).execute()
        return False

# ============================================================
# QUEUE PROCESSOR
# ============================================================
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
        lands = supabase.table("lands").select("*").eq("tenant_id", tenant_id).in_("id", land_ids).execute().data or []
        tile = supabase.table("satellite_tiles").select("*").eq("tile_id", tile_id).limit(1).execute().data
        tile = tile[0] if tile else None

        processed, failed = 0, 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_farmer_land, land, tile) for land in lands]
            for f in as_completed(futures):
                ok = f.result()
                if ok:
                    processed += 1
                else:
                    failed += 1

        status = "completed" if failed == 0 else "failed"
        supabase.table("ndvi_request_queue").update({
            "status": status,
            "processed_count": processed,
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
    logger.info(f"🚀 NDVI Worker v4.1.0 starting | limit={limit}, threads={workers}")
    process_queue(limit=limit, max_workers=workers)
    logger.info("🏁 NDVI Worker finished")

if __name__ == "__main__":
    main()
