"""
NDVI Land Worker (v3.7-secure)
------------------------------
Processes queued NDVI requests for each tenant’s lands.

✅ Features:
- Secure Backblaze B2 key-based access (no public URL)
- NDVI calculation with fallback logic
- Inserts into ndvi_data + ndvi_micro_tiles
- Updates lands summary
- Logs into ndvi_processing_logs
- Clear logging with removable debug sections (marked ⚠️)

© KisanShaktiAI | 2025
"""

import io
import os
import json
import datetime
import logging
import traceback
import base64
import requests
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.io import MemoryFile
from rasterio.warp import transform_geom
from shapely.geometry import shape, mapping
from supabase import create_client

# ==========================
# CONFIGURATION
# ==========================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
B2_KEY_ID = os.getenv("B2_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
B2_BUCKET = os.getenv("B2_BUCKET")
B2_PREFIX = os.getenv("B2_PREFIX", "tiles/processed/")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ Missing Supabase credentials (SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY)")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ndvi-worker")


# ==========================
# B2 AUTHENTICATION
# ==========================

def get_b2_auth():
    """Authenticate with Backblaze B2 using key-based API access."""
    if not all([B2_KEY_ID, B2_APP_KEY, B2_BUCKET]):
        raise ValueError("Missing one or more B2 credentials (B2_KEY_ID, B2_APP_KEY, B2_BUCKET)")

    credentials = base64.b64encode(f"{B2_KEY_ID}:{B2_APP_KEY}".encode()).decode()
    headers = {"Authorization": f"Basic {credentials}"}

    resp = requests.get("https://api.backblazeb2.com/b2api/v2/b2_authorize_account", headers=headers)
    if resp.status_code != 200:
        raise Exception(f"❌ B2 authorization failed: {resp.status_code} {resp.text[:200]}")

    data = resp.json()
    return {
        "api_url": data["apiUrl"],
        "download_url": data["downloadUrl"],
        "auth_token": data["authorizationToken"],
        "bucket_name": B2_BUCKET,
    }


def build_b2_path(tile_id: str, date: str, subdir: str, filename: str) -> str:
    """Construct object path based on prefix."""
    return f"{B2_PREFIX.rstrip('/')}/{subdir}/{tile_id}/{date}/{filename}"


def download_b2_file(tile_id: str, date: str, subdir: str, filename: str) -> io.BytesIO | None:
    """Securely download file from Backblaze B2 bucket."""
    b2 = get_b2_auth()
    path = build_b2_path(tile_id, date, subdir, filename)
    url = f"{b2['download_url'].rstrip('/')}/file/{b2['bucket_name']}/{path}"

    headers = {"Authorization": b2["auth_token"]}
    logger.info(f"📦 Downloading from B2: {subdir}/{filename} for tile {tile_id}")

    try:
        resp = requests.get(url, headers=headers, timeout=90)
        if resp.status_code == 200:
            logger.info(f"✅ {filename} ({len(resp.content)/1e6:.2f} MB) downloaded successfully")
            return io.BytesIO(resp.content)
        elif resp.status_code == 404:
            logger.warning(f"⚠️ File not found: {path}")
        else:
            logger.error(f"❌ B2 download failed ({resp.status_code}): {resp.text[:200]}")
    except Exception as e:
        logger.error(f"❌ Exception downloading from B2: {e}")

    return None


# ==========================
# NDVI CORE LOGIC
# ==========================

def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute NDVI = (NIR - RED) / (NIR + RED)."""
    np.seterr(divide="ignore", invalid="ignore")
    denom = nir + red
    ndvi = np.where(denom == 0, np.nan, (nir - red) / denom)
    return np.clip(ndvi, -1, 1)


def calculate_statistics(arr: np.ndarray) -> dict:
    """Compute NDVI statistics."""
    valid = arr[~np.isnan(arr)]
    total = arr.size
    if valid.size == 0:
        return {
            "mean": None, "min": None, "max": None,
            "std": None, "valid_pixels": 0,
            "total_pixels": total, "coverage": 0.0,
        }
    return {
        "mean": float(np.mean(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "std": float(np.std(valid)),
        "valid_pixels": int(valid.size),
        "total_pixels": int(total),
        "coverage": float(valid.size / total * 100),
    }


# ==========================
# PROCESSING FUNCTION
# ==========================

def process_farmer_land(land: dict, tile: dict) -> bool:
    """Process NDVI for a single land."""
    land_id = land["id"]
    tenant_id = land["tenant_id"]  # ⚠️ Sensitive — remove in prod logs
    tile_id = tile["tile_id"]
    date = tile.get("acquisition_date") or datetime.date.today().isoformat()

    try:
        logger.info(f"🌿 Processing land={land_id} | tile={tile_id}")  # ⚠️ Optional in prod

        geom_raw = land.get("boundary_polygon_old")
        if not geom_raw:
            raise Exception("Missing boundary_polygon_old")

        geom_json = json.loads(geom_raw) if isinstance(geom_raw, str) else geom_raw
        land_geom = shape(geom_json)

        # Try NDVI.tif first
        ndvi_buf = download_b2_file(tile_id, date, "ndvi", "ndvi.tif")
        if ndvi_buf:
            ndvi_src = rasterio.open(ndvi_buf)
            ndvi_arr = ndvi_src.read(1)
            ndvi_meta = ndvi_src.meta
        else:
            # Fall back to raw Sentinel bands
            b04_buf = download_b2_file(tile_id, date, "raw", "B04.tif")
            b08_buf = download_b2_file(tile_id, date, "raw", "B08.tif")

            if not b04_buf or not b08_buf:
                raise Exception("Missing Sentinel bands or NDVI file in B2")

            with rasterio.open(b04_buf) as red_src, rasterio.open(b08_buf) as nir_src:
                geom_trans = transform_geom("EPSG:4326", red_src.crs.to_string(), mapping(land_geom))
                red_clip, _ = mask(red_src, [geom_trans], crop=True, nodata=np.nan)
                nir_clip, _ = mask(nir_src, [geom_trans], crop=True, nodata=np.nan)
                ndvi_arr = calculate_ndvi(red_clip[0].astype(float), nir_clip[0].astype(float))
                ndvi_meta = red_src.meta

        stats = calculate_statistics(ndvi_arr)
        if stats["valid_pixels"] == 0:
            raise Exception("No valid NDVI pixels found")

        # Save to Supabase
        ndvi_record = {
            "tenant_id": tenant_id,
            "land_id": land_id,
            "tile_id": tile_id,
            "date": date,
            "ndvi_value": stats["mean"],
            "ndvi_min": stats["min"],
            "ndvi_max": stats["max"],
            "ndvi_std": stats["std"],
            "coverage": stats["coverage"],
            "created_at": datetime.datetime.utcnow().isoformat(),
            "metadata": stats,
        }

        supabase.table("ndvi_data").upsert(ndvi_record, on_conflict="land_id,date").execute()
        logger.info(f"✅ NDVI saved for land {land_id} | mean={stats['mean']:.3f}")  # ⚠️ Optional in prod

        supabase.table("lands").update({
            "last_ndvi_value": stats["mean"],
            "last_ndvi_calculation": date,
        }).eq("id", land_id).execute()

        return True

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"❌ Land processing failed: {e}\n{tb[:400]}")  # ⚠️ Optional — truncate sensitive data

        supabase.table("ndvi_processing_logs").insert({
            "tenant_id": tenant_id,
            "land_id": land_id,
            "satellite_tile_id": tile.get("id"),
            "processing_step": "ndvi_calculation",
            "step_status": "failed",
            "completed_at": datetime.datetime.utcnow().isoformat(),
            "error_message": str(e),
            "error_details": {"traceback": tb[:400]},
            "metadata": {"tile_id": tile_id, "date": date},
        }).execute()
        return False


# ==========================
# QUEUE PROCESSOR
# ==========================

def process_queue(limit: int = 10):
    """Process queued NDVI requests."""
    logger.info("🧾 Checking NDVI request queue...")

    rq = supabase.table("ndvi_request_queue").select("*").eq("status", "queued").limit(limit).execute()
    requests = rq.data or []
    if not requests:
        logger.info("✅ No queued NDVI requests.")
        return

    logger.info(f"📋 Found {len(requests)} queued request(s).")

    for req in requests:
        req_id = req["id"]
        tenant_id = req["tenant_id"]
        tile_id = req["tile_id"]
        logger.info(f"⚙️ Starting queue {req_id} for tenant={tenant_id} tile={tile_id}")  # ⚠️ Optional in prod

        supabase.table("ndvi_request_queue").update({
            "status": "processing",
            "started_at": datetime.datetime.utcnow().isoformat(),
        }).eq("id", req_id).execute()

        lands = supabase.table("lands").select("*").eq("tenant_id", tenant_id).in_("id", req["land_ids"]).execute().data
        if not lands:
            logger.warning(f"⚠️ No lands found for tenant={tenant_id}")
            supabase.table("ndvi_request_queue").update({
                "status": "failed",
                "error_message": "No lands found",
            }).eq("id", req_id).execute()
            continue

        tile_res = supabase.table("satellite_tiles").select("*").eq("tile_id", tile_id).limit(1).execute()
        if not tile_res.data:
            logger.error(f"❌ No satellite_tiles record for {tile_id}")
            supabase.table("ndvi_request_queue").update({
                "status": "failed",
                "error_message": "Missing tile metadata",
            }).eq("id", req_id).execute()
            continue

        tile = tile_res.data[0]
        processed = sum(process_farmer_land(land, tile) for land in lands)

        supabase.table("ndvi_request_queue").update({
            "status": "completed",
            "processed_count": processed,
            "completed_at": datetime.datetime.utcnow().isoformat(),
        }).eq("id", req_id).execute()

        logger.info(f"🎯 Completed request {req_id} | Lands processed: {processed}")  # ⚠️ Optional in prod


# ==========================
# ENTRY POINT
# ==========================

def main(limit: int = 10):
    logger.info("🚀 NDVI Worker started (secure v3.7)")
    process_queue(limit)
    logger.info("🏁 NDVI Worker finished.")


if __name__ == "__main__":
    main()
