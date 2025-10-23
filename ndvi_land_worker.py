"""
NDVI Land Worker (v4.3.1-PRODUCTION)
------------------------------------
🌿 KisanShaktiAI NDVI Processor - Private B2 Bucket Compatible

✅ Fix: Uses authorized B2 API download URLs
✅ Automatically authenticates and downloads B04/B08/NDVI.tif securely
✅ Works with both raw and precomputed NDVI files
✅ Uploads vegetation PNG to Supabase
"""

import os, io, json, datetime, logging, traceback, base64
import numpy as np, requests, rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
from shapely.geometry import shape, mapping
from PIL import Image
import matplotlib.cm as cm
from supabase import create_client

# ------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://qfklkkzxemsbeniyugiz.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_NDVI_BUCKET = os.getenv("SUPABASE_NDVI_BUCKET", "ndvi-thumbnails")

B2_KEY_ID = os.getenv("B2_KEY_ID")
B2_APPLICATION_KEY = os.getenv("B2_APPLICATION_KEY")
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME", "kisanshakti-ndvi-tiles")

if not (SUPABASE_KEY and B2_KEY_ID and B2_APPLICATION_KEY):
    raise ValueError("❌ Missing Supabase or B2 credentials!")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ndvi-worker-v4.3.1")

# ------------------------------------------------------
# B2 AUTH + DOWNLOAD
# ------------------------------------------------------
def b2_authorize():
    logger.info("🔑 Authorizing Backblaze B2 account...")
    url = "https://api.backblazeb2.com/b2api/v2/b2_authorize_account"
    auth = requests.auth.HTTPBasicAuth(B2_KEY_ID, B2_APPLICATION_KEY)
    res = requests.get(url, auth=auth)
    res.raise_for_status()
    data = res.json()
    logger.info("✅ B2 authorization success")
    return {
        "auth_token": data["authorizationToken"],
        "download_url": data["downloadUrl"],
        "api_url": data["apiUrl"],
    }

def b2_download_file(file_path: str, auth_data: dict) -> bytes:
    """Downloads file securely from B2 using authorization token."""
    url = f"{auth_data['download_url']}/file/{B2_BUCKET_NAME}/{file_path}"
    headers = {"Authorization": auth_data["auth_token"]}
    res = requests.get(url, headers=headers, timeout=120)
    if res.status_code == 200:
        return io.BytesIO(res.content)
    else:
        raise FileNotFoundError(f"❌ Failed to fetch {file_path} ({res.status_code})")

# ------------------------------------------------------
# NDVI PROCESSING UTILITIES
# ------------------------------------------------------
def calculate_ndvi(red, nir):
    np.seterr(divide="ignore", invalid="ignore")
    ndvi = (nir - red) / (nir + red)
    return np.clip(ndvi, -1, 1)

def create_colorized_ndvi_png(ndvi, cmap="RdYlGn") -> bytes:
    normalized = np.clip((ndvi + 1) / 2, 0, 1)
    rgba = (cm.get_cmap(cmap)(normalized) * 255).astype(np.uint8)
    rgba[..., 3][np.isnan(ndvi)] = 0
    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    buf.seek(0)
    return buf.getvalue()

def upload_thumbnail_to_supabase(path: str, png_bytes: bytes) -> str:
    supabase.storage.from_(SUPABASE_NDVI_BUCKET).upload(
        path, io.BytesIO(png_bytes), {"content-type": "image/png", "upsert": True}
    )
    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_NDVI_BUCKET}/{path}"

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

# ------------------------------------------------------
# MAIN PROCESSOR
# ------------------------------------------------------
def process_farmer_land(land, tile):
    land_id = land["id"]
    tenant_id = land["tenant_id"]
    try:
        geom_raw = land.get("boundary_polygon_old") or land.get("boundary_polygon")
        if not geom_raw:
            raise ValueError("Missing boundary polygon")
        land_geom = shape(geom_raw if isinstance(geom_raw, dict) else json.loads(geom_raw))

        # Fetch tile metadata
        tile_id = tile["tile_id"]
        date = tile["acquisition_date"]

        b04_path = tile.get("red_band_path") or f"tiles/raw/{tile_id}/{date}/B04.tif"
        b08_path = tile.get("nir_band_path") or f"tiles/raw/{tile_id}/{date}/B08.tif"
        ndvi_path = tile.get("ndvi_path") or f"tiles/ndvi/{tile_id}/{date}/ndvi.tif"

        logger.info(f"🛰️ Tile {tile_id} | Date {date}")
        logger.info(f"B04: {b04_path}")
        logger.info(f"B08: {b08_path}")
        logger.info(f"NDVI: {ndvi_path}")

        auth_data = b2_authorize()

        # Try NDVI first
        try:
            ndvi_buf = b2_download_file(ndvi_path, auth_data)
            logger.info("🟢 Using precomputed NDVI file")
            with rasterio.open(ndvi_buf) as src:
                geom_t = transform_geom("EPSG:4326", src.crs.to_string(), mapping(land_geom))
                arr, _ = mask(src, [geom_t], crop=True, all_touched=True, nodata=np.nan)
                ndvi = arr[0]
        except Exception:
            logger.info("⚠️ Precomputed NDVI not found, computing from raw bands")
            red_buf = b2_download_file(b04_path, auth_data)
            nir_buf = b2_download_file(b08_path, auth_data)
            with rasterio.open(red_buf) as red, rasterio.open(nir_buf) as nir:
                geom_t = transform_geom("EPSG:4326", red.crs.to_string(), mapping(land_geom))
                arr_r, _ = mask(red, [geom_t], crop=True, all_touched=True, nodata=np.nan)
                arr_n, _ = mask(nir, [geom_t], crop=True, all_touched=True, nodata=np.nan)
                ndvi = calculate_ndvi(arr_r[0], arr_n[0])

        stats = calculate_statistics(ndvi)
        if stats["valid_pixels"] == 0:
            raise ValueError("No valid NDVI pixels found")

        png_bytes = create_colorized_ndvi_png(ndvi)
        path = f"{tenant_id}/{land_id}/{date}/vegetation_map.png"
        image_url = upload_thumbnail_to_supabase(path, png_bytes)
        now = datetime.datetime.utcnow().isoformat()

        # Save results
        supabase.table("ndvi_data").upsert({
            "tenant_id": tenant_id,
            "land_id": land_id,
            "tile_id": tile_id,
            "date": date,
            "ndvi_value": stats["mean"],
            "ndvi_min": stats["min"],
            "ndvi_max": stats["max"],
            "ndvi_std": stats["std"],
            "image_url": image_url,
            "updated_at": now
        }, on_conflict="land_id,date").execute()

        supabase.table("micro_tile_thumbnail").upsert({
            "tenant_id": tenant_id,
            "land_id": land_id,
            "acquisition_date": date,
            "ndvi_thumbnail_url": image_url,
            "bbox": mapping(land_geom),
            "created_at": now
        }, on_conflict="land_id,acquisition_date").execute()

        logger.info(f"✅ Land {land_id} NDVI success (mean={stats['mean']:.3f})")
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
