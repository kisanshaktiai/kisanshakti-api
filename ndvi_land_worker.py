"""
NDVI Land Worker v4.4.0 — KisanShaktiAI Production
---------------------------------------------------
✅ Fully uses environment-based config (no hardcoded creds)
✅ Caches Backblaze B2 authorization for high performance
✅ Handles private B2 bucket downloads securely
✅ Auto-transforms EPSG:4326 → raster CRS
✅ Adds polygon buffer to prevent overlap errors
✅ Validates land-raster intersection before masking
✅ Generates accurate vegetation NDVI PNGs
✅ Saves NDVI stats + thumbnails to Supabase
✅ Detailed structured logging + error capture
"""

import os, io, json, datetime, logging, traceback, functools
import numpy as np, requests, rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
from shapely.geometry import shape, mapping
from PIL import Image
import matplotlib.cm as cm
from supabase import create_client

# ---------------- Config ----------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

B2_APP_KEY_ID = os.environ.get("B2_KEY_ID")
B2_APP_KEY = os.environ.get("B2_APP_KEY")
B2_BUCKET_NAME = os.environ.get("B2_BUCKET_RAW", "kisanshakti-ndvi-tiles")
B2_PREFIX = os.environ.get("B2_PREFIX", "tiles/")
SUPABASE_NDVI_BUCKET = os.environ.get("SUPABASE_NDVI_BUCKET", "ndvi-thumbnails")

# ---------------- Validations ----------------
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("❌ SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY missing in environment.")
if not (B2_APP_KEY_ID and B2_APP_KEY):
    raise RuntimeError("❌ B2_KEY_ID or B2_APP_KEY missing in environment.")

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ndvi-worker-v4.4.0")

# ---------------- Supabase ----------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- Backblaze B2 ----------------
@functools.lru_cache(maxsize=1)
def b2_authorize_cached():
    """Authorize Backblaze B2 account and cache credentials."""
    logger.info("🔑 Authorizing Backblaze B2 account (cached)...")
    res = requests.get(
        "https://api.backblazeb2.com/b2api/v2/b2_authorize_account",
        auth=requests.auth.HTTPBasicAuth(B2_APP_KEY_ID, B2_APP_KEY),
        timeout=30
    )
    res.raise_for_status()
    data = res.json()
    logger.info("✅ B2 authorization successful (token cached)")
    return {"auth_token": data["authorizationToken"], "download_url": data["downloadUrl"]}

def b2_download_file(file_path: str) -> io.BytesIO:
    """Download a file from Backblaze B2 securely using cached token."""
    auth_data = b2_authorize_cached()
    url = f"{auth_data['download_url']}/file/{B2_BUCKET_NAME}/{file_path}"
    headers = {"Authorization": auth_data["auth_token"]}
    logger.info(f"📥 Downloading from B2: {file_path}")
    res = requests.get(url, headers=headers, timeout=120)
    if res.status_code == 200:
        buf = io.BytesIO(res.content)
        buf.seek(0)
        logger.info(f"✅ Downloaded {len(res.content)/1024/1024:.2f}MB from B2")
        return buf
    raise FileNotFoundError(f"⚠️ Failed to fetch {file_path} (status={res.status_code})")

# ---------------- NDVI Utilities ----------------
def calculate_ndvi(red, nir):
    """Compute NDVI safely."""
    np.seterr(divide="ignore", invalid="ignore")
    ndvi = (nir - red) / (nir + red)
    return np.clip(ndvi, -1, 1)

def calculate_ndvi_stats(arr):
    """Compute NDVI statistics safely."""
    valid = arr[~np.isnan(arr)]
    total = arr.size
    if valid.size == 0:
        return None
    return {
        "mean": float(np.mean(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "std": float(np.std(valid)),
        "coverage": float(valid.size / total * 100),
        "valid": int(valid.size),
        "total": int(total)
    }

def create_colorized_ndvi_png(ndvi, cmap="RdYlGn") -> bytes:
    """Generate a vegetation color PNG from NDVI values."""
    norm = np.clip((ndvi + 1) / 2, 0, 1)
    rgba = (cm.get_cmap(cmap)(norm) * 255).astype(np.uint8)
    rgba[..., 3][np.isnan(ndvi)] = 0
    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    buf.seek(0)
    return buf.getvalue()

def upload_thumbnail_to_supabase(tenant_id, land_id, date, png_bytes):
    """Upload vegetation PNG to Supabase Storage and return public URL."""
    path = f"{tenant_id}/{land_id}/{date}/vegetation_map.png"
    supabase.storage.from_(SUPABASE_NDVI_BUCKET).upload(
        path, io.BytesIO(png_bytes),
        {"content-type": "image/png", "upsert": True}
    )
    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_NDVI_BUCKET}/{path}"

# ---------------- Core NDVI Thumbnail Generator ----------------
def process_land_ndvi_thumbnail(land, tile):
    """
    Extract NDVI for given land geometry from tile’s NDVI.tif in B2,
    generate vegetation thumbnail, and save to Supabase.
    """
    land_id = land["id"]
    tenant_id = land["tenant_id"]
    tile_id = tile["tile_id"]
    acq_date = tile["acquisition_date"]

    try:
        geom_raw = land.get("boundary_polygon_old") or land.get("boundary_polygon")
        if not geom_raw:
            raise ValueError("Missing boundary polygon")

        land_geom = shape(geom_raw if isinstance(geom_raw, dict) else json.loads(geom_raw))

        ndvi_path = f"{B2_PREFIX}ndvi/{tile_id}/{acq_date}/ndvi.tif"
        logger.info(f"🛰️ Starting NDVI extraction for land {land_id} (tile {tile_id}, {acq_date})")

        ndvi_buf = b2_download_file(ndvi_path)
        with rasterio.open(ndvi_buf) as src:
            src_crs = src.crs.to_string()
            logger.info(f"🗺️ NDVI raster CRS: {src_crs}")

            # Buffer polygon slightly (~50m in degrees)
            buffered = land_geom.buffer(0.0005)

            # Transform to raster CRS
            geom_t = transform_geom("EPSG:4326", src_crs, mapping(buffered))

            # Check overlap
            r_bounds = src.bounds
            g_bounds = shape(geom_t).bounds
            if (
                g_bounds[2] < r_bounds.left or g_bounds[0] > r_bounds.right or
                g_bounds[3] < r_bounds.bottom or g_bounds[1] > r_bounds.top
            ):
                raise ValueError("Input shapes do not overlap raster bounds")

            arr, _ = mask(src, [geom_t], crop=True, all_touched=True, nodata=np.nan)
            ndvi = arr[0]

        stats = calculate_ndvi_stats(ndvi)
        if not stats:
            raise ValueError("No valid NDVI pixels found inside boundary")

        logger.info(f"🌿 NDVI mean={stats['mean']:.3f}, coverage={stats['coverage']:.1f}%")

        png_bytes = create_colorized_ndvi_png(ndvi)
        image_url = upload_thumbnail_to_supabase(tenant_id, land_id, acq_date, png_bytes)

        now = datetime.datetime.utcnow().isoformat()

        # ✅ Update micro_tile_thumbnail table
        supabase.table("micro_tile_thumbnail").upsert({
            "tenant_id": tenant_id,
            "land_id": land_id,
            "acquisition_date": acq_date,
            "tile_id": tile_id,
            "image_url": image_url,
            "ndvi_mean": stats["mean"],
            "ndvi_min": stats["min"],
            "ndvi_max": stats["max"],
            "ndvi_std_dev": stats["std"],
            "valid_pixels": stats["valid"],
            "total_pixels": stats["total"],
            "coverage_percent": stats["coverage"],
            "created_at": now,
            "updated_at": now
        }, on_conflict="land_id,acquisition_date").execute()

        # ✅ Also record in ndvi_data table (summary)
        supabase.table("ndvi_data").upsert({
            "tenant_id": tenant_id,
            "land_id": land_id,
            "tile_id": tile_id,
            "date": acq_date,
            "ndvi_value": stats["mean"],
            "ndvi_min": stats["min"],
            "ndvi_max": stats["max"],
            "ndvi_std": stats["std"],
            "image_url": image_url,
            "updated_at": now
        }, on_conflict="land_id,date").execute()

        logger.info(f"✅ NDVI thumbnail saved for land {land_id} (mean={stats['mean']:.3f})")
        return True

    except Exception as e:
        logger.error(f"❌ Land {land_id} NDVI processing failed: {e}")
        supabase.table("ndvi_processing_logs").insert({
            "tenant_id": tenant_id,
            "land_id": land_id,
            "processing_step": "thumbnail_generation",
            "step_status": "failed",
            "completed_at": datetime.datetime.utcnow().isoformat(),
            "error_message": str(e)[:400],
            "error_details": {"traceback": traceback.format_exc()[:1000]},
        }).execute()
        return False
