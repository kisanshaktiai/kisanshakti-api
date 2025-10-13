"""
NDVI Land Worker (v3.9-complete)
--------------------------------
Processes queued NDVI requests and updates all related tables.

✅ Enhancements (v3.9):
- Correct B2 path builder (matches tiles/raw & tiles/ndvi layout)
- Uses preprocessed ndvi.tif if available, otherwise computes NDVI from raw bands
- Generates colorized NDVI PNG (RdYlGn) with transparency for nodata
- Uploads color PNG to Supabase Storage (configurable bucket)
- Inserts/updates ndvi_data, ndvi_micro_tiles, lands, ndvi_processing_logs
- Core NDVI math and pipeline logic preserved

© KisanShaktiAI | 2025
"""

import io
import os
import json
import datetime
import logging
import traceback
import base64
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

# ==========================
# CONFIGURATION
# ==========================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Backblaze B2 settings
B2_KEY_ID = os.getenv("B2_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
# Default bucket name you provided
B2_BUCKET = os.getenv("B2_BUCKET", "kisanshakti-ndvi-tiles")
# The prefix inside the bucket where tiles are stored (your example shows 'tiles')
B2_PREFIX = os.getenv("B2_PREFIX", "tiles")

# Supabase storage bucket for thumbnails (public)
SUPABASE_NDVI_BUCKET = os.getenv("SUPABASE_NDVI_BUCKET", "ndvi-thumbnails")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ Missing Supabase credentials (SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY)")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ndvi-worker-v3.9")


# ==========================
# BACKBLAZE B2 HELPERS
# ==========================
def get_b2_auth() -> dict:
    """
    Authorize with Backblaze B2 using application key ID and application key.
    Returns a dict with apiUrl, downloadUrl, authorizationToken and bucket_name.
    """
    if not all([B2_KEY_ID, B2_APP_KEY, B2_BUCKET]):
        raise ValueError("Missing B2 credentials (B2_KEY_ID, B2_APP_KEY, B2_BUCKET).")

    credentials = base64.b64encode(f"{B2_KEY_ID}:{B2_APP_KEY}".encode()).decode()
    headers = {"Authorization": f"Basic {credentials}"}

    resp = requests.get("https://api.backblazeb2.com/b2api/v2/b2_authorize_account", headers=headers, timeout=30)
    if resp.status_code != 200:
        raise Exception(f"B2 authorization failed: {resp.status_code} {resp.text[:200]}")
    data = resp.json()
    return {
        "api_url": data.get("apiUrl"),
        "download_url": data.get("downloadUrl"),
        "auth_token": data.get("authorizationToken"),
        "bucket_name": B2_BUCKET,
    }


def build_b2_path(tile_id: str, date: str, subdir: str, filename: str) -> str:
    """
    Build B2 object path consistent with your storage layout:
      tiles/raw/{tile_id}/{date}/{filename}
      tiles/ndvi/{tile_id}/{date}/{filename}
    """
    # Ensure prefix does not have duplicated slashes
    prefix = B2_PREFIX.rstrip("/")
    return f"{prefix}/{subdir}/{tile_id}/{date}/{filename}"


def download_b2_file(tile_id: str, date: str, subdir: str, filename: str) -> Optional[io.BytesIO]:
    """
    Download a file from B2 using the downloadUrl from authorization.
    Returns BytesIO on success, None if not found or error.
    """
    try:
        b2 = get_b2_auth()
        path = build_b2_path(tile_id, date, subdir, filename)
        url = f"{b2['download_url'].rstrip('/')}/file/{b2['bucket_name']}/{path}"
        headers = {"Authorization": b2["auth_token"]}

        logger.debug(f"Attempting B2 download: {url}")
        resp = requests.get(url, headers=headers, timeout=120)
        if resp.status_code == 200:
            logger.info(f"✅ B2 file downloaded: {subdir}/{filename} ({tile_id} @ {date})")
            return io.BytesIO(resp.content)
        elif resp.status_code == 404:
            logger.debug(f"⚠️ B2 file not found: {path}")
            return None
        else:
            logger.error(f"❌ B2 download error {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as ex:
        logger.error(f"❌ Exception during B2 download: {ex}")
        return None


# ==========================
# NDVI Math & Stats
# ==========================
def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - RED) / (NIR + RED) with NaN where denom == 0"""
    np.seterr(divide="ignore", invalid="ignore")
    red = red.astype(float)
    nir = nir.astype(float)
    denom = nir + red
    ndvi = np.where(denom == 0, np.nan, (nir - red) / denom)
    return np.clip(ndvi, -1.0, 1.0)


def calculate_statistics(arr: np.ndarray) -> dict:
    """Return mean, min, max, std, valid pixel count, coverage %"""
    valid = arr[~np.isnan(arr)]
    total = int(arr.size)
    if valid.size == 0:
        return {
            "mean": None,
            "min": None,
            "max": None,
            "std": None,
            "valid_pixels": 0,
            "total_pixels": total,
            "coverage": 0.0,
        }
    return {
        "mean": float(np.mean(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "std": float(np.std(valid)),
        "valid_pixels": int(valid.size),
        "total_pixels": total,
        "coverage": float(valid.size / total * 100),
    }


# ==========================
# Visualization & Upload
# ==========================
def create_colorized_ndvi_png(ndvi_array: np.ndarray, cmap_name: str = "RdYlGn") -> bytes:
    """
    Convert NDVI array (-1..1 with NaN) to a colorized PNG bytes using matplotlib colormap.
    Transparent where NDVI is NaN.
    Returns PNG bytes.
    """
    # Normalize to 0..1
    ndvi_normalized = np.clip((ndvi_array + 1.0) / 2.0, 0.0, 1.0)  # NaN stays NaN

    # Use matplotlib colormap
    cmap = cm.get_cmap(cmap_name)
    # cmap expects array in 0..1; produces RGBA floats 0..1
    rgba = cmap(ndvi_normalized)  # shape H x W x 4, float [0..1], NaN -> returns nan in channels

    # Convert nan mask to fully transparent
    nan_mask = np.isnan(ndvi_array)
    # Convert to 0..255 uint8
    rgba_uint8 = (rgba * 255).astype(np.uint8)
    # If any NaNs exist in rgba (due to ndvi_normalized NaN) set alpha=0 at those pixels
    if nan_mask.any():
        rgba_uint8[..., 3][nan_mask] = 0
    else:
        # Ensure alpha channel is fully opaque if not NaN
        rgba_uint8[..., 3] = 255

    # Create PIL Image from RGBA array
    img = Image.fromarray(rgba_uint8, mode="RGBA")

    # Save to PNG bytes
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.getvalue()


def upload_thumbnail_to_supabase(land_id: str, date: str, png_bytes: bytes) -> Optional[str]:
    """
    Upload png_bytes to Supabase Storage under SUPABASE_NDVI_BUCKET at path {land_id}/{date}/ndvi_color.png.
    Returns public URL on success, None on failure.
    """
    path = f"{land_id}/{date}/ndvi_color.png"
    try:
        # supabase.storage.from_(bucket).upload expects file-like or bytes depending on client version
        # Use a BytesIO for safety
        file_obj = io.BytesIO(png_bytes)
        # Attempt upload; supabase-py returns dict like {'data': ..., 'error': ...}
        res = supabase.storage.from_(SUPABASE_NDVI_BUCKET).upload(path, file_obj, {"content-type": "image/png", "upsert": True})
        # Handle different supabase client return styles
        if isinstance(res, dict):
            # new-style: {'data': {...}, 'error': None}
            if res.get("error"):
                logger.error(f"❌ Supabase upload error: {res.get('error')}")
                return None
        else:
            # older clients may return a HTTPResponse-like object; attempt to check
            try:
                status = getattr(res, "status_code", None)
                text = getattr(res, "text", "")
                if status and status not in (200, 201):
                    logger.error(f"❌ Supabase upload failed status {status}: {text[:200]}")
                    return None
            except Exception:
                # ignore and proceed to construct URL
                pass

        # Construct public URL for supabase storage (public bucket assumed)
        public_url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/public/{SUPABASE_NDVI_BUCKET}/{path}"
        logger.info(f"🖼️ Uploaded NDVI thumbnail to Supabase: {public_url}")
        return public_url
    except Exception as e:
        logger.error(f"❌ Exception uploading thumbnail to Supabase: {e}")
        return None


# ==========================
# Processing per land
# ==========================
def process_farmer_land(land: dict, tile: dict) -> bool:
    """
    Process NDVI for a single land:
     - get geometry
     - fetch ndvi.tif from B2 or compute from raw bands
     - crop/mask the raster to land geometry
     - compute statistics
     - create colorized PNG and upload
     - upsert ndvi_data, ndvi_micro_tiles, update lands
    """
    land_id = land.get("id")
    tenant_id = land.get("tenant_id")
    tile_id = tile.get("tile_id")
    date = tile.get("acquisition_date") or datetime.date.today().isoformat()

    try:
        logger.info(f"🌿 Processing land {land_id} with tile {tile_id} ({date})")

        # get geometry (prefer boundary_polygon_old like your worker used)
        geom_raw = land.get("boundary_polygon_old") or land.get("boundary")
        if not geom_raw:
            raise Exception("Missing boundary polygon for land")

        geom_json = json.loads(geom_raw) if isinstance(geom_raw, str) else geom_raw
        land_geom = shape(geom_json)

        # Step 1: Try precomputed NDVI product
        ndvi_buf = download_b2_file(tile_id, date, "ndvi", "ndvi.tif")
        ndvi_array = None
        ndvi_meta = None

        if ndvi_buf:
            # rasterio can open a BytesIO directly via MemoryFile
            with rasterio.io.MemoryFile(ndvi_buf.read()) as memfile:
                with memfile.open() as src:
                    # Transform polygon from EPSG:4326 to source CRS
                    geom_trans = transform_geom("EPSG:4326", src.crs.to_string(), mapping(land_geom))
                    clipped, _ = mask(src, [geom_trans], crop=True, all_touched=True, nodata=np.nan)
                    # read returns shape (bands, h, w) — NDVI must be single band
                    ndvi_array = clipped[0].astype(float)
                    ndvi_meta = src.meta
        else:
            # Step 2: Fall back to raw bands B04 (red) & B08 (nir)
            b04_buf = download_b2_file(tile_id, date, "raw", "B04.tif")
            b08_buf = download_b2_file(tile_id, date, "raw", "B08.tif")
            if not b04_buf or not b08_buf:
                raise Exception("NDVI product missing and raw bands (B04/B08) unavailable in B2")

            # Use MemoryFile to open BytesIO with rasterio
            with rasterio.io.MemoryFile(b04_buf.read()) as red_mem, rasterio.io.MemoryFile(b08_buf.read()) as nir_mem:
                with red_mem.open() as red_src, nir_mem.open() as nir_src:
                    # Reproject transform geometry into red_src CRS
                    geom_trans = transform_geom("EPSG:4326", red_src.crs.to_string(), mapping(land_geom))
                    red_clip, _ = mask(red_src, [geom_trans], crop=True, all_touched=True, nodata=np.nan)
                    nir_clip, _ = mask(nir_src, [geom_trans], crop=True, all_touched=True, nodata=np.nan)
                    # select first band (index 0)
                    red_band = red_clip[0].astype(float)
                    nir_band = nir_clip[0].astype(float)
                    ndvi_array = calculate_ndvi(red_band, nir_band)
                    ndvi_meta = red_src.meta

        if ndvi_array is None:
            raise Exception("Failed to obtain NDVI array for land")

        # Statistics
        stats = calculate_statistics(ndvi_array)
        if stats["valid_pixels"] == 0:
            raise Exception("No valid NDVI pixels after mask")

        # Create colorized PNG bytes
        png_bytes = create_colorized_ndvi_png(ndvi_array, cmap_name="RdYlGn")

        # Upload to Supabase storage
        image_url = upload_thumbnail_to_supabase(land_id, date, png_bytes)

        # Prepare NDVI record (main summary)
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
            "image_url": image_url,
            "created_at": datetime.datetime.utcnow().isoformat(),
            "metadata": stats,
        }

        # Upsert main ndvi_data table
        supabase.table("ndvi_data").upsert(ndvi_record, on_conflict="land_id,date").execute()

        # Upsert micro tile record
        micro_tile_record = {
            "tenant_id": tenant_id,
            "land_id": land_id,
            "tile_id": tile_id,
            "acquisition_date": date,
            "image_url": image_url,
            "ndvi_mean": stats["mean"],
            "ndvi_min": stats["min"],
            "ndvi_max": stats["max"],
            "ndvi_std": stats["std"],
            "coverage": stats["coverage"],
            "created_at": datetime.datetime.utcnow().isoformat(),
        }
        # Use composite conflict key if available; adjust on_conflict to your table PK/index as needed
        supabase.table("ndvi_micro_tiles").upsert(micro_tile_record, on_conflict="land_id,acquisition_date").execute()

        # Update lands summary
        supabase.table("lands").update({
            "last_ndvi_value": stats["mean"],
            "last_ndvi_calculation": date,
            "last_ndvi_image_url": image_url,
            "updated_at": datetime.datetime.utcnow().isoformat()
        }).eq("id", land_id).execute()

        logger.info(f"✅ Processed land {land_id} | mean={stats['mean']:.3f} | image_url={image_url}")
        return True

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"❌ Failed to process land {land_id}: {e}\n{tb[:400]}")
        # Log to processing logs table
        try:
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
        except Exception as log_exc:
            logger.warning(f"⚠️ Failed to write processing log: {log_exc}")
        return False


# ==========================
# Queue Processor (entry point)
# ==========================
def process_queue(limit: int = 10):
    """
    Process queued NDVI requests from the ndvi_request_queue table.
    """
    logger.info("🧾 Checking NDVI request queue...")
    try:
        rq = supabase.table("ndvi_request_queue").select("*").eq("status", "queued").limit(limit).execute()
        requests = rq.data or []
    except Exception as e:
        logger.error(f"❌ Failed to read ndvi_request_queue: {e}")
        return

    if not requests:
        logger.info("✅ No queued NDVI requests.")
        return

    logger.info(f"📋 Found {len(requests)} queued request(s).")

    for req in requests:
        req_id = req.get("id")
        tenant_id = req.get("tenant_id")
        tile_id = req.get("tile_id")

        logger.info(f"⚙️ Starting request {req_id} for tenant={tenant_id} tile={tile_id}")
        # Mark as processing
        supabase.table("ndvi_request_queue").update({
            "status": "processing",
            "started_at": datetime.datetime.utcnow().isoformat(),
        }).eq("id", req_id).execute()

        # Fetch lands
        lands_res = supabase.table("lands").select("*").eq("tenant_id", tenant_id).in_("id", req.get("land_ids", [])).execute()
        lands = lands_res.data or []
        if not lands:
            logger.warning(f"⚠️ No lands found for request {req_id}")
            supabase.table("ndvi_request_queue").update({
                "status": "failed",
                "error_message": "No lands found for tenant",
            }).eq("id", req_id).execute()
            continue

        # Fetch tile metadata
        tile_res = supabase.table("satellite_tiles").select("*").eq("tile_id", tile_id).limit(1).execute()
        if not tile_res.data:
            logger.error(f"❌ Satellite tile metadata missing for tile_id={tile_id}")
            supabase.table("ndvi_request_queue").update({
                "status": "failed",
                "error_message": "Missing tile metadata",
            }).eq("id", req_id).execute()
            continue

        tile = tile_res.data[0]
        processed_count = 0
        for land in lands:
            ok = process_farmer_land(land, tile)
            if ok:
                processed_count += 1

        # Mark completed
        supabase.table("ndvi_request_queue").update({
            "status": "completed",
            "processed_count": processed_count,
            "completed_at": datetime.datetime.utcnow().isoformat(),
        }).eq("id", req_id).execute()

        logger.info(f"🎯 Completed request {req_id} | processed_count={processed_count}")


# ==========================
# CLI / Entry
# ==========================
def main(limit: int = 10):
    logger.info("🚀 NDVI Worker v3.9 starting")
    process_queue(limit)
    logger.info("🏁 NDVI Worker v3.9 finished")


if __name__ == "__main__":
    # Allow override limit via environment or CLI by editing call
    main(limit=int(os.getenv("NDVI_WORKER_LIMIT", "10")))
