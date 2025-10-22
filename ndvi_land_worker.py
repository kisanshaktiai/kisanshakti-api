"""
NDVI Land Worker (v3.9.3-STABLE)
--------------------------------
🚀 Production-grade NDVI processor for KisanShaktiAI
- Fixed B2 path layout (tiles/raw & tiles/ndvi)
- Auto-detects satellite tile intersections for each land
- Handles invalid or missing geometries gracefully
- Integrates fully with NDVI Land API v3.9.1-FIXED
- Safely writes NDVI data, micro-tiles, and logs errors

© 2025 KisanShaktiAI | Amarsinh Patil
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
from shapely.wkt import loads as wkt_loads
from supabase import create_client
from PIL import Image
import matplotlib.cm as cm

# ============================================================
# CONFIGURATION
# ============================================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

B2_KEY_ID = os.getenv("B2_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
B2_BUCKET = os.getenv("B2_BUCKET", "kisanshakti-ndvi-tiles")
B2_PREFIX = os.getenv("B2_PREFIX", "tiles")
SUPABASE_NDVI_BUCKET = os.getenv("SUPABASE_NDVI_BUCKET", "ndvi-thumbnails")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ Missing Supabase credentials (SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY)")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ndvi-worker-v3.9.3")

# ============================================================
# BACKBLAZE B2 HELPERS (Final - matches actual cloud layout)
# ============================================================
def get_b2_auth() -> dict:
    """Authorize Backblaze B2 account using application key."""
    if not all([B2_KEY_ID, B2_APP_KEY, B2_BUCKET]):
        raise ValueError("❌ Missing B2 credentials")

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
    Correctly construct B2 path:
      kisanshakti-ndvi-tiles/tiles/raw/{tile_id}/{date}/{filename}
      kisanshakti-ndvi-tiles/tiles/ndvi/{tile_id}/{date}/{filename}
    """
    subdir = subdir.strip("/")
    return f"tiles/{subdir}/{tile_id}/{date}/{filename}"


def download_b2_file(tile_id: str, date: str, subdir: str, filename: str) -> Optional[io.BytesIO]:
    """Download file from Backblaze B2."""
    try:
        b2 = get_b2_auth()
        path = build_b2_path(tile_id, date, subdir, filename)
        url = f"{b2['download_url'].rstrip('/')}/file/{b2['bucket_name']}/{path}"
        headers = {"Authorization": b2["auth_token"]}

        resp = requests.get(url, headers=headers, timeout=120)
        if resp.status_code == 200:
            logger.info(f"✅ B2 file found: {path}")
            return io.BytesIO(resp.content)
        elif resp.status_code == 404:
            logger.warning(f"⚠️ B2 file not found: {path}")
            return None
        else:
            logger.error(f"❌ B2 download error {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as e:
        logger.error(f"❌ B2 download exception for {tile_id}/{filename}: {e}")
        return None

# ============================================================
# NDVI MATH & VISUALIZATION
# ============================================================
def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute NDVI = (NIR - RED) / (NIR + RED)."""
    np.seterr(divide="ignore", invalid="ignore")
    denom = nir + red
    ndvi = np.where(denom == 0, np.nan, (nir - red) / denom)
    return np.clip(ndvi, -1.0, 1.0)


def calculate_statistics(arr: np.ndarray) -> dict:
    """Compute NDVI summary statistics."""
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
    """Convert NDVI array into colorized PNG with transparent nodata."""
    ndvi_normalized = np.clip((ndvi_array + 1.0) / 2.0, 0.0, 1.0)
    cmap = cm.get_cmap(cmap_name)
    rgba = (cmap(ndvi_normalized) * 255).astype(np.uint8)
    rgba[..., 3][np.isnan(ndvi_array)] = 0  # transparent for NaN
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.getvalue()


def upload_thumbnail_to_supabase(land_id: str, date: str, png_bytes: bytes) -> Optional[str]:
    """Upload NDVI PNG to Supabase public bucket."""
    path = f"{land_id}/{date}/ndvi_color.png"
    try:
        file_obj = io.BytesIO(png_bytes)
        supabase.storage.from_(SUPABASE_NDVI_BUCKET).upload(path, file_obj, {"content-type": "image/png", "upsert": True})
        public_url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/public/{SUPABASE_NDVI_BUCKET}/{path}"
        logger.info(f"🖼️ Uploaded NDVI thumbnail: {public_url}")
        return public_url
    except Exception as e:
        logger.error(f"❌ Failed to upload thumbnail: {e}")
        return None

# ============================================================
# TILE DETECTION HELPER
# ============================================================
def find_intersecting_tile(land_geom, land):
    """Find satellite tile that intersects with given land polygon."""
    try:
        state = land.get("state")
        district = land.get("district")
        query = supabase.table("satellite_tiles").select("tile_id,geometry,state,district").limit(500)
        if state:
            query = query.eq("state", state)
            if district:
                query = query.eq("district", district)
        tiles = query.execute().data or []

        for t in tiles:
            geom = t.get("geometry")
            try:
                geom_shape = shape(geom if isinstance(geom, dict) else json.loads(geom))
                if geom_shape.intersects(land_geom):
                    logger.info(f"🛰️ Tile {t['tile_id']} intersects land {land['id']}")
                    return t
            except Exception:
                continue
        return None
    except Exception as e:
        logger.error(f"❌ Tile lookup failed: {e}")
        return None

# ============================================================
# MAIN LAND PROCESSOR
# ============================================================
def process_farmer_land(land: dict, tile: Optional[dict] = None) -> bool:
    """Process NDVI for a single land."""
    land_id = land.get("id")
    tenant_id = land.get("tenant_id")
    ndvi_array = None
    stats = None

    try:
        logger.info(f"🌿 [START] Land {land_id} NDVI processing")

        # Step 1: Parse geometry
        raw_geom = land.get("boundary_polygon_old") or land.get("boundary_polygon") or land.get("boundary")
        if not raw_geom:
            raise ValueError("Missing land boundary")

        if isinstance(raw_geom, str):
            try:
                geom_json = json.loads(raw_geom)
            except json.JSONDecodeError:
                geom_json = json.loads(json.dumps(raw_geom))
        elif isinstance(raw_geom, dict):
            geom_json = raw_geom
        else:
            raise ValueError(f"Invalid geometry type: {type(raw_geom)}")

        land_geom = shape(geom_json)
        if not land_geom.is_valid:
            land_geom = land_geom.buffer(0)

        # Step 2: Determine satellite tile
        if not tile or not tile.get("tile_id"):
            tile = find_intersecting_tile(land_geom, land)
            if not tile:
                raise ValueError("Unable to determine satellite tile for this land.")

        tile_id = tile["tile_id"]
        date = tile.get("acquisition_date") or datetime.date.today().isoformat()
        logger.info(f"🛰️ Using tile {tile_id} ({date}) for land {land_id}")

        # Step 3: Fetch NDVI or compute from raw bands
        ndvi_buf = download_b2_file(tile_id, date, "ndvi", "ndvi.tif")
        if ndvi_buf:
            try:
                with rasterio.io.MemoryFile(ndvi_buf.read()) as memfile:
                    with memfile.open() as src:
                        geom_t = transform_geom("EPSG:4326", src.crs.to_string(), mapping(land_geom))
                        arr, _ = mask(src, [geom_t], crop=True, all_touched=True, nodata=np.nan)
                        ndvi_array = arr[0].astype(float)
            except Exception as e:
                logger.warning(f"⚠️ Precomputed NDVI failed: {e}")
                ndvi_array = None

        if ndvi_array is None:
            b04_buf = download_b2_file(tile_id, date, "raw", "B04.tif")
            b08_buf = download_b2_file(tile_id, date, "raw", "B08.tif")
            if not b04_buf or not b08_buf:
                raise FileNotFoundError(f"Raw bands missing for {tile_id}")
            with rasterio.io.MemoryFile(b04_buf.read()) as red_mem, rasterio.io.MemoryFile(b08_buf.read()) as nir_mem:
                with red_mem.open() as red, nir_mem.open() as nir:
                    geom_t = transform_geom("EPSG:4326", red.crs.to_string(), mapping(land_geom))
                    r_clip, _ = mask(red, [geom_t], crop=True, all_touched=True, nodata=np.nan)
                    n_clip, _ = mask(nir, [geom_t], crop=True, all_touched=True, nodata=np.nan)
                    ndvi_array = calculate_ndvi(r_clip[0], n_clip[0])

        # Step 4: Compute stats & save
        stats = calculate_statistics(ndvi_array)
        if stats["valid_pixels"] == 0:
            raise ValueError("No valid NDVI pixels after masking")

        png_bytes = create_colorized_ndvi_png(ndvi_array)
        image_url = upload_thumbnail_to_supabase(land_id, date, png_bytes)
        now = datetime.datetime.utcnow().isoformat()

        ndvi_record = {
            "tenant_id": tenant_id,
            "land_id": land_id,
            "tile_id": tile_id,
            "date": date,
            "ndvi_value": stats["mean"],
            "ndvi_min": stats["min"],
            "ndvi_max": stats["max"],
            "ndvi_std": stats["std"],
            "coverage_percentage": stats["coverage"],
            "image_url": image_url,
            "metadata": stats,
            "created_at": now,
            "updated_at": now,
        }

        supabase.table("ndvi_data").upsert(ndvi_record, on_conflict="land_id,date").execute()

        supabase.table("ndvi_micro_tiles").upsert({
            "tenant_id": tenant_id,
            "land_id": land_id,
            "acquisition_date": date,
            "ndvi_mean": stats["mean"],
            "ndvi_min": stats["min"],
            "ndvi_max": stats["max"],
            "ndvi_std_dev": stats["std"],
            "ndvi_thumbnail_url": image_url,
            "cloud_cover": 0,
            "created_at": now,
        }, on_conflict="land_id,acquisition_date").execute()

        supabase.table("lands").update({
            "last_ndvi_value": stats["mean"],
            "last_ndvi_calculation": date,
            "ndvi_thumbnail_url": image_url,
            "ndvi_tested": True,
            "last_processed_at": now,
        }).eq("id", land_id).execute()

        logger.info(f"✅ Land {land_id} processed successfully (mean={stats['mean']:.3f})")
        return True

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"❌ [FAILED] Land {land_id}: {e}")
        try:
            supabase.table("ndvi_processing_logs").insert({
                "tenant_id": tenant_id,
                "land_id": land_id,
                "processing_step": "ndvi_calculation",
                "step_status": "failed",
                "completed_at": datetime.datetime.utcnow().isoformat(),
                "error_message": str(e)[:500],
                "error_details": {"traceback": tb[:1000]},
            }).execute()
        except Exception as log_err:
            logger.warning(f"⚠️ Failed to write log: {log_err}")
        return False
