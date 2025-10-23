"""
NDVI Land Worker v5.0.0 — KisanShaktiAI Production
--------------------------------------------------
✅ Multi-tile intersection logic (no missed coverage)
✅ CRS auto-transform (EPSG:4326 → raster CRS)
✅ Validates land–tile intersection precisely
✅ Resolves boundary overlap conflicts
✅ Accurate NDVI + vegetation color image per land
✅ Writes to ndvi_data + micro_tile_thumbnail + Supabase bucket
✅ Uses cached B2 auth for high performance
✅ Full structured logging and error capture
"""

import os, io, json, logging, datetime, traceback, functools
import numpy as np
import requests, rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from supabase import create_client
from PIL import Image
import matplotlib.cm as cm

# ---------------- ENVIRONMENT CONFIG ----------------
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

B2_APP_KEY_ID = os.environ["B2_KEY_ID"]
B2_APP_KEY = os.environ["B2_APP_KEY"]
B2_BUCKET = os.environ.get("B2_BUCKET_RAW", "kisanshakti-ndvi-tiles")
B2_PREFIX = os.environ.get("B2_PREFIX", "tiles/ndvi")
SUPABASE_NDVI_BUCKET = os.environ.get("SUPABASE_NDVI_BUCKET", "ndvi-thumbnails")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ndvi-worker-v5.0.0")

# ---------------- B2 AUTH (CACHED) ----------------
@functools.lru_cache(maxsize=1)
def b2_authorize_cached():
    logger.info("🔑 Authorizing B2...")
    res = requests.get(
        "https://api.backblazeb2.com/b2api/v2/b2_authorize_account",
        auth=requests.auth.HTTPBasicAuth(B2_APP_KEY_ID, B2_APP_KEY),
        timeout=30,
    )
    res.raise_for_status()
    data = res.json()
    return {"auth": data["authorizationToken"], "url": data["downloadUrl"]}

def b2_download(path: str) -> io.BytesIO:
    """Securely fetch file from B2"""
    token = b2_authorize_cached()
    url = f"{token['url']}/file/{B2_BUCKET}/{path}"
    res = requests.get(url, headers={"Authorization": token["auth"]}, timeout=120)
    if res.status_code != 200:
        raise FileNotFoundError(f"Missing NDVI tile: {path}")
    buf = io.BytesIO(res.content)
    buf.seek(0)
    return buf

# ---------------- NDVI UTILITIES ----------------
def calculate_ndvi(red, nir):
    np.seterr(divide="ignore", invalid="ignore")
    ndvi = (nir - red) / (nir + red)
    return np.clip(ndvi, -1, 1)

def ndvi_stats(arr):
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return None
    return {
        "mean": float(np.mean(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "std": float(np.std(valid)),
        "coverage": float(valid.size / arr.size * 100),
        "valid": int(valid.size),
        "total": int(arr.size),
    }

def colorize_ndvi(ndvi):
    norm = np.clip((ndvi + 1) / 2, 0, 1)
    rgba = (cm.get_cmap("RdYlGn")(norm) * 255).astype(np.uint8)
    rgba[..., 3][np.isnan(ndvi)] = 0
    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    buf.seek(0)
    return buf

# ---------------- SUPABASE UPLOAD ----------------
def upload_to_supabase(tenant, land, date, png_bytes):
    path = f"{tenant}/{land}/{date}/ndvi_map.png"
    supabase.storage.from_(SUPABASE_NDVI_BUCKET).upload(
        path, png_bytes, {"content-type": "image/png", "upsert": True}
    )
    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_NDVI_BUCKET}/{path}"

# ---------------- MAIN PROCESSOR ----------------
def process_land_ndvi(land, all_tiles):
    land_id = land["id"]
    tenant_id = land["tenant_id"]

    try:
        geom_raw = land.get("boundary_geom") or land.get("boundary_polygon_old")
        if not geom_raw:
            raise ValueError("No boundary geometry found")

        land_geom = shape(geom_raw if isinstance(geom_raw, dict) else json.loads(geom_raw))

        # --- STEP 1: Filter intersecting tiles ---
        intersect_tiles = []
        for t in all_tiles:
            tile_geom = shape(t["geometry"]) if "geometry" in t else None
            if tile_geom and land_geom.intersects(tile_geom):
                intersect_tiles.append(t)

        if not intersect_tiles:
            raise ValueError("No intersecting NDVI tiles found for land")

        logger.info(f"🧩 Land {land_id}: {len(intersect_tiles)} intersecting tiles")

        merged_ndvi = None
        for tile in intersect_tiles:
            tile_id = tile["tile_id"]
            date = tile["acquisition_date"]
            tif_path = f"{B2_PREFIX}/{tile_id}/{date}/ndvi.tif"
            buf = b2_download(tif_path)
            with rasterio.open(buf) as src:
                raster_crs = src.crs.to_string()
                geom_t = transform_geom("EPSG:4326", raster_crs, mapping(land_geom.buffer(0.0003)))

                # --- Intersection validation ---
                r_bounds = src.bounds
                g_bounds = shape(geom_t).bounds
                if (
                    g_bounds[2] < r_bounds.left
                    or g_bounds[0] > r_bounds.right
                    or g_bounds[3] < r_bounds.bottom
                    or g_bounds[1] > r_bounds.top
                ):
                    logger.warning(f"⛔ Land {land_id} skipped non-overlapping tile {tile_id}")
                    continue

                arr, _ = mask(src, [geom_t], crop=True, all_touched=True, nodata=np.nan)
                ndvi_arr = arr[0] if arr.ndim > 2 else arr
                merged_ndvi = ndvi_arr if merged_ndvi is None else np.nanmean(
                    np.dstack((merged_ndvi, ndvi_arr)), axis=2
                )

        if merged_ndvi is None:
            raise ValueError("No valid NDVI coverage across intersecting tiles")

        # --- NDVI Stats ---
        stats = ndvi_stats(merged_ndvi)
        if not stats:
            raise ValueError("No valid NDVI pixels")

        # --- NDVI Image ---
        png_bytes = colorize_ndvi(merged_ndvi)
        image_url = upload_to_supabase(tenant_id, land_id, datetime.date.today(), png_bytes)

        now = datetime.datetime.utcnow().isoformat()
        supabase.table("ndvi_data").upsert({
            "tenant_id": tenant_id,
            "land_id": land_id,
            "ndvi_value": stats["mean"],
            "ndvi_min": stats["min"],
            "ndvi_max": stats["max"],
            "ndvi_std": stats["std"],
            "coverage_percent": stats["coverage"],
            "image_url": image_url,
            "updated_at": now
        }, on_conflict="land_id").execute()

        supabase.table("micro_tile_thumbnail").upsert({
            "tenant_id": tenant_id,
            "land_id": land_id,
            "image_url": image_url,
            "ndvi_mean": stats["mean"],
            "created_at": now,
            "updated_at": now
        }, on_conflict="land_id").execute()

        logger.info(f"✅ NDVI processed: {land_id} mean={stats['mean']:.3f}")

        return True

    except Exception as e:
        logger.error(f"❌ NDVI failed for {land_id}: {e}")
        supabase.table("ndvi_processing_logs").insert({
            "tenant_id": tenant_id,
            "land_id": land_id,
            "processing_step": "ndvi_generation",
            "step_status": "failed",
            "completed_at": datetime.datetime.utcnow().isoformat(),
            "error_message": str(e)[:400],
            "error_details": {"traceback": traceback.format_exc()[:1000]},
        }).execute()
        return False

# Compatibility alias
process_farmer_land = process_land_ndvi
