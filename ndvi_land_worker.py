"""
NDVI Land Worker v5.1.0 — Unified Table Fix
--------------------------------------------
✅ Writes all NDVI + thumbnails to ndvi_micro_tiles
✅ Compatible with NDVI API v4.1.0
✅ Auto-detects correct tiles per land
✅ Validates overlap before processing
✅ Backblaze B2 + Supabase unified output

© 2025 KisanShaktiAI
"""

import os, io, json, datetime, logging, traceback, functools
import numpy as np, requests, rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
from shapely.geometry import shape, mapping, box
from PIL import Image
import matplotlib.cm as cm
from supabase import create_client

# ============ CONFIG ============
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
B2_APP_KEY_ID = os.environ.get("B2_KEY_ID")
B2_APP_KEY = os.environ.get("B2_APP_KEY")
B2_BUCKET_NAME = os.environ.get("B2_BUCKET_RAW", "kisanshakti-ndvi-tiles")
B2_PREFIX = os.environ.get("B2_PREFIX", "tiles/")
SUPABASE_NDVI_BUCKET = os.environ.get("SUPABASE_NDVI_BUCKET", "ndvi-thumbnails")

if not all([SUPABASE_URL, SUPABASE_KEY, B2_APP_KEY_ID, B2_APP_KEY]):
    raise RuntimeError("❌ Missing environment variables")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ndvi-worker-v5.1")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============ B2 AUTH ============
@functools.lru_cache(maxsize=1)
def b2_auth():
    logger.info("🔑 Authorizing B2...")
    res = requests.get(
        "https://api.backblazeb2.com/b2api/v2/b2_authorize_account",
        auth=requests.auth.HTTPBasicAuth(B2_APP_KEY_ID, B2_APP_KEY),
        timeout=30,
    )
    res.raise_for_status()
    data = res.json()
    logger.info("✅ B2 authorized")
    return {"token": data["authorizationToken"], "url": data["downloadUrl"]}


def b2_download(path: str) -> io.BytesIO:
    data = b2_auth()
    url = f"{data['url']}/file/{B2_BUCKET_NAME}/{path}"
    res = requests.get(url, headers={"Authorization": data["token"]}, timeout=120)
    if res.status_code == 200:
        buf = io.BytesIO(res.content)
        buf.seek(0)
        logger.info(f"✅ Downloaded {len(res.content)/1024/1024:.2f}MB from B2")
        return buf
    raise FileNotFoundError(f"❌ File not found: {path}")


# ============ UTILITIES ============
def calculate_ndvi(red, nir):
    np.seterr(divide="ignore", invalid="ignore")
    ndvi = (nir - red) / (nir + red)
    return np.clip(ndvi, -1, 1)


def calculate_stats(arr):
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


def ndvi_to_png(ndvi, cmap="RdYlGn"):
    norm = np.clip((ndvi + 1) / 2, 0, 1)
    rgba = (cm.get_cmap(cmap)(norm) * 255).astype(np.uint8)
    rgba[..., 3][np.isnan(ndvi)] = 0
    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    buf.seek(0)
    return buf.getvalue()


def upload_thumbnail(tenant_id, land_id, date, png_bytes):
    path = f"{tenant_id}/{land_id}/{date}/ndvi.png"
    supabase.storage.from_(SUPABASE_NDVI_BUCKET).upload(
        path, io.BytesIO(png_bytes), {"content-type": "image/png", "upsert": "true"}
    )
    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_NDVI_BUCKET}/{path}"


# ============ CORE NDVI PROCESSOR ============
def process_land_ndvi_thumbnail(land, tile):
    land_id, tenant_id = land["id"], land["tenant_id"]
    name = land.get("name", "Unknown")

    try:
        geom = land.get("boundary_polygon_old") or land.get("boundary_polygon")
        if not geom:
            raise ValueError("Missing boundary polygon")

        land_geom = shape(geom if isinstance(geom, dict) else json.loads(geom))
        acq_date = tile["acquisition_date"]
        tile_id = tile["tile_id"]

        logger.info(f"🌾 Processing {name} ({land_id[:8]}) for tile {tile_id}")

        # Download NDVI raster
        ndvi_path = f"{B2_PREFIX}ndvi/{tile_id}/{acq_date}/ndvi.tif"
        buf = b2_download(ndvi_path)

        with rasterio.open(buf) as src:
            geom_t = transform_geom("EPSG:4326", src.crs.to_string(), mapping(land_geom))
            rb = box(*src.bounds)
            lb = shape(geom_t)

            if not rb.intersects(lb):
                raise ValueError(f"Land does not overlap raster: {tile_id} ({acq_date})")

            arr, _ = mask(src, [geom_t], crop=True, all_touched=True, nodata=np.nan)
            ndvi = arr[0]

        stats = calculate_stats(ndvi)
        if not stats:
            raise ValueError("No valid NDVI pixels")

        png = ndvi_to_png(ndvi)
        img_url = upload_thumbnail(tenant_id, land_id, acq_date, png)

        now = datetime.datetime.utcnow().isoformat()
        supabase.table("ndvi_micro_tiles").upsert(
            {
                "tenant_id": tenant_id,
                "land_id": land_id,
                "acquisition_date": acq_date,
                "tile_id": tile_id,
                "ndvi_mean": stats["mean"],
                "ndvi_min": stats["min"],
                "ndvi_max": stats["max"],
                "ndvi_std_dev": stats["std"],
                "ndvi_thumbnail_url": img_url,
                "thumbnail_size_kb": round(len(png) / 1024, 2),
                "cloud_cover": tile.get("cloud_cover"),
                "resolution_meters": tile.get("resolution_meters"),
                "bbox": tile.get("bbox"),
                "updated_at": now,
            },
            on_conflict="land_id,acquisition_date",
        ).execute()

        supabase.table("lands").update(
            {
                "last_ndvi_value": round(stats["mean"], 3),
                "last_ndvi_calculation": acq_date,
                "ndvi_tested": True,
                "ndvi_thumbnail_url": img_url,
                "last_processed_at": now,
            }
        ).eq("id", land_id).execute()

        logger.info(f"✅ NDVI completed for {name}")
        return True

    except Exception as e:
        logger.error(f"❌ {name}: {e}")
        supabase.table("ndvi_processing_logs").insert(
            {
                "tenant_id": tenant_id,
                "land_id": land_id,
                "processing_step": "ndvi_extraction",
                "step_status": "failed",
                "completed_at": datetime.datetime.utcnow().isoformat(),
                "error_message": str(e)[:400],
            }
        ).execute()
        return False
