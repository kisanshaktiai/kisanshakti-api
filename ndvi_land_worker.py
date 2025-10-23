"""
NDVI Land Worker (v4.3.0-FIXED-B2-PATHS)
----------------------------------------
🌿 KisanShaktiAI NDVI Processor - Fixed B2 URL Resolution

✅ FIX: Now uses actual B2 paths from satellite_tiles table
✅ Converts b2:// protocol paths to HTTP URLs
✅ Validates file existence before processing
✅ Better error messages showing actual URLs tried

© 2025 KisanShaktiAI | Fixed by Claude
"""

import io, os, json, base64, datetime, logging, traceback
import numpy as np, requests, rasterio
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
SUPABASE_NDVI_BUCKET = os.getenv("SUPABASE_NDVI_BUCKET", "ndvi-thumbnails")
B2_BUCKET = "kisanshakti-ndvi-tiles"
B2_REGION_URL = "https://f005.backblazeb2.com/file"  # friendly URL base

if not SUPABASE_KEY:
    raise ValueError("❌ Missing Supabase credentials!")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ndvi-worker-v4.3.0")

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

def upload_thumbnail_to_supabase(path: str, png_bytes: bytes) -> str:
    supabase.storage.from_(SUPABASE_NDVI_BUCKET).upload(
        path, io.BytesIO(png_bytes), {"content-type": "image/png", "upsert": True}
    )
    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_NDVI_BUCKET}/{path}"

# ============================================================
# B2 PATH CONVERSION
# ============================================================
def convert_b2_path_to_url(b2_path: str) -> str:
    """
    Convert b2:// protocol path to HTTP URL
    
    Input: b2://kisanshakti-ndvi-tiles/tiles/raw/43QCU/2025-10-05/B04.tif
    Output: https://f005.backblazeb2.com/file/kisanshakti-ndvi-tiles/tiles/raw/43QCU/2025-10-05/B04.tif
    """
    if b2_path.startswith("b2://"):
        # Remove b2:// prefix and prepend HTTP base URL
        path = b2_path.replace("b2://", "")
        return f"{B2_REGION_URL}/{path}"
    elif b2_path.startswith("http"):
        # Already an HTTP URL
        return b2_path
    else:
        # Assume it's a relative path
        return f"{B2_REGION_URL}/{B2_BUCKET}/{b2_path}"

def check_b2_file_exists(url: str) -> bool:
    """Check if file exists and is accessible at B2 URL"""
    try:
        resp = requests.head(url, timeout=30)
        exists = resp.status_code == 200
        if not exists:
            logger.warning(f"⚠️ B2 file check failed (status {resp.status_code}): {url}")
        return exists
    except Exception as e:
        logger.warning(f"⚠️ B2 file check error: {e}")
        return False

# ============================================================
# TILE DATA RETRIEVAL
# ============================================================
def get_tile_data_for_date(tile_id: str, date: str = None) -> dict:
    """
    Get tile data from satellite_tiles table with actual B2 paths.
    If date is None, gets the latest available tile.
    """
    try:
        query = supabase.table("satellite_tiles") \
            .select("*") \
            .eq("tile_id", tile_id) \
            .eq("status", "ready")
        
        if date:
            query = query.eq("acquisition_date", date)
        else:
            query = query.order("acquisition_date", desc=True).limit(1)
        
        resp = query.execute()
        
        if resp.data and len(resp.data) > 0:
            tile_data = resp.data[0]
            logger.info(f"✅ Found tile metadata for {tile_id} on {tile_data['acquisition_date']}")
            return tile_data
        else:
            logger.error(f"❌ No tile data found for {tile_id}" + (f" on {date}" if date else ""))
            return None
            
    except Exception as e:
        logger.error(f"❌ Error querying tile data: {e}")
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
# MAIN PROCESSOR
# ============================================================
def process_farmer_land(land: dict, tile: dict = None) -> bool:
    land_id = land.get("id")
    tenant_id = land.get("tenant_id")
    try:
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

        # 🔥 FIX: Get tile data with actual B2 paths from database
        tile_data = get_tile_data_for_date(tile_id)
        if not tile_data:
            raise ValueError(f"No satellite data found for tile {tile_id}")
        
        date = tile_data["acquisition_date"]
        logger.info(f"🛰️ Processing tile {tile_id} for date {date}")

        # Convert B2 paths to HTTP URLs
        b04_url = convert_b2_path_to_url(tile_data.get("red_band_path", ""))
        b08_url = convert_b2_path_to_url(tile_data.get("nir_band_path", ""))
        ndvi_url = convert_b2_path_to_url(tile_data.get("ndvi_path", ""))

        # Log URLs being tried
        logger.info(f"🔍 Red band URL: {b04_url}")
        logger.info(f"🔍 NIR band URL: {b08_url}")
        logger.info(f"🔍 NDVI URL: {ndvi_url}")

        # Check which files exist
        b04_exists = check_b2_file_exists(b04_url)
        b08_exists = check_b2_file_exists(b08_url)
        ndvi_exists = check_b2_file_exists(ndvi_url)

        logger.info(f"📊 File availability: B04={b04_exists}, B08={b08_exists}, NDVI={ndvi_exists}")

        if b04_exists and b08_exists:
            logger.info("✅ Found raw bands, computing NDVI")
            with rasterio.open(b04_url) as red, rasterio.open(b08_url) as nir:
                geom_t = transform_geom("EPSG:4326", red.crs.to_string(), mapping(land_geom))
                arr_r, _ = mask(red, [geom_t], crop=True, all_touched=True, nodata=np.nan)
                arr_n, _ = mask(nir, [geom_t], crop=True, all_touched=True, nodata=np.nan)
                ndvi = calculate_ndvi(arr_r[0], arr_n[0])
        elif ndvi_exists:
            logger.info("🟢 Using precomputed NDVI file")
            with rasterio.open(ndvi_url) as src:
                geom_t = transform_geom("EPSG:4326", src.crs.to_string(), mapping(land_geom))
                arr, _ = mask(src, [geom_t], crop=True, all_touched=True, nodata=np.nan)
                ndvi = arr[0]
        else:
            # Provide detailed error message
            error_details = []
            if not b04_exists:
                error_details.append(f"B04 not found: {b04_url}")
            if not b08_exists:
                error_details.append(f"B08 not found: {b08_url}")
            if not ndvi_exists:
                error_details.append(f"NDVI not found: {ndvi_url}")
            
            error_msg = "No valid NDVI or raw bands accessible. " + " | ".join(error_details)
            raise FileNotFoundError(error_msg)

        # Compute stats and save image
        stats = calculate_statistics(ndvi)
        if stats["valid_pixels"] == 0:
            raise ValueError("No valid NDVI pixels after masking")

        png_bytes = create_colorized_ndvi_png(ndvi)
        path = f"{tenant_id}/{land_id}/{date}/vegetation_map.png"
        image_url = upload_thumbnail_to_supabase(path, png_bytes)
        now = datetime.datetime.utcnow().isoformat()

        # Save to ndvi_data
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
            "valid_pixels": stats["valid_pixels"],
            "total_pixels": stats["total_pixels"],
            "coverage_percentage": stats["coverage"],
            "image_url": image_url,
            "updated_at": now
        }, on_conflict="land_id,date").execute()

        # Save to ndvi_micro_tiles
        supabase.table("ndvi_micro_tiles").upsert({
            "tenant_id": tenant_id,
            "land_id": land_id,
            "acquisition_date": date,
            "cloud_cover": tile_data.get("cloud_cover", 0),
            "ndvi_mean": stats["mean"],
            "ndvi_min": stats["min"],
            "ndvi_max": stats["max"],
            "ndvi_std_dev": stats["std"],
            "ndvi_thumbnail_url": image_url,
            "resolution_meters": 10,
            "bbox": mapping(land_geom),
            "created_at": now
        }, on_conflict="land_id,acquisition_date").execute()

        logger.info(f"✅ Land {land_id} NDVI done (mean={stats['mean']:.3f})")
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

        supabase.table("ndvi_request_queue").update({
            "status": "completed" if failed == 0 else "failed",
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
    logger.info(f"🚀 NDVI Worker v4.3.0 starting | limit={limit}, threads={workers}")
    process_queue(limit=limit, max_workers=workers)
    logger.info("🏁 NDVI Worker finished")

if __name__ == "__main__":
    main()
