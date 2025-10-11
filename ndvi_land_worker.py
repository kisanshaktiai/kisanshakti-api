"""
NDVI Land Worker
----------------
Background processor for clipping and calculating NDVI for each tenant's lands
using globally shared satellite tiles stored in Backblaze B2.

Tables used:
- lands (tenant-scoped)
- satellite_tiles (public)
- ndvi_data
- ndvi_micro_tiles
- ndvi_request_queue
- ndvi_processing_logs
"""

import io
import os
import json
import datetime
import logging
import traceback
import numpy as np
import requests
import rasterio
from rasterio.mask import mask
from rasterio.io import MemoryFile
from rasterio.warp import transform_geom
from shapely.geometry import shape, mapping
from supabase import create_client

# === CONFIG ===
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
B2_BUCKET_URL = os.getenv("B2_BUCKET_URL")  # Example: https://f002.backblazeb2.com/file/your-bucket

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ndvi-worker")

# ---------------- Utility Functions ----------------
def download_tile_from_b2(tile_path: str) -> io.BytesIO:
    """Download GeoTIFF from B2 bucket"""
    if not tile_path:
        raise ValueError("Tile path not provided")
    url = f"{B2_BUCKET_URL.rstrip('/')}/{tile_path.lstrip('/')}"
    logger.info(f"Downloading tile: {url}")
    resp = requests.get(url, timeout=90)
    if resp.status_code != 200:
        raise Exception(f"Failed to download tile {url} (HTTP {resp.status_code})")
    return io.BytesIO(resp.content)

def open_raster_from_bytesio(buf: io.BytesIO):
    """Open a raster from in-memory bytes"""
    buf.seek(0)
    return MemoryFile(buf.read()).open()

def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    np.seterr(divide='ignore', invalid='ignore')
    denom = nir + red
    ndvi = np.where(denom == 0, np.nan, (nir - red) / denom)
    return np.clip(ndvi, -1, 1)

def calculate_statistics(arr: np.ndarray) -> dict:
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
        "total_pixels": int(total),
        "coverage": float(valid.size / total * 100)
    }

def generate_ndvi_visualization(ndvi: np.ndarray) -> bytes:
    import matplotlib.pyplot as plt
    ndvi_clean = np.nan_to_num(ndvi, nan=0.0)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.imshow(ndvi_clean, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def resolve_tile_tif_path(tile: dict) -> str:
    """Choose correct field for raster path"""
    for key in ["ndvi_path", "product_path", "combined_path", "nir_band_path", "b2_path"]:
        if tile.get(key):
            return tile[key]
    return None

# ---------------- Core Land Processing ----------------
def process_farmer_land(land: dict, tile: dict) -> bool:
    """Clip satellite tile with land polygon and calculate NDVI"""
    land_id = land["id"]
    tenant_id = land["tenant_id"]
    farmer_id = land.get("farmer_id")

    try:
        geom_raw = land.get("boundary_polygon_old")
        if not geom_raw:
            raise Exception("Missing boundary_polygon_old")

        geom_json = json.loads(geom_raw) if isinstance(geom_raw, str) else geom_raw
        land_geom = shape(geom_json)

        tif_path = resolve_tile_tif_path(tile)
        if not tif_path:
            raise Exception("No valid tile path")

        tif_bytes = download_tile_from_b2(tif_path)
        with open_raster_from_bytesio(tif_bytes) as src:
            geom_transformed = transform_geom("EPSG:4326", src.crs.to_string(), mapping(land_geom))
            clipped, transform = mask(src, [geom_transformed], crop=True, all_touched=True, nodata=np.nan)

            if src.count == 1:
                ndvi = clipped[0].astype(float)
            else:
                red = clipped[min(3, src.count - 1)]
                nir = clipped[min(7, src.count - 1)]
                ndvi = calculate_ndvi(red, nir)

        stats = calculate_statistics(ndvi)
        if stats["valid_pixels"] == 0:
            raise Exception("No valid NDVI pixels")

        # Visualization
        img_bytes = generate_ndvi_visualization(ndvi)
        acquisition = tile.get("acquisition_date") or datetime.date.today().isoformat()
        filename = f"{tenant_id}_{land_id}_{acquisition}.png"
        storage_path = f"ndvi-thumbnails/{filename}"

        # Upload image
        supabase.storage.from_("ndvi-thumbnails").upload(storage_path, img_bytes, {"content-type": "image/png", "upsert": "true"})
        public_url = f"{SUPABASE_URL}/storage/v1/object/public/ndvi-thumbnails/{storage_path}"

        # Store results
        supabase.table("ndvi_data").insert({
            "tenant_id": tenant_id,
            "farmer_id": farmer_id,
            "land_id": land_id,
            "date": acquisition,
            "ndvi_mean": stats["mean"],
            "stats": stats,
            "thumbnail_url": public_url,
            "tile_id": tile.get("tile_id"),
            "created_at": datetime.datetime.utcnow().isoformat()
        }).execute()

        supabase.table("lands").update({
            "last_ndvi_value": stats["mean"],
            "last_ndvi_calculation": acquisition,
            "updated_at": datetime.datetime.utcnow().isoformat()
        }).eq("id", land_id).execute()

        logger.info(f"✅ Land {land_id} processed: mean={stats['mean']:.4f}")
        return True

    except Exception as e:
        logger.error(f"❌ Land {land_id} failed: {e}")
        supabase.table("ndvi_processing_logs").insert({
            "tenant_id": tenant_id,
            "land_id": land_id,
            "tile_id": tile.get("tile_id"),
            "processing_step": "ndvi_calc",
            "step_status": "failed",
            "error_message": str(e),
            "completed_at": datetime.datetime.utcnow().isoformat()
        }).execute()
        return False

# ---------------- Queue Runner ----------------
def process_queue(limit: int = 10):
    """Process queued NDVI requests"""
    rq = supabase.table("ndvi_request_queue").select("*").eq("status", "queued").limit(limit).execute()
    requests = rq.data or []
    if not requests:
        logger.info("No queued NDVI requests")
        return

    for req in requests:
        req_id = req["id"]
        tenant_id = req["tenant_id"]
        tile_id = req["tile_id"]

        supabase.table("ndvi_request_queue").update({"status": "processing"}).eq("id", req_id).execute()
        lands = supabase.table("lands").select("*").eq("tenant_id", tenant_id).in_("id", req["land_ids"]).execute().data
        tile = supabase.table("satellite_tiles").select("*").eq("tile_id", tile_id).limit(1).execute().data[0]

        processed = 0
        for land in lands:
            if process_farmer_land(land, tile):
                processed += 1

        supabase.table("ndvi_request_queue").update({
            "status": "completed",
            "processed_count": processed,
            "completed_at": datetime.datetime.utcnow().isoformat()
        }).eq("id", req_id).execute()

# ---------------- Entry ----------------
def main(limit: int = 10):
    logger.info("🚀 NDVI Worker started")
    process_queue(limit)

if __name__ == "__main__":
    main()
