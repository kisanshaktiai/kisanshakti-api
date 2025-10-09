# farmer_land_ndvi_worker.py

import os
import io
import logging
import datetime
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import shape, mapping
from supabase import create_client
import requests
from PIL import Image

# === CONFIG ===
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
B2_BUCKET_URL = os.getenv("B2_BUCKET_URL")  # e.g. https://f123.backblazeb2.com/file/ndvi-tiles

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
logging.basicConfig(level=logging.INFO)


# ---------- Utility Functions ----------
def download_tile_from_b2(tile_path: str) -> io.BytesIO:
    """Download GeoTIFF tile from Backblaze B2 bucket"""
    url = f"{B2_BUCKET_URL}/{tile_path}"
    resp = requests.get(url)
    if resp.status_code != 200:
        raise Exception(f"Failed to download tile: {url}")
    return io.BytesIO(resp.content)


def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - RED) / (NIR + RED)"""
    np.seterr(divide='ignore', invalid='ignore')
    return (nir.astype(float) - red.astype(float)) / (nir + red)


def generate_thumbnail(ndvi_array: np.ndarray, output_size=(256, 256)) -> (bytes, int):
    """Generate thumbnail image from NDVI numpy array"""
    ndvi_norm = ((ndvi_array - np.nanmin(ndvi_array)) /
                 (np.nanmax(ndvi_array) - np.nanmin(ndvi_array) + 1e-6) * 255).astype(np.uint8)
    img = Image.fromarray(ndvi_norm).convert("L")
    img.thumbnail(output_size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    size_kb = len(buf.getvalue()) / 1024
    return buf.getvalue(), size_kb


# ---------- Core Land Processor ----------
def process_farmer_land(land: dict, tile: dict) -> bool:
    """Process NDVI for a single farmer land with a specific tile"""
    try:
        land_id = land["id"]
        farmer_id = land.get("farmer_id")
        tenant_id = land.get("tenant_id")
        land_geom = shape(land["boundary"])  # GeoJSON polygon

        # Download tile
        tif_data = download_tile_from_b2(tile["b2_path"])

        # Clip raster
        with rasterio.open(tif_data) as src:
            clipped, transform = mask(src, [mapping(land_geom)], crop=True)
            red = clipped[2]  # assuming band 3 = red
            nir = clipped[3]  # assuming band 4 = nir

        # NDVI calculation
        ndvi = calculate_ndvi(red, nir)
        ndvi_mean, ndvi_min, ndvi_max, ndvi_std = (
            float(np.nanmean(ndvi)),
            float(np.nanmin(ndvi)),
            float(np.nanmax(ndvi)),
            float(np.nanstd(ndvi))
        )

        # Thumbnail
        thumb_bytes, thumb_size = generate_thumbnail(ndvi)
        thumb_path = f"ndvi-thumbnails/{land_id}_{tile['acquisition_date']}.png"
        supabase.storage.from_("ndvi-thumbnails").upload(thumb_path, thumb_bytes, {"content-type": "image/png"})
        thumb_url = f"{SUPABASE_URL}/storage/v1/object/public/ndvi-thumbnails/{thumb_path}"

        # Save record in ndvi_micro_tiles
        supabase.table("ndvi_micro_tiles").insert({
            "land_id": land_id,
            "farmer_id": farmer_id,
            "tenant_id": tenant_id,
            "bbox": land["boundary"],
            "acquisition_date": tile["acquisition_date"],
            "cloud_cover": tile.get("cloud_cover"),
            "ndvi_mean": ndvi_mean,
            "ndvi_min": ndvi_min,
            "ndvi_max": ndvi_max,
            "ndvi_std_dev": ndvi_std,
            "ndvi_thumbnail_url": thumb_url,
            "thumbnail_size_kb": thumb_size,
            "resolution_meters": tile.get("resolution", 10)
        }).execute()

        # Update lands table
        supabase.table("lands").update({
            "last_ndvi_calculation": datetime.date.today().isoformat(),
            "last_ndvi_value": ndvi_mean,
            "ndvi_thumbnail_url": thumb_url
        }).eq("id", land_id).execute()

        # Log success
        supabase.table("ndvi_processing_logs").insert({
            "satellite_tile_id": tile["id"],
            "processing_step": "ndvi_calculation",
            "step_status": "success",
            "metadata": {"land_id": land_id, "ndvi_mean": ndvi_mean}
        }).execute()

        logging.info(f"✅ NDVI processed for land {land_id}")
        return True

    except Exception as e:
        logging.error(f"❌ Failed processing land {land.get('id')}: {e}")
        supabase.table("ndvi_processing_logs").insert({
            "satellite_tile_id": tile["id"] if tile else None,
            "processing_step": "ndvi_calculation",
            "step_status": "failed",
            "error_message": str(e),
            "metadata": {"land_id": land.get("id")}
        }).execute()
        return False


# ---------- Batch Processor ----------
def process_queue(limit: int = 10):
    """Process requests from ndvi_request_queue"""
    requests_q = supabase.table("ndvi_request_queue") \
        .select("*") \
        .eq("status", "queued") \
        .order("priority", desc=False) \
        .limit(limit) \
        .execute()

    for req in requests_q.data:
        req_id = req["id"]
        land_ids = req["land_ids"]

        # Mark request as processing
        supabase.table("ndvi_request_queue").update({
            "status": "processing",
            "started_at": datetime.datetime.utcnow().isoformat()
        }).eq("id", req_id).execute()

        processed_count = 0

        # Fetch lands
        lands = supabase.table("lands").select("*").in_("id", land_ids).execute()

        # Fetch tiles for request date range
        tiles = supabase.table("satellite_tiles").select("*") \
            .gte("acquisition_date", req["date_from"]) \
            .lte("acquisition_date", req["date_to"]) \
            .eq("tile_id", req["tile_id"]) \
            .execute()

        if not tiles.data:
            logging.warning(f"No tiles found for request {req_id}")
            supabase.table("ndvi_request_queue").update({
                "status": "failed",
                "error_message": "No tiles found for given range"
            }).eq("id", req_id).execute()
            continue

        latest_tile = sorted(tiles.data, key=lambda t: t["acquisition_date"], reverse=True)[0]

        for land in lands.data:
            if process_farmer_land(land, latest_tile):
                processed_count += 1

        # Mark request completed
        supabase.table("ndvi_request_queue").update({
            "status": "completed",
            "completed_at": datetime.datetime.utcnow().isoformat(),
            "processed_count": processed_count
        }).eq("id", req_id).execute()

        logging.info(f"🎯 Request {req_id} completed: {processed_count}/{len(land_ids)} lands")


def main(limit: int = None, use_queue: bool = True):
    """Main entrypoint"""
    if use_queue:
        logging.info("🔄 Processing NDVI request queue...")
        process_queue(limit or 5)
    else:
        logging.info("🔄 Processing all lands directly...")
        lands = supabase.table("lands").select("*").limit(limit or 100).execute()
        count = 0
        for land in lands.data:
            # Pick latest tile automatically
            tiles = supabase.table("satellite_tiles").select("*").order("acquisition_date", desc=True).limit(1).execute()
            if tiles.data:
                if process_farmer_land(land, tiles.data[0]):
                    count += 1
        logging.info(f"Processed {count}/{len(lands.data)} lands")
        return count
