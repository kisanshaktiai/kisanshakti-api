import os
import io
import logging
import datetime
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
from shapely.geometry import shape, mapping
from supabase import create_client
import requests
import json
import matplotlib.pyplot as plt

# === CONFIG ===
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
B2_BUCKET_URL = os.getenv("B2_BUCKET_URL")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
logging.basicConfig(level=logging.INFO)


# ---------- Utility Functions ----------
def download_tile_from_b2(tile_path: str) -> io.BytesIO:
    """Download GeoTIFF tile from Backblaze B2 bucket"""
    url = f"{B2_BUCKET_URL}/{tile_path}"
    logging.info(f"Downloading tile from: {url}")
    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        raise Exception(f"Failed to download tile: {url} (Status: {resp.status_code})")
    return io.BytesIO(resp.content)


def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - RED) / (NIR + RED), range [-1, 1]"""
    np.seterr(divide='ignore', invalid='ignore')
    denominator = nir.astype(float) + red.astype(float)
    ndvi = np.where(denominator == 0, 0, (nir - red) / denominator)
    return np.clip(ndvi, -1, 1)


def calculate_vegetation_indices(red: np.ndarray, nir: np.ndarray, blue: np.ndarray = None) -> dict:
    """Calculate multiple vegetation indices"""
    indices = {"ndvi": calculate_ndvi(red, nir)}

    # EVI (Enhanced Vegetation Index)
    if blue is not None:
        np.seterr(divide='ignore', invalid='ignore')
        evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
        indices["evi"] = np.clip(evi, -1, 1)

    # SAVI (Soil Adjusted Vegetation Index)
    L = 0.5
    savi = ((nir - red) / (nir + red + L)) * (1 + L)
    indices["savi"] = np.clip(savi, -1, 1)

    return indices


def generate_ndvi_visualization(ndvi_array: np.ndarray, output_size=(512, 512)) -> bytes:
    """
    Generate NDVI visualization using matplotlib instead of Pillow
    Color map: RdYlGn (-1=red, 0=yellow, 1=green)
    """
    ndvi_clean = np.nan_to_num(ndvi_array, nan=0.0)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    cax = ax.imshow(ndvi_clean, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.axis("off")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close(fig)
    buf.seek(0)

    return buf.getvalue()


def calculate_statistics(array: np.ndarray) -> dict:
    """Calculate statistics for vegetation indices"""
    valid_data = array[~np.isnan(array)]

    if len(valid_data) == 0:
        return {
            "mean": None,
            "min": None,
            "max": None,
            "std": None,
            "median": None,
            "q25": None,
            "q75": None,
            "valid_pixels": 0,
            "total_pixels": array.size,
            "coverage_percentage": 0.0,
        }

    return {
        "mean": float(np.mean(valid_data)),
        "min": float(np.min(valid_data)),
        "max": float(np.max(valid_data)),
        "std": float(np.std(valid_data)),
        "median": float(np.median(valid_data)),
        "q25": float(np.percentile(valid_data, 25)),
        "q75": float(np.percentile(valid_data, 75)),
        "valid_pixels": int(len(valid_data)),
        "total_pixels": int(array.size),
        "coverage_percentage": float(len(valid_data) / array.size * 100),
    }


def transform_geometry(geom_geojson: dict, src_crs: str, dst_crs: str) -> dict:
    """Transform geometry between CRS"""
    return transform_geom(src_crs, dst_crs, geom_geojson)


# ---------- Core Land Processor ----------
def process_farmer_land(land: dict, tile: dict) -> bool:
    """Process NDVI for a single farmer land with a specific tile"""
    land_id = land["id"]
    farmer_id = land.get("farmer_id")
    tenant_id = land.get("tenant_id")

    try:
        logging.info(f"Processing land {land_id} with tile {tile['id']}")

        # Get land boundary
        if "boundary" in land and land["boundary"]:
            boundary_data = land["boundary"]
            if isinstance(boundary_data, str):
                boundary_data = json.loads(boundary_data)
            land_geom = shape(boundary_data)
        elif "boundary_polygon_old" in land and land["boundary_polygon_old"]:
            land_geom = shape(land["boundary_polygon_old"])
        else:
            raise Exception(f"No valid boundary found for land {land_id}")

        # Download NDVI tile from B2
        tif_data = download_tile_from_b2(tile["b2_path"])

        # Clip raster to land boundary
        with rasterio.open(tif_data) as src:
            land_geom_transformed = transform_geom("EPSG:4326", src.crs, mapping(land_geom))
            clipped, transform = mask(
                src, [land_geom_transformed], crop=True, all_touched=True, nodata=0
            )

            num_bands = src.count
            if num_bands >= 4:
                blue = clipped[1]  # Band 2
                red = clipped[3]   # Band 4
                nir = clipped[7] if num_bands >= 8 else clipped[3]  # Band 8
            else:
                logging.warning(f"Insufficient bands ({num_bands}) for land {land_id}")
                red = clipped[0]
                nir = clipped[1] if num_bands >= 2 else clipped[0]
                blue = None

        # Calculate vegetation indices
        indices = calculate_vegetation_indices(red, nir, blue)
        ndvi = indices["ndvi"]

        # Statistics
        ndvi_stats = calculate_statistics(ndvi)
        if ndvi_stats["valid_pixels"] == 0:
            raise Exception("No valid NDVI data")

        # Visualization
        viz_bytes = generate_ndvi_visualization(ndvi)
        viz_filename = f"{land_id}_{tile['acquisition_date']}.png"
        viz_path = f"ndvi-thumbnails/{viz_filename}"

        supabase.storage.from_("ndvi-thumbnails").upload(
            viz_path, viz_bytes, {"content-type": "image/png", "upsert": "true"}
        )

        viz_url = f"{SUPABASE_URL}/storage/v1/object/public/ndvi-thumbnails/{viz_path}"

        # Prepare metadata & DB updates
        metadata = {
            "tile_id": tile.get("tile_id"),
            "resolution": tile.get("resolution", 10),
            "cloud_cover": tile.get("cloud_cover"),
            "processing_timestamp": datetime.datetime.utcnow().isoformat(),
            "band_configuration": "sentinel-2-l2a",
            "statistics": ndvi_stats,
        }

        if "evi" in indices:
            metadata["evi_stats"] = calculate_statistics(indices["evi"])
        if "savi" in indices:
            metadata["savi_stats"] = calculate_statistics(indices["savi"])

        # Save results in Supabase tables
        supabase.table("ndvi_micro_tiles").upsert(
            {
                "land_id": land_id,
                "farmer_id": farmer_id,
                "tenant_id": tenant_id,
                "bbox": mapping(land_geom),
                "acquisition_date": tile["acquisition_date"],
                "cloud_cover": tile.get("cloud_cover"),
                "ndvi_mean": ndvi_stats["mean"],
                "ndvi_min": ndvi_stats["min"],
                "ndvi_max": ndvi_stats["max"],
                "ndvi_std_dev": ndvi_stats["std"],
                "ndvi_thumbnail_url": viz_url,
                "resolution_meters": tile.get("resolution", 10),
            },
            on_conflict="land_id,acquisition_date",
        ).execute()

        supabase.table("lands").update(
            {
                "last_ndvi_calculation": tile["acquisition_date"],
                "last_ndvi_value": ndvi_stats["mean"],
                "updated_at": datetime.datetime.utcnow().isoformat(),
            }
        ).eq("id", land_id).execute()

        logging.info(f"✅ Land {land_id} processed, mean NDVI={ndvi_stats['mean']:.3f}")
        return True

    except Exception as e:
        logging.error(f"❌ Failed processing land {land_id}: {e}")
        supabase.table("ndvi_processing_logs").insert(
            {
                "satellite_tile_id": tile.get("id"),
                "processing_step": "ndvi_calculation",
                "step_status": "failed",
                "completed_at": datetime.datetime.utcnow().isoformat(),
                "error_message": str(e),
                "metadata": {"land_id": land_id, "tenant_id": tenant_id},
            }
        ).execute()
        return False


def process_queue(limit: int = 10):
    """Process lands from ndvi_request_queue"""
    requests_q = supabase.table("ndvi_request_queue").select("*").eq("status", "queued").limit(limit).execute()
    if not requests_q.data:
        logging.info("No queued requests")
        return

    for req in requests_q.data:
        req_id = req["id"]
        land_ids = req["land_ids"]

        supabase.table("ndvi_request_queue").update(
            {"status": "processing", "started_at": datetime.datetime.utcnow().isoformat()}
        ).eq("id", req_id).execute()

        lands = supabase.table("lands").select("*").in_("id", land_ids).execute()
        tiles = supabase.table("satellite_tiles").select("*") \
            .eq("tile_id", req["tile_id"]) \
            .gte("acquisition_date", req["date_from"]) \
            .lte("acquisition_date", req["date_to"]) \
            .lte("cloud_cover", req.get("cloud_coverage", 20)) \
            .order("acquisition_date", desc=True).execute()

        if not tiles.data:
            supabase.table("ndvi_request_queue").update(
                {"status": "failed", "error_message": "No tiles found"}
            ).eq("id", req_id).execute()
            continue

        latest_tile = tiles.data[0]
        success_count = 0

        for land in lands.data:
            if process_farmer_land(land, latest_tile):
                success_count += 1

        supabase.table("ndvi_request_queue").update(
            {"status": "completed", "completed_at": datetime.datetime.utcnow().isoformat(), "processed_count": success_count}
        ).eq("id", req_id).execute()


def main(limit: int = None, use_queue: bool = True):
    if use_queue:
        logging.info("🔄 Processing NDVI request queue")
        process_queue(limit or 5)
    else:
        lands = supabase.table("lands").select("*").limit(limit or 50).execute()
        tiles = supabase.table("satellite_tiles").select("*").order("acquisition_date", desc=True).limit(1).execute()
        if not tiles.data:
            logging.warning("No tiles available")
            return

        latest_tile = tiles.data[0]
        for land in lands.data:
            process_farmer_land(land, latest_tile)
