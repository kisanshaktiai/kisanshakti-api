"""
NDVI Land Worker (v3.6-debug)
-----------------------------
Enhanced background processor for clipping and calculating NDVI for each tenant’s lands
using globally shared Sentinel-2 tiles stored in Backblaze B2.

Debug Patch Features:
✅ Full verbose logging
✅ Uses precomputed NDVI.tif if available
✅ Handles missing B04/B08 gracefully
✅ Detailed error handling with Supabase updates
✅ Writes to ndvi_data / ndvi_micro_tiles / ndvi_processing_logs
✅ Compatible with MPC pipeline (B2 tile structure)
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
B2_BUCKET_URL = os.getenv("B2_BUCKET_URL")  # e.g., https://f005.backblazeb2.com/file/kisanshakti-ndvi-tiles

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ndvi-worker")


# ---------------- Utility Functions ----------------
def build_b2_path(tile_id: str, date: str, subdir: str, filename: str) -> str:
    """Construct Backblaze B2 full path for given subdir (raw/ndvi)."""
    return f"tiles/{subdir}/{tile_id}/{date}/{filename}"


def download_b2_file(tile_id: str, date: str, subdir: str, filename: str) -> io.BytesIO | None:
    """Download file from B2 with error tracing."""
    if not B2_BUCKET_URL:
        raise ValueError("B2_BUCKET_URL environment variable not set")

    path = build_b2_path(tile_id, date, subdir, filename)
    url = f"{B2_BUCKET_URL.rstrip('/')}/{path}"

    logger.info(f"📥 Fetching {filename} from {url}")
    try:
        resp = requests.get(url, timeout=90)
        if resp.status_code != 200:
            logger.warning(f"⚠️ Missing B2 file: {filename} ({resp.status_code})")
            return None
        return io.BytesIO(resp.content)
    except Exception as e:
        logger.error(f"❌ B2 download error for {filename}: {e}")
        return None


def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute NDVI = (NIR - RED) / (NIR + RED)."""
    np.seterr(divide="ignore", invalid="ignore")
    ndvi = (nir - red) / (nir + red)
    return np.clip(ndvi, -1, 1)


def calculate_statistics(arr: np.ndarray) -> dict:
    """Compute NDVI statistics."""
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return {"mean": None, "min": None, "max": None, "std": None, "coverage": 0.0, "valid_pixels": 0}
    return {
        "mean": float(np.mean(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "std": float(np.std(valid)),
        "coverage": float(valid.size / arr.size * 100),
        "valid_pixels": int(valid.size),
    }


def generate_ndvi_visualization(ndvi: np.ndarray) -> bytes:
    """Generate NDVI heatmap as PNG bytes."""
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


# ---------------- Core Processing ----------------
def process_farmer_land(land: dict, tile: dict) -> bool:
    land_id = land["id"]
    tenant_id = land["tenant_id"]
    farmer_id = land.get("farmer_id")
    tile_id = tile["tile_id"]
    date = tile.get("acquisition_date") or datetime.date.today().isoformat()

    logger.info(f"🌍 Starting NDVI for land={land_id} | tile={tile_id}")

    try:
        # Validate geometry
        geom_raw = land.get("boundary_polygon_old")
        if not geom_raw:
            raise Exception("Missing boundary_polygon_old")
        geom_json = json.loads(geom_raw) if isinstance(geom_raw, str) else geom_raw
        land_geom = shape(geom_json)

        # --- Try precomputed NDVI ---
        ndvi_buf = download_b2_file(tile_id, date, "ndvi", "ndvi.tif")
        if ndvi_buf:
            logger.info(f"✅ Found precomputed NDVI.tif for tile {tile_id}")
            ndvi_src = rasterio.open(ndvi_buf)
            ndvi_clip, _ = mask(ndvi_src, [transform_geom("EPSG:4326", ndvi_src.crs.to_string(), mapping(land_geom))], crop=True)
            ndvi = ndvi_clip[0]
        else:
            logger.info(f"⚠️ No precomputed NDVI.tif found — falling back to raw bands (B04/B08)")
            red_buf = download_b2_file(tile_id, date, "raw", "B04.tif")
            nir_buf = download_b2_file(tile_id, date, "raw", "B08.tif")

            if not red_buf or not nir_buf:
                raise Exception(f"Missing B04/B08 for tile {tile_id}")

            with rasterio.open(red_buf) as red_src, rasterio.open(nir_buf) as nir_src:
                geom_transformed = transform_geom("EPSG:4326", red_src.crs.to_string(), mapping(land_geom))
                red_clip, _ = mask(red_src, [geom_transformed], crop=True, nodata=np.nan)
                nir_clip, _ = mask(nir_src, [geom_transformed], crop=True, nodata=np.nan)
                ndvi = calculate_ndvi(red_clip[0].astype(float), nir_clip[0].astype(float))

        stats = calculate_statistics(ndvi)
        if not stats["valid_pixels"]:
            raise Exception("No valid NDVI pixels after clipping")

        # --- Visualization Upload ---
        try:
            img_bytes = generate_ndvi_visualization(ndvi)
            storage_path = f"ndvi-thumbnails/{tenant_id}_{land_id}_{date}.png"
            upload_resp = supabase.storage.from_("ndvi-thumbnails").upload(
                storage_path, io.BytesIO(img_bytes), {"content-type": "image/png", "upsert": "true"}
            )
            img_url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/public/ndvi-thumbnails/{storage_path}"
            logger.info(f"⬆️ Uploaded thumbnail: {img_url}")
        except Exception as e:
            logger.error(f"❌ Thumbnail upload failed for land {land_id}: {e}")
            img_url = None

        # --- Upsert into ndvi_micro_tiles ---
        try:
            supabase.table("ndvi_micro_tiles").upsert(
                {
                    "tenant_id": tenant_id,
                    "farmer_id": farmer_id,
                    "land_id": land_id,
                    "acquisition_date": date,
                    "ndvi_mean": stats["mean"],
                    "ndvi_min": stats["min"],
                    "ndvi_max": stats["max"],
                    "ndvi_std_dev": stats["std"],
                    "ndvi_thumbnail_url": img_url,
                    "resolution_meters": 10,
                    "bbox": mapping(land_geom),
                    "created_at": datetime.datetime.utcnow().isoformat(),
                },
                on_conflict="land_id,acquisition_date",
            ).execute()
            logger.info(f"🧩 Saved ndvi_micro_tile for land {land_id}")
        except Exception as e:
            logger.error(f"❌ ndvi_micro_tiles insert failed: {e}")

        # --- Upsert into ndvi_data ---
        try:
            ndvi_payload = {
                "tenant_id": tenant_id,
                "farmer_id": farmer_id,
                "land_id": land_id,
                "date": date,
                "tile_id": tile_id,
                "ndvi_value": stats["mean"],
                "mean_ndvi": stats["mean"],
                "min_ndvi": stats["min"],
                "max_ndvi": stats["max"],
                "coverage_percentage": stats["coverage"],
                "image_url": img_url,
                "spatial_resolution": 10,
                "created_at": datetime.datetime.utcnow().isoformat(),
                "metadata": stats,
            }
            supabase.table("ndvi_data").upsert(ndvi_payload, on_conflict="land_id,date").execute()
            logger.info(f"💾 NDVI data inserted for land {land_id}")
        except Exception as e:
            logger.error(f"❌ ndvi_data upsert failed: {e}")

        # --- Update land summary ---
        supabase.table("lands").update(
            {
                "last_ndvi_value": stats["mean"],
                "last_ndvi_calculation": date,
                "updated_at": datetime.datetime.utcnow().isoformat(),
            }
        ).eq("id", land_id).execute()

        logger.info(f"✅ Land {land_id} processed successfully (mean NDVI={stats['mean']:.4f})")
        return True

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"❌ Failed processing land {land_id}: {e}\n{tb}")
        supabase.table("ndvi_processing_logs").insert(
            {
                "tenant_id": tenant_id,
                "land_id": land_id,
                "satellite_tile_id": tile.get("id"),
                "processing_step": "ndvi_calculation",
                "step_status": "failed",
                "completed_at": datetime.datetime.utcnow().isoformat(),
                "error_message": str(e),
                "error_details": {"traceback": tb},
                "metadata": {"tile_id": tile_id, "date": date},
            }
        ).execute()
        return False


# ---------------- Queue Processing ----------------
def process_queue(limit: int = 10):
    logger.info("🧾 Checking NDVI request queue...")
    rq = supabase.table("ndvi_request_queue").select("*").eq("status", "queued").limit(limit).execute()
    requests = rq.data or []
    logger.info(f"📋 Found {len(requests)} queued request(s)")

    if not requests:
        logger.info("🟢 No queued NDVI jobs found — idle worker.")
        return

    for req in requests:
        req_id = req["id"]
        tenant_id = req["tenant_id"]
        tile_id = req["tile_id"]
        land_ids = req.get("land_ids", [])
        logger.info(f"⚙️ Starting queue {req_id} for tenant={tenant_id} tile={tile_id}")

        # Update to processing
        supabase.table("ndvi_request_queue").update(
            {"status": "processing", "started_at": datetime.datetime.utcnow().isoformat()}
        ).eq("id", req_id).execute()

        # Validate lands
        lands = supabase.table("lands").select("*").eq("tenant_id", tenant_id).in_("id", land_ids).execute().data
        if not lands:
            logger.warning(f"⚠️ No lands found for tenant={tenant_id}, skipping queue {req_id}")
            supabase.table("ndvi_request_queue").update(
                {"status": "failed", "error_message": "No lands found for tenant."}
            ).eq("id", req_id).execute()
            continue

        # Validate tile
        tile_res = supabase.table("satellite_tiles").select("*").eq("tile_id", tile_id).limit(1).execute()
        if not getattr(tile_res, "data", None):
            logger.error(f"❌ No satellite tile metadata found for tile_id={tile_id}")
            supabase.table("ndvi_request_queue").update(
                {"status": "failed", "error_message": "Missing tile metadata."}
            ).eq("id", req_id).execute()
            continue

        tile = tile_res.data[0]
        processed = 0

        for land in lands:
            if process_farmer_land(land, tile):
                processed += 1

        supabase.table("ndvi_request_queue").update(
            {
                "status": "completed",
                "processed_count": processed,
                "completed_at": datetime.datetime.utcnow().isoformat(),
            }
        ).eq("id", req_id).execute()

        logger.info(f"🎯 Completed queue {req_id} | Lands processed: {processed}")


# ---------------- Entry ----------------
def main(limit: int = 10):
    logger.info("🚀 NDVI Worker started (debug mode)")
    process_queue(limit)
    logger.info("🏁 NDVI Worker finished")


if __name__ == "__main__":
    main()
