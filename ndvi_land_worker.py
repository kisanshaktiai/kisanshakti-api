"""
NDVI Land Worker (v3.6)
-----------------------
Background processor for clipping and calculating NDVI for each tenant’s lands
using globally shared Sentinel-2 tiles stored in Backblaze B2.

Data flow:
1. Pull queued NDVI jobs from ndvi_request_queue (tenant-scoped)
2. For each land_id -> download B04 (red) + B08 (NIR) from B2
3. Clip by land boundary_polygon_old
4. Compute NDVI, generate color NDVI thumbnail
5. Save NDVI stats & image URL in ndvi_data / ndvi_micro_tiles
6. Update land summary (mean NDVI)
7. Log status in ndvi_processing_logs
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
def build_b2_band_path(tile_id: str, date: str, band: str) -> str:
    """Construct Backblaze B2 path for Sentinel-2 bands."""
    return f"tiles/raw/{tile_id}/{date}/{band}.tif"


def download_b2_band(tile_id: str, date: str, band: str) -> io.BytesIO:
    """Download a specific band (B04 or B08) from B2 private bucket with error checks."""
    if not B2_BUCKET_URL:
        raise ValueError("B2_BUCKET_URL environment variable not set")

    path = build_b2_band_path(tile_id, date, band)
    url = f"{B2_BUCKET_URL.rstrip('/')}/{path}"
    logger.info(f"📥 Downloading {band} from {url}")

    resp = requests.get(url, timeout=90)
    if resp.status_code != 200:
        body_preview = resp.content[:200].decode("utf-8", errors="replace")
        logger.error(f"❌ B2 download failed for {band}: HTTP {resp.status_code}, Preview: {body_preview}")
        raise Exception(f"Failed to download {band} (HTTP {resp.status_code})")

    logger.info(f"✅ Downloaded {band} ({len(resp.content)/1e6:.2f} MB)")
    return io.BytesIO(resp.content)


def open_raster_from_bytesio(buf: io.BytesIO):
    """Open raster from memory buffer."""
    buf.seek(0)
    return MemoryFile(buf.read()).open()


def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute NDVI = (NIR - RED) / (NIR + RED)."""
    np.seterr(divide="ignore", invalid="ignore")
    denom = nir + red
    ndvi = np.where(denom == 0, np.nan, (nir - red) / denom)
    return np.clip(ndvi, -1, 1)


def calculate_statistics(arr: np.ndarray) -> dict:
    """Compute NDVI statistics."""
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
        "coverage": float(valid.size / total * 100),
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
    """Clip tile bands by land boundary, compute NDVI, and store results."""
    land_id = land["id"]
    tenant_id = land["tenant_id"]
    farmer_id = land.get("farmer_id")
    tile_id = tile["tile_id"]
    date = tile.get("acquisition_date") or datetime.date.today().isoformat()

    try:
        logger.info(f"🌍 Processing land {land_id} using tile {tile_id}")

        # --- Geometry ---
        geom_raw = land.get("boundary_polygon_old")
        if not geom_raw:
            raise Exception("Missing boundary_polygon_old")

        geom_json = json.loads(geom_raw) if isinstance(geom_raw, str) else geom_raw
        land_geom = shape(geom_json)

        # --- Download Sentinel bands ---
        red_buf = download_b2_band(tile_id, date, "B04")
        nir_buf = download_b2_band(tile_id, date, "B08")

        with rasterio.open(red_buf) as red_src, rasterio.open(nir_buf) as nir_src:
            if red_src.crs is None:
                raise Exception("Missing CRS in source image")

            geom_transformed = transform_geom("EPSG:4326", red_src.crs.to_string(), mapping(land_geom))
            red_clip, _ = mask(red_src, [geom_transformed], crop=True, nodata=np.nan)
            nir_clip, _ = mask(nir_src, [geom_transformed], crop=True, nodata=np.nan)

            ndvi = calculate_ndvi(red_clip[0].astype(float), nir_clip[0].astype(float))

        # --- NDVI Stats ---
        stats = calculate_statistics(ndvi)
        if stats["valid_pixels"] == 0:
            raise Exception("No valid NDVI pixels in clipped area")

        # --- Visualization Upload ---
        try:
            img_bytes = generate_ndvi_visualization(ndvi)
            filename = f"{tenant_id}_{land_id}_{date}.png"
            storage_path = f"ndvi-thumbnails/{filename}"

            img_buffer = io.BytesIO(img_bytes)
            img_buffer.seek(0)
            upload_resp = supabase.storage.from_("ndvi-thumbnails").upload(
                storage_path,
                img_buffer,
                {"content-type": "image/png", "upsert": "true"},
            )

            upload_error = None
            if hasattr(upload_resp, "error") and upload_resp.error:
                upload_error = upload_resp.error
            elif isinstance(upload_resp, dict) and upload_resp.get("error"):
                upload_error = upload_resp["error"]
            if upload_error:
                raise Exception(f"Upload error: {upload_error}")

            img_url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/public/ndvi-thumbnails/{storage_path}"
            logger.info(f"⬆️ Uploaded NDVI thumbnail: {img_url}")
        except Exception as exc:
            logger.error(f"❌ Thumbnail upload failed for {land_id}: {exc}")
            img_url = None

        # --- Upsert into ndvi_micro_tiles ---
        try:
            upsert_resp = supabase.table("ndvi_micro_tiles").upsert(
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
            if getattr(upsert_resp, "error", None):
                raise Exception(upsert_resp.error)
        except Exception as exc:
            logger.error(f"❌ ndvi_micro_tiles upsert failed for {land_id}: {exc}")

        # --- Upsert into ndvi_data ---
        try:
            ndvi_payload = {
                "tenant_id": tenant_id,
                "farmer_id": farmer_id,
                "land_id": land_id,
                "date": date,
                "ndvi_value": stats["mean"],
                "min_ndvi": stats["min"],
                "max_ndvi": stats["max"],
                "mean_ndvi": stats["mean"],
                "coverage_percentage": stats["coverage"],
                "image_url": img_url,
                "tile_id": tile_id,
                "spatial_resolution": 10,
                "created_at": datetime.datetime.utcnow().isoformat(),
                "metadata": stats,
            }
            ndvi_resp = supabase.table("ndvi_data").upsert(ndvi_payload, on_conflict="land_id,date").execute()
            if getattr(ndvi_resp, "error", None):
                raise Exception(ndvi_resp.error)
        except Exception as exc:
            logger.error(f"❌ ndvi_data upsert failed for {land_id}: {exc}")

        # --- Update land summary ---
        supabase.table("lands").update(
            {
                "last_ndvi_value": stats["mean"],
                "last_ndvi_calculation": date,
                "updated_at": datetime.datetime.utcnow().isoformat(),
            }
        ).eq("id", land_id).execute()

        logger.info(f"✅ Land {land_id} processed | Mean NDVI={stats['mean']:.4f}")
        return True

    except Exception as e:
        tb = traceback.format_exc()
        logger.exception(f"❌ Failed processing land {land_id}: {e}\n{tb}")
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
    """Process queued NDVI requests."""
    logger.info("🧾 Checking NDVI request queue...")
    rq = supabase.table("ndvi_request_queue").select("*").eq("status", "queued").limit(limit).execute()
    requests = rq.data or []
    if not requests:
        logger.info("No queued NDVI requests found.")
        return

    for req in requests:
        req_id = req["id"]
        tenant_id = req["tenant_id"]
        tile_id = req["tile_id"]
        logger.info(f"⚙️ Processing queue request {req_id} for tenant {tenant_id}")

        # Update to processing
        supabase.table("ndvi_request_queue").update(
            {"status": "processing", "started_at": datetime.datetime.utcnow().isoformat()}
        ).eq("id", req_id).execute()

        lands = supabase.table("lands").select("*").eq("tenant_id", tenant_id).in_("id", req["land_ids"]).execute().data
        if not lands:
            supabase.table("ndvi_request_queue").update(
                {"status": "failed", "error_message": "No lands found"}
            ).eq("id", req_id).execute()
            continue

        tile_res = supabase.table("satellite_tiles").select("*").eq("tile_id", tile_id).limit(1).execute()
        if not getattr(tile_res, "data", None):
            logger.error(f"❌ No satellite_tiles record found for tile_id={tile_id}")
            supabase.table("ndvi_request_queue").update(
                {"status": "failed", "error_message": "Missing tile metadata"}
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

        logger.info(f"✅ Completed request {req_id} | Lands processed: {processed}")


# ---------------- Entry ----------------
def main(limit: int = 10):
    logger.info("🚀 NDVI Worker started")
    process_queue(limit)


if __name__ == "__main__":
    main()
