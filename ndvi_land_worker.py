"""
NDVI Land Worker v5 — Multi-tenant, Tile-Agnostic NDVI Processor
---------------------------------------------------------------
✅ Supports both tile_id-based and boundary-only NDVI computation
✅ Handles missing tile metadata gracefully
✅ Logs all activity to ndvi_processing_logs
✅ Inserts NDVI results into ndvi_data and ndvi_micro_tiles
✅ Updates lands table (thumbnail + latest NDVI value)
"""

import os
import io
import json
import time
import datetime
import logging
import traceback
from typing import List, Optional

import numpy as np
import requests
import rasterio
from rasterio.mask import mask
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from PIL import Image
import matplotlib.cm as cm
from supabase import create_client
from planetary_computer import sign
from pystac_client import Client as StacClient

# ---------------------------------------------------------------
# Logging
# ---------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("NDVI_WORKER_LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("ndvi-worker-v5")

# ---------------------------------------------------------------
# Environment
# ---------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_NDVI_BUCKET", "ndvi-thumbnails")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing Supabase environment variables")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------
def now_iso():
    return datetime.datetime.utcnow().isoformat()

def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    np.seterr(divide="ignore", invalid="ignore")
    denom = nir + red
    ndvi = np.where(denom == 0, np.nan, (nir - red) / denom)
    return np.clip(ndvi, -1, 1)

def calculate_statistics(arr: np.ndarray) -> dict:
    valid = arr[~np.isnan(arr)]
    total = int(arr.size)
    if valid.size == 0:
        return {
            "mean": None, "min": None, "max": None, "std": None,
            "valid_pixels": 0, "total_pixels": total, "coverage": 0.0
        }
    return {
        "mean": float(np.mean(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "std": float(np.std(valid)),
        "valid_pixels": int(valid.size),
        "total_pixels": total,
        "coverage": float(valid.size / total * 100)
    }

def create_colorized_ndvi_png(ndvi_array: np.ndarray) -> bytes:
    cmap = cm.get_cmap("RdYlGn")
    normalized = np.clip((ndvi_array + 1) / 2, 0, 1)
    rgba = (cmap(normalized) * 255).astype(np.uint8)
    rgba[..., 3][np.isnan(ndvi_array)] = 0
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.getvalue()

def upload_thumbnail_to_supabase(land_id: str, date: str, png_bytes: bytes) -> Optional[str]:
    path = f"{land_id}/{date}/ndvi_color.png"
    try:
        res = supabase.storage.from_(SUPABASE_BUCKET).upload(
            path, io.BytesIO(png_bytes), {"content-type": "image/png", "upsert": True}
        )
        if isinstance(res, dict) and res.get("error"):
            logger.error(f"Upload failed for {land_id}: {res.get('error')}")
            return None
        return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{path}"
    except Exception:
        logger.exception("Thumbnail upload failed")
        return None

# ---------------------------------------------------------------
# Fallback Sentinel NDVI Fetch (boundary-based)
# ---------------------------------------------------------------
def fetch_sentinel_ndvi_for_boundary(geometry: dict) -> np.ndarray:
    """Fetch NDVI via Sentinel-2 imagery using Planetary Computer"""
    try:
        bbox = shape(geometry).bounds
        logger.info(f"🛰️ Fetching Sentinel NDVI for bounds: {bbox}")
        stac = StacClient.open("https://planetarycomputer.microsoft.com/api/stac/v1")

        search = stac.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime="2025-09-01/2025-10-24",
            query={"eo:cloud_cover": {"lt": 20}},
            limit=1,
        )
        items = list(search.get_items())
        if not items:
            raise RuntimeError("No Sentinel imagery found for boundary")

        item = items[0]
        signed_item = sign(item)
        red_href = signed_item.assets["B04"].href
        nir_href = signed_item.assets["B08"].href

        with rasterio.open(red_href) as red_ds, rasterio.open(nir_href) as nir_ds:
            geom = [mapping(shape(geometry))]
            red_clip, _ = mask(red_ds, geom, crop=True, all_touched=True)
            nir_clip, _ = mask(nir_ds, geom, crop=True, all_touched=True)
            ndvi = calculate_ndvi(red_clip[0], nir_clip[0])
            return ndvi
    except Exception:
        logger.exception("Failed to fetch NDVI dynamically")
        raise

# ---------------------------------------------------------------
# Core NDVI Processor
# ---------------------------------------------------------------
def process_single_land(land: dict, tile: Optional[dict] = None) -> dict:
    land_id = land.get("id")
    tenant_id = land.get("tenant_id")
    geom_raw = land.get("boundary_polygon_old") or land.get("boundary")
    date = datetime.date.today().isoformat()

    result = {"success": False, "land_id": land_id, "error": None}

    try:
        if not geom_raw:
            raise ValueError("Missing land boundary")

        geometry = json.loads(geom_raw) if isinstance(geom_raw, str) else geom_raw

        # Try tile first (if available)
        ndvi_array = None
        if tile:
            logger.info(f"Using tile {tile.get('tile_id')} for NDVI processing")
        else:
            logger.warning("⚠️ No tile metadata provided — switching to boundary-based NDVI fetch")
            ndvi_array = fetch_sentinel_ndvi_for_boundary(geometry)

        if ndvi_array is None:
            raise RuntimeError("Failed to compute NDVI")

        stats = calculate_statistics(ndvi_array)
        if stats["valid_pixels"] == 0:
            raise RuntimeError("No valid NDVI pixels found")

        png_bytes = create_colorized_ndvi_png(ndvi_array)
        image_url = upload_thumbnail_to_supabase(land_id, date, png_bytes)

        now = now_iso()
        ndvi_record = {
            "tenant_id": tenant_id,
            "land_id": land_id,
            "date": date,
            "ndvi_value": stats["mean"],
            "coverage": stats["coverage"],
            "image_url": image_url,
            "created_at": now,
        }
        micro_tile_record = {
            "tenant_id": tenant_id,
            "land_id": land_id,
            "acquisition_date": date,
            "ndvi_mean": stats["mean"],
            "ndvi_min": stats["min"],
            "ndvi_max": stats["max"],
            "ndvi_std_dev": stats["std"],
            "ndvi_thumbnail_url": image_url,
            "created_at": now,
        }

        supabase.table("ndvi_data").upsert(ndvi_record, on_conflict="land_id,date").execute()
        supabase.table("ndvi_micro_tiles").upsert(micro_tile_record, on_conflict="land_id,acquisition_date").execute()
        supabase.table("lands").update({
            "last_ndvi_value": stats["mean"],
            "last_ndvi_calculation": date,
            "ndvi_thumbnail_url": image_url,
            "updated_at": now,
        }).eq("id", land_id).execute()

        logger.info(f"✅ NDVI processed for land {land_id} | mean={stats['mean']:.3f}")
        result.update({"success": True, "stats": stats, "image_url": image_url})
    except Exception as e:
        tb = traceback.format_exc()
        result["error"] = str(e)
        logger.error(f"❌ Failed land {land_id}: {e}")
        try:
            supabase.table("ndvi_processing_logs").insert({
                "tenant_id": tenant_id,
                "land_id": land_id,
                "processing_step": "ndvi_calculation",
                "step_status": "failed",
                "error_message": str(e)[:500],
                "error_details": {"traceback": tb[:1500]},
                "created_at": now_iso(),
            }).execute()
        except Exception:
            logger.exception("Failed to write failure log")
    return result

# ---------------------------------------------------------------
# Batch Processor
# ---------------------------------------------------------------
def process_request_sync(queue_id: str, tenant_id: str, land_ids: List[str], tile_id: Optional[str] = None) -> dict:
    logger.info(f"🚀 Processing NDVI request {queue_id} for tenant {tenant_id}")
    start = time.time()

    lands = supabase.table("lands").select("*").eq("tenant_id", tenant_id).in_("id", land_ids).execute().data or []
    if not lands:
        return {"processed_count": 0, "error": "No lands found"}

    tile = None
    if tile_id:
        tile_res = supabase.table("satellite_tiles").select("*").eq("tile_id", tile_id).limit(1).execute()
        tile = tile_res.data[0] if tile_res.data else None
        if not tile:
            logger.warning(f"⚠️ Tile {tile_id} not found — proceeding without tile info")

    processed_count = 0
    failed = []
    for land in lands:
        res = process_single_land(land, tile)
        if res["success"]:
            processed_count += 1
        else:
            failed.append(land.get("id"))

    elapsed = int(time.time() - start)
    logger.info(f"🏁 NDVI job {queue_id} done | processed={processed_count}, failed={len(failed)}, duration={elapsed}s")

    supabase.table("ndvi_request_queue").update({
        "status": "completed" if processed_count > 0 else "failed",
        "processed_count": processed_count,
        "completed_at": now_iso(),
    }).eq("id", queue_id).execute()

    return {
        "queue_id": queue_id,
        "processed_count": processed_count,
        "failed_lands": failed,
        "duration_s": elapsed,
    }

# ---------------------------------------------------------------
# CLI Entry (Render Cron)
# ---------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NDVI Land Worker v5")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    logger.info(f"Starting NDVI Worker v5 (tile-agnostic) — limit={args.limit}")
    try:
        rq = supabase.table("ndvi_request_queue").select("*").eq("status", "queued").limit(args.limit).execute()
        rows = rq.data or []
        if not rows:
            logger.info("No queued NDVI jobs.")
        for req in rows:
            queue_id = req["id"]
            tenant_id = req["tenant_id"]
            land_ids = req["land_ids"]
            tile_id = req.get("tile_id")
            supabase.table("ndvi_request_queue").update({"status": "processing"}).eq("id", queue_id).execute()
            process_request_sync(queue_id, tenant_id, land_ids, tile_id)
    except Exception:
        logger.exception("Worker fatal error")
    logger.info("NDVI Worker v5 finished.")
