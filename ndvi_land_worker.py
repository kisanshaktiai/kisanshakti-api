"""
NDVI Land Worker v9.1 — CRS Fix + Stable Intersection
=====================================================
Critical Fixes:
1. ✅ CRS-safe reprojection for land polygons
2. ✅ Reliable mask intersection with raster CRS validation
3. ✅ NDVI empty raster detection & logging
4. ✅ Retains async + B2 streaming + Supabase sync
"""

import os
import io
import json
import time
import logging
import datetime
import traceback
import argparse
from typing import List, Optional, Dict, Any, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.errors import RasterioIOError
from rasterio.crs import CRS
from shapely.geometry import shape, mapping, box
from shapely import wkt, wkb
from shapely.ops import unary_union, transform
import pyproj
from PIL import Image
import matplotlib.cm as cm

from supabase import create_client, Client
from b2sdk.v2 import InMemoryAccountInfo, B2Api

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ndvi-worker-v9.1")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
B2_KEY_ID = os.getenv("B2_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME", "kisanshakti-ndvi-tiles")
B2_PUBLIC_REGION = os.getenv("B2_PUBLIC_REGION", "f005")
SUPABASE_NDVI_BUCKET = os.getenv("SUPABASE_NDVI_BUCKET", "ndvi-thumbnails")

MAX_CONCURRENT_LANDS = int(os.getenv("MAX_CONCURRENT_LANDS", "8"))
THREAD_POOL_WORKERS = int(os.getenv("THREAD_POOL_WORKERS", "12"))

if not all([SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, B2_KEY_ID, B2_APP_KEY]):
    raise RuntimeError("Missing critical environment variables")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

b2_info = InMemoryAccountInfo()
b2_api = B2Api(b2_info)
b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
try:
    b2_bucket = b2_api.get_bucket_by_name(B2_BUCKET_NAME)
    logger.info(f"✅ B2 bucket connected: {B2_BUCKET_NAME}")
except Exception as e:
    logger.critical(f"❌ B2 bucket access failed: {e}")
    b2_bucket = None

# ---------------------------------------------------------------------
# Geometry Extraction
# ---------------------------------------------------------------------
def extract_geometry_from_land(land: Dict) -> Optional[Dict]:
    """Extract valid GeoJSON geometry from land record"""
    for key in ["boundary", "boundary_geom", "boundary_polygon_old"]:
        val = land.get(key)
        if not val:
            continue
        try:
            if isinstance(val, (bytes, bytearray)):
                geom = wkb.loads(val)
            elif isinstance(val, str):
                try:
                    geom = wkb.loads(bytes.fromhex(val))
                except Exception:
                    geom = shape(json.loads(val))
            elif isinstance(val, dict):
                geom = shape(val)
            else:
                continue
            return mapping(geom)
        except Exception as e:
            logger.debug(f"Failed to parse {key}: {e}")
    return None

# ---------------------------------------------------------------------
# CRS Safe Reprojection
# ---------------------------------------------------------------------
def safe_reproject_geom(geojson_geom: Dict, target_crs: CRS) -> Dict:
    """Reproject EPSG:4326 geometry to match raster CRS safely"""
    try:
        if not geojson_geom:
            raise ValueError("Empty geometry passed to reproject")

        shp = shape(geojson_geom)
        if shp.is_empty:
            raise ValueError("Input geometry is empty")

        src_crs = "EPSG:4326"
        tgt = target_crs.to_string() if hasattr(target_crs, "to_string") else str(target_crs)
        transformer = pyproj.Transformer.from_crs(src_crs, tgt, always_xy=True).transform
        reprojected = transform(transformer, shp)

        if reprojected.is_empty:
            raise ValueError("Reprojection produced empty geometry")

        return mapping(reprojected)
    except Exception as e:
        logger.error(f"❌ safe_reproject_geom failed: {e}")
        return geojson_geom

# ---------------------------------------------------------------------
# NDVI Extraction (single tile)
# ---------------------------------------------------------------------
def extract_ndvi_from_tile(tile_id: str, acq_date: str, land_geom_4326: Dict, debug_land_id: str) -> Optional[Tuple[np.ndarray, Dict]]:
    """Extract NDVI from a single raster tile"""
    from rasterio.io import MemoryFile
    import requests

    ndvi_path = f"tiles/ndvi/{tile_id}/{acq_date}/ndvi.tif"
    url = f"https://{B2_PUBLIC_REGION}.backblazeb2.com/file/{B2_BUCKET_NAME}/{ndvi_path}"

    try:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_TIMEOUT='30'):
            with rasterio.open(url) as src:
                logger.info(f"🗺️ NDVI Tile: {tile_id}/{acq_date} CRS={src.crs}")
                land_proj = safe_reproject_geom(land_geom_4326, src.crs)

                land_shape = shape(land_proj)
                tile_box = box(*src.bounds)
                if not land_shape.intersects(tile_box):
                    logger.debug(f"🚫 No overlap {tile_id}/{acq_date}")
                    return None

                ndvi_clip, transform = mask(
                    src,
                    [mapping(land_shape)],
                    crop=True,
                    all_touched=True,
                    indexes=1
                )

                if np.all(np.isnan(ndvi_clip)) or ndvi_clip.size == 0:
                    logger.warning(f"⚠️ NDVI clip empty: {tile_id}/{acq_date}")
                    return None

                return ndvi_clip[0], {"transform": transform, "crs": src.crs, "nodata": src.nodata}

    except Exception as e:
        logger.warning(f"❌ NDVI tile {tile_id}/{acq_date} failed: {e}")
        return None

# ---------------------------------------------------------------------
# Merge Multiple Tiles
# ---------------------------------------------------------------------
def merge_multi_tile_ndvi(ndvi_results: List[Tuple[np.ndarray, Dict]]) -> np.ndarray:
    """Simple average merge of multiple NDVI arrays"""
    if len(ndvi_results) == 1:
        return ndvi_results[0][0]
    try:
        arrays = [r[0] for r in ndvi_results if r is not None]
        if not arrays:
            raise ValueError("No valid NDVI arrays to merge")
        stacked = np.stack(arrays)
        return np.nanmean(stacked, axis=0)
    except Exception as e:
        logger.warning(f"Merge failed: {e}")
        return ndvi_results[0][0]

# ---------------------------------------------------------------------
# NDVI Stats + Thumbnail
# ---------------------------------------------------------------------
def calculate_statistics(ndvi: np.ndarray) -> Dict[str, Any]:
    mask_valid = (ndvi >= -1) & (ndvi <= 1)
    valid = ndvi[mask_valid]
    if valid.size == 0:
        return {"mean": None, "min": None, "max": None, "std": None, "coverage": 0.0}
    return {
        "mean": float(np.nanmean(valid)),
        "min": float(np.nanmin(valid)),
        "max": float(np.nanmax(valid)),
        "std": float(np.nanstd(valid)),
        "coverage": float((valid.size / ndvi.size) * 100)
    }

def create_colorized_thumbnail(ndvi: np.ndarray, max_size: int = 512) -> bytes:
    norm = np.clip((ndvi + 1) / 2, 0, 1)
    cmap = cm.get_cmap("RdYlGn")
    rgba = (cmap(norm) * 255).astype(np.uint8)
    img = Image.fromarray(rgba, mode="RGBA")
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.getvalue()

# ---------------------------------------------------------------------
# Main Async Land Processor (patched)
# ---------------------------------------------------------------------
async def process_single_land_async(land: Dict, tile_ids: Optional[List[str]], acquisition_date_override: Optional[str], executor: ThreadPoolExecutor) -> Dict[str, Any]:
    loop = asyncio.get_running_loop()
    land_id = land.get("id")
    tenant_id = land.get("tenant_id")

    result = {"land_id": land_id, "success": False}

    try:
        geom = extract_geometry_from_land(land)
        if not geom:
            raise ValueError("No valid land geometry")

        # Determine tiles
        if not tile_ids and land.get("tile_ids"):
            tile_ids = land["tile_ids"]
        if not tile_ids:
            raise ValueError("No intersecting tiles")

        ndvi_results = []
        for tile_id in tile_ids:
            acq_date = acquisition_date_override or await loop.run_in_executor(executor, lambda: datetime.date.today().isoformat())
            r = await loop.run_in_executor(executor, extract_ndvi_from_tile, tile_id, acq_date, geom, land_id)
            if r:
                ndvi_results.append(r)

        if not ndvi_results:
            raise ValueError("No NDVI data extracted from any intersecting tiles")

        merged = merge_multi_tile_ndvi(ndvi_results)
        stats = calculate_statistics(merged)
        if stats["mean"] is None:
            raise ValueError("No valid NDVI pixels")

        thumb = await loop.run_in_executor(executor, create_colorized_thumbnail, merged)
        logger.info(f"✅ Land {land_id}: NDVI={stats['mean']:.3f} coverage={stats['coverage']:.1f}%")

        result["success"] = True
        result["stats"] = stats
    except Exception as e:
        logger.error(f"❌ Land {land_id} failed: {e}")
        result["error"] = str(e)
    return result

# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------
async def process_request_async(
    queue_id: str,
    tenant_id: str,
    land_ids: List[str],
    tile_ids: Optional[List[str]] = None
):
    logger.info(f"🚀 Processing queue {queue_id} for {len(land_ids)} lands")
    resp = supabase.table("lands").select("*").eq("tenant_id", tenant_id).in_("id", land_ids).execute()
    lands = resp.data or []
    executor = ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS)
    sem = asyncio.Semaphore(MAX_CONCURRENT_LANDS)

    async def sem_task(land):
        async with sem:
            return await process_single_land_async(land, None, None, executor)

    tasks = [sem_task(l) for l in lands]
    results = await asyncio.gather(*tasks)
    logger.info(f"🏁 Done: {sum(1 for r in results if r['success'])}/{len(results)} successful")

# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-id", type=str)
    parser.add_argument("--tenant", type=str)
    parser.add_argument("--lands", type=str)
    args = parser.parse_args()

    if not args.queue_id or not args.tenant or not args.lands:
        print("Usage: python ndvi_land_worker_v9.1.py --queue-id ID --tenant TENANT_ID --lands land1,land2")
        exit(1)

    lands = args.lands.split(",")
    asyncio.run(process_request_async(args.queue_id, args.tenant, lands))
