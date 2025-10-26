"""
NDVI Land Worker v6.0 - Production NDVI Processing Engine
==========================================================
Processes NDVI from already-downloaded Sentinel-2 tiles stored in B2.

Architecture:
- Finds all intersecting tiles for each land parcel
- Downloads .tif files from B2 cloud storage
- Clips rasters to land boundaries using PostGIS geometries
- Merges multi-tile NDVI for lands spanning multiple tiles
- Computes vegetation statistics (mean, min, max, std)
- Generates colorized PNG thumbnails
- Uploads results to Supabase Storage
- Updates ndvi_data, ndvi_micro_tiles, and lands tables

Features:
✅ Multi-tile support (lands spanning multiple MGRS tiles)
✅ Efficient raster clipping with rasterio
✅ Beautiful NDVI visualizations (RdYlGn colormap)
✅ Comprehensive error handling & retry logic
✅ Detailed processing logs
✅ Memory-efficient streaming from B2
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

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.io import MemoryFile
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import shape, mapping
from PIL import Image
import matplotlib.cm as cm

from supabase import create_client, Client
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from b2sdk.download_dest import DownloadDestBytes

# =============================================================================
# Configuration & Logging
# =============================================================================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ndvi-worker")

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
B2_APP_KEY_ID = os.getenv("B2_APP_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME", "kisanshakti-ndvi-tiles")
SUPABASE_NDVI_BUCKET = os.getenv("SUPABASE_NDVI_BUCKET", "ndvi-thumbnails")

# Validate environment
if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("❌ Missing Supabase credentials")
if not B2_APP_KEY_ID or not B2_APP_KEY:
    raise RuntimeError("❌ Missing B2 credentials")

# Initialize clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
logger.info(f"✅ Supabase client initialized")

# Initialize B2 API
b2_info = InMemoryAccountInfo()
b2_api = B2Api(b2_info)
b2_api.authorize_account("production", B2_APP_KEY_ID, B2_APP_KEY)
b2_bucket = b2_api.get_bucket_by_name(B2_BUCKET_NAME)
logger.info(f"✅ B2 API initialized: bucket={B2_BUCKET_NAME}")

# =============================================================================
# Utility Functions
# =============================================================================
def now_iso() -> str:
    """Current UTC timestamp in ISO format"""
    return datetime.datetime.utcnow().isoformat()

def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Calculate NDVI from red and NIR bands.
    NDVI = (NIR - Red) / (NIR + Red)
    Range: [-1, 1] where higher values indicate healthier vegetation
    """
    np.seterr(divide='ignore', invalid='ignore')
    ndvi = (nir.astype(float) - red.astype(float)) / (nir.astype(float) + red.astype(float))
    ndvi = np.clip(ndvi, -1, 1)
    ndvi[np.isnan(ndvi)] = -1  # Set NaN to -1 (no data)
    return ndvi

def calculate_statistics(ndvi_array: np.ndarray) -> Dict[str, Any]:
    """Compute comprehensive statistics from NDVI raster"""
    valid_mask = (ndvi_array >= -1) & (ndvi_array <= 1) & ~np.isnan(ndvi_array)
    valid_pixels = ndvi_array[valid_mask]
    
    total_pixels = int(ndvi_array.size)
    valid_count = int(valid_pixels.size)
    
    if valid_count == 0:
        return {
            "mean": None, "min": None, "max": None, "std": None,
            "valid_pixels": 0, "total_pixels": total_pixels,
            "coverage": 0.0, "health_category": "no_data"
        }
    
    mean_ndvi = float(np.mean(valid_pixels))
    
    # Categorize vegetation health
    if mean_ndvi < 0.2:
        health = "poor"
    elif mean_ndvi < 0.4:
        health = "fair"
    elif mean_ndvi < 0.6:
        health = "good"
    else:
        health = "excellent"
    
    return {
        "mean": float(mean_ndvi),
        "min": float(np.min(valid_pixels)),
        "max": float(np.max(valid_pixels)),
        "std": float(np.std(valid_pixels)),
        "valid_pixels": valid_count,
        "total_pixels": total_pixels,
        "coverage": float(valid_count / total_pixels * 100),
        "health_category": health
    }

def create_colorized_thumbnail(ndvi_array: np.ndarray, max_size: int = 512) -> bytes:
    """
    Generate a beautiful colorized PNG thumbnail from NDVI array.
    Uses RdYlGn (Red-Yellow-Green) colormap for vegetation visualization.
    """
    # Normalize NDVI to 0-1 range
    normalized = np.clip((ndvi_array + 1) / 2, 0, 1)
    
    # Apply colormap
    cmap = cm.get_cmap('RdYlGn')
    rgba = (cmap(normalized) * 255).astype(np.uint8)
    
    # Set transparent for no-data pixels
    rgba[..., 3][ndvi_array == -1] = 0
    
    # Create PIL Image
    img = Image.fromarray(rgba, mode='RGBA')
    
    # Resize if too large
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Save to bytes
    buffer = io.BytesIO()
    img.save(buffer, format='PNG', optimize=True)
    buffer.seek(0)
    
    return buffer.getvalue()

def upload_thumbnail_to_supabase(land_id: str, date: str, png_bytes: bytes) -> Optional[str]:
    """Upload NDVI thumbnail to Supabase Storage"""
    try:
        path = f"{land_id}/{date}/ndvi_colorized.png"
        
        # Upload with upsert
        result = supabase.storage.from_(SUPABASE_NDVI_BUCKET).upload(
            path=path,
            file=png_bytes,
            file_options={"content-type": "image/png", "upsert": "true"}
        )
        
        # Generate public URL
        public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_NDVI_BUCKET}/{path}"
        
        logger.debug(f"📤 Thumbnail uploaded: {path}")
        return public_url
        
    except Exception as e:
        logger.error(f"❌ Thumbnail upload failed for land {land_id}: {e}")
        return None

# =============================================================================
# B2 Data Access
# =============================================================================
def download_from_b2(tile_id: str, subdir: str, filename: str) -> Optional[bytes]:
    """
    Download a file from B2 cloud storage.
    
    Args:
        tile_id: MGRS tile ID (e.g., '43RGN')
        subdir: Subdirectory ('ndvi', 'raw', 'composite')
        filename: File name (e.g., '2024-01-15/ndvi.tif', '2024-01-15/B04.tif')
    
    Returns:
        File bytes or None if not found
    """
    try:
        b2_path = f"tiles/{subdir}/{tile_id}/{filename}"
        logger.debug(f"⬇️  Downloading from B2: {b2_path}")
        
        dest = DownloadDestBytes()
        b2_bucket.download_file_by_name(b2_path, dest)
        
        data = dest.get_bytes_written()
        logger.debug(f"✅ Downloaded {len(data)} bytes from {b2_path}")
        
        return data
        
    except Exception as e:
        logger.warning(f"⚠️  B2 download failed: {b2_path} | {e}")
        return None

def get_latest_tile_date(tile_id: str) -> Optional[str]:
    """Find the most recent acquisition date for a tile in satellite_tiles table"""
    try:
        response = supabase.table("satellite_tiles").select("acquisition_date").eq(
            "tile_id", tile_id
        ).eq("status", "completed").order("acquisition_date", desc=True).limit(1).execute()
        
        if response.data:
            return response.data[0]["acquisition_date"]
        
        logger.warning(f"⚠️  No completed tiles found for {tile_id}")
        return None
        
    except Exception as e:
        logger.error(f"❌ Failed to query satellite_tiles: {e}")
        return None

# =============================================================================
# Geospatial Processing
# =============================================================================
def get_intersecting_tiles_for_land(land_geometry: Dict) -> List[Dict[str, Any]]:
    """
    Find all MGRS tiles that intersect a land boundary.
    Uses PostGIS spatial intersection via Supabase RPC or fallback query.
    """
    try:
        # Try RPC function first
        try:
            response = supabase.rpc(
                "get_intersecting_tiles",
                {"land_geom": json.dumps(land_geometry)}
            ).execute()
            
            if response.data:
                logger.debug(f"🗺️  Found {len(response.data)} intersecting tiles via RPC")
                return response.data
        except Exception as rpc_error:
            logger.debug(f"RPC not available, using fallback: {rpc_error}")
        
        # Fallback: Query all tiles and check intersection with shapely
        all_tiles_response = supabase.table("mgrs_tiles").select(
            "tile_id, geometry, is_agri"
        ).eq("is_agri", True).execute()
        
        land_shape = shape(land_geometry)
        intersecting = []
        
        for tile in all_tiles_response.data or []:
            try:
                tile_geom = tile.get("geometry") or tile.get("bbox")
                if tile_geom:
                    tile_shape = shape(tile_geom)
                    if land_shape.intersects(tile_shape):
                        intersecting.append(tile)
            except Exception:
                continue
        
        logger.debug(f"🗺️  Found {len(intersecting)} intersecting tiles (fallback)")
        return intersecting
        
    except Exception as e:
        logger.error(f"❌ Failed to find intersecting tiles: {e}")
        return []

def clip_raster_to_geometry(raster_bytes: bytes, geometry: Dict) -> Optional[np.ndarray]:
    """
    Clip a raster (GeoTIFF) to a polygon geometry.
    
    Args:
        raster_bytes: GeoTIFF file bytes
        geometry: GeoJSON polygon geometry
    
    Returns:
        Clipped numpy array or None if failed
    """
    try:
        with MemoryFile(raster_bytes) as memfile:
            with memfile.open() as src:
                # Clip to geometry
                out_image, out_transform = mask(src, [geometry], crop=True, all_touched=True)
                
                # Return first band (NDVI is single band)
                return out_image[0]
                
    except Exception as e:
        logger.error(f"❌ Raster clipping failed: {e}")
        return None

def merge_ndvi_rasters(ndvi_arrays: List[np.ndarray]) -> np.ndarray:
    """
    Merge multiple NDVI rasters into a single array.
    Used when a land parcel spans multiple tiles.
    """
    if len(ndvi_arrays) == 0:
        raise ValueError("No NDVI arrays to merge")
    
    if len(ndvi_arrays) == 1:
        return ndvi_arrays[0]
    
    # Use rasterio merge for proper handling
    try:
        # Create temporary datasets in memory
        datasets = []
        for arr in ndvi_arrays:
            memfile = MemoryFile()
            dataset = memfile.open(
                driver='GTiff',
                height=arr.shape[0],
                width=arr.shape[1],
                count=1,
                dtype=arr.dtype
            )
            dataset.write(arr, 1)
            datasets.append(dataset)
        
        # Merge
        mosaic, out_trans = merge(datasets)
        
        # Close datasets
        for ds in datasets:
            ds.close()
        
        return mosaic[0]
        
    except Exception as e:
        logger.error(f"❌ Raster merge failed: {e}")
        # Fallback: simple averaging
        return np.mean(np.stack(ndvi_arrays), axis=0)

# =============================================================================
# Core Processing Logic
# =============================================================================
def process_single_land(
    land: Dict[str, Any],
    tile_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Process NDVI for a single land parcel.
    
    Steps:
    1. Find intersecting MGRS tiles
    2. Download NDVI rasters from B2 for each tile
    3. Clip rasters to land boundary
    4. Merge if multiple tiles
    5. Calculate statistics
    6. Generate thumbnail
    7. Save to database
    
    Args:
        land: Land record from database (must include boundary_polygon_old or boundary)
        tile_ids: Optional list of tile IDs to process (auto-detected if None)
    
    Returns:
        Result dictionary with success status and statistics
    """
    land_id = land["id"]
    tenant_id = land["tenant_id"]
    
    result = {
        "land_id": land_id,
        "success": False,
        "error": None,
        "stats": None
    }
    
    try:
        logger.info(f"🔄 Processing land: {land_id}")
        
        # Extract geometry
        geom_raw = land.get("boundary_polygon_old") or land.get("boundary")
        if not geom_raw:
            raise ValueError("Land has no boundary geometry")
        
        # Parse geometry
        if isinstance(geom_raw, str):
            geometry = json.loads(geom_raw)
        else:
            geometry = geom_raw
        
        # Find intersecting tiles
        if not tile_ids:
            tiles = get_intersecting_tiles_for_land(geometry)
            if not tiles:
                raise ValueError("No intersecting tiles found")
            tile_ids = [t["tile_id"] for t in tiles]
        
        logger.info(f"🗺️  Land {land_id} intersects tiles: {tile_ids}")
        
        # Process each tile
        ndvi_clips = []
        acquisition_date = datetime.date.today().isoformat()
        
        for tile_id in tile_ids:
            # Get latest date for this tile
            tile_date = get_latest_tile_date(tile_id)
            if not tile_date:
                logger.warning(f"⚠️  No data for tile {tile_id}, skipping")
                continue
            
            acquisition_date = tile_date  # Use most recent
            
            # Download NDVI raster from B2
            ndvi_bytes = download_from_b2(tile_id, "ndvi", f"{tile_date}/ndvi.tif")
            
            if not ndvi_bytes:
                # Fallback: Download B04 (Red) and B08 (NIR) and calculate NDVI
                logger.info(f"📥 NDVI not found, downloading raw bands for {tile_id}")
                
                red_bytes = download_from_b2(tile_id, "raw", f"{tile_date}/B04.tif")
                nir_bytes = download_from_b2(tile_id, "raw", f"{tile_date}/B08.tif")
                
                if not (red_bytes and nir_bytes):
                    logger.warning(f"⚠️  Missing bands for tile {tile_id}")
                    continue
                
                # Clip both bands and calculate NDVI
                red_clip = clip_raster_to_geometry(red_bytes, geometry)
                nir_clip = clip_raster_to_geometry(nir_bytes, geometry)
                
                if red_clip is None or nir_clip is None:
                    continue
                
                ndvi_clip = calculate_ndvi(red_clip, nir_clip)
            else:
                # Clip pre-computed NDVI
                ndvi_clip = clip_raster_to_geometry(ndvi_bytes, geometry)
            
            if ndvi_clip is not None:
                ndvi_clips.append(ndvi_clip)
        
        if not ndvi_clips:
            raise ValueError("No NDVI data could be extracted from any tile")
        
        # Merge if multiple tiles
        final_ndvi = merge_ndvi_rasters(ndvi_clips) if len(ndvi_clips) > 1 else ndvi_clips[0]
        
        # Calculate statistics
        stats = calculate_statistics(final_ndvi)
        
        if stats["valid_pixels"] == 0:
            raise ValueError("No valid NDVI pixels after processing")
        
        logger.info(f"📊 NDVI stats for {land_id}: mean={stats['mean']:.3f}, coverage={stats['coverage']:.1f}%")
        
        # Generate thumbnail
        thumbnail_bytes = create_colorized_thumbnail(final_ndvi)
        thumbnail_url = upload_thumbnail_to_supabase(land_id, acquisition_date, thumbnail_bytes)
        
        # Save to ndvi_data table
        ndvi_data_record = {
            "tenant_id": tenant_id,
            "land_id": land_id,
            "date": acquisition_date,
            "ndvi_value": stats["mean"],
            "mean_ndvi": stats["mean"],
            "min_ndvi": stats["min"],
            "max_ndvi": stats["max"],
            "ndvi_std": stats["std"],
            "valid_pixels": stats["valid_pixels"],
            "total_pixels": stats["total_pixels"],
            "coverage_percentage": stats["coverage"],
            "image_url": thumbnail_url,
            "satellite_source": "SENTINEL-2",
            "processing_level": "L2A",
            "created_at": now_iso(),
            "computed_at": now_iso()
        }
        
        supabase.table("ndvi_data").upsert(ndvi_data_record, on_conflict="land_id,date").execute()
        
        # Save to ndvi_micro_tiles table
        bbox = {
            "type": "Polygon",
            "coordinates": geometry.get("coordinates", [])
        }
        
        micro_tile_record = {
            "tenant_id": tenant_id,
            "land_id": land_id,
            "farmer_id": land.get("farmer_id"),
            "acquisition_date": acquisition_date,
            "ndvi_mean": stats["mean"],
            "ndvi_min": stats["min"],
            "ndvi_max": stats["max"],
            "ndvi_std_dev": stats["std"],
            "ndvi_thumbnail_url": thumbnail_url,
            "bbox": bbox,
            "cloud_cover": 0,  # Placeholder
            "created_at": now_iso()
        }
        
        supabase.table("ndvi_micro_tiles").upsert(micro_tile_record).execute()
        
        # Update lands table with latest NDVI
        supabase.table("lands").update({
            "last_ndvi_value": stats["mean"],
            "last_ndvi_calculation": acquisition_date,
            "ndvi_thumbnail_url": thumbnail_url,
            "updated_at": now_iso()
        }).eq("id", land_id).execute()
        
        result["success"] = True
        result["stats"] = stats
        result["thumbnail_url"] = thumbnail_url
        
        logger.info(f"✅ Successfully processed land {land_id}")
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Failed to process land {land_id}: {error_msg}")
        
        result["error"] = error_msg
        
        # Log to ndvi_processing_logs
        try:
            supabase.table("ndvi_processing_logs").insert({
                "tenant_id": tenant_id,
                "land_id": land_id,
                "processing_step": "ndvi_calculation",
                "step_status": "failed",
                "error_message": error_msg[:500],
                "error_details": {"traceback": traceback.format_exc()[:1000]},
                "created_at": now_iso()
            }).execute()
        except:
            pass
    
    return result

def process_request_sync(
    queue_id: str,
    tenant_id: str,
    land_ids: List[str],
    tile_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Synchronous processing of a queue request.
    Called by API for instant processing or by cron worker.
    
    Args:
        queue_id: UUID of queue entry
        tenant_id: Tenant UUID
        land_ids: List of land UUIDs to process
        tile_ids: Optional list of tile IDs (auto-detected if None)
    
    Returns:
        Summary dict with processed count and errors
    """
    logger.info(f"🚀 Starting NDVI processing: queue_id={queue_id}, tenant={tenant_id}, lands={len(land_ids)}")
    
    start_time = time.time()
    
    # Fetch land records
    lands_response = supabase.table("lands").select("*").eq(
        "tenant_id", tenant_id
    ).in_("id", land_ids).execute()
    
    lands = lands_response.data or []
    
    if not lands:
        error_msg = f"No lands found for tenant {tenant_id}"
        logger.error(f"❌ {error_msg}")
        return {
            "queue_id": queue_id,
            "processed_count": 0,
            "failed_count": len(land_ids),
            "error": error_msg
        }
    
    # Process each land
    processed = 0
    failed = []
    
    for land in lands:
        result = process_single_land(land, tile_ids)
        
        if result["success"]:
            processed += 1
        else:
            failed.append({
                "land_id": land["id"],
                "error": result["error"]
            })
    
    duration = int((time.time() - start_time) * 1000)  # milliseconds
    
    logger.info(f"🏁 Processing complete: queue_id={queue_id} | processed={processed}/{len(lands)} | duration={duration}ms")
    
    return {
        "queue_id": queue_id,
        "processed_count": processed,
        "failed_count": len(failed),
        "failed_lands": failed,
        "duration_ms": duration
    }

# =============================================================================
# Cron Worker Main Loop
# =============================================================================
def run_cron_worker(limit: int = 10, max_retries: int = 3):
    """
    Cron worker that processes queued NDVI requests.
    This should be called by a scheduled job (e.g., every 5 minutes).
    
    Args:
        limit: Maximum number of queue items to process in one run
        max_retries: Maximum retry attempts for failed items
    """
    logger.info(f"🔄 NDVI Worker starting: limit={limit}, max_retries={max_retries}")
    
    try:
        # Fetch queued items
        queue_response = supabase.table("ndvi_request_queue").select("*").eq(
            "status", "queued"
        ).order("priority", desc=False).order("created_at", desc=False).limit(limit).execute()
        
        queue_items = queue_response.data or []
        
        if not queue_items:
            logger.info("✅ No queued items to process")
            return
        
        logger.info(f"📋 Found {len(queue_items)} queued items")
        
        for item in queue_items:
            queue_id = item["id"]
            tenant_id = item["tenant_id"]
            land_ids = item["land_ids"]
            tile_id = item.get("tile_id")
            
            # Check retry count
            retry_count = item.get("retry_count", 0)
            if retry_count >= max_retries:
                logger.warning(f"⚠️  Max retries reached for {queue_id}, marking as failed")
                
                supabase.table("ndvi_request_queue").update({
                    "status": "failed",
                    "last_error": f"Max retries ({max_retries}) exceeded",
                    "completed_at": now_iso()
                }).eq("id", queue_id).execute()
                
                continue
            
            # Mark as processing
            supabase.table("ndvi_request_queue").update({
                "status": "processing",
                "started_at": now_iso(),
                "retry_count": retry_count + 1
            }).eq("id", queue_id).execute()
            
            # Process
            try:
                result = process_request_sync(
                    queue_id=queue_id,
                    tenant_id=tenant_id,
                    land_ids=land_ids,
                    tile_ids=[tile_id] if tile_id else None
                )
                
                final_status = "completed" if result["processed_count"] > 0 else "failed"
                
                supabase.table("ndvi_request_queue").update({
                    "status": final_status,
                    "processed_count": result["processed_count"],
                    "processing_duration_ms": result.get("duration_ms"),
                    "completed_at": now_iso(),
                    "last_error": result.get("error")
                }).eq("id", queue_id).execute()
                
                logger.info(f"✅ Queue item {queue_id} completed: {final_status}")
                
            except Exception as e:
                logger.exception(f"❌ Failed to process queue item {queue_id}")
                
                supabase.table("ndvi_request_queue").update({
                    "status": "queued",  # Re-queue for retry
                    "last_error": str(e)[:500]
                }).eq("id", queue_id).execute()
    
    except Exception as e:
        logger.exception("❌ Fatal error in cron worker")

# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NDVI Land Worker v6.0")
    parser.add_argument("--limit", type=int, default=10, help="Max queue items to process")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retry attempts")
    parser.add_argument("--mode", choices=["cron", "single"], default="cron", help="Worker mode")
    parser.add_argument("--queue-id", type=str, help="Process specific queue ID (single mode)")
    
    args = parser.parse_args()
    
    logger.info(f"🚀 NDVI Land Worker v6.0 starting in {args.mode} mode")
    
    if args.mode == "cron":
        run_cron_worker(limit=args.limit, max_retries=args.max_retries)
    elif args.mode == "single" and args.queue_id:
        # Process single queue item
        queue_item = supabase.table("ndvi_request_queue").select("*").eq("id", args.queue_id).single().execute()
        if queue_item.data:
            process_request_sync(
                queue_id=queue_item.data["id"],
                tenant_id=queue_item.data["tenant_id"],
                land_ids=queue_item.data["land_ids"]
            )
        else:
            logger.error(f"❌ Queue item {args.queue_id} not found")
    
    logger.info("🏁 NDVI Land Worker finished")
