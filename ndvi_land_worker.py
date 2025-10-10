# farmer_land_ndvi_worker.py

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
from PIL import Image
import json

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
    """
    NDVI = (NIR - RED) / (NIR + RED)
    Returns values between -1 and 1
    """
    np.seterr(divide='ignore', invalid='ignore')
    denominator = nir.astype(float) + red.astype(float)
    ndvi = np.where(
        denominator == 0,
        0,
        (nir.astype(float) - red.astype(float)) / denominator
    )
    # Clip to valid NDVI range
    ndvi = np.clip(ndvi, -1, 1)
    return ndvi


def calculate_vegetation_indices(red: np.ndarray, nir: np.ndarray, blue: np.ndarray = None) -> dict:
    """Calculate multiple vegetation indices"""
    indices = {}
    
    # NDVI
    indices['ndvi'] = calculate_ndvi(red, nir)
    
    # EVI (Enhanced Vegetation Index) - requires blue band
    if blue is not None:
        np.seterr(divide='ignore', invalid='ignore')
        evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
        indices['evi'] = np.clip(evi, -1, 1)
    
    # SAVI (Soil Adjusted Vegetation Index) - L=0.5 for moderate vegetation
    L = 0.5
    savi = ((nir - red) / (nir + red + L)) * (1 + L)
    indices['savi'] = np.clip(savi, -1, 1)
    
    return indices


def generate_ndvi_visualization(ndvi_array: np.ndarray, output_size=(512, 512)) -> bytes:
    """
    Generate color-mapped NDVI visualization
    Red: Low vegetation, Yellow: Medium, Green: High vegetation
    """
    # Remove NaN values
    ndvi_clean = np.nan_to_num(ndvi_array, nan=0.0)
    
    # Normalize to 0-255
    ndvi_norm = ((ndvi_clean + 1) / 2 * 255).astype(np.uint8)
    
    # Create RGB image with color mapping
    img_rgb = np.zeros((*ndvi_norm.shape, 3), dtype=np.uint8)
    
    # Color mapping: Red (low) -> Yellow (medium) -> Green (high)
    # Low NDVI (< 0.2): Red tones
    mask_low = ndvi_clean < 0.2
    img_rgb[mask_low] = [255, int(ndvi_norm[mask_low].mean()), 0]
    
    # Medium NDVI (0.2-0.5): Yellow to light green
    mask_med = (ndvi_clean >= 0.2) & (ndvi_clean < 0.5)
    img_rgb[mask_med] = [200, 255, 100]
    
    # High NDVI (>= 0.5): Green
    mask_high = ndvi_clean >= 0.5
    img_rgb[mask_high] = [0, 255, 0]
    
    img = Image.fromarray(img_rgb, mode='RGB')
    img.thumbnail(output_size, Image.Resampling.LANCZOS)
    
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.getvalue()


def calculate_statistics(array: np.ndarray) -> dict:
    """Calculate comprehensive statistics for an array"""
    valid_data = array[~np.isnan(array)]
    
    if len(valid_data) == 0:
        return {
            'mean': None,
            'min': None,
            'max': None,
            'std': None,
            'median': None,
            'q25': None,
            'q75': None,
            'valid_pixels': 0,
            'total_pixels': array.size,
            'coverage_percentage': 0.0
        }
    
    return {
        'mean': float(np.mean(valid_data)),
        'min': float(np.min(valid_data)),
        'max': float(np.max(valid_data)),
        'std': float(np.std(valid_data)),
        'median': float(np.median(valid_data)),
        'q25': float(np.percentile(valid_data, 25)),
        'q75': float(np.percentile(valid_data, 75)),
        'valid_pixels': int(len(valid_data)),
        'total_pixels': int(array.size),
        'coverage_percentage': float(len(valid_data) / array.size * 100)
    }


def transform_geometry(geom_geojson: dict, src_crs: str, dst_crs: str) -> dict:
    """Transform geometry from one CRS to another"""
    return transform_geom(src_crs, dst_crs, geom_geojson)


# ---------- Core Land Processor ----------
def process_farmer_land(land: dict, tile: dict) -> bool:
    """Process NDVI for a single farmer land with a specific tile"""
    land_id = land["id"]
    farmer_id = land.get("farmer_id")
    tenant_id = land.get("tenant_id")
    
    try:
        logging.info(f"Processing land {land_id} with tile {tile['id']}")
        
        # Get land geometry - handle both 'boundary' (geography) and GeoJSON formats
        if 'boundary' in land and land['boundary']:
            # PostGIS geography type - need to convert
            boundary_data = land['boundary']
            if isinstance(boundary_data, str):
                boundary_data = json.loads(boundary_data)
            land_geom = shape(boundary_data)
        elif 'boundary_polygon_old' in land and land['boundary_polygon_old']:
            land_geom = shape(land['boundary_polygon_old'])
        else:
            raise Exception(f"No valid boundary found for land {land_id}")

        # Download tile
        tif_data = download_tile_from_b2(tile["b2_path"])

        # Open and process raster
        with rasterio.open(tif_data) as src:
            # Transform land geometry to match raster CRS
            land_geom_transformed = transform_geom(
                'EPSG:4326',
                src.crs,
                mapping(land_geom)
            )
            
            # Clip raster to land boundary
            try:
                clipped, transform = mask(
                    src, 
                    [land_geom_transformed], 
                    crop=True,
                    all_touched=True,
                    nodata=0
                )
            except Exception as mask_error:
                logging.error(f"Mask error for land {land_id}: {mask_error}")
                raise

            # Extract bands - adjust indices based on your satellite data
            # Sentinel-2: Band 4 (Red), Band 8 (NIR), Band 2 (Blue)
            num_bands = src.count
            
            if num_bands >= 4:
                blue = clipped[1]  # Band 2
                red = clipped[3]   # Band 4
                nir = clipped[7] if num_bands >= 8 else clipped[3]  # Band 8 or fallback
            else:
                logging.warning(f"Insufficient bands ({num_bands}) for land {land_id}")
                red = clipped[0]
                nir = clipped[1] if num_bands >= 2 else clipped[0]
                blue = None

        # Calculate vegetation indices
        indices = calculate_vegetation_indices(red, nir, blue)
        ndvi = indices['ndvi']
        
        # Calculate statistics
        ndvi_stats = calculate_statistics(ndvi)
        
        if ndvi_stats['valid_pixels'] == 0:
            logging.warning(f"No valid NDVI pixels for land {land_id}")
            raise Exception("No valid NDVI data - possibly outside tile coverage")

        # Generate visualization
        viz_bytes = generate_ndvi_visualization(ndvi)
        viz_size_kb = len(viz_bytes) / 1024
        
        # Upload to storage
        viz_filename = f"{land_id}_{tile['acquisition_date']}.png"
        viz_path = f"ndvi-thumbnails/{viz_filename}"
        
        # Upload with proper error handling
        try:
            supabase.storage.from_("ndvi-thumbnails").upload(
                viz_path, 
                viz_bytes, 
                {"content-type": "image/png", "upsert": "true"}
            )
        except Exception as upload_error:
            logging.warning(f"Upload error (trying upsert): {upload_error}")
            # Try updating if exists
            supabase.storage.from_("ndvi-thumbnails").update(
                viz_path,
                viz_bytes,
                {"content-type": "image/png"}
            )
        
        viz_url = f"{SUPABASE_URL}/storage/v1/object/public/ndvi-thumbnails/{viz_path}"

        # Prepare metadata
        metadata = {
            'tile_id': tile.get('tile_id'),
            'resolution': tile.get('resolution', 10),
            'cloud_cover': tile.get('cloud_cover'),
            'processing_timestamp': datetime.datetime.utcnow().isoformat(),
            'band_configuration': 'sentinel-2-l2a',
            'statistics': ndvi_stats
        }
        
        if 'evi' in indices:
            evi_stats = calculate_statistics(indices['evi'])
            metadata['evi_stats'] = evi_stats
        
        if 'savi' in indices:
            savi_stats = calculate_statistics(indices['savi'])
            metadata['savi_stats'] = savi_stats

        # === 1. Insert into ndvi_micro_tiles ===
        micro_tile_data = {
            "land_id": land_id,
            "farmer_id": farmer_id,
            "tenant_id": tenant_id,
            "bbox": mapping(land_geom),
            "acquisition_date": tile["acquisition_date"],
            "cloud_cover": tile.get("cloud_cover"),
            "ndvi_mean": ndvi_stats['mean'],
            "ndvi_min": ndvi_stats['min'],
            "ndvi_max": ndvi_stats['max'],
            "ndvi_std_dev": ndvi_stats['std'],
            "ndvi_thumbnail_url": viz_url,
            "thumbnail_size_kb": round(viz_size_kb, 2),
            "resolution_meters": tile.get("resolution", 10),
            "statistics_only": False,
            "processing_units_used": round(land.get('area_acres', 1) * 0.1, 3)
        }
        
        supabase.table("ndvi_micro_tiles").upsert(
            micro_tile_data,
            on_conflict="land_id,acquisition_date"
        ).execute()

        # === 2. Insert into ndvi_data (legacy table) ===
        ndvi_data = {
            "land_id": land_id,
            "tenant_id": tenant_id,
            "date": tile["acquisition_date"],
            "ndvi_value": round(ndvi_stats['mean'], 3),
            "satellite_source": "sentinel-2",
            "collection_id": "sentinel-2-l2a",
            "scene_id": tile.get("scene_id"),
            "tile_id": tile.get("tile_id"),
            "cloud_cover": tile.get("cloud_cover"),
            "cloud_coverage": tile.get("cloud_cover"),
            "processing_level": "L2A",
            "spatial_resolution": tile.get("resolution", 10),
            "image_url": viz_url,
            "min_ndvi": ndvi_stats['min'],
            "max_ndvi": ndvi_stats['max'],
            "mean_ndvi": ndvi_stats['mean'],
            "valid_pixels": ndvi_stats['valid_pixels'],
            "total_pixels": ndvi_stats['total_pixels'],
            "coverage_percentage": ndvi_stats['coverage_percentage'],
            "metadata": metadata
        }
        
        if 'evi' in indices:
            ndvi_data['evi_value'] = round(calculate_statistics(indices['evi'])['mean'], 3)
        if 'savi' in indices:
            ndvi_data['savi_value'] = round(calculate_statistics(indices['savi'])['mean'], 3)
        
        supabase.table("ndvi_data").upsert(
            ndvi_data,
            on_conflict="land_id,date"
        ).execute()

        # === 3. Update lands table ===
        supabase.table("lands").update({
            "last_ndvi_calculation": tile["acquisition_date"],
            "last_ndvi_value": ndvi_stats['mean'],
            "updated_at": datetime.datetime.utcnow().isoformat()
        }).eq("id", land_id).execute()

        # === 4. Update land_tile_mapping ===
        mapping_update = {
            "last_ndvi_request_date": tile["acquisition_date"],
            "last_ndvi_value": ndvi_stats['mean'],
            "ndvi_cache_expiry": (datetime.datetime.utcnow() + datetime.timedelta(days=7)).isoformat(),
            "needs_refresh": False,
            "updated_at": datetime.datetime.utcnow().isoformat()
        }
        
        supabase.table("land_tile_mapping").update(mapping_update)\
            .eq("land_id", land_id)\
            .eq("tile_id", tile.get("tile_id"))\
            .execute()

        # === 5. Log success ===
        supabase.table("ndvi_processing_logs").insert({
            "satellite_tile_id": tile["id"],
            "processing_step": "ndvi_calculation",
            "step_status": "success",
            "completed_at": datetime.datetime.utcnow().isoformat(),
            "metadata": {
                "land_id": land_id,
                "farmer_id": farmer_id,
                "tenant_id": tenant_id,
                "ndvi_mean": ndvi_stats['mean'],
                "coverage": ndvi_stats['coverage_percentage'],
                "valid_pixels": ndvi_stats['valid_pixels']
            }
        }).execute()

        logging.info(f"✅ NDVI processed for land {land_id}: mean={ndvi_stats['mean']:.3f}, coverage={ndvi_stats['coverage_percentage']:.1f}%")
        return True

    except Exception as e:
        error_msg = str(e)
        logging.error(f"❌ Failed processing land {land_id}: {error_msg}")
        
        # Log failure
        supabase.table("ndvi_processing_logs").insert({
            "satellite_tile_id": tile.get("id") if tile else None,
            "processing_step": "ndvi_calculation",
            "step_status": "failed",
            "completed_at": datetime.datetime.utcnow().isoformat(),
            "error_message": error_msg,
            "error_details": {
                "land_id": land_id,
                "exception_type": type(e).__name__
            },
            "metadata": {
                "land_id": land_id,
                "farmer_id": farmer_id,
                "tenant_id": tenant_id
            }
        }).execute()
        return False


# ---------- Batch Processor ----------
def process_queue(limit: int = 10):
    """Process requests from ndvi_request_queue"""
    requests_q = supabase.table("ndvi_request_queue") \
        .select("*") \
        .eq("status", "queued") \
        .order("priority") \
        .order("scheduled_for") \
        .limit(limit) \
        .execute()

    if not requests_q.data:
        logging.info("No queued requests found")
        return

    for req in requests_q.data:
        req_id = req["id"]
        land_ids = req["land_ids"]
        
        logging.info(f"Processing request {req_id} with {len(land_ids)} lands")

        # Mark request as processing
        supabase.table("ndvi_request_queue").update({
            "status": "processing",
            "started_at": datetime.datetime.utcnow().isoformat()
        }).eq("id", req_id).execute()

        processed_count = 0
        processing_units = 0.0

        try:
            # Fetch lands
            lands = supabase.table("lands").select("*").in_("id", land_ids).execute()
            
            if not lands.data:
                raise Exception("No lands found for given IDs")

            # Fetch tiles for request date range
            tiles = supabase.table("satellite_tiles").select("*") \
                .gte("acquisition_date", req["date_from"]) \
                .lte("acquisition_date", req["date_to"]) \
                .eq("tile_id", req["tile_id"]) \
                .lte("cloud_cover", req.get("cloud_coverage", 20)) \
                .order("acquisition_date", desc=True) \
                .execute()

            if not tiles.data:
                raise Exception(f"No tiles found for tile_id={req['tile_id']} between {req['date_from']} and {req['date_to']}")

            # Use the latest tile with lowest cloud cover
            latest_tile = sorted(
                tiles.data,
                key=lambda t: (t["acquisition_date"], -t.get("cloud_cover", 100)),
                reverse=True
            )[0]
            
            logging.info(f"Using tile from {latest_tile['acquisition_date']} with {latest_tile.get('cloud_cover', 'N/A')}% cloud cover")

            # Process each land
            for land in lands.data:
                if process_farmer_land(land, latest_tile):
                    processed_count += 1
                    processing_units += land.get('area_acres', 1) * 0.1

            # Mark request completed
            supabase.table("ndvi_request_queue").update({
                "status": "completed",
                "completed_at": datetime.datetime.utcnow().isoformat(),
                "processed_count": processed_count,
                "processing_units_consumed": round(processing_units, 3)
            }).eq("id", req_id).execute()

            logging.info(f"🎯 Request {req_id} completed: {processed_count}/{len(land_ids)} lands")

        except Exception as e:
            error_msg = str(e)
            logging.error(f"❌ Request {req_id} failed: {error_msg}")
            
            supabase.table("ndvi_request_queue").update({
                "status": "failed",
                "completed_at": datetime.datetime.utcnow().isoformat(),
                "error_message": error_msg,
                "processed_count": processed_count
            }).eq("id", req_id).execute()


def main(limit: int = None, use_queue: bool = True):
    """Main entrypoint"""
    if use_queue:
        logging.info("🔄 Processing NDVI request queue...")
        process_queue(limit or 5)
    else:
        logging.info("🔄 Processing all lands directly...")
        lands = supabase.table("lands")\
            .select("*")\
            .eq("is_active", True)\
            .is_("deleted_at", "null")\
            .limit(limit or 100)\
            .execute()
        
        count = 0
        for land in lands.data:
            # Pick latest tile automatically
            tiles = supabase.table("satellite_tiles")\
                .select("*")\
                .order("acquisition_date", desc=True)\
                .limit(1)\
                .execute()
            
            if tiles.data:
                if process_farmer_land(land, tiles.data[0]):
                    count += 1
        
        logging.info(f"Processed {count}/{len(lands.data)} lands")
        return count
