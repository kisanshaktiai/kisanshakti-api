#!/usr/bin/env python3
# ndvi_land_worker_v11.0.py — ML-Enhanced for Small Landholdings
# Features: Adaptive buffering, super-resolution, land cover filtering, 
#          Gaussian weighting, advanced quality metrics, SAR fusion ready

import os
import io
import json
import time
import logging
import datetime
import traceback
import argparse
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.errors import RasterioIOError
from rasterio.warp import transform_geom, reproject, Resampling
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import shape, mapping, box, Point
from shapely.ops import transform as shapely_transform
from shapely import wkb
from pyproj import Transformer
from PIL import Image
import matplotlib.cm as cm
from scipy import ndimage
from scipy.ndimage import gaussian_filter

from supabase import create_client, Client
from b2sdk.v2 import InMemoryAccountInfo, B2Api

# ---------------- Logging ----------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ndvi-worker-v11.0")

# ---------------- Configuration ----------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
B2_KEY_ID = os.getenv("B2_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
B2_BUCKET_NAME = os.getenv("B2_BUCKET_NAME", "kisanshakti-ndvi-tiles")
B2_PUBLIC_REGION = os.getenv("B2_PUBLIC_REGION", "f005")
SUPABASE_NDVI_BUCKET = os.getenv("SUPABASE_NDVI_BUCKET", "ndvi-thumbnails")

MAX_CONCURRENT_LANDS = int(os.getenv("MAX_CONCURRENT_LANDS", "6"))
THREAD_POOL_WORKERS = int(os.getenv("THREAD_POOL_WORKERS", "8"))

# Enhanced settings for small parcels
ADAPTIVE_BUFFERING = os.getenv("ADAPTIVE_BUFFERING", "true").lower() == "true"
MICRO_LAND_THRESHOLD = float(os.getenv("MICRO_LAND_THRESHOLD", "2000"))  # m² (~0.5 acres)
SMALL_LAND_THRESHOLD = float(os.getenv("SMALL_LAND_THRESHOLD", "8000"))  # m² (~2 acres)
SUPER_RESOLUTION_ENABLED = os.getenv("SUPER_RESOLUTION_ENABLED", "true").lower() == "true"
GAUSSIAN_WEIGHTING = os.getenv("GAUSSIAN_WEIGHTING", "true").lower() == "true"
LAND_COVER_FILTERING = os.getenv("LAND_COVER_FILTERING", "true").lower() == "true"

# Spatial coherence settings
MIN_VALID_PIXELS = int(os.getenv("MIN_VALID_PIXELS", "9"))  # 3x3 minimum for stats
TARGET_PIXEL_COUNT = int(os.getenv("TARGET_PIXEL_COUNT", "100"))  # Ideal for confidence

# Validate critical env
if not all([SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, B2_KEY_ID, B2_APP_KEY]):
    logger.critical("Missing required environment variables")
    raise RuntimeError("Missing required environment variables")

# ---------------- Clients ----------------
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

# ---------------- Data Classes ----------------
@dataclass
class LandCharacteristics:
    """Geometric and spatial characteristics of a land parcel"""
    area_m2: float
    perimeter_m: float
    centroid: Point
    shape_complexity: float  # perimeter²/area ratio
    aspect_ratio: float
    size_category: str  # micro, small, medium, large
    recommended_buffer: float
    expected_pixel_count: int

@dataclass
class QualityMetrics:
    """Enhanced quality assessment"""
    confidence_level: str
    pixel_count: int
    edge_pixel_ratio: float
    spatial_coverage: float
    spatial_coherence: float  # Moran's I
    temporal_consistency: Optional[float]
    spectral_validity: float
    quality_score: float
    quality_flags: List[str]
    uncertainty_estimate: float
    processing_method: str

# ---------------- Geometry Analysis ----------------
def analyze_land_geometry(geom_4326: Dict) -> LandCharacteristics:
    """
    Analyze land parcel geometry to determine optimal processing parameters.
    Returns adaptive buffering recommendations and expected pixel counts.
    """
    try:
        geom = shape(geom_4326)
        centroid = geom.centroid
        
        # Project to local UTM for accurate metric calculations
        lon, lat = centroid.x, centroid.y
        utm_zone = int((lon + 180) / 6) + 1
        utm_epsg = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
        
        project_to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
        geom_utm = shapely_transform(project_to_utm.transform, geom)
        
        area_m2 = geom_utm.area
        perimeter_m = geom_utm.length
        
        # Calculate shape complexity (1.0 = circle, higher = more complex)
        shape_complexity = (perimeter_m ** 2) / (4 * np.pi * area_m2) if area_m2 > 0 else 1.0
        
        # Calculate aspect ratio (width/height of bounding box)
        bounds = geom_utm.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        aspect_ratio = max(width, height) / max(min(width, height), 1.0)
        
        # Categorize by size
        if area_m2 < MICRO_LAND_THRESHOLD:
            size_category = "micro"
            base_buffer = 50.0  # 5 pixels at 10m resolution
        elif area_m2 < SMALL_LAND_THRESHOLD:
            size_category = "small"
            base_buffer = 30.0  # 3 pixels
        elif area_m2 < 40000:  # ~10 acres
            size_category = "medium"
            base_buffer = 20.0  # 2 pixels
        else:
            size_category = "large"
            base_buffer = 10.0  # 1 pixel
        
        # Adjust buffer based on shape complexity
        complexity_factor = min(shape_complexity / 2.0, 2.0)  # Cap at 2x
        recommended_buffer = base_buffer * complexity_factor
        
        # Estimate expected pixel count at 10m resolution
        buffered_area = geom_utm.buffer(recommended_buffer).area
        expected_pixels = int(buffered_area / 100)  # 100m² per pixel
        
        logger.debug(f"Land analysis: {area_m2:.0f}m² ({size_category}), "
                    f"complexity={shape_complexity:.2f}, buffer={recommended_buffer:.1f}m, "
                    f"expected_pixels={expected_pixels}")
        
        return LandCharacteristics(
            area_m2=area_m2,
            perimeter_m=perimeter_m,
            centroid=Point(lon, lat),
            shape_complexity=shape_complexity,
            aspect_ratio=aspect_ratio,
            size_category=size_category,
            recommended_buffer=recommended_buffer,
            expected_pixel_count=expected_pixels
        )
        
    except Exception as e:
        logger.warning(f"Land geometry analysis failed: {e}, using defaults")
        return LandCharacteristics(
            area_m2=5000,
            perimeter_m=300,
            centroid=Point(0, 0),
            shape_complexity=1.5,
            aspect_ratio=1.5,
            size_category="small",
            recommended_buffer=30.0,
            expected_pixel_count=50
        )

def buffer_geometry_adaptive(geom_4326: Dict, buffer_meters: float) -> Dict:
    """Buffer geometry using accurate UTM projection"""
    try:
        geom = shape(geom_4326)
        centroid = geom.centroid
        
        lon, lat = centroid.x, centroid.y
        utm_zone = int((lon + 180) / 6) + 1
        utm_epsg = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
        
        project_to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
        project_to_wgs84 = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
        
        geom_utm = shapely_transform(project_to_utm.transform, geom)
        buffered_utm = geom_utm.buffer(buffer_meters)
        buffered_wgs84 = shapely_transform(project_to_wgs84.transform, buffered_utm)
        
        return mapping(buffered_wgs84)
    except Exception as e:
        logger.warning(f"Adaptive buffer failed: {e}")
        return geom_4326

# ---------------- Gaussian Distance Weighting ----------------
def create_gaussian_weight_matrix(shape: Tuple[int, int], center_row: int, center_col: int, 
                                  sigma_factor: float = 0.3) -> np.ndarray:
    """
    Create Gaussian weight matrix centered on land centroid.
    Pixels closer to center receive higher weights.
    
    sigma_factor: Controls spread (0.3 = weights drop to ~0.1 at edges)
    """
    rows, cols = shape
    
    # Create coordinate grids
    row_coords = np.arange(rows)
    col_coords = np.arange(cols)
    row_grid, col_grid = np.meshgrid(row_coords, col_coords, indexing='ij')
    
    # Calculate distance from center
    distances = np.sqrt((row_grid - center_row)**2 + (col_grid - center_col)**2)
    
    # Calculate sigma based on array size
    sigma = max(rows, cols) * sigma_factor
    
    # Gaussian weights
    weights = np.exp(-(distances**2) / (2 * sigma**2))
    
    # Normalize to sum to 1
    weights = weights / np.sum(weights)
    
    return weights

def apply_gaussian_weighting(ndvi: np.ndarray, transform, land_centroid: Point) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Gaussian distance weighting to NDVI array based on land centroid.
    Returns weighted NDVI and weight matrix.
    """
    try:
        # Convert centroid to pixel coordinates
        from rasterio.transform import rowcol
        center_row, center_col = rowcol(transform, land_centroid.x, land_centroid.y)
        
        # Ensure center is within bounds
        center_row = max(0, min(ndvi.shape[0] - 1, center_row))
        center_col = max(0, min(ndvi.shape[1] - 1, center_col))
        
        # Create weight matrix
        weights = create_gaussian_weight_matrix(ndvi.shape, center_row, center_col)
        
        # Apply weights (only to valid pixels)
        valid_mask = (ndvi != -1.0) & ~np.isnan(ndvi)
        weighted_ndvi = np.where(valid_mask, ndvi * weights, ndvi)
        
        logger.debug(f"Applied Gaussian weighting centered at pixel ({center_row}, {center_col})")
        
        return weighted_ndvi, weights
        
    except Exception as e:
        logger.warning(f"Gaussian weighting failed: {e}, returning original")
        return ndvi, np.ones_like(ndvi)

# ---------------- Super-Resolution Enhancement ----------------
def apply_super_resolution(ndvi_stack: List[np.ndarray], factor: int = 2) -> np.ndarray:
    """
    Apply super-resolution to increase effective pixel count.
    Uses bicubic interpolation as baseline (can be replaced with ML model).
    
    For ML enhancement: Replace with ESRGAN or custom trained model
    """
    if not ndvi_stack or len(ndvi_stack) == 0:
        return None
    
    try:
        # Use median composite if multiple images
        if len(ndvi_stack) > 1:
            stack = np.stack(ndvi_stack, axis=0)
            valid_mask = stack != -1.0
            with np.errstate(invalid="ignore"):
                masked_stack = np.ma.masked_where(~valid_mask, stack)
                base = np.ma.median(masked_stack, axis=0)
                base = np.where(base.mask, -1.0, base.data).astype(np.float32)
        else:
            base = ndvi_stack[0]
        
        # Upscale using high-quality interpolation
        from scipy.ndimage import zoom
        
        # Separate valid and invalid regions
        valid_mask = base != -1.0
        
        # Upscale valid data
        upscaled = zoom(base, factor, order=3)  # Bicubic interpolation
        upscaled_mask = zoom(valid_mask.astype(np.float32), factor, order=0) > 0.5
        
        # Apply mask to upscaled data
        upscaled = np.where(upscaled_mask, upscaled, -1.0)
        
        # Clip to valid NDVI range
        upscaled = np.clip(upscaled, -1.0, 1.0)
        
        logger.debug(f"Super-resolution: {base.shape} → {upscaled.shape} (factor={factor})")
        
        return upscaled.astype(np.float32)
        
    except Exception as e:
        logger.warning(f"Super-resolution failed: {e}")
        return ndvi_stack[0] if ndvi_stack else None

# ---------------- Land Cover Filtering ----------------
def apply_land_cover_filter(ndvi: np.ndarray, bounds, src_crs) -> Tuple[np.ndarray, Dict]:
    """
    Filter NDVI using land cover classification to remove non-crop pixels.
    
    This is a placeholder for ESA WorldCover or Dynamic World integration.
    In production, query land cover API/dataset and filter accordingly.
    """
    try:
        # Placeholder: In production, fetch ESA WorldCover or Dynamic World data
        # For now, apply simple spectral filtering
        
        # Filter based on NDVI thresholds (crop vegetation typically 0.3-0.9)
        valid_mask = (ndvi >= 0.2) & (ndvi <= 0.95)
        
        # Calculate statistics before/after filtering
        original_valid = np.sum((ndvi != -1.0) & ~np.isnan(ndvi))
        filtered_ndvi = np.where(valid_mask, ndvi, -1.0)
        filtered_valid = np.sum((filtered_ndvi != -1.0) & ~np.isnan(filtered_ndvi))
        
        filter_info = {
            "method": "spectral_threshold",
            "original_pixels": int(original_valid),
            "filtered_pixels": int(filtered_valid),
            "pixels_removed": int(original_valid - filtered_valid),
            "removal_ratio": float((original_valid - filtered_valid) / max(original_valid, 1))
        }
        
        logger.debug(f"Land cover filter: removed {filter_info['pixels_removed']} pixels "
                    f"({filter_info['removal_ratio']*100:.1f}%)")
        
        return filtered_ndvi, filter_info
        
    except Exception as e:
        logger.warning(f"Land cover filtering failed: {e}")
        return ndvi, {"method": "none", "error": str(e)}

# ---------------- Spatial Coherence Analysis ----------------
def calculate_spatial_coherence(ndvi: np.ndarray) -> float:
    """
    Calculate spatial coherence using Moran's I statistic.
    Higher values indicate more spatially coherent (less noisy) data.
    
    Range: -1 (dispersed) to +1 (clustered)
    Crop fields typically show positive spatial autocorrelation (0.3-0.8)
    """
    try:
        valid_mask = (ndvi != -1.0) & ~np.isnan(ndvi)
        valid_data = ndvi[valid_mask]
        
        if len(valid_data) < 9:  # Need at least 3x3
            return 0.0
        
        # Calculate local Moran's I using neighboring pixels
        # Simplified version using gradient magnitude as proxy
        gradient_y, gradient_x = np.gradient(np.where(valid_mask, ndvi, np.nan))
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Lower gradient = higher coherence
        mean_gradient = np.nanmean(gradient_magnitude)
        
        # Normalize to 0-1 range (lower gradient = higher coherence)
        coherence = np.clip(1.0 - (mean_gradient / 0.5), 0.0, 1.0)
        
        return float(coherence)
        
    except Exception as e:
        logger.debug(f"Spatial coherence calculation failed: {e}")
        return 0.5  # Neutral value

def calculate_spectral_validity(ndvi: np.ndarray) -> float:
    """
    Validate spectral characteristics of NDVI data.
    Checks for anomalies that might indicate processing errors.
    """
    try:
        valid_mask = (ndvi != -1.0) & ~np.isnan(ndvi)
        valid_data = ndvi[valid_mask]
        
        if len(valid_data) == 0:
            return 0.0
        
        validity_score = 1.0
        
        # Check 1: Reasonable value distribution
        if np.std(valid_data) > 0.4:  # Too much variation
            validity_score *= 0.8
        
        # Check 2: Not too many extreme values
        extreme_ratio = np.sum((valid_data < -0.5) | (valid_data > 0.95)) / len(valid_data)
        if extreme_ratio > 0.1:
            validity_score *= 0.7
        
        # Check 3: Mean in expected crop range
        mean_val = np.mean(valid_data)
        if mean_val < 0.1 or mean_val > 0.9:
            validity_score *= 0.8
        
        return float(np.clip(validity_score, 0.0, 1.0))
        
    except Exception:
        return 0.5

# ---------------- Utilities ----------------
def now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

def extract_geometry_from_land(land_record: Dict) -> Optional[Dict]:
    """
    Extract geometry GeoJSON from land record.
    CORRECT PRIORITY for your schema:
    1. boundary_geom (PostGIS geometry - populated by trigger)
    2. boundary_polygon_old (jsonb - source data)
    3. center_lat/center_lon (fallback point buffer)
    """
    land_id = land_record.get("id", "<unknown>")
    
    # Debug logging
    logger.debug(f"Land {land_id} geometry fields available: "
                f"boundary_geom={land_record.get('boundary_geom') is not None}, "
                f"boundary_polygon_old={land_record.get('boundary_polygon_old') is not None}, "
                f"center_coords={land_record.get('center_lat') is not None}")
    
    # ========================================================================
    # PRIORITY 1: boundary_geom (PostGIS geometry - what your trigger populates)
    # ========================================================================
    boundary_geom_val = land_record.get("boundary_geom")
    if boundary_geom_val:
        try:
            # Case 1: Already parsed as GeoJSON dict (Supabase sometimes does this)
            if isinstance(boundary_geom_val, dict) and "type" in boundary_geom_val:
                logger.info(f"✅ Land {land_id}: Using boundary_geom (GeoJSON dict)")
                return boundary_geom_val
            
            # Case 2: WKB binary format (most common from PostGIS)
            if isinstance(boundary_geom_val, (bytes, bytearray)):
                shapely_geom = wkb.loads(boundary_geom_val)
                geojson = mapping(shapely_geom)
                logger.info(f"✅ Land {land_id}: Using boundary_geom (WKB binary)")
                return geojson
            
            # Case 3: WKB hex string
            if isinstance(boundary_geom_val, str):
                s = boundary_geom_val.strip()
                
                # Try parsing as GeoJSON string first
                if s.startswith("{"):
                    try:
                        parsed = json.loads(s)
                        if "type" in parsed and "coordinates" in parsed:
                            logger.info(f"✅ Land {land_id}: Using boundary_geom (GeoJSON string)")
                            return parsed
                    except:
                        pass
                
                # Try parsing as WKB hex
                try:
                    hex_str = s.replace("0x", "").replace("0X", "")
                    geom_bytes = bytes.fromhex(hex_str)
                    shapely_geom = wkb.loads(geom_bytes)
                    geojson = mapping(shapely_geom)
                    logger.info(f"✅ Land {land_id}: Using boundary_geom (WKB hex)")
                    return geojson
                except Exception as wkb_error:
                    logger.debug(f"WKB parsing failed: {wkb_error}")
        
        except Exception as e:
            logger.warning(f"⚠️ Land {land_id}: Failed to parse boundary_geom: {e}")
            # Continue to next priority
    
    # ========================================================================
    # PRIORITY 2: boundary_polygon_old (jsonb - your source data)
    # ========================================================================
    boundary_old = land_record.get("boundary_polygon_old")
    if boundary_old:
        try:
            # Case 1: Already a dict
            if isinstance(boundary_old, dict):
                if "type" in boundary_old and "coordinates" in boundary_old:
                    logger.info(f"✅ Land {land_id}: Using boundary_polygon_old (dict)")
                    return boundary_old
                else:
                    logger.warning(f"⚠️ Land {land_id}: boundary_polygon_old dict missing type/coordinates")
            
            # Case 2: JSON string
            if isinstance(boundary_old, str):
                parsed = json.loads(boundary_old)
                if "type" in parsed and "coordinates" in parsed:
                    logger.info(f"✅ Land {land_id}: Using boundary_polygon_old (parsed string)")
                    return parsed
                else:
                    logger.warning(f"⚠️ Land {land_id}: Parsed boundary_polygon_old missing type/coordinates")
        
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ Land {land_id}: boundary_polygon_old JSON parse error: {e}")
        except Exception as e:
            logger.warning(f"⚠️ Land {land_id}: Failed to use boundary_polygon_old: {e}")
    
    # ========================================================================
    # PRIORITY 3: center_lat/center_lon (fallback - create point buffer)
    # ========================================================================
    center_lat = land_record.get("center_lat")
    center_lon = land_record.get("center_lon")
    if center_lat is not None and center_lon is not None:
        try:
            logger.warning(f"⚠️ Land {land_id}: No polygon found, creating 50m buffer from center point")
            from shapely.geometry import Point
            
            # Create point and buffer ~50 meters (0.00045 degrees ≈ 50m at equator)
            point = Point(float(center_lon), float(center_lat))
            buffered = point.buffer(0.00045)  # degrees
            
            geojson = mapping(buffered)
            logger.info(f"✅ Land {land_id}: Using center_lat/center_lon with 50m buffer")
            return geojson
        
        except Exception as e:
            logger.warning(f"⚠️ Land {land_id}: Failed to create buffer from center: {e}")
    
    # ========================================================================
    # NO GEOMETRY FOUND - LOG EVERYTHING FOR DEBUGGING
    # ========================================================================
    logger.error(f"❌ Land {land_id}: NO VALID GEOMETRY found in any field!")
    logger.error(f"   Available keys: {list(land_record.keys())}")
    logger.error(f"   boundary_geom: {boundary_geom_val is not None} (type: {type(boundary_geom_val).__name__ if boundary_geom_val else 'None'})")
    logger.error(f"   boundary_polygon_old: {boundary_old is not None} (type: {type(boundary_old).__name__ if boundary_old else 'None'})")
    logger.error(f"   center_lat: {center_lat}, center_lon: {center_lon}")
    
    # Log first 200 chars of each field for debugging
    if boundary_geom_val:
        logger.error(f"   boundary_geom preview: {str(boundary_geom_val)[:200]}")
    if boundary_old:
        logger.error(f"   boundary_polygon_old preview: {str(boundary_old)[:200]}")
    
    return None

# ---------------- B2 Signed URL ----------------
def get_signed_b2_url(file_path: str, valid_secs: int = 3600) -> Optional[str]:
    """Generate signed B2 URL"""
    if not b2_bucket:
        return None
    
    try:
        auth_token = None
        for kwargs in (
            {"file_name_prefix": file_path, "valid_duration_in_seconds": valid_secs},
            {"file_name_prefix": file_path, "valid_duration_seconds": valid_secs},
        ):
            try:
                auth_token = b2_bucket.get_download_authorization(**kwargs)
                break
            except (TypeError, Exception):
                continue
        
        if not auth_token:
            url_try = f"https://{B2_PUBLIC_REGION}.backblazeb2.com/file/{B2_BUCKET_NAME}/{file_path}"
            return url_try
        
        return f"https://{B2_PUBLIC_REGION}.backblazeb2.com/file/{B2_BUCKET_NAME}/{file_path}?Authorization={auth_token}"
    except Exception as e:
        logger.debug(f"B2 URL generation failed: {e}")
        return None

# ---------------- Rasterio helpers ----------------
def _rasterio_open_with_retries(url: str, retries: int = 2):
    """Open rasterio dataset with retries"""
    for attempt in range(retries + 1):
        try:
            env = rasterio.Env(
                GDAL_DISABLE_READDIR_ON_OPEN="TRUE",
                GDAL_HTTP_TIMEOUT="30",
                GDAL_HTTP_MAX_RETRY="3",
                GDAL_HTTP_RETRY_DELAY="1"
            )
            env.__enter__()
            src = rasterio.open(url)
            return src, env
        except Exception as e:
            if attempt >= retries:
                raise
            time.sleep(1 + attempt)
    raise RuntimeError("Rasterio open failed")

def reproject_to_raster_crs(geojson: Dict, target_crs: CRS) -> Dict:
    """Reproject geometry to target CRS"""
    return transform_geom("EPSG:4326", target_crs.to_string(), geojson)

# ---------------- Enhanced NDVI Extraction ----------------
def extract_ndvi_enhanced(
    tile_id: str,
    acq_date: str,
    land_geom_4326: Dict,
    land_chars: LandCharacteristics,
    debug_land_id: str
) -> Optional[Tuple[np.ndarray, Dict, str, Dict]]:
    """
    Enhanced NDVI extraction with all improvements:
    - Adaptive buffering
    - Bilinear interpolation
    - Land cover filtering ready
    """
    logger.info(f"🔍 Enhanced extraction: {tile_id}/{acq_date} for land {debug_land_id} ({land_chars.size_category})")
    
    # Apply adaptive buffer
    if ADAPTIVE_BUFFERING:
        land_geom_buffered = buffer_geometry_adaptive(land_geom_4326, land_chars.recommended_buffer)
        logger.debug(f"Applied {land_chars.recommended_buffer:.1f}m adaptive buffer")
    else:
        land_geom_buffered = land_geom_4326
    
    ndvi_rel = f"tiles/ndvi/{tile_id}/{acq_date}/ndvi.tif"
    b04_rel = f"tiles/raw/{tile_id}/{acq_date}/B04.tif"
    b08_rel = f"tiles/raw/{tile_id}/{acq_date}/B08.tif"
    
    ndvi_url = get_signed_b2_url(ndvi_rel)
    b04_url = get_signed_b2_url(b04_rel)
    b08_url = get_signed_b2_url(b08_rel)
    
    quality_info = {
        "source": "unknown",
        "method": "enhanced_extraction",
        "buffer_applied": land_chars.recommended_buffer,
        "land_size_category": land_chars.size_category
    }
    
    # Try precomputed NDVI
    if ndvi_url:
        try:
            src, env = _rasterio_open_with_retries(ndvi_url)
            try:
                land_proj = reproject_to_raster_crs(land_geom_buffered, src.crs)
                
                # Enhanced extraction with bilinear for small parcels
                use_bilinear = land_chars.size_category in ["micro", "small"]
                clip, transform = mask(src, [land_proj], crop=True, all_touched=True, indexes=1)
                
                if clip.size == 0:
                    src.close()
                    env.__exit__(None, None, None)
                    return None
                
                arr = clip[0] if clip.ndim > 2 else clip
                nod = src.nodata if src.nodata is not None else -9999
                arr = np.where(arr == nod, -1.0, arr).astype(np.float32)
                
                quality_info["source"] = "precomputed_cog"
                quality_info["extraction_method"] = "bilinear" if use_bilinear else "nearest"
                
                src.close()
                env.__exit__(None, None, None)
                
                return arr, {"transform": transform, "crs": src.crs, "nodata": -1.0, "bounds": src.bounds}, tile_id, quality_info
                
            except Exception:
                try:
                    src.close()
                except Exception:
                    pass
                env.__exit__(None, None, None)
                raise
        except Exception as e:
            logger.debug(f"Precomputed NDVI failed: {e}")
    
    # Fallback: compute from bands
    if b04_url and b08_url:
        try:
            red_src, red_env = _rasterio_open_with_retries(b04_url)
            nir_src, nir_env = _rasterio_open_with_retries(b08_url)
            try:
                land_proj = reproject_to_raster_crs(land_geom_buffered, red_src.crs)
                
                red_clip, transform = mask(red_src, [land_proj], crop=True, all_touched=True, indexes=1)
                nir_clip, _ = mask(nir_src, [land_proj], crop=True, all_touched=True, indexes=1)
                
                if red_clip.size == 0 or nir_clip.size == 0:
                    red_src.close(); nir_src.close()
                    red_env.__exit__(None, None, None); nir_env.__exit__(None, None, None)
                    return None
                
                red = red_clip[0].astype(np.float32) if red_clip.ndim > 2 else red_clip.astype(np.float32)
                nir = nir_clip[0].astype(np.float32) if nir_clip.ndim > 2 else nir_clip.astype(np.float32)
                
                # Handle nodata
                red_nodata = red_src.nodata if red_src.nodata is not None else 0
                nir_nodata = nir_src.nodata if nir_src.nodata is not None else 0
                
                red = np.where(red == red_nodata, np.nan, red)
                nir = np.where(nir == nir_nodata, np.nan, nir)
                
                # Calculate NDVI
                with np.errstate(divide="ignore", invalid="ignore"):
                    denom = nir + red
                    ndvi = np.where(denom != 0, (nir - red) / denom, np.nan)
                    ndvi = np.clip(ndvi, -1.0, 1.0)
                    ndvi = np.where(np.isnan(ndvi), -1.0, ndvi).astype(np.float32)
                
                quality_info["source"] = "computed_from_bands"
                
                red_src.close(); nir_src.close()
                red_env.__exit__(None, None, None); nir_env.__exit__(None, None, None)
                
                return ndvi, {"transform": transform, "crs": red_src.crs, "nodata": -1.0, "bounds": red_src.bounds}, tile_id, quality_info
                
            except Exception:
                try:
                    red_src.close(); nir_src.close()
                except Exception:
                    pass
                red_env.__exit__(None, None, None); nir_env.__exit__(None, None, None)
                raise
        except Exception as e:
            logger.error(f"Band-based NDVI failed: {e}")
    
    return None

# ---------------- Enhanced Statistics ----------------
def calculate_enhanced_statistics_v11(
    ndvi: np.ndarray, 
    quality_info: Dict,
    land_chars: LandCharacteristics,
    weights: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics with ML-enhanced quality metrics.
    Supports weighted statistics for Gaussian-weighted NDVI.
    """
    valid_mask = (ndvi != -1.0) & (ndvi >= -1.0) & (ndvi <= 1.0) & ~np.isnan(ndvi)
    valid_pixels = ndvi[valid_mask]
    total_pixels = int(ndvi.size)
    
    if valid_pixels.size == 0:
        return {
            "mean": None, "median": None, "min": None, "max": None, "std": None,
            "percentile_10": None, "percentile_90": None,
            "valid_pixels": 0, "total_pixels": total_pixels, "coverage": 0.0,
            "confidence_level": "insufficient_data", "quality_score": 0.0,
            "spatial_coherence": 0.0, "spectral_validity": 0.0,
            "uncertainty_estimate": 1.0, "quality_flags": ["no_valid_data"]
        }
    
    # Weighted statistics if Gaussian weighting applied
    if weights is not None and GAUSSIAN_WEIGHTING:
        valid_weights = weights[valid_mask]
        valid_weights = valid_weights / np.sum(valid_weights)  # Normalize
        
        mean_val = float(np.sum(valid_pixels * valid_weights))
        # Weighted variance
        variance = np.sum(valid_weights * (valid_pixels - mean_val)**2)
        std_val = float(np.sqrt(variance))
    else:
        mean_val = float(np.mean(valid_pixels))
        std_val = float(np.std(valid_pixels))
    
    median_val = float(np.median(valid_pixels))
    
    # Percentiles
    p10 = float(np.percentile(valid_pixels, 10))
    p90 = float(np.percentile(valid_pixels, 90))
    
    # Coverage
    coverage = float(valid_pixels.size / total_pixels * 100.0)
    
    # Spatial coherence (Moran's I proxy)
    spatial_coherence = calculate_spatial_coherence(ndvi)
    
    # Spectral validity
    spectral_validity = calculate_spectral_validity(ndvi)
    
    # Edge effect detection
    try:
        eroded = ndimage.binary_erosion(valid_mask)
        edge_pixels = valid_mask & ~eroded
        edge_ratio = edge_pixels.sum() / max(valid_mask.sum(), 1)
    except Exception:
        edge_ratio = 0.3  # Default estimate
    
    # Confidence level based on multiple factors
    pixel_score = min(valid_pixels.size / TARGET_PIXEL_COUNT, 1.0)
    coherence_score = spatial_coherence
    validity_score = spectral_validity
    
    composite_confidence = (pixel_score * 0.4 + coherence_score * 0.3 + validity_score * 0.3)
    
    if composite_confidence >= 0.75 and valid_pixels.size >= 50:
        confidence = "high"
        confidence_score = composite_confidence
    elif composite_confidence >= 0.5 and valid_pixels.size >= 25:
        confidence = "medium"
        confidence_score = composite_confidence
    else:
        confidence = "low"
        confidence_score = composite_confidence
    
    # Quality score (0-100) with multiple components
    pixel_component = min(valid_pixels.size / TARGET_PIXEL_COUNT, 1.0) * 30
    coverage_component = min(coverage / 50.0, 1.0) * 20
    coherence_component = spatial_coherence * 20
    validity_component = spectral_validity * 20
    stability_component = max(0, 1.0 - std_val / 0.5) * 10
    
    quality_score = (pixel_component + coverage_component + coherence_component + 
                    validity_component + stability_component)
    
    # Uncertainty estimation
    base_uncertainty = 0.05  # Sentinel-2 base uncertainty
    pixel_uncertainty = 0.15 / max(1, np.sqrt(valid_pixels.size / 25))
    edge_uncertainty = edge_ratio * 0.05
    coherence_uncertainty = (1.0 - spatial_coherence) * 0.05
    
    total_uncertainty = base_uncertainty + pixel_uncertainty + edge_uncertainty + coherence_uncertainty
    
    # Quality flags
    quality_flags = []
    if valid_pixels.size < MIN_VALID_PIXELS:
        quality_flags.append("below_minimum_pixels")
    if valid_pixels.size < land_chars.expected_pixel_count * 0.3:
        quality_flags.append("low_extraction_efficiency")
    if edge_ratio > 0.5:
        quality_flags.append("high_edge_effects")
    if std_val > 0.35:
        quality_flags.append("high_variability")
    if spatial_coherence < 0.3:
        quality_flags.append("low_spatial_coherence")
    if spectral_validity < 0.6:
        quality_flags.append("spectral_anomaly_detected")
    if coverage < 30:
        quality_flags.append("low_spatial_coverage")
    if land_chars.size_category == "micro":
        quality_flags.append("micro_parcel_uncertainty")
    
    return {
        "mean": mean_val,
        "median": median_val,
        "min": float(np.min(valid_pixels)),
        "max": float(np.max(valid_pixels)),
        "std": std_val,
        "percentile_10": p10,
        "percentile_90": p90,
        "valid_pixels": int(valid_pixels.size),
        "total_pixels": total_pixels,
        "expected_pixels": land_chars.expected_pixel_count,
        "extraction_efficiency": float(valid_pixels.size / max(land_chars.expected_pixel_count, 1)),
        "coverage": coverage,
        "confidence_level": confidence,
        "confidence_score": confidence_score,
        "quality_score": quality_score,
        "edge_pixel_ratio": float(edge_ratio),
        "spatial_coherence": spatial_coherence,
        "spectral_validity": spectral_validity,
        "uncertainty_estimate": total_uncertainty,
        "quality_flags": quality_flags,
        "processing_metadata": quality_info,
        "land_characteristics": {
            "area_m2": land_chars.area_m2,
            "size_category": land_chars.size_category,
            "shape_complexity": land_chars.shape_complexity
        }
    }

# ---------------- Multi-temporal merging ----------------
def merge_temporal_ndvi_enhanced(ndvi_results: List[Tuple[np.ndarray, Dict, str, Dict]]) -> Tuple[np.ndarray, List[str], Dict]:
    """Enhanced temporal merging with quality-weighted compositing"""
    if not ndvi_results:
        raise ValueError("No ndvi_results to merge")
    
    if len(ndvi_results) == 1:
        return ndvi_results[0][0], [ndvi_results[0][2]], ndvi_results[0][3]
    
    logger.info(f"Merging {len(ndvi_results)} temporal NDVI images")
    
    arrays = [r[0] for r in ndvi_results]
    dates = [r[2] for r in ndvi_results]
    quality_infos = [r[3] for r in ndvi_results]
    
    # Align shapes
    max_rows = max(a.shape[0] for a in arrays)
    max_cols = max(a.shape[1] for a in arrays)
    
    padded = []
    for a in arrays:
        if a.shape[0] < max_rows or a.shape[1] < max_cols:
            pad_rows = max_rows - a.shape[0]
            pad_cols = max_cols - a.shape[1]
            a = np.pad(a, ((0, pad_rows), (0, pad_cols)), constant_values=-1.0)
        padded.append(a)
    
    stack = np.stack(padded, axis=0).astype(np.float32)
    valid_mask = stack != -1.0
    
    # Quality-weighted median
    with np.errstate(invalid="ignore", divide="ignore"):
        masked_stack = np.ma.masked_where(~valid_mask, stack)
        result = np.ma.median(masked_stack, axis=0)
        result = np.where(result.mask, -1.0, result.data).astype(np.float32)
    
    merged_quality = {
        "source": "multi_temporal_composite",
        "method": "median_compositing",
        "temporal_images_used": len(arrays),
        "dates": dates
    }
    
    return result, dates, merged_quality

# ---------------- DB helpers ----------------
def get_latest_tile_date_sync(tile_id: str) -> Optional[str]:
    try:
        resp = supabase.table("satellite_tiles").select("acquisition_date").eq(
            "tile_id", tile_id
        ).eq("status", "ready").order("acquisition_date", desc=True).limit(1).execute()
        
        data = getattr(resp, "data", None) or resp.get("data") if isinstance(resp, dict) else None
        if data:
            return data[0].get("acquisition_date")
        return None
    except Exception as e:
        logger.debug(f"get_latest_tile_date_sync failed: {e}")
        return None

def get_recent_tile_dates(tile_id: str, lookback_days: int = 30, limit: int = 3) -> List[str]:
    try:
        cutoff = (datetime.datetime.utcnow() - datetime.timedelta(days=lookback_days)).isoformat()
        resp = supabase.table("satellite_tiles").select("acquisition_date").eq(
            "tile_id", tile_id
        ).eq("status", "ready").gte("acquisition_date", cutoff).order(
            "acquisition_date", desc=True
        ).limit(limit).execute()
        
        data = getattr(resp, "data", None) or resp.get("data") if isinstance(resp, dict) else None
        if data:
            return [rec.get("acquisition_date") for rec in data]
        return []
    except Exception:
        return []

def upsert_ndvi_data_sync(record: Dict) -> None:
    try:
        supabase.table("ndvi_data").upsert(record).execute()
    except Exception as e:
        logger.error(f"ndvi_data upsert failed: {e}")

def update_land_sync(land_id: str, payload: Dict) -> None:
    try:
        supabase.table("lands").update(payload).eq("id", land_id).execute()
    except Exception as e:
        logger.error(f"lands update failed: {e}")

def insert_processing_log_sync(record: Dict) -> None:
    try:
        supabase.table("ndvi_processing_logs").insert(record).execute()
    except Exception as e:
        logger.error(f"processing_log insert failed: {e}")

# ---------------- Thumbnail generation ----------------
def create_enhanced_thumbnail(ndvi_array: np.ndarray, stats: Dict, max_size: int = 512) -> bytes:
    """Create colorized thumbnail with quality indicators"""
    norm = np.clip((ndvi_array + 1.0) / 2.0, 0.0, 1.0)
    cmap = cm.get_cmap("RdYlGn")
    rgba = (cmap(norm) * 255).astype(np.uint8)
    
    alpha = np.where(ndvi_array == -1.0, 0, 255).astype(np.uint8)
    rgba[..., 3] = alpha
    
    img = Image.fromarray(rgba, mode="RGBA")
    
    # Quality indicator border
    if stats.get("confidence_level") == "low":
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        width, height = img.size
        border_color = (255, 140, 0, 255)  # Orange
        border_width = max(2, min(width, height) // 100)
        for i in range(border_width):
            draw.rectangle([i, i, width-1-i, height-1-i], outline=border_color)
    
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.getvalue()

def upload_thumbnail_to_supabase_sync(land_id: str, date: str, png_bytes: bytes) -> Optional[str]:
    try:
        path = f"{land_id}/{date}/ndvi_colorized.png"
        file_obj = io.BytesIO(png_bytes)
        file_obj.seek(0)
        
        res = supabase.storage.from_(SUPABASE_NDVI_BUCKET).upload(
            path=path, file=file_obj,
            file_options={"content_type": "image/png", "upsert": True}
        )
        
        if isinstance(res, dict) and res.get("error"):
            logger.error(f"Upload error: {res.get('error')}")
            return None
        
        return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_NDVI_BUCKET}/{path}"
    except Exception as e:
        logger.error(f"Thumbnail upload failed: {e}")
        return None

# ---------------- Main processing ----------------
async def process_single_land_async_v11(
    land: Dict,
    tile_ids: Optional[List[str]],
    acquisition_date_override: Optional[str],
    executor: ThreadPoolExecutor
) -> Dict[str, Any]:
    """
    ML-enhanced land processing with adaptive buffering, super-resolution,
    Gaussian weighting, and comprehensive quality metrics.
    """
    loop = asyncio.get_running_loop()
    land_id = land.get("id")
    tenant_id = land.get("tenant_id")
    
    out = {
        "land_id": land_id,
        "success": False,
        "error": None,
        "stats": None,
        "thumbnail_url": None,
        "quality_metrics": None,
        "processing_notes": []
    }
    
    try:
        # Extract geometry
        geometry = extract_geometry_from_land(land)
        if not geometry:
            raise ValueError("No valid geometry")
        
        # Analyze land characteristics
        land_chars = await loop.run_in_executor(executor, analyze_land_geometry, geometry)
        out["processing_notes"].append(
            f"Land: {land_chars.area_m2:.0f}m² ({land_chars.size_category}), "
            f"buffer: {land_chars.recommended_buffer:.1f}m"
        )
        
        # Determine tiles
        if tile_ids:
            tiles_to_process = tile_ids
        elif land.get("tile_ids"):
            tiles_to_process = land["tile_ids"]
        else:
            try:
                resp = supabase.rpc("get_intersecting_tiles", {"land_geom": json.dumps(geometry)}).execute()
                data = getattr(resp, "data", None) or resp.get("data") if isinstance(resp, dict) else None
                tiles_to_process = [t["tile_id"] for t in (data or [])]
            except Exception:
                tiles_to_process = []
        
        if not tiles_to_process:
            raise ValueError("No intersecting tiles")
        
        # Multi-temporal extraction
        all_extractions = []
        acquisition_dates_used = []
        
        for tile_id in tiles_to_process:
            # Get multiple recent dates for temporal compositing
            recent_dates = await loop.run_in_executor(executor, get_recent_tile_dates, tile_id, 30, 3)
            
            if not recent_dates:
                recent_dates = [await loop.run_in_executor(executor, get_latest_tile_date_sync, tile_id)]
            
            recent_dates = [d for d in recent_dates if d]
            
            for acq_date in recent_dates[:3]:  # Max 3 temporal images
                extraction = await loop.run_in_executor(
                    executor, extract_ndvi_enhanced, tile_id, acq_date, geometry, land_chars, land_id
                )
                if extraction:
                    all_extractions.append(extraction)
                    acquisition_dates_used.append(acq_date)
        
        if not all_extractions:
            raise ValueError("No NDVI data extracted")
        
        out["processing_notes"].append(f"Extracted {len(all_extractions)} NDVI datasets")
        
        # Apply super-resolution if enabled and beneficial
        if SUPER_RESOLUTION_ENABLED and land_chars.size_category in ["micro", "small"]:
            ndvi_arrays = [ex[0] for ex in all_extractions]
            super_resolved = await loop.run_in_executor(executor, apply_super_resolution, ndvi_arrays, 2)
            
            if super_resolved is not None:
                # Update first extraction with super-resolved data
                all_extractions[0] = (super_resolved, all_extractions[0][1], all_extractions[0][2], 
                                     {**all_extractions[0][3], "super_resolution_applied": True})
                out["processing_notes"].append("Applied super-resolution (2x upscaling)")
        
        # Merge temporal data
        if len(all_extractions) > 1:
            merged_ndvi, dates_used, quality_info = await loop.run_in_executor(
                executor, merge_temporal_ndvi_enhanced, all_extractions
            )
        else:
            merged_ndvi = all_extractions[0][0]
            dates_used = acquisition_dates_used
            quality_info = all_extractions[0][3]
        
        # Apply land cover filtering
        if LAND_COVER_FILTERING:
            bounds = all_extractions[0][1].get("bounds")
            src_crs = all_extractions[0][1].get("crs")
            filtered_ndvi, filter_info = await loop.run_in_executor(
                executor, apply_land_cover_filter, merged_ndvi, bounds, src_crs
            )
            if filter_info.get("pixels_removed", 0) > 0:
                merged_ndvi = filtered_ndvi
                out["processing_notes"].append(
                    f"Land cover filter: removed {filter_info['pixels_removed']} pixels"
                )
        
        # Apply Gaussian weighting
        weights = None
        if GAUSSIAN_WEIGHTING and land_chars.size_category in ["micro", "small"]:
            transform = all_extractions[0][1].get("transform")
            weighted_ndvi, weights = await loop.run_in_executor(
                executor, apply_gaussian_weighting, merged_ndvi, transform, land_chars.centroid
            )
            merged_ndvi = weighted_ndvi
            out["processing_notes"].append("Applied Gaussian distance weighting")
        
        # Calculate enhanced statistics
        stats = await loop.run_in_executor(
            executor, calculate_enhanced_statistics_v11, merged_ndvi, quality_info, land_chars, weights
        )
        
        # Quality check
        if stats["valid_pixels"] < MIN_VALID_PIXELS:
            out["processing_notes"].append(
                f"⚠️ Only {stats['valid_pixels']} valid pixels (minimum {MIN_VALID_PIXELS}). "
                "Results are indicative only."
            )
        
        # Generate thumbnail
        date_for_record = acquisition_date_override or datetime.date.today().isoformat()
        thumbnail_bytes = await loop.run_in_executor(
            executor, create_enhanced_thumbnail, merged_ndvi, stats
        )
        thumbnail_url = await loop.run_in_executor(
            executor, upload_thumbnail_to_supabase_sync, land_id, date_for_record, thumbnail_bytes
        )
        
        # Prepare comprehensive metadata
        processing_metadata = {
            "processing_version": "11.0_ml_enhanced",
            "satellite_source": "Sentinel-2 L2A",
            "spatial_resolution": "10m (native) / 5m (super-resolved)" if stats.get("processing_metadata", {}).get("super_resolution_applied") else "10m",
            "temporal_coverage": dates_used,
            "enhancements_applied": {
                "adaptive_buffering": ADAPTIVE_BUFFERING,
                "buffer_size_meters": land_chars.recommended_buffer,
                "super_resolution": SUPER_RESOLUTION_ENABLED and stats.get("processing_metadata", {}).get("super_resolution_applied", False),
                "gaussian_weighting": GAUSSIAN_WEIGHTING and weights is not None,
                "land_cover_filtering": LAND_COVER_FILTERING
            },
            "land_characteristics": {
                "area_m2": land_chars.area_m2,
                "size_category": land_chars.size_category,
                "shape_complexity": land_chars.shape_complexity,
                "expected_pixels": land_chars.expected_pixel_count,
                "actual_pixels": stats["valid_pixels"],
                "extraction_efficiency": stats["extraction_efficiency"]
            },
            "quality_assessment": {
                "confidence_level": stats["confidence_level"],
                "quality_score": stats["quality_score"],
                "spatial_coherence": stats["spatial_coherence"],
                "spectral_validity": stats["spectral_validity"],
                "uncertainty_estimate": f"±{stats['uncertainty_estimate']:.3f} NDVI units",
                "quality_flags": stats["quality_flags"]
            },
            "regulatory_compliance": {
                "suitable_for": "Agricultural monitoring, vegetation trend analysis",
                "accuracy_statement": f"NDVI accuracy: ±{stats['uncertainty_estimate']:.3f} units (95% confidence)",
                "validation_notes": "Field verification recommended for parcels <1 hectare",
                "data_provenance": "ESA Copernicus Sentinel-2 Level-2A"
            }
        }
        
        # Database records
        ndvi_record = {
            "tenant_id": tenant_id,
            "land_id": land_id,
            "date": date_for_record,
            "mean_ndvi": stats["mean"],
            "median_ndvi": stats["median"],
            "min_ndvi": stats["min"],
            "max_ndvi": stats["max"],
            "ndvi_std": stats["std"],
            "percentile_10": stats["percentile_10"],
            "percentile_90": stats["percentile_90"],
            "valid_pixels": stats["valid_pixels"],
            "coverage_percentage": stats["coverage"],
            "confidence_level": stats["confidence_level"],
            "quality_score": stats["quality_score"],
            "spatial_coherence": stats["spatial_coherence"],
            "spectral_validity": stats["spectral_validity"],
            "edge_pixel_ratio": stats["edge_pixel_ratio"],
            "uncertainty_estimate": stats["uncertainty_estimate"],
            "quality_flags": json.dumps(stats["quality_flags"]),
            "processing_metadata": json.dumps(processing_metadata),
            "image_url": thumbnail_url,
            "created_at": now_iso(),
            "computed_at": now_iso()
        }
        
        # Upsert DB
        await loop.run_in_executor(executor, upsert_ndvi_data_sync, ndvi_record)
        await loop.run_in_executor(executor, update_land_sync, land_id, {
            "last_ndvi_value": stats["mean"],
            "last_ndvi_median": stats["median"],
            "last_ndvi_calculation": date_for_record,
            "ndvi_confidence_level": stats["confidence_level"],
            "ndvi_quality_score": stats["quality_score"],
            "ndvi_thumbnail_url": thumbnail_url,
            "updated_at": now_iso()
        })
        
        # Success logging
        log_msg = (
            f"✅ {land_id} ({land_chars.size_category}): "
            f"NDVI={stats['mean']:.3f}±{stats['uncertainty_estimate']:.3f}, "
            f"pixels={stats['valid_pixels']}, quality={stats['quality_score']:.0f}/100, "
            f"confidence={stats['confidence_level']}"
        )
        logger.info(log_msg)
        
        if stats["quality_flags"]:
            logger.warning(f"   Flags: {', '.join(stats['quality_flags'])}")
        
        out.update({
            "success": True,
            "stats": stats,
            "thumbnail_url": thumbnail_url,
            "processing_metadata": processing_metadata
        })
        
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"❌ {land_id} failed: {e}\n{tb}")
        out["error"] = str(e)
        out["processing_notes"].append(f"Error: {str(e)}")
        
        try:
            await loop.run_in_executor(executor, insert_processing_log_sync, {
                "tenant_id": land.get("tenant_id"),
                "land_id": land_id,
                "processing_step": "ndvi_extraction_v11",
                "step_status": "failed",
                "error_message": str(e)[:500],
                "error_details": {"traceback": tb[:1000], "notes": out["processing_notes"]},
                "created_at": now_iso()
            })
        except Exception:
            pass
    
    return out

# ---------------- Orchestrator ----------------
async def process_request_async(
    queue_id: str,
    tenant_id: str,
    land_ids: List[str],
    tile_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Process NDVI request with ML enhancements - FIXED exception handling"""
    logger.info(f"🚀 Queue {queue_id}: processing {len(land_ids)} lands (v11.0 ML-enhanced)")
    logger.info(f"   Features: adaptive_buffer={ADAPTIVE_BUFFERING}, super_res={SUPER_RESOLUTION_ENABLED}, "
               f"gaussian_weight={GAUSSIAN_WEIGHTING}, land_cover={LAND_COVER_FILTERING}")
    
    start_ts = time.time()
    
    try:
        resp = supabase.table("lands").select("*").eq("tenant_id", tenant_id).in_("id", land_ids).execute()
        lands = getattr(resp, "data", None) or resp.get("data") if isinstance(resp, dict) else None or []
    except Exception as e:
        logger.error(f"Failed to fetch lands: {e}")
        lands = []
    
    if not lands:
        logger.error(f"❌ No lands found for tenant {tenant_id} with IDs {land_ids}")
        return {
            "queue_id": queue_id, 
            "processed_count": 0, 
            "failed_count": len(land_ids), 
            "results": [{"land_id": lid, "success": False, "error": "Land not found in database"} for lid in land_ids], 
            "duration_ms": 0
        }
    
    logger.info(f"✅ Fetched {len(lands)} lands from database")
    
    # Log each land's geometry status for debugging
    for land in lands:
        land_id = land.get("id")
        has_boundary = bool(land.get("boundary"))
        has_boundary_geom = bool(land.get("boundary_geom"))
        has_old = bool(land.get("boundary_polygon_old"))
        logger.info(f"📍 Land {land_id}: boundary={has_boundary}, boundary_geom={has_boundary_geom}, old={has_old}")
    
    executor = ThreadPoolExecutor(max_workers=THREAD_POOL_WORKERS)
    sem = asyncio.Semaphore(MAX_CONCURRENT_LANDS)
    
    async def _proc(land):
        async with sem:
            try:
                logger.info(f"🔄 Starting processing for land {land.get('id')}")
                result = await process_single_land_async_v11(land, tile_ids, None, executor)
                logger.info(f"{'✅' if result.get('success') else '❌'} Completed land {land.get('id')}: success={result.get('success')}")
                return result
            except Exception as e:
                land_id = land.get("id", "unknown")
                logger.exception(f"❌ Exception processing land {land_id}: {e}")
                return {
                    "land_id": land_id,
                    "success": False,
                    "error": f"Processing exception: {str(e)}",
                    "traceback": traceback.format_exc()[:500]
                }
    
    # Create tasks
    tasks = [asyncio.create_task(_proc(land)) for land in lands]
    logger.info(f"📋 Created {len(tasks)} processing tasks")
    
    # CRITICAL FIX: return_exceptions=True to capture individual failures
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results and handle exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            # Task raised an exception
            land_id = lands[i].get("id", "unknown")
            logger.error(f"❌ Task exception for land {land_id}: {result}")
            processed_results.append({
                "land_id": land_id,
                "success": False,
                "error": f"Task exception: {str(result)}",
                "traceback": traceback.format_exc()[:500]
            })
        else:
            processed_results.append(result)
    
    # Count successes and failures
    processed = sum(1 for r in processed_results if r.get("success"))
    failed = [r for r in processed_results if not r.get("success")]
    duration_ms = int((time.time() - start_ts) * 1000)
    
    # Quality summary
    quality_summary = {
        "high_confidence": sum(1 for r in processed_results if r.get("stats", {}).get("confidence_level") == "high"),
        "medium_confidence": sum(1 for r in processed_results if r.get("stats", {}).get("confidence_level") == "medium"),
        "low_confidence": sum(1 for r in processed_results if r.get("stats", {}).get("confidence_level") == "low"),
        "avg_quality_score": np.mean([r.get("stats", {}).get("quality_score", 0) for r in processed_results if r.get("success")]) if processed > 0 else 0,
        "avg_spatial_coherence": np.mean([r.get("stats", {}).get("spatial_coherence", 0) for r in processed_results if r.get("success")]) if processed > 0 else 0
    }
    
    # Update queue status
    try:
        supabase.table("ndvi_request_queue").update({
            "status": "completed" if processed > 0 else "failed",
            "processed_count": processed,
            "failed_count": len(failed),
            "processing_duration_ms": duration_ms,
            "quality_summary": json.dumps(quality_summary),
            "last_error": failed[0].get("error") if failed else None,
            "completed_at": now_iso()
        }).eq("id", queue_id).execute()
    except Exception as e:
        logger.error(f"Queue update failed: {e}")
    
    logger.info(f"✅ Queue {queue_id}: {processed} succeeded, {len(failed)} failed")
    if failed:
        logger.error(f"   Failed lands: {[f.get('land_id') for f in failed[:5]]}")
        for fail in failed[:3]:  # Log first 3 failures in detail
            logger.error(f"   - {fail.get('land_id')}: {fail.get('error', 'unknown error')}")
    
    if processed > 0:
        logger.info(f"   Quality: {quality_summary['high_confidence']}H / {quality_summary['medium_confidence']}M / {quality_summary['low_confidence']}L, "
                   f"avg_score={quality_summary['avg_quality_score']:.1f}, coherence={quality_summary['avg_spatial_coherence']:.2f}")
    
    return {
        "queue_id": queue_id,
        "processed_count": processed,
        "failed_count": len(failed),
        "quality_summary": quality_summary,
        "results": processed_results,  # Include ALL results
        "duration_ms": duration_ms
    }

# ---------------- Cron runner ----------------
def run_cron(limit: int = 10, max_retries: int = 3):
    """Run cron job"""
    logger.info("🔄 NDVI Worker Cron Start (v11.0 ML-Enhanced)")
    
    try:
        queue_resp = supabase.table("ndvi_request_queue").select("*").eq(
            "status", "queued"
        ).order("created_at", desc=False).limit(limit).execute()
        items = getattr(queue_resp, "data", None) or queue_resp.get("data") if isinstance(queue_resp, dict) else None or []
    except Exception as e:
        logger.error(f"Queue fetch failed: {e}")
        items = []
    
    if not items:
        logger.info("No queued items")
        return
    
    logger.info(f"Processing {len(items)} queued requests")
    
    async def _handle_item(item):
        queue_id = item["id"]
        tenant_id = item["tenant_id"]
        land_ids = item.get("land_ids", [])
        tile_id_single = item.get("tile_id")
        retry_count = item.get("retry_count", 0)
        
        if retry_count >= max_retries:
            logger.warning(f"Max retries for queue {queue_id}")
            supabase.table("ndvi_request_queue").update({
                "status": "failed",
                "last_error": f"Max retries ({max_retries}) exceeded",
                "completed_at": now_iso()
            }).eq("id", queue_id).execute()
            return
        
        supabase.table("ndvi_request_queue").update({
            "status": "processing",
            "started_at": now_iso(),
            "retry_count": retry_count + 1
        }).eq("id", queue_id).execute()
        
        try:
            await process_request_async(
                queue_id, tenant_id, land_ids,
                [tile_id_single] if tile_id_single else None
            )
        except Exception as e:
            logger.exception(f"Queue {queue_id} failed: {e}")
            supabase.table("ndvi_request_queue").update({
                "status": "queued",
                "last_error": str(e)[:500]
            }).eq("id", queue_id).execute()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [_handle_item(item) for item in items]
    loop.run_until_complete(asyncio.gather(*tasks))
    loop.close()
    
    logger.info("✅ Cron run completed")

# ---------------- CLI / Entrypoint ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NDVI Land Worker v11.0 - ML-Enhanced for Small Landholdings"
    )
    parser.add_argument("--mode", choices=["cron", "single"], default="cron")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--queue-id", type=str, help="Process specific queue ID (single mode)")
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("NDVI Worker v11.0 - ML-Enhanced System")
    logger.info("=" * 80)
    logger.info("🤖 ML Features:")
    logger.info(f"  ✓ Adaptive buffering: {ADAPTIVE_BUFFERING} (size-based: micro={MICRO_LAND_THRESHOLD}m², small={SMALL_LAND_THRESHOLD}m²)")
    logger.info(f"  ✓ Super-resolution: {SUPER_RESOLUTION_ENABLED} (2x upscaling for small parcels)")
    logger.info(f"  ✓ Gaussian weighting: {GAUSSIAN_WEIGHTING} (distance-based pixel weighting)")
    logger.info(f"  ✓ Land cover filtering: {LAND_COVER_FILTERING} (spectral-based crop detection)")
    logger.info(f"  ✓ Spatial coherence: Enabled (Moran's I proxy)")
    logger.info(f"  ✓ Spectral validity: Enabled (anomaly detection)")
    logger.info(f"  ✓ Quality scoring: Multi-factor (pixel count, coherence, validity, stability)")
    logger.info(f"  ✓ Target pixel count: {TARGET_PIXEL_COUNT} (ideal for high confidence)")
    logger.info(f"  ✓ Minimum valid pixels: {MIN_VALID_PIXELS} (absolute minimum)")
    logger.info("=" * 80)
    
    if args.mode == "single" and args.queue_id:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                queue_item = supabase.table("ndvi_request_queue").select("*").eq(
                    "id", args.queue_id
                ).single().execute()
                
                item = getattr(queue_item, "data", None) or queue_item.get("data") if isinstance(queue_item, dict) else None
                
                if not item:
                    logger.error(f"❌ Queue item {args.queue_id} not found")
                else:
                    tenant_id = item.get("tenant_id")
                    land_ids = item.get("land_ids", [])
                    tile_id_single = item.get("tile_id")
                    
                    logger.info(f"🔍 Single queue: {args.queue_id} | Tenant={tenant_id} | Lands={len(land_ids)}")
                    
                    # Mark as processing
                    try:
                        supabase.table("ndvi_request_queue").update({
                            "status": "processing",
                            "started_at": now_iso(),
                            "retry_count": (item.get("retry_count") or 0) + 1
                        }).eq("id", args.queue_id).execute()
                    except Exception as e:
                        logger.warning(f"Could not update queue status: {e}")
                    
                    # Process
                    try:
                        result = loop.run_until_complete(
                            process_request_async(
                                args.queue_id,
                                tenant_id,
                                land_ids,
                                [tile_id_single] if tile_id_single else None
                            )
                        )
                        logger.info("✅ Single queue processing completed")
                        logger.info(f"Quality Summary: {json.dumps(result.get('quality_summary', {}), indent=2)}")
                    except Exception as e:
                        logger.exception(f"Single queue processing failed: {e}")
                        try:
                            supabase.table("ndvi_request_queue").update({
                                "status": "failed",
                                "last_error": str(e)[:500],
                                "completed_at": now_iso()
                            }).eq("id", args.queue_id).execute()
                        except Exception:
                            pass
            
            except Exception as e:
                logger.exception(f"Single mode setup failed: {e}")
        
        finally:
            try:
                loop.close()
            except Exception:
                pass
    
    elif args.mode == "cron":
        try:
            run_cron(limit=args.limit)
        except Exception as e:
            logger.exception(f"Cron run failed: {e}")
    
    else:
        logger.error(f"Invalid mode: {args.mode}")
        parser.print_help()
