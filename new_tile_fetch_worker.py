"""
Production-ready FastAPI for Sentinel-2 Satellite Tile Management
- Fetches tiles from Microsoft Planetary Computer
- Stores compressed TIFFs in B2 Cloud Storage
- Validates and prevents duplicate downloads
- Computes NDVI statistics
- Enterprise-grade error handling and logging
"""

import os
import json
import time
import math
import tempfile
import datetime
import logging
import traceback
import threading
import asyncio
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import requests
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely import wkb, wkt
from shapely.geometry import mapping, shape
import planetary_computer as pc
from supabase import create_client
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from requests.adapters import HTTPAdapter, Retry

# ===================== Configuration =====================
class Config:
    # Database
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    
    # B2 Cloud Storage
    B2_KEY_ID = os.getenv("B2_KEY_ID", "")
    B2_APP_KEY = os.getenv("B2_APP_KEY", "")
    B2_BUCKET_NAME = os.getenv("B2_BUCKET_RAW", "kisanshakti-ndvi-tiles")
    B2_PREFIX = os.getenv("B2_PREFIX", "tiles/")
    
    # Planetary Computer
    MPC_COLLECTION = os.getenv("MPC_COLLECTION", "sentinel-2-l2a")
    MPC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    
    # Processing defaults
    CLOUD_COVER_DEFAULT = int(os.getenv("DEFAULT_CLOUD_COVER_MAX", "80"))
    LOOKBACK_DAYS_DEFAULT = int(os.getenv("MAX_SCENE_LOOKBACK_DAYS", "90"))
    TIFF_COMPRESSION = os.getenv("TIFF_COMPRESSION", "LZW")
    BLOCK_ROWS = int(os.getenv("BLOCK_ROWS", "1024"))
    
    # B2 Upload
    B2_UPLOAD_RETRIES = int(os.getenv("B2_UPLOAD_RETRIES", "3"))
    B2_LARGE_THRESHOLD = int(os.getenv("B2_LARGE_THRESHOLD_BYTES", str(100 * 1024 * 1024)))
    
    # Concurrency
    MAX_CONCURRENT_TILES = int(os.getenv("MAX_CONCURRENT_TILES", "3"))
    THREAD_POOL_SIZE = int(os.getenv("THREAD_POOL_SIZE", "5"))

config = Config()

# ===================== Logging Setup =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("satellite_tile_api")

# ===================== Global Clients =====================
supabase_client = None
b2_api = None
b2_bucket = None
http_session = None
executor = None

def initialize_clients():
    """Initialize all external service clients"""
    global supabase_client, b2_api, b2_bucket, http_session, executor
    
    try:
        # Supabase
        supabase_client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
        logger.info("✅ Supabase client initialized")
        
        # B2 Cloud Storage
        info = InMemoryAccountInfo()
        b2_api = B2Api(info)
        b2_api.authorize_account("production", config.B2_KEY_ID, config.B2_APP_KEY)
        b2_bucket = b2_api.get_bucket_by_name(config.B2_BUCKET_NAME)
        logger.info(f"✅ B2 bucket connected: {config.B2_BUCKET_NAME}")
        
        # HTTP Session with retry logic
        http_session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        http_session.mount("https://", HTTPAdapter(max_retries=retries))
        logger.info("✅ HTTP session initialized")
        
        # Thread pool for concurrent processing
        executor = ThreadPoolExecutor(max_workers=config.THREAD_POOL_SIZE)
        logger.info(f"✅ Thread pool initialized: {config.THREAD_POOL_SIZE} workers")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize clients: {e}")
        raise

def cleanup_clients():
    """Cleanup all client connections"""
    global executor
    if executor:
        executor.shutdown(wait=True)
        logger.info("✅ Thread pool shut down")

# ===================== Utility Functions =====================
def safe_float(v, decimals=None):
    """Safely convert value to float with optional rounding"""
    try:
        if v is None:
            return None
        f = float(v)
        return round(f, decimals) if decimals is not None else f
    except Exception:
        return None

def to_int(v):
    """Safely convert value to integer"""
    try:
        return int(v) if v is not None else None
    except Exception:
        return None

def decode_geometry_to_geojson(row: Dict) -> Optional[Dict]:
    """Decode geometry from various formats to GeoJSON"""
    geom = row.get("geojson_geometry") or row.get("geometry")
    if not geom:
        return None
    
    try:
        # Already a dict/GeoJSON
        if isinstance(geom, dict):
            return geom
        
        # Binary WKB
        if isinstance(geom, (bytes, bytearray)):
            return mapping(wkb.loads(geom))
        
        # String (JSON or WKT)
        if isinstance(geom, str):
            s = geom.strip()
            # Try JSON parse
            if s.startswith("{"):
                return json.loads(s)
            # Try WKT
            try:
                return mapping(wkt.loads(s))
            except:
                # Try hex-encoded WKB
                return mapping(wkb.loads(bytes.fromhex(s)))
    
    except Exception as e:
        logger.warning(f"Failed to decode geometry: {e}")
    
    return None

def extract_bbox_polygon(geom: Dict) -> Optional[Dict]:
    """Extract bounding box as polygon from geometry"""
    if not geom:
        return None
    
    try:
        s = shape(geom)
        minx, miny, maxx, maxy = s.bounds
        return {
            "type": "Polygon",
            "coordinates": [[
                [minx, miny],
                [maxx, miny],
                [maxx, maxy],
                [minx, maxy],
                [minx, miny]
            ]]
        }
    except Exception as e:
        logger.warning(f"Failed to extract bbox: {e}")
        return None

# ===================== B2 Storage Operations =====================
class B2StorageManager:
    """Manage B2 Cloud Storage operations"""
    
    @staticmethod
    def get_b2_path(tile_id: str, acq_date: str, subdir: str, filename: str) -> str:
        """Generate B2 storage path"""
        return f"{config.B2_PREFIX}{subdir}/{tile_id}/{acq_date}/{filename}"
    
    @staticmethod
    def check_file_exists(b2_path: str) -> tuple:
        """Check if file exists in B2 and return size"""
        try:
            file_info = b2_bucket.get_file_info_by_name(b2_path)
            return True, int(getattr(file_info, "size", 0))
        except Exception:
            return False, None
    
    @staticmethod
    def upload_with_retry(local_path: str, b2_path: str) -> Optional[int]:
        """Upload file to B2 with retry logic"""
        for attempt in range(1, config.B2_UPLOAD_RETRIES + 1):
            try:
                if not os.path.exists(local_path):
                    raise Exception(f"Local file missing: {local_path}")
                
                size = os.path.getsize(local_path)
                if size < 1024:
                    raise Exception(f"File too small ({size} bytes): {local_path}")
                
                # Upload to B2
                b2_bucket.upload_local_file(local_path, b2_path)
                logger.info(f"✅ Uploaded {b2_path} ({size / 1024 / 1024:.2f} MB)")
                return size
                
            except Exception as e:
                logger.warning(f"Upload attempt {attempt} failed: {e}")
                if attempt < config.B2_UPLOAD_RETRIES:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"❌ Upload failed after {config.B2_UPLOAD_RETRIES} attempts: {b2_path}")
        
        return None
    
    @staticmethod
    def delete_file(b2_path: str):
        """Delete file from B2"""
        try:
            file_info = b2_bucket.get_file_info_by_name(b2_path)
            b2_api.delete_file_version(file_info.id_, b2_path)
            logger.info(f"🗑️ Deleted {b2_path}")
        except Exception as e:
            logger.warning(f"Failed to delete {b2_path}: {e}")

b2_storage = B2StorageManager()

# ===================== TIFF Validation =====================
def verify_tiff_validity(path: str) -> bool:
    """Verify TIFF file is valid and readable"""
    if not os.path.exists(path):
        raise Exception(f"TIFF file not found: {path}")
    
    size = os.path.getsize(path)
    if size < 1024:
        raise Exception(f"TIFF file too small ({size} bytes): {path}")
    
    try:
        with rasterio.open(path) as src:
            _ = src.count
            _ = src.height
            _ = src.width
        return True
    except Exception as e:
        raise Exception(f"Corrupt TIFF file {path}: {e}")

# ===================== NDVI Calculation =====================
def compute_ndvi_with_stats(red_path: str, nir_path: str, output_profile: Dict) -> tuple:
    """
    Compute NDVI from RED and NIR bands with streaming processing
    Returns: (ndvi_path, statistics_dict)
    """
    ndvi_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    ndvi_tmp.close()
    
    try:
        # Verify input TIFFs
        verify_tiff_validity(red_path)
        verify_tiff_validity(nir_path)
        
        # Prepare output profile
        profile = output_profile.copy()
        profile.update(
            dtype=rasterio.float32,
            count=1,
            compress=config.TIFF_COMPRESSION,
            tiled=True,
            blockxsize=256,
            blockysize=256
        )
        
        # Initialize statistics
        total_pixels = 0
        valid_pixels = 0
        vegetation_pixels = 0
        sum_ndvi = 0.0
        sum_squares = 0.0
        min_ndvi = float('inf')
        max_ndvi = float('-inf')
        
        # Process in blocks for memory efficiency
        with rasterio.open(red_path) as red_src, \
             rasterio.open(nir_path) as nir_src, \
             rasterio.open(ndvi_tmp.name, 'w', **profile) as dst:
            
            for row_start in range(0, red_src.height, config.BLOCK_ROWS):
                row_count = min(config.BLOCK_ROWS, red_src.height - row_start)
                window = Window(0, row_start, red_src.width, row_count)
                
                # Read blocks
                red_block = red_src.read(1, window=window).astype('float32')
                nir_block = nir_src.read(1, window=window).astype('float32')
                
                # Calculate NDVI
                with np.errstate(divide='ignore', invalid='ignore'):
                    denominator = nir_block + red_block
                    ndvi_block = np.where(
                        denominator != 0,
                        (nir_block - red_block) / denominator,
                        np.nan
                    )
                
                # Write NDVI block
                dst.write(ndvi_block.astype(np.float32), 1, window=window)
                
                # Update statistics
                valid_mask = np.isfinite(ndvi_block)
                valid_count = valid_mask.sum()
                total_pixels += ndvi_block.size
                
                if valid_count > 0:
                    valid_values = ndvi_block[valid_mask]
                    valid_pixels += valid_count
                    sum_ndvi += valid_values.sum()
                    sum_squares += (valid_values ** 2).sum()
                    min_ndvi = min(min_ndvi, valid_values.min())
                    max_ndvi = max(max_ndvi, valid_values.max())
                    
                    # Vegetation threshold: NDVI > 0.3
                    vegetation_pixels += (valid_values > 0.3).sum()
        
        # Calculate final statistics
        if valid_pixels == 0:
            stats = {
                "ndvi_min": None,
                "ndvi_max": None,
                "ndvi_mean": None,
                "ndvi_std_dev": None,
                "vegetation_health_score": None,
                "vegetation_coverage_percent": 0.0,
                "data_completeness_percent": 0.0,
                "pixel_count": total_pixels,
                "valid_pixel_count": 0
            }
        else:
            mean = sum_ndvi / valid_pixels
            variance = (sum_squares / valid_pixels) - (mean ** 2)
            std_dev = math.sqrt(max(0, variance))
            
            veg_coverage = (vegetation_pixels / valid_pixels) * 100
            data_completeness = (valid_pixels / total_pixels) * 100
            
            # Health score: weighted combination
            # - NDVI mean (normalized 0-100): 50%
            # - Vegetation coverage: 30%
            # - Data completeness: 20%
            health_score = (
                ((mean + 1) / 2 * 100) * 0.5 +
                veg_coverage * 0.3 +
                data_completeness * 0.2
            )
            
            stats = {
                "ndvi_min": safe_float(min_ndvi, 3),
                "ndvi_max": safe_float(max_ndvi, 3),
                "ndvi_mean": safe_float(mean, 3),
                "ndvi_std_dev": safe_float(std_dev, 3),
                "vegetation_coverage_percent": safe_float(veg_coverage, 2),
                "data_completeness_percent": safe_float(data_completeness, 2),
                "pixel_count": to_int(total_pixels),
                "valid_pixel_count": to_int(valid_pixels),
                "vegetation_health_score": safe_float(health_score, 2)
            }
        
        return ndvi_tmp.name, stats
        
    except Exception as e:
        logger.error(f"NDVI calculation failed: {e}")
        if os.path.exists(ndvi_tmp.name):
            os.unlink(ndvi_tmp.name)
        return None, None

# ===================== Database Operations =====================
class DatabaseManager:
    """Manage Supabase database operations"""
    
    @staticmethod
    def fetch_agri_tiles(
        tile_ids: Optional[List[str]] = None,
        country_id: Optional[str] = None,
        state: Optional[str] = None,
        district: Optional[str] = None
    ) -> List[Dict]:
        """Fetch agricultural MGRS tiles from database"""
        try:
            query = supabase_client.table("mgrs_tiles").select(
                "id, tile_id, geometry, geojson_geometry, country_id, state, district"
            ).eq("is_agri", True).eq("is_land_contain", True)
            
            if tile_ids:
                query = query.in_("tile_id", tile_ids)
            if country_id:
                query = query.eq("country_id", country_id)
            if state:
                query = query.eq("state", state)
            if district:
                query = query.eq("district", district)
            
            response = query.execute()
            tiles = response.data or []
            logger.info(f"📍 Fetched {len(tiles)} agricultural tiles")
            return tiles
            
        except Exception as e:
            logger.error(f"Failed to fetch tiles: {e}")
            return []
    
    @staticmethod
    def check_tile_exists(tile_id: str, acquisition_date: str) -> bool:
        """Check if tile already exists for acquisition date"""
        try:
            response = supabase_client.table("satellite_tiles").select("id").eq(
                "tile_id", tile_id
            ).eq("acquisition_date", acquisition_date).execute()
            
            exists = len(response.data or []) > 0
            if exists:
                logger.info(f"⏭️ Tile {tile_id} for {acquisition_date} already exists")
            return exists
            
        except Exception as e:
            logger.warning(f"Error checking tile existence: {e}")
            return False
    
    @staticmethod
    def get_latest_acquisition(tile_id: str) -> Optional[str]:
        """Get latest acquisition date for a tile"""
        try:
            response = supabase_client.table("satellite_tiles").select(
                "acquisition_date"
            ).eq("tile_id", tile_id).order(
                "acquisition_date", desc=True
            ).limit(1).execute()
            
            if response.data:
                return response.data[0]["acquisition_date"]
            return None
            
        except Exception as e:
            logger.warning(f"Error fetching latest acquisition: {e}")
            return None
    
    @staticmethod
    def insert_satellite_tile(payload: Dict) -> bool:
        """Insert or update satellite tile record"""
        try:
            supabase_client.table("satellite_tiles").upsert(
                payload,
                on_conflict="tile_id,acquisition_date,collection"
            ).execute()
            
            logger.info(f"💾 Saved tile: {payload['tile_id']} - {payload['acquisition_date']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save tile: {e}")
            return False
    
    @staticmethod
    def update_mgrs_tile_status(mgrs_id: str):
        """Update MGRS tile last NDVI update timestamp"""
        try:
            now = datetime.datetime.utcnow().isoformat() + "Z"
            supabase_client.table("mgrs_tiles").update({
                "last_ndvi_update": now,
                "is_ndvi_ready": True
            }).eq("id", mgrs_id).execute()
            
            logger.info(f"✅ Updated MGRS tile status: {mgrs_id}")
            
        except Exception as e:
            logger.warning(f"Failed to update MGRS status: {e}")

db_manager = DatabaseManager()

# ===================== Tile Processing =====================
def process_single_tile(
    tile: Dict,
    cloud_cover: int,
    lookback_days: int
) -> Dict:
    """
    Process a single MGRS tile:
    1. Fetch geometry and bounds
    2. Search Planetary Computer for new imagery
    3. Download RED and NIR bands
    4. Calculate NDVI with statistics
    5. Upload to B2 Cloud Storage
    6. Save metadata to database
    """
    tile_id = tile["tile_id"]
    mgrs_id = tile["id"]
    
    red_tmp = None
    nir_tmp = None
    ndvi_path = None
    
    try:
        logger.info(f"🛰️ Processing tile: {tile_id}")
        
        # Decode geometry
        geom = decode_geometry_to_geojson(tile)
        if not geom:
            raise Exception("No valid geometry found")
        
        bbox_polygon = extract_bbox_polygon(geom)
        if not bbox_polygon:
            raise Exception("Failed to extract bbox")
        
        # Define date range
        today = datetime.date.today()
        start_date = (today - datetime.timedelta(days=lookback_days)).isoformat()
        end_date = today.isoformat()
        
        # Search Planetary Computer
        search_payload = {
            "collections": [config.MPC_COLLECTION],
            "intersects": geom,
            "datetime": f"{start_date}/{end_date}",
            "query": {
                "eo:cloud_cover": {"lt": cloud_cover}
            }
        }
        
        response = http_session.post(
            config.MPC_API_URL,
            json=search_payload,
            timeout=60
        )
        response.raise_for_status()
        
        scenes = response.json().get("features", [])
        if not scenes:
            logger.info(f"📭 No new scenes for {tile_id}")
            return {
                "tile_id": tile_id,
                "status": "no_scenes",
                "message": "No imagery found in date range"
            }
        
        # Sort by cloud cover (lowest first)
        scene = sorted(scenes, key=lambda s: s["properties"]["eo:cloud_cover"])[0]
        
        # Extract scene metadata
        acquisition_date = scene["properties"]["datetime"].split("T")[0]
        cloud_pct = scene["properties"].get("eo:cloud_cover", 0)
        assets = scene["assets"]
        
        # Check if already exists
        if db_manager.check_tile_exists(tile_id, acquisition_date):
            return {
                "tile_id": tile_id,
                "status": "exists",
                "acquisition_date": acquisition_date,
                "message": "Tile already downloaded"
            }
        
        # Get signed URLs for RED and NIR bands
        red_asset = assets.get("red") or assets.get("B04")
        nir_asset = assets.get("nir") or assets.get("B08")
        
        if not red_asset or not nir_asset:
            raise Exception("RED or NIR band not found in scene")
        
        red_url = pc.sign(red_asset["href"])
        nir_url = pc.sign(nir_asset["href"])
        
        # Download RED band
        logger.info(f"⬇️ Downloading RED band for {tile_id}")
        red_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        red_tmp.close()
        
        with http_session.get(red_url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(red_tmp.name, 'wb') as f:
                for chunk in r.iter_content(chunk_size=512 * 1024):
                    f.write(chunk)
        
        # Download NIR band
        logger.info(f"⬇️ Downloading NIR band for {tile_id}")
        nir_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        nir_tmp.close()
        
        with http_session.get(nir_url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(nir_tmp.name, 'wb') as f:
                for chunk in r.iter_content(chunk_size=512 * 1024):
                    f.write(chunk)
        
        # Verify downloads
        verify_tiff_validity(red_tmp.name)
        verify_tiff_validity(nir_tmp.name)
        
        # Get raster profile
        with rasterio.open(red_tmp.name) as src:
            profile = src.profile.copy()
        
        # Calculate NDVI
        logger.info(f"🧮 Calculating NDVI for {tile_id}")
        ndvi_path, ndvi_stats = compute_ndvi_with_stats(red_tmp.name, nir_tmp.name, profile)
        
        if not ndvi_path or not ndvi_stats:
            raise Exception("NDVI calculation failed")
        
        # Upload to B2
        logger.info(f"☁️ Uploading to B2 for {tile_id}")
        red_b2_path = b2_storage.get_b2_path(tile_id, acquisition_date, "raw", "B04.tif")
        nir_b2_path = b2_storage.get_b2_path(tile_id, acquisition_date, "raw", "B08.tif")
        ndvi_b2_path = b2_storage.get_b2_path(tile_id, acquisition_date, "ndvi", "ndvi.tif")
        
        red_size = b2_storage.upload_with_retry(red_tmp.name, red_b2_path)
        nir_size = b2_storage.upload_with_retry(nir_tmp.name, nir_b2_path)
        ndvi_size = b2_storage.upload_with_retry(ndvi_path, ndvi_b2_path)
        
        if not all([red_size, nir_size, ndvi_size]):
            raise Exception("Failed to upload one or more files to B2")
        
        total_size_mb = safe_float((red_size + nir_size + ndvi_size) / (1024 * 1024), 2)
        
        # Prepare database payload
        now = datetime.datetime.utcnow().isoformat() + "Z"
        
        payload = {
            "tile_id": tile_id,
            "acquisition_date": acquisition_date,
            "collection": config.MPC_COLLECTION.upper(),
            "cloud_cover": safe_float(cloud_pct, 2),
            "file_size_mb": total_size_mb,
            "red_band_path": f"b2://{config.B2_BUCKET_NAME}/{red_b2_path}",
            "nir_band_path": f"b2://{config.B2_BUCKET_NAME}/{nir_b2_path}",
            "ndvi_path": f"b2://{config.B2_BUCKET_NAME}/{ndvi_b2_path}",
            "red_band_size_bytes": red_size,
            "nir_band_size_bytes": nir_size,
            "ndvi_size_bytes": ndvi_size,
            "status": "ready",
            "actual_download_status": "downloaded",
            "processing_stage": "completed",
            "processing_level": "L2A",
            "resolution": "R10m",
            "processing_method": "cog_streaming",
            "api_source": "planetary_computer",
            "updated_at": now,
            "processing_completed_at": now,
            "ndvi_calculation_timestamp": now,
            "bbox": json.dumps(bbox_polygon),
            "country_id": tile.get("country_id"),
            "mgrs_tile_id": mgrs_id
        }
        
        # Add NDVI statistics
        if ndvi_stats:
            payload.update(ndvi_stats)
        
        # Save to database
        if not db_manager.insert_satellite_tile(payload):
            raise Exception("Failed to save tile to database")
        
        # Update MGRS tile status
        db_manager.update_mgrs_tile_status(mgrs_id)
        
        logger.info(f"✅ Successfully processed {tile_id} - {acquisition_date}")
        
        return {
            "tile_id": tile_id,
            "status": "success",
            "acquisition_date": acquisition_date,
            "cloud_cover": cloud_pct,
            "file_size_mb": total_size_mb,
            "ndvi_stats": ndvi_stats
        }
        
    except Exception as e:
        error_msg = f"Failed to process {tile_id}: {str(e)}"
        logger.error(f"❌ {error_msg}")
        logger.debug(traceback.format_exc())
        
        return {
            "tile_id": tile_id,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        
    finally:
        # Cleanup temporary files
        for tmp_file in [red_tmp, nir_tmp, ndvi_path]:
            if tmp_file:
                path = tmp_file.name if hasattr(tmp_file, 'name') else tmp_file
                if isinstance(path, str) and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {path}: {e}")

# ===================== FastAPI Application =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting Satellite Tile API...")
    initialize_clients()
    yield
    # Shutdown
    logger.info("🛑 Shutting down Satellite Tile API...")
    cleanup_clients()

app = FastAPI(
    title="Satellite Tile Management API",
    description="Production API for Sentinel-2 tile downloading, processing, and management",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ===================== Request Models =====================
class TileProcessRequest(BaseModel):
    tile_ids: Optional[List[str]] = Field(None, description="Specific MGRS tile IDs")
    country_id: Optional[str] = Field(None, description="Filter by country ID")
    state: Optional[str] = Field(None, description="Filter by state")
    district: Optional[str] = Field(None, description="Filter by district")
    cloud_cover: int = Field(80, ge=0, le=100, description="Maximum cloud cover percentage")
    lookback_days: int = Field(90, ge=1, le=365, description="Days to look back for imagery")
    async_mode: bool = Field(True, description="Run in background (async)")

class TileStatusRequest(BaseModel):
    tile_ids: Optional[List[str]] = Field(None, description="Filter by tile IDs")
    acquisition_date: Optional[str] = Field(None, description="Filter by acquisition date")
    status: Optional[str] = Field(None, description="Filter by status")
    limit: int = Field(100, ge=1, le=1000)

class BatchProcessResult(BaseModel):
    status: str
    total_tiles: int
    successful: int
    failed: int
    no_scenes: int
    already_exists: int
    results: List[Dict]

# ===================== API Endpoints =====================

@app.get("/", tags=["General"])
async def root():
    """API root with service information"""
    return {
        "service": "Satellite Tile Management API",
        "version": "2.0.0",
        "description": "Download and process Sentinel-2 tiles from Microsoft Planetary Computer",
        "endpoints": {
            "GET /": "Service information",
            "GET /health": "Health check",
            "POST /tiles/process": "Process tiles (download, compute NDVI, upload to B2)",
            "GET /tiles/status": "Query tile status",
            "GET /tiles/{tile_id}": "Get specific tile information",
            "DELETE /tiles/{tile_id}/{acquisition_date}": "Delete tile data"
        },
        "storage": {
            "backend": "B2 Cloud Storage",
            "bucket": config.B2_BUCKET_NAME,
            "compression": config.TIFF_COMPRESSION
        },
        "planetary_computer": {
            "collection": config.MPC_COLLECTION,
            "api": config.MPC_API_URL
        }
    }

@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Test database connection
        db_manager.fetch_agri_tiles(tile_ids=["TEST"])
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    try:
        # Test B2 connection
        b2_bucket.get_bucket_info()
        b2_status = "healthy"
    except Exception as e:
        b2_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "ok",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "services": {
            "database": db_status,
            "b2_storage": b2_status,
            "http_session": "healthy"
        }
    }

@app.post("/tiles/process", response_model=Dict, tags=["Tile Processing"])
async def process_tiles(
    request: TileProcessRequest,
    background_tasks: BackgroundTasks
):
    """
    Process satellite tiles: download, compute NDVI, and upload to B2
    
    This endpoint will:
    - Fetch MGRS tiles from database based on filters
    - Search Microsoft Planetary Computer for new imagery
    - Download RED and NIR bands
    - Calculate NDVI with comprehensive statistics
    - Compress and upload to B2 Cloud Storage
    - Save metadata to satellite_tiles table
    - Skip tiles that already exist for the same acquisition date
    
    Args:
        request: TileProcessRequest with filters and parameters
        
    Returns:
        Processing status and results
    """
    try:
        logger.info(f"📥 Received process request: {request.dict()}")
        
        # Fetch tiles
        tiles = db_manager.fetch_agri_tiles(
            tile_ids=request.tile_ids,
            country_id=request.country_id,
            state=request.state,
            district=request.district
        )
        
        if not tiles:
            raise HTTPException(
                status_code=404,
                detail="No agricultural tiles found matching the criteria"
            )
        
        logger.info(f"📍 Found {len(tiles)} tiles to process")
        
        # Async mode - run in background
        if request.async_mode:
            def background_job():
                results = []
                for tile in tiles:
                    result = process_single_tile(
                        tile,
                        request.cloud_cover,
                        request.lookback_days
                    )
                    results.append(result)
                
                # Summarize results
                success = sum(1 for r in results if r.get("status") == "success")
                failed = sum(1 for r in results if r.get("status") == "failed")
                no_scenes = sum(1 for r in results if r.get("status") == "no_scenes")
                exists = sum(1 for r in results if r.get("status") == "exists")
                
                logger.info(
                    f"✨ Background processing complete: "
                    f"{success} successful, {failed} failed, "
                    f"{no_scenes} no scenes, {exists} already exist"
                )
            
            # Start background thread
            thread = threading.Thread(target=background_job, daemon=True)
            thread.start()
            
            return JSONResponse(
                status_code=202,
                content={
                    "status": "started",
                    "message": "Processing started in background",
                    "total_tiles": len(tiles),
                    "parameters": {
                        "cloud_cover": request.cloud_cover,
                        "lookback_days": request.lookback_days
                    }
                }
            )
        
        # Synchronous mode - process and wait
        else:
            results = []
            
            # Process tiles with concurrency control
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            with ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_TILES) as pool:
                futures = {
                    pool.submit(
                        process_single_tile,
                        tile,
                        request.cloud_cover,
                        request.lookback_days
                    ): tile for tile in tiles
                }
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        tile = futures[future]
                        logger.error(f"Exception processing tile {tile['tile_id']}: {e}")
                        results.append({
                            "tile_id": tile["tile_id"],
                            "status": "failed",
                            "error": str(e)
                        })
            
            # Summarize results
            summary = {
                "status": "completed",
                "total_tiles": len(tiles),
                "successful": sum(1 for r in results if r.get("status") == "success"),
                "failed": sum(1 for r in results if r.get("status") == "failed"),
                "no_scenes": sum(1 for r in results if r.get("status") == "no_scenes"),
                "already_exists": sum(1 for r in results if r.get("status") == "exists"),
                "results": results
            }
            
            return JSONResponse(content=summary)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Process tiles failed: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tiles/status", tags=["Tile Query"])
async def get_tiles_status(
    tile_ids: Optional[str] = None,
    acquisition_date: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100
):
    """
    Query satellite tiles status from database
    
    Args:
        tile_ids: Comma-separated list of tile IDs
        acquisition_date: Filter by acquisition date (YYYY-MM-DD)
        status: Filter by processing status
        limit: Maximum results to return
        
    Returns:
        List of tile records
    """
    try:
        query = supabase_client.table("satellite_tiles").select("*")
        
        if tile_ids:
            tile_list = [t.strip() for t in tile_ids.split(",")]
            query = query.in_("tile_id", tile_list)
        
        if acquisition_date:
            query = query.eq("acquisition_date", acquisition_date)
        
        if status:
            query = query.eq("status", status)
        
        query = query.order("created_at", desc=True).limit(limit)
        
        response = query.execute()
        tiles = response.data or []
        
        return {
            "count": len(tiles),
            "tiles": tiles
        }
    
    except Exception as e:
        logger.error(f"Failed to query tiles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tiles/{tile_id}", tags=["Tile Query"])
async def get_tile_info(tile_id: str):
    """
    Get detailed information for a specific MGRS tile
    
    Args:
        tile_id: MGRS tile identifier
        
    Returns:
        Tile information including all acquisitions
    """
    try:
        # Get MGRS tile info
        mgrs_response = supabase_client.table("mgrs_tiles").select("*").eq(
            "tile_id", tile_id
        ).execute()
        
        if not mgrs_response.data:
            raise HTTPException(status_code=404, detail=f"MGRS tile {tile_id} not found")
        
        mgrs_tile = mgrs_response.data[0]
        
        # Get all satellite tile acquisitions
        sat_response = supabase_client.table("satellite_tiles").select("*").eq(
            "tile_id", tile_id
        ).order("acquisition_date", desc=True).execute()
        
        acquisitions = sat_response.data or []
        
        return {
            "mgrs_tile": mgrs_tile,
            "acquisition_count": len(acquisitions),
            "acquisitions": acquisitions,
            "latest_acquisition": acquisitions[0] if acquisitions else None
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get tile info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/tiles/{tile_id}/{acquisition_date}", tags=["Tile Management"])
async def delete_tile(tile_id: str, acquisition_date: str):
    """
    Delete tile data from B2 and database
    
    Args:
        tile_id: MGRS tile identifier
        acquisition_date: Acquisition date (YYYY-MM-DD)
        
    Returns:
        Deletion status
    """
    try:
        # Get tile record
        response = supabase_client.table("satellite_tiles").select("*").eq(
            "tile_id", tile_id
        ).eq("acquisition_date", acquisition_date).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=404,
                detail=f"Tile {tile_id} for {acquisition_date} not found"
            )
        
        tile = response.data[0]
        
        # Delete from B2
        deleted_files = []
        for path_key in ["red_band_path", "nir_band_path", "ndvi_path"]:
            path = tile.get(path_key)
            if path and path.startswith("b2://"):
                b2_path = path.replace(f"b2://{config.B2_BUCKET_NAME}/", "")
                try:
                    b2_storage.delete_file(b2_path)
                    deleted_files.append(b2_path)
                except Exception as e:
                    logger.warning(f"Failed to delete {b2_path}: {e}")
        
        # Delete from database
        supabase_client.table("satellite_tiles").delete().eq(
            "tile_id", tile_id
        ).eq("acquisition_date", acquisition_date).execute()
        
        return {
            "status": "deleted",
            "tile_id": tile_id,
            "acquisition_date": acquisition_date,
            "deleted_files": deleted_files
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete tile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tiles/reprocess/{tile_id}/{acquisition_date}", tags=["Tile Management"])
async def reprocess_tile(tile_id: str, acquisition_date: str):
    """
    Reprocess existing tile (recalculate NDVI from stored bands)
    
    Args:
        tile_id: MGRS tile identifier
        acquisition_date: Acquisition date (YYYY-MM-DD)
        
    Returns:
        Reprocessing status
    """
    try:
        # Get tile record
        response = supabase_client.table("satellite_tiles").select("*").eq(
            "tile_id", tile_id
        ).eq("acquisition_date", acquisition_date).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=404,
                detail=f"Tile {tile_id} for {acquisition_date} not found"
            )
        
        tile = response.data[0]
        
        # Download bands from B2
        red_path = tile.get("red_band_path")
        nir_path = tile.get("nir_band_path")
        
        if not red_path or not nir_path:
            raise HTTPException(
                status_code=400,
                detail="Band paths not found in tile record"
            )
        
        # TODO: Implement B2 download and reprocessing
        # This would involve:
        # 1. Download RED and NIR from B2
        # 2. Recalculate NDVI
        # 3. Upload new NDVI
        # 4. Update database record
        
        return {
            "status": "not_implemented",
            "message": "Reprocessing feature coming soon"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reprocess tile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats/summary", tags=["Statistics"])
async def get_summary_stats():
    """
    Get summary statistics of all processed tiles
    
    Returns:
        Overall system statistics
    """
    try:
        # Total tiles processed
        total_response = supabase_client.table("satellite_tiles").select(
            "id", count="exact"
        ).execute()
        total_count = total_response.count or 0
        
        # Tiles by status
        status_response = supabase_client.table("satellite_tiles").select(
            "status"
        ).execute()
        
        status_counts = {}
        for tile in (status_response.data or []):
            status = tile.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Storage usage
        size_response = supabase_client.table("satellite_tiles").select(
            "file_size_mb"
        ).execute()
        
        total_size_mb = sum(
            float(t.get("file_size_mb") or 0)
            for t in (size_response.data or [])
        )
        
        # Recent acquisitions (last 7 days)
        week_ago = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()
        recent_response = supabase_client.table("satellite_tiles").select(
            "id", count="exact"
        ).gte("acquisition_date", week_ago).execute()
        recent_count = recent_response.count or 0
        
        return {
            "total_tiles": total_count,
            "status_breakdown": status_counts,
            "total_storage_mb": round(total_size_mb, 2),
            "total_storage_gb": round(total_size_mb / 1024, 2),
            "recent_acquisitions_7days": recent_count,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get summary stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tiles/ndvi-stats/{tile_id}", tags=["Statistics"])
async def get_ndvi_statistics(tile_id: str, limit: int = 10):
    """
    Get NDVI statistics history for a specific tile
    
    Args:
        tile_id: MGRS tile identifier
        limit: Number of recent acquisitions to return
        
    Returns:
        NDVI statistics over time
    """
    try:
        response = supabase_client.table("satellite_tiles").select(
            "acquisition_date, ndvi_min, ndvi_max, ndvi_mean, ndvi_std_dev, "
            "vegetation_health_score, vegetation_coverage_percent, cloud_cover"
        ).eq("tile_id", tile_id).order(
            "acquisition_date", desc=True
        ).limit(limit).execute()
        
        if not response.data:
            raise HTTPException(
                status_code=404,
                detail=f"No NDVI data found for tile {tile_id}"
            )
        
        return {
            "tile_id": tile_id,
            "count": len(response.data),
            "statistics": response.data
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get NDVI stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===================== Run Application =====================
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"🚀 Starting server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
