"""
NDVI Land Processing Worker
============================
Core worker for satellite data discovery, download, NDVI computation,
and overlay generation for agricultural lands.

Handles:
- Microsoft Planetary Computer STAC queries
- Satellite tile download (B04, B08 bands)
- NDVI computation and clipping to land polygons
- Colorized overlay generation for Google Maps
- Database updates and cloud storage uploads

Author: KisanShakti Team
Version: 1.0.0
"""

import os
import io
import json
import logging
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID

import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from rasterio.io import MemoryFile
import requests
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import shape, mapping, Polygon, box
from shapely.ops import transform
import pyproj
from b2sdk.v2 import B2Api, InMemoryAccountInfo
from supabase import create_client, Client
import planetary_computer as pc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NDVILandWorker:
    """
    Worker class for processing NDVI data for agricultural lands.
    """
    
    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        b2_key_id: str,
        b2_app_key: str,
        b2_bucket: str,
        mpc_stac_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
    ):
        """
        Initialize the NDVI worker.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase service role key
            b2_key_id: Backblaze B2 key ID
            b2_app_key: Backblaze B2 application key
            b2_bucket: B2 bucket name for raw tiles
            mpc_stac_url: Microsoft Planetary Computer STAC API URL
        """
        self.supabase = create_client(supabase_url, supabase_key)
        self.mpc_stac_url = mpc_stac_url
        self.b2_bucket_name = b2_bucket
        
        # Initialize B2
        self.b2_info = InMemoryAccountInfo()
        self.b2_api = B2Api(self.b2_info)
        self.b2_api.authorize_account("production", b2_key_id, b2_app_key)
        self.b2_bucket = self.b2_api.get_bucket_by_name(b2_bucket)
        
        # NDVI colormap (brown -> yellow -> green)
        self.ndvi_colormap = LinearSegmentedColormap.from_list(
            'ndvi',
            [
                (0.0, '#8B4513'),   # Brown (bare soil/rock)
                (0.2, '#D2B48C'),   # Tan
                (0.3, '#F4A460'),   # Sandy brown
                (0.4, '#FFFF00'),   # Yellow (sparse vegetation)
                (0.5, '#9ACD32'),   # Yellow-green
                (0.6, '#7CFC00'),   # Lawn green
                (0.7, '#32CD32'),   # Lime green
                (0.8, '#228B22'),   # Forest green
                (1.0, '#006400')    # Dark green (dense vegetation)
            ]
        )
        
        logger.info("NDVILandWorker initialized successfully")
    
    def process_land_ndvi(
        self,
        land_id: str,
        tenant_id: str,
        lookback_days: int = 15,
        max_cloud_cover: float = 20.0
    ) -> Dict[str, Any]:
        """
        Main processing pipeline for land NDVI.
        
        Args:
            land_id: UUID of the land to process
            tenant_id: UUID of the tenant (for multi-tenancy)
            lookback_days: Days to search back for imagery
            max_cloud_cover: Maximum acceptable cloud cover (%)
        
        Returns:
            Dictionary containing processing results
        """
        logger.info(f"Processing NDVI for land {land_id}, tenant {tenant_id}")
        
        try:
            # 1. Fetch land data
            land = self._fetch_land_data(land_id, tenant_id)
            
            # 2. Parse land polygon
            land_polygon = self._parse_land_boundary(land)
            
            # 3. Find or discover satellite tiles
            tiles = self._find_satellite_tiles(
                land_polygon, lookback_days, max_cloud_cover
            )
            
            if not tiles:
                raise ValueError(
                    f"No suitable satellite imagery found for land {land_id} "
                    f"in the last {lookback_days} days with cloud cover < {max_cloud_cover}%"
                )
            
            logger.info(f"Found {len(tiles)} suitable satellite tiles")
            
            # 4. Process each tile
            results = []
            for tile in tiles[:3]:  # Limit to 3 most recent tiles
                try:
                    result = self._process_tile(
                        tile, land, land_polygon, tenant_id
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process tile {tile['tile_id']}: {e}")
                    continue
            
            if not results:
                raise ValueError("Failed to process any tiles successfully")
            
            # 5. Aggregate results
            aggregated = self._aggregate_results(results, land_id, tenant_id)
            
            logger.info(f"NDVI processing completed for land {land_id}")
            return aggregated
        
        except Exception as e:
            logger.error(f"NDVI processing failed for land {land_id}: {e}", exc_info=True)
            raise
    
    def _fetch_land_data(self, land_id: str, tenant_id: str) -> Dict[str, Any]:
        """Fetch land data from database."""
        response = self.supabase.table("lands").select(
            "id, tenant_id, farmer_id, name, area_acres, boundary, "
            "center_lat, center_lon, tile_id, current_crop"
        ).eq("id", land_id).eq("tenant_id", tenant_id).execute()
        
        if not response.data:
            raise ValueError(f"Land {land_id} not found")
        
        return response.data[0]
    
    def _parse_land_boundary(self, land: Dict[str, Any]) -> Polygon:
        """Parse land boundary from GeoJSON to Shapely polygon."""
        boundary = land.get("boundary")
        
        if not boundary:
            raise ValueError("Land boundary not defined")
        
        # boundary is stored as PostGIS geography, returned as GeoJSON string or dict
        if isinstance(boundary, str):
            boundary = json.loads(boundary)
        
        # Handle different GeoJSON formats
        if "type" in boundary:
            geom = shape(boundary)
        elif "geometry" in boundary:
            geom = shape(boundary["geometry"])
        else:
            raise ValueError("Invalid boundary format")
        
        # Ensure it's a valid polygon
        if not isinstance(geom, Polygon):
            raise ValueError(f"Boundary must be a Polygon, got {type(geom)}")
        
        if not geom.is_valid:
            geom = geom.buffer(0)  # Fix invalid geometry
        
        return geom
    
    def _find_satellite_tiles(
        self,
        land_polygon: Polygon,
        lookback_days: int,
        max_cloud_cover: float
    ) -> List[Dict[str, Any]]:
        """
        Find suitable satellite tiles for the land polygon.
        First checks database, then queries STAC if needed.
        """
        # Get bounding box of land
        bounds = land_polygon.bounds  # (minx, miny, maxx, maxy)
        
        # Search date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Check existing tiles in database
        existing_tiles = self._check_existing_tiles(
            bounds, start_date, end_date, max_cloud_cover
        )
        
        if existing_tiles:
            logger.info(f"Found {len(existing_tiles)} existing tiles in database")
            return existing_tiles
        
        # Query STAC for new tiles
        logger.info("Querying Microsoft Planetary Computer STAC...")
        stac_tiles = self._query_stac(
            land_polygon, start_date, end_date, max_cloud_cover
        )
        
        if stac_tiles:
            logger.info(f"Found {len(stac_tiles)} tiles from STAC")
            # Store in database for future use
            self._store_satellite_tiles(stac_tiles)
        
        return stac_tiles
    
    def _check_existing_tiles(
        self,
        bounds: Tuple[float, float, float, float],
        start_date: datetime,
        end_date: datetime,
        max_cloud_cover: float
    ) -> List[Dict[str, Any]]:
        """Check for existing tiles in database that intersect the bounds."""
        try:
            response = self.supabase.table("satellite_tiles").select("*").gte(
                "acquisition_date", start_date.date().isoformat()
            ).lte(
                "acquisition_date", end_date.date().isoformat()
            ).lte(
                "cloud_cover", max_cloud_cover
            ).eq(
                "status", "completed"
            ).order("acquisition_date", desc=True).execute()
            
            # Filter by spatial intersection (if bbox_geom available)
            tiles = []
            land_box = box(*bounds)
            
            for tile in response.data:
                if tile.get("bbox"):
                    bbox = tile["bbox"]
                    # bbox format: [minx, miny, maxx, maxy]
                    if isinstance(bbox, list) and len(bbox) == 4:
                        tile_box = box(*bbox)
                        if tile_box.intersects(land_box):
                            tiles.append(tile)
                else:
                    # No bbox, include it
                    tiles.append(tile)
            
            return tiles
        
        except Exception as e:
            logger.error(f"Error checking existing tiles: {e}")
            return []
    
    def _query_stac(
        self,
        land_polygon: Polygon,
        start_date: datetime,
        end_date: datetime,
        max_cloud_cover: float
    ) -> List[Dict[str, Any]]:
        """Query Microsoft Planetary Computer STAC API."""
        # Convert polygon to GeoJSON
        geojson = mapping(land_polygon)
        
        # STAC query
        query = {
            "collections": ["sentinel-2-l2a"],
            "intersects": geojson,
            "datetime": f"{start_date.isoformat()}Z/{end_date.isoformat()}Z",
            "query": {
                "eo:cloud_cover": {"lt": max_cloud_cover}
            },
            "limit": 10
        }
        
        try:
            response = requests.post(self.mpc_stac_url, json=query, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            features = data.get("features", [])
            
            # Process STAC items into our tile format
            tiles = []
            for item in features:
                tile = self._parse_stac_item(item)
                if tile:
                    tiles.append(tile)
            
            return tiles
        
        except Exception as e:
            logger.error(f"STAC query failed: {e}")
            return []
    
    def _parse_stac_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse STAC item into our tile format."""
        try:
            properties = item.get("properties", {})
            assets = item.get("assets", {})
            
            # Get tile ID from s2:mgrs_tile property
            tile_id = properties.get("s2:mgrs_tile") or properties.get("grid:code")
            
            if not tile_id:
                logger.warning("No tile ID found in STAC item")
                return None
            
            # Get acquisition date
            acq_date_str = properties.get("datetime")
            acq_date = datetime.fromisoformat(acq_date_str.replace("Z", ""))
            
            # Get cloud cover
            cloud_cover = properties.get("eo:cloud_cover", 0)
            
            # Get band assets (with Planetary Computer signing)
            b04_asset = assets.get("B04", {})
            b08_asset = assets.get("B08", {})
            
            if not b04_asset or not b08_asset:
                logger.warning(f"Missing required bands for tile {tile_id}")
                return None
            
            # Sign URLs with Planetary Computer
            b04_url = pc.sign(b04_asset.get("href"))
            b08_url = pc.sign(b08_asset.get("href"))
            
            # Get bounding box
            bbox = item.get("bbox", [])
            
            return {
                "tile_id": tile_id,
                "acquisition_date": acq_date.date(),
                "collection": "sentinel-2-l2a",
                "cloud_cover": cloud_cover,
                "b04_url": b04_url,
                "b08_url": b08_url,
                "bbox": bbox,
                "scene_id": item.get("id"),
                "stac_item": item
            }
        
        except Exception as e:
            logger.error(f"Failed to parse STAC item: {e}")
            return None
    
    def _store_satellite_tiles(self, tiles: List[Dict[str, Any]]):
        """Store satellite tile metadata in database."""
        for tile in tiles:
            try:
                data = {
                    "tile_id": tile["tile_id"],
                    "acquisition_date": tile["acquisition_date"].isoformat(),
                    "collection": tile["collection"],
                    "cloud_cover": tile["cloud_cover"],
                    "bbox": tile.get("bbox"),
                    "scene_id": tile.get("scene_id"),
                    "status": "pending",
                    "api_source": "planetary_computer"
                }
                
                # Upsert (insert or update)
                self.supabase.table("satellite_tiles").upsert(
                    data,
                    on_conflict="tile_id,acquisition_date"
                ).execute()
                
            except Exception as e:
                logger.error(f"Failed to store tile {tile['tile_id']}: {e}")
    
    def _process_tile(
        self,
        tile: Dict[str, Any],
        land: Dict[str, Any],
        land_polygon: Polygon,
        tenant_id: str
    ) -> Dict[str, Any]:
        """
        Process a single satellite tile for the land.
        Downloads bands, computes NDVI, clips to polygon, generates overlay.
        """
        tile_id = tile["tile_id"]
        acq_date = tile["acquisition_date"]
        
        logger.info(f"Processing tile {tile_id} from {acq_date}")
        
        # Download and process bands
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Download B04 (Red) and B08 (NIR)
            b04_data, b04_profile = self._download_band(
                tile, "B04", tmpdir_path
            )
            b08_data, b08_profile = self._download_band(
                tile, "B08", tmpdir_path
            )
            
            # Compute NDVI
            ndvi_data, ndvi_profile = self._compute_ndvi(
                b04_data, b08_data, b04_profile
            )
            
            # Clip NDVI to land polygon
            clipped_ndvi, clipped_profile, clipped_geom = self._clip_to_polygon(
                ndvi_data, ndvi_profile, land_polygon
            )
            
            # Calculate statistics
            stats = self._calculate_statistics(clipped_ndvi)
            
            # Generate colorized overlay
            overlay_png = self._generate_overlay(
                clipped_ndvi, clipped_profile, land_polygon
            )
            
            # Generate thumbnail
            thumbnail_png = self._generate_thumbnail(clipped_ndvi, 256)
            
            # Upload to B2 and Supabase Storage
            b04_path = self._upload_to_b2(
                b04_data, b04_profile, tile_id, acq_date, "B04"
            )
            b08_path = self._upload_to_b2(
                b08_data, b08_profile, tile_id, acq_date, "B08"
            )
            ndvi_path = self._upload_to_b2(
                ndvi_data, ndvi_profile, tile_id, acq_date, "ndvi"
            )
            
            thumbnail_url = self._upload_thumbnail(
                thumbnail_png, land["id"], tile_id, acq_date
            )
            
            # Store in database
            self._store_ndvi_data(
                land["id"], tenant_id, acq_date, tile_id, stats
            )
            
            self._store_micro_tile(
                land["id"], land["farmer_id"], tenant_id, tile_id,
                acq_date, clipped_geom, stats, thumbnail_url
            )
            
            # Update satellite_tiles status
            self._update_tile_status(tile_id, acq_date, "completed", ndvi_path)
        
        return {
            "tile_id": tile_id,
            "acquisition_date": acq_date.isoformat(),
            "thumbnail_url": thumbnail_url,
            "overlay_url": thumbnail_url,
            "bbox": tile.get("bbox", []),
            "ndvi_stats": stats,
            "geometry": mapping(clipped_geom)
        }
    
    def _download_band(
        self,
        tile: Dict[str, Any],
        band: str,
        tmpdir: Path
    ) -> Tuple[np.ndarray, Dict]:
        """Download a satellite band (B04 or B08)."""
        url_key = f"{band.lower()}_url"
        
        if url_key in tile:
            # Download from signed URL
            url = tile[url_key]
            logger.info(f"Downloading {band} from Planetary Computer")
            
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            # Read with rasterio from bytes
            with MemoryFile(response.content) as memfile:
                with memfile.open() as src:
                    data = src.read(1)  # Read first band
                    profile = src.profile.copy()
            
            return data, profile
        
        else:
            # Try to load from B2 if already downloaded
            path_key = f"{band.lower()}_band_path"
            if path_key in tile and tile[path_key]:
                logger.info(f"Loading {band} from B2: {tile[path_key]}")
                file_info = self.b2_bucket.get_file_info_by_name(tile[path_key])
                download_dest = tmpdir / f"{band}.tif"
                self.b2_bucket.download_file_by_name(
                    tile[path_key], download_dest
                )
                
                with rasterio.open(download_dest) as src:
                    data = src.read(1)
                    profile = src.profile.copy()
                
                return data, profile
            
            else:
                raise ValueError(f"No URL or path available for band {band}")
    
    def _compute_ndvi(
        self,
        red: np.ndarray,
        nir: np.ndarray,
        profile: Dict
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute NDVI from RED and NIR bands.
        NDVI = (NIR - RED) / (NIR + RED)
        """
        # Convert to float32 for computation
        red = red.astype(np.float32)
        nir = nir.astype(np.float32)
        
        # Handle division by zero
        denominator = nir + red
        ndvi = np.where(
            denominator != 0,
            (nir - red) / denominator,
            0
        )
        
        # Clip to valid NDVI range [-1, 1]
        ndvi = np.clip(ndvi, -1, 1)
        
        # Update profile for NDVI
        ndvi_profile = profile.copy()
        ndvi_profile.update({
            'dtype': 'float32',
            'count': 1,
            'nodata': -9999
        })
        
        return ndvi, ndvi_profile
    
    def _clip_to_polygon(
        self,
        data: np.ndarray,
        profile: Dict,
        polygon: Polygon
    ) -> Tuple[np.ndarray, Dict, Polygon]:
        """Clip raster data to polygon boundary."""
        # Ensure polygon is in same CRS as raster
        raster_crs = CRS.from_dict(profile['crs'])
        
        # Transform polygon to raster CRS if needed
        if raster_crs != CRS.from_epsg(4326):
            project = pyproj.Transformer.from_crs(
                "EPSG:4326", raster_crs, always_xy=True
            ).transform
            polygon_transformed = transform(project, polygon)
        else:
            polygon_transformed = polygon
        
        # Create temporary in-memory raster
        with MemoryFile() as memfile:
            with memfile.open(**profile) as src_mem:
                src_mem.write(data, 1)
                
                # Mask/clip
                clipped_data, clipped_transform = mask(
                    src_mem,
                    [mapping(polygon_transformed)],
                    crop=True,
                    all_touched=True,
                    nodata=-9999
                )
        
        # Update profile
        clipped_profile = profile.copy()
        clipped_profile.update({
            'height': clipped_data.shape[1],
            'width': clipped_data.shape[2],
            'transform': clipped_transform
        })
        
        return clipped_data[0], clipped_profile, polygon_transformed
    
    def _calculate_statistics(self, ndvi: np.ndarray) -> Dict[str, Any]:
        """Calculate NDVI statistics."""
        # Mask out nodata values
        valid_mask = (ndvi != -9999) & (~np.isnan(ndvi)) & (ndvi >= -1) & (ndvi <= 1)
        valid_ndvi = ndvi[valid_mask]
        
        if len(valid_ndvi) == 0:
            return {
                "min_ndvi": None,
                "max_ndvi": None,
                "mean_ndvi": None,
                "median_ndvi": None,
                "std_ndvi": None,
                "valid_pixels": 0,
                "total_pixels": ndvi.size,
                "coverage_percentage": 0.0
            }
        
        return {
            "min_ndvi": float(np.min(valid_ndvi)),
            "max_ndvi": float(np.max(valid_ndvi)),
            "mean_ndvi": float(np.mean(valid_ndvi)),
            "median_ndvi": float(np.median(valid_ndvi)),
            "std_ndvi": float(np.std(valid_ndvi)),
            "valid_pixels": int(len(valid_ndvi)),
            "total_pixels": int(ndvi.size),
            "coverage_percentage": float(len(valid_ndvi) / ndvi.size * 100)
        }
    
    def _generate_overlay(
        self,
        ndvi: np.ndarray,
        profile: Dict,
        polygon: Polygon
    ) -> bytes:
        """Generate colorized PNG overlay for Google Maps."""
        # Normalize NDVI to 0-1 range for colormap
        ndvi_norm = (ndvi + 1) / 2  # From [-1,1] to [0,1]
        
        # Apply colormap
        colored = self.ndvi_colormap(ndvi_norm)
        
        # Convert to RGBA (0-255)
        rgba = (colored * 255).astype(np.uint8)
        
        # Set transparency for nodata
        nodata_mask = (ndvi == -9999) | np.isnan(ndvi)
        rgba[nodata_mask, 3] = 0  # Fully transparent
        
        # Convert to PIL Image
        img = Image.fromarray(rgba, mode='RGBA')
        
        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format='PNG', optimize=True)
        buffer.seek(0)
        
        return buffer.getvalue()
    
    def _generate_thumbnail(self, ndvi: np.ndarray, size: int) -> bytes:
        """Generate small thumbnail of NDVI."""
        # Normalize and colorize
        ndvi_norm = np.clip((ndvi + 1) / 2, 0, 1)
        colored = self.ndvi_colormap(ndvi_norm)
        rgba = (colored * 255).astype(np.uint8)
        
        # Transparency for nodata
        nodata_mask = (ndvi == -9999) | np.isnan(ndvi)
        rgba[nodata_mask, 3] = 0
        
        # Create image and resize
        img = Image.fromarray(rgba, mode='RGBA')
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        
        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format='PNG', optimize=True)
        buffer.seek(0)
        
        return buffer.getvalue()
    
    def _upload_to_b2(
        self,
        data: np.ndarray,
        profile: Dict,
        tile_id: str,
        acq_date,
        band_type: str
    ) -> str:
        """Upload raster to Backblaze B2."""
        # Create path
        date_str = acq_date.strftime("%Y%m%d") if hasattr(acq_date, 'strftime') else str(acq_date)
        
        if band_type in ["B04", "B08"]:
            path = f"tiles/raw/{tile_id}/{date_str}/{band_type}.tif"
        else:
            path = f"tiles/ndvi/{tile_id}/{date_str}/{band_type}.tif"
        
        # Write to memory
        with MemoryFile() as memfile:
            with memfile.open(**profile) as dst:
                dst.write(data, 1)
            
            # Upload
            memfile.seek(0)
            self.b2_bucket.upload_bytes(
                memfile.read(),
                path
            )
        
        logger.info(f"Uploaded {band_type} to B2: {path}")
        return path
    
    def _upload_thumbnail(
        self,
        thumbnail_bytes: bytes,
        land_id: str,
        tile_id: str,
        acq_date
    ) -> str:
        """Upload thumbnail to Supabase Storage."""
        date_str = acq_date.strftime("%Y%m%d") if hasattr(acq_date, 'strftime') else str(acq_date)
        file_name = f"{land_id}_{tile_id}_{date_str}.png"
        path = f"thumbnails/{land_id}/{file_name}"
        
        try:
            # Upload to Supabase Storage
            response = self.supabase.storage.from_("ndvi-thumbnails").upload(
                path,
                thumbnail_bytes,
                {"content-type": "image/png", "upsert": "true"}
            )
            
            # Get public URL
            public_url = self.supabase.storage.from_("ndvi-thumbnails").get_public_url(path)
            
            logger.info(f"Uploaded thumbnail: {public_url}")
            return public_url
        
        except Exception as e:
            logger.error(f"Failed to upload thumbnail: {e}")
            return None
    
    def _store_ndvi_data(
        self,
        land_id: str,
        tenant_id: str,
        acq_date,
        tile_id: str,
        stats: Dict[str, Any]
    ):
        """Store NDVI data record in database."""
        data = {
            "land_id": land_id,
            "tenant_id": tenant_id,
            "date": acq_date.isoformat() if hasattr(acq_date, 'isoformat') else str(acq_date),
            "ndvi_value": stats["mean_ndvi"],
            "min_ndvi": stats["min_ndvi"],
            "max_ndvi": stats["max_ndvi"],
            "mean_ndvi": stats["mean_ndvi"],
            "ndvi_std": stats["std_ndvi"],
            "valid_pixels": stats["valid_pixels"],
            "total_pixels": stats["total_pixels"],
            "coverage_percentage": stats["coverage_percentage"],
            "satellite_source": "sentinel-2",
            "collection_id": "sentinel-2-l2a",
            "tile_id": tile_id,
            "computed_at": datetime.utcnow().isoformat()
        }
        
        try:
            self.supabase.table("ndvi_data").upsert(
                data,
                on_conflict="land_id,date"
            ).execute()
            logger.info(f"Stored NDVI data for land {land_id}, date {acq_date}")
        except Exception as e:
            logger.error(f"Failed to store NDVI data: {e}")
    
    def _store_micro_tile(
        self,
        land_id: str,
        farmer_id: str,
        tenant_id: str,
        tile_id: str,
        acq_date,
        geometry: Polygon,
        stats: Dict[str, Any],
        thumbnail_url: str
    ):
        """Store micro-tile record in database."""
        data = {
            "land_id": land_id,
            "farmer_id": farmer_id,
            "tenant_id": tenant_id,
            "bbox": list(geometry.bounds),  # [minx, miny, maxx, maxy]
            "acquisition_date": acq_date.isoformat() if hasattr(acq_date, 'isoformat') else str(acq_date),
            "ndvi_mean": stats["mean_ndvi"],
            "ndvi_min": stats["min_ndvi"],
            "ndvi_max": stats["max_ndvi"],
            "ndvi_std_dev": stats["std_ndvi"],
            "ndvi_thumbnail_url": thumbnail_url,
            "statistics_only": False,
            "resolution_meters": 10
        }
        
        try:
            self.supabase.table("ndvi_micro_tiles").upsert(
                data,
                on_conflict="land_id,acquisition_date"
            ).execute()
            logger.info(f"Stored micro-tile for land {land_id}")
        except Exception as e:
            logger.error(f"Failed to store micro-tile: {e}")
    
    def _update_tile_status(
        self,
        tile_id: str,
        acq_date,
        status: str,
        ndvi_path: str
    ):
        """Update satellite tile status in database."""
        try:
            self.supabase.table("satellite_tiles").update({
                "status": status,
                "ndvi_path": ndvi_path,
                "processing_completed_at": datetime.utcnow().isoformat(),
                "actual_download_status": "downloaded"
            }).eq("tile_id", tile_id).eq(
                "acquisition_date",
                acq_date.isoformat() if hasattr(acq_date, 'isoformat') else str(acq_date)
            ).execute()
        except Exception as e:
            logger.error(f"Failed to update tile status: {e}")
    
    def _aggregate_results(
        self,
        results: List[Dict[str, Any]],
        land_id: str,
        tenant_id: str
    ) -> Dict[str, Any]:
        """Aggregate results from multiple tiles."""
        # Calculate overall statistics (use most recent tile)
        latest_result = results[0]
        
        # Average stats across all tiles (weighted by valid pixels)
        total_valid_pixels = sum(r["ndvi_stats"]["valid_pixels"] for r in results)
        
        if total_valid_pixels > 0:
            overall_mean = sum(
                r["ndvi_stats"]["mean_ndvi"] * r["ndvi_stats"]["valid_pixels"]
                for r in results if r["ndvi_stats"]["mean_ndvi"] is not None
            ) / total_valid_pixels
            
            overall_min = min(
                r["ndvi_stats"]["min_ndvi"] for r in results
                if r["ndvi_stats"]["min_ndvi"] is not None
            )
            overall_max = max(
                r["ndvi_stats"]["max_ndvi"] for r in results
                if r["ndvi_stats"]["max_ndvi"] is not None
            )
        else:
            overall_mean = None
            overall_min = None
            overall_max = None
        
        overall_stats = {
            "min_ndvi": overall_min,
            "max_ndvi": overall_max,
            "mean_ndvi": overall_mean,
            "median_ndvi": latest_result["ndvi_stats"].get("median_ndvi"),
            "std_ndvi": latest_result["ndvi_stats"].get("std_ndvi"),
            "valid_pixels": total_valid_pixels,
            "total_pixels": sum(r["ndvi_stats"]["total_pixels"] for r in results),
            "coverage_percentage": sum(
                r["ndvi_stats"]["coverage_percentage"] for r in results
            ) / len(results)
        }
        
        # Build overlay GeoJSON
        overlay_geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": r["geometry"],
                    "properties": {
                        "tile_id": r["tile_id"],
                        "acquisition_date": r["acquisition_date"],
                        "mean_ndvi": r["ndvi_stats"]["mean_ndvi"],
                        "thumbnail_url": r["thumbnail_url"]
                    }
                }
                for r in results
            ]
        }
        
        return {
            "overall_stats": overall_stats,
            "micro_tiles": results,
            "overlay_geojson": overlay_geojson
        }


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python ndvi_land_worker.py <land_id> <tenant_id>")
        sys.exit(1)
    
    land_id = sys.argv[1]
    tenant_id = sys.argv[2]
    
    worker = NDVILandWorker(
        supabase_url=os.getenv("SUPABASE_URL"),
        supabase_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
        b2_key_id=os.getenv("B2_KEY_ID"),
        b2_app_key=os.getenv("B2_APP_KEY"),
        b2_bucket=os.getenv("B2_BUCKET_RAW"),
        mpc_stac_url=os.getenv("MPC_STAC_BASE")
    )
    
    result = worker.process_land_ndvi(land_id, tenant_id)
    print(json.dumps(result, indent=2))
