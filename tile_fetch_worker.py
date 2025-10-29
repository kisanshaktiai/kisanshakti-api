# ──────────────────────────────────────────────────────────────────────────────
# File: tile_fetch_worker.py
# Version: Resotred Old Dated 10 Oct 25
# Author: Amarsinh Patil
# Purpose:
#   NDVI tile processor for Sentinel-2 scenes.
#   Downloads RED/NIR bands from Microsoft Planetary Computer (MPC),
#   compresses as COG, computes NDVI, uploads to Backblaze B2 bucket
#   (kisanshakti-ndvi-tiles), and updates Supabase satellite_tiles table.
# ──────────────────────────────────────────────────────────────────────────────

import os, requests, json, datetime, tempfile, logging, traceback
from supabase import create_client
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from requests.adapters import HTTPAdapter, Retry
from shapely import wkb, wkt
from shapely.geometry import mapping, shape
import planetary_computer as pc
import rasterio
from rasterio.windows import Window
import numpy as np

# ---------------- Config ----------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

B2_APP_KEY_ID = os.environ.get("B2_KEY_ID")
B2_APP_KEY = os.environ.get("B2_APP_KEY")
B2_BUCKET_NAME = os.environ.get("B2_BUCKET_RAW", "kisanshakti-ndvi-tiles")
B2_PREFIX = os.environ.get("B2_PREFIX", "tiles/")

MPC_STAC = os.environ.get("MPC_STAC_BASE", "https://planetarycomputer.microsoft.com/api/stac/v1/search")
MPC_COLLECTION = os.environ.get("MPC_COLLECTION", "sentinel-2-l2a")
CLOUD_COVER = int(os.environ.get("DEFAULT_CLOUD_COVER_MAX", "20"))
LOOKBACK_DAYS = int(os.environ.get("MAX_SCENE_LOOKBACK_DAYS", "5"))

# Memory optimization: downsample factor (1=no downsample, 2=half size, 4=quarter size)
DOWNSAMPLE_FACTOR = int(os.environ.get("DOWNSAMPLE_FACTOR", "4"))

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------- Clients ----------------
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY missing in env.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

if not B2_APP_KEY_ID or not B2_APP_KEY:
    raise RuntimeError("B2_KEY_ID or B2_APP_KEY missing in env.")

info = InMemoryAccountInfo()
b2_api = B2Api(info)
b2_api.authorize_account("production", B2_APP_KEY_ID, B2_APP_KEY)
bucket = b2_api.get_bucket_by_name(B2_BUCKET_NAME)

session = requests.Session()
retries = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

# ---------------- Helpers ----------------
def fetch_agri_tiles():
    """Get MGRS tiles where is_agri=True, including country_id"""
    try:
        resp = supabase.table("mgrs_tiles") \
            .select("tile_id, geometry, country_id, id") \
            .eq("is_agri", True) \
            .execute()
        tiles = resp.data or []
        logging.info(f"Fetched {len(tiles)} agri tiles")
        return tiles
    except Exception as e:
        logging.error(f"Failed to fetch agri tiles: {e}")
        return []

def decode_geom_to_geojson(geom_value):
    """Decode geometry into GeoJSON dict"""
    try:
        if geom_value is None:
            return None
        if isinstance(geom_value, dict) and "type" in geom_value and "coordinates" in geom_value:
            return geom_value
        if isinstance(geom_value, (bytes, bytearray)):
            return mapping(wkb.loads(geom_value))
        if isinstance(geom_value, str):
            s = geom_value.strip()
            if s.startswith("{") and s.endswith("}"):
                try:
                    return json.loads(s)
                except:
                    pass
            try:
                return mapping(wkt.loads(s))
            except:
                pass
            try:
                return mapping(wkb.loads(bytes.fromhex(s)))
            except:
                pass
        return None
    except Exception as e:
        logging.error(f"decode_geom_to_geojson failed: {e}\n{traceback.format_exc()}")
        return None

def extract_bbox(geom_json):
    """Extract bounding box from GeoJSON geometry"""
    try:
        if not geom_json:
            return None
        geom = shape(geom_json)
        bounds = geom.bounds  # (minx, miny, maxx, maxy)
        return {
            "type": "Polygon",
            "coordinates": [[
                [bounds[0], bounds[1]],  # SW
                [bounds[2], bounds[1]],  # SE
                [bounds[2], bounds[3]],  # NE
                [bounds[0], bounds[3]],  # NW
                [bounds[0], bounds[1]]   # Close
            ]]
        }
    except Exception as e:
        logging.error(f"Failed to extract bbox: {e}")
        return None

def query_mpc(tile_geom, start_date, end_date):
    try:
        geom_json = decode_geom_to_geojson(tile_geom)
        if not geom_json:
            return []
        body = {
            "collections": [MPC_COLLECTION],
            "intersects": geom_json,
            "datetime": f"{start_date}/{end_date}",
            "query": {"eo:cloud_cover": {"lt": CLOUD_COVER}}
        }
        logging.info(f"STAC query: {MPC_COLLECTION}, {start_date}->{end_date}, cloud<{CLOUD_COVER}")
        resp = session.post(MPC_STAC, json=body, timeout=45)
        if not resp.ok:
            logging.error(f"STAC error {resp.status_code}: {resp.text}")
            return []
        return resp.json().get("features", [])
    except Exception as e:
        logging.error(f"MPC query failed: {e}\n{traceback.format_exc()}")
        return []

def _signed_asset_url(assets, primary_key, fallback_key=None):
    """Get signed PC asset URL"""
    href = None
    if primary_key in assets and "href" in assets[primary_key]:
        href = assets[primary_key]["href"]
    elif fallback_key and fallback_key in assets and "href" in assets[fallback_key]:
        href = assets[fallback_key]["href"]
    if not href:
        return None
    try:
        return pc.sign(href)
    except:
        return href

def check_b2_file_exists(b2_path):
    """Check if file exists in B2 bucket and return file info"""
    try:
        file_info = bucket.get_file_info_by_name(b2_path)
        return True, file_info.size if hasattr(file_info, 'size') else None
    except Exception:
        return False, None

def download_from_b2(b2_path):
    """Download file from B2 to local temp file"""
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        logging.info(f"📥 Downloading from B2: {b2_path}")
        bucket.download_file_by_name(b2_path, temp_file)
        temp_file.close()
        logging.info(f"✅ Downloaded from B2 to: {temp_file.name}")
        return temp_file.name
    except Exception as e:
        logging.error(f"❌ Failed to download from B2: {e}")
        return None

def get_b2_paths(tile_id, acq_date):
    """Generate B2 paths for red, nir, and ndvi"""
    return {
        "red": f"{B2_PREFIX}raw/{tile_id}/{acq_date}/B04.tif",
        "nir": f"{B2_PREFIX}raw/{tile_id}/{acq_date}/B08.tif",
        "ndvi": f"{B2_PREFIX}ndvi/{tile_id}/{acq_date}/ndvi.tif"
    }

def check_existing_files(tile_id, acq_date):
    """Check which files already exist in B2 and get their sizes"""
    paths = get_b2_paths(tile_id, acq_date)
    
    red_exists, red_size = check_b2_file_exists(paths["red"])
    nir_exists, nir_size = check_b2_file_exists(paths["nir"])
    ndvi_exists, ndvi_size = check_b2_file_exists(paths["ndvi"])
    
    exists = {
        "red": red_exists,
        "nir": nir_exists,
        "ndvi": ndvi_exists
    }
    
    sizes = {
        "red": red_size,
        "nir": nir_size,
        "ndvi": ndvi_size
    }
    
    logging.info(f"📂 Existing files in B2: Red={red_exists}, NIR={nir_exists}, NDVI={ndvi_exists}")
    if red_size:
        logging.info(f"📊 File sizes: Red={red_size/1024/1024:.2f}MB, NIR={nir_size/1024/1024:.2f}MB, NDVI={ndvi_size/1024/1024:.2f}MB" if ndvi_size else f"Red={red_size/1024/1024:.2f}MB, NIR={nir_size/1024/1024:.2f}MB")
    
    return exists, paths, sizes

def get_file_size(filepath):
    """Get file size in bytes"""
    try:
        return os.path.getsize(filepath)
    except:
        return None

def download_band(url, b2_path):
    """Download band → compress → upload → return compressed local, b2_uri, and size"""
    raw_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    compressed_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    try:
        # Download raw file with streaming to minimize memory
        logging.info(f"📥 Downloading from {url[:80]}...")
        r = session.get(url, stream=True, timeout=120)
        if not r.ok:
            logging.error(f"❌ Download failed: {r.status_code}")
            return None, None, None
        
        with open(raw_tmp.name, "wb") as f:
            for chunk in r.iter_content(chunk_size=512*1024):  # 512KB chunks
                if chunk:
                    f.write(chunk)
        
        logging.info(f"💾 Downloaded to temp: {raw_tmp.name}")

        # Compress with windowed processing for memory efficiency
        with rasterio.open(raw_tmp.name) as src:
            meta = src.meta.copy()
            
            # Apply downsampling if needed
            if DOWNSAMPLE_FACTOR > 1:
                out_shape = (src.height // DOWNSAMPLE_FACTOR, src.width // DOWNSAMPLE_FACTOR)
                data = src.read(
                    1,
                    out_shape=out_shape,
                    resampling=rasterio.enums.Resampling.average
                )
                meta.update({
                    'height': out_shape[0],
                    'width': out_shape[1],
                    'transform': src.transform * src.transform.scale(
                        (src.width / out_shape[1]),
                        (src.height / out_shape[0])
                    )
                })
            else:
                data = src.read(1)
            
            meta.update(
                compress="LZW",
                tiled=True,
                blockxsize=256,
                blockysize=256
            )
            
            with rasterio.open(compressed_tmp.name, "w", **meta) as dst:
                dst.write(data, 1)
            
            # Clear data from memory
            del data

        # Get file size before upload
        file_size = get_file_size(compressed_tmp.name)

        # Upload compressed version
        logging.info(f"☁️  Uploading to B2: {b2_path} ({file_size/1024/1024:.2f}MB)")
        bucket.upload_local_file(local_file=compressed_tmp.name, file_name=b2_path)
        logging.info(f"✅ Upload complete")

        return compressed_tmp.name, f"b2://{B2_BUCKET_NAME}/{b2_path}", file_size

    except Exception as e:
        logging.error(f"❌ Download/compress/upload failed: {e}\n{traceback.format_exc()}")
        return None, None, None
    finally:
        try: 
            os.remove(raw_tmp.name)
        except: 
            pass

def _record_exists(tile_id, acq_date):
    try:
        resp = supabase.table("satellite_tiles") \
            .select("id,status,created_at") \
            .eq("tile_id", tile_id) \
            .eq("acquisition_date", acq_date) \
            .eq("collection", MPC_COLLECTION.upper()) \
            .limit(1).execute()
        rows = resp.data or []
        return (True, rows[0]) if rows else (False, None)
    except Exception as e:
        logging.error(f"Error checking record existence: {e}")
        return False, None

def pick_best_scene(scenes):
    try:
        return sorted(
            scenes,
            key=lambda s: (
                s["properties"].get("eo:cloud_cover", 100),
                -datetime.datetime.fromisoformat(
                    s["properties"]["datetime"].replace("Z", "+00:00")
                ).timestamp()
            )
        )[0] if scenes else None
    except:
        return None

def calculate_vegetation_health_score(ndvi_mean, ndvi_std_dev, veg_coverage, data_completeness):
    """
    Calculate vegetation health score (0-100)
    Based on NDVI mean, vegetation coverage, and data quality
    """
    try:
        # Normalize NDVI mean from [-1, 1] to [0, 100]
        ndvi_score = ((ndvi_mean + 1) / 2) * 100
        
        # Weight factors
        health_score = (
            ndvi_score * 0.5 +           # 50% weight on NDVI value
            veg_coverage * 0.3 +         # 30% weight on coverage
            data_completeness * 0.2      # 20% weight on data quality
        )
        
        return round(health_score, 2)
    except:
        return None

# ---------- NDVI Calculation ----------
def compute_and_upload_ndvi(tile_id, acq_date, red_local, nir_local):
    """Compute NDVI with minimal memory footprint and upload to B2"""
    ndvi_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    try:
        with rasterio.open(red_local) as rsrc, rasterio.open(nir_local) as nsrc:
            meta = rsrc.meta.copy()
            height = rsrc.height
            width = rsrc.width
            total_pixels = height * width
            
            # Process in chunks to reduce memory
            chunk_size = 1024  # Process 1024 rows at a time
            ndvi_full = np.empty((height, width), dtype=np.float32)
            
            valid_pixels = 0
            sum_ndvi = 0.0
            sum_sq_ndvi = 0.0
            min_val = float('inf')
            max_val = float('-inf')
            veg_pixels = 0
            
            for i in range(0, height, chunk_size):
                rows = min(chunk_size, height - i)
                window = Window(0, i, width, rows)
                
                red = rsrc.read(1, window=window).astype('float32')
                nir = nsrc.read(1, window=window).astype('float32')
                
                np.seterr(divide="ignore", invalid="ignore")
                denominator = nir + red
                ndvi_chunk = np.where(denominator != 0, (nir - red) / denominator, np.nan)
                
                ndvi_full[i:i+rows, :] = ndvi_chunk
                
                # Update statistics
                valid = ~np.isnan(ndvi_chunk)
                if valid.sum() > 0:
                    valid_pixels += valid.sum()
                    chunk_valid = ndvi_chunk[valid]
                    sum_ndvi += chunk_valid.sum()
                    sum_sq_ndvi += (chunk_valid ** 2).sum()
                    min_val = min(min_val, chunk_valid.min())
                    max_val = max(max_val, chunk_valid.max())
                    veg_pixels += (chunk_valid > 0.3).sum()
                
                # Free memory
                del red, nir, ndvi_chunk
            
            # Calculate final statistics
            stats = {
                "ndvi_min": None,
                "ndvi_max": None,
                "ndvi_mean": None,
                "ndvi_std_dev": None,
                "vegetation_coverage_percent": None,
                "data_completeness_percent": None,
                "pixel_count": total_pixels,
                "valid_pixel_count": valid_pixels,
                "vegetation_health_score": None
            }
            
            if valid_pixels > 0:
                mean = sum_ndvi / valid_pixels
                variance = (sum_sq_ndvi / valid_pixels) - (mean ** 2)
                std_dev = np.sqrt(max(0, variance))
                veg_coverage = veg_pixels / valid_pixels * 100.0
                data_completeness = valid_pixels / total_pixels * 100.0
                
                stats = {
                    "ndvi_min": float(min_val),
                    "ndvi_max": float(max_val),
                    "ndvi_mean": float(mean),
                    "ndvi_std_dev": float(std_dev),
                    "vegetation_coverage_percent": float(veg_coverage),
                    "data_completeness_percent": float(data_completeness),
                    "pixel_count": total_pixels,
                    "valid_pixel_count": valid_pixels,
                    "vegetation_health_score": calculate_vegetation_health_score(
                        mean, std_dev, veg_coverage, data_completeness
                    )
                }
                logging.info(f"📈 NDVI stats: min={stats['ndvi_min']:.3f}, max={stats['ndvi_max']:.3f}, mean={stats['ndvi_mean']:.3f}")
                logging.info(f"🌱 Vegetation: coverage={stats['vegetation_coverage_percent']:.1f}%, health={stats['vegetation_health_score']:.1f}")
            
            # Write NDVI
            meta.update(dtype=rasterio.float32, count=1, compress="LZW", tiled=True, blockxsize=256, blockysize=256)
            with rasterio.open(ndvi_tmp.name, "w", **meta) as dst:
                dst.write(ndvi_full.astype(rasterio.float32), 1)
            
            # Free memory
            del ndvi_full

        # Get file size
        ndvi_size = get_file_size(ndvi_tmp.name)

        ndvi_b2_path = f"{B2_PREFIX}ndvi/{tile_id}/{acq_date}/ndvi.tif"
        logging.info(f"☁️  Uploading NDVI to B2: {ndvi_b2_path} ({ndvi_size/1024/1024:.2f}MB)")
        bucket.upload_local_file(local_file=ndvi_tmp.name, file_name=ndvi_b2_path)
        logging.info(f"✅ NDVI uploaded successfully")
        
        return f"b2://{B2_BUCKET_NAME}/{ndvi_b2_path}", stats, ndvi_size
        
    except Exception as e:
        logging.error(f"❌ NDVI computation failed: {e}\n{traceback.format_exc()}")
        return None, None, None
    finally:
        try: 
            os.remove(ndvi_tmp.name)
        except: 
            pass

# ---------------- Main process ----------------
def process_tile(tile):
    """Process a single tile: fetch scene, download bands if needed, compute NDVI, save to DB"""
    try:
        tile_id = tile["tile_id"]
        geom_value = tile["geometry"]
        country_id = tile.get("country_id")
        mgrs_tile_id = tile.get("id")

        today = datetime.date.today()
        start_date = (today - datetime.timedelta(days=LOOKBACK_DAYS)).isoformat()
        end_date = today.isoformat()

        scenes = query_mpc(geom_value, start_date, end_date)
        if not scenes:
            logging.info(f"🔍 No scenes for {tile_id} in {start_date}..{end_date}")
            return False

        scene = pick_best_scene(scenes)
        if not scene:
            logging.info(f"⚠️ No valid scene after sorting for {tile_id}")
            return False

        acq_date = scene["properties"]["datetime"].split("T")[0]
        cloud_cover = scene["properties"].get("eo:cloud_cover")
        
        # Extract bbox from geometry
        geom_json = decode_geom_to_geojson(geom_value)
        bbox = extract_bbox(geom_json)
        
        # Check if files already exist in B2
        exists, paths, file_sizes = check_existing_files(tile_id, acq_date)
        
        # Check DB record
        db_exists, row = _record_exists(tile_id, acq_date)
        original_created_at = row.get("created_at") if row else None
        
        # If all files exist in B2 and DB record exists with ready status, just update timestamp
        if exists["red"] and exists["nir"] and exists["ndvi"] and db_exists:
            logging.info(f"✅ All files exist for {tile_id} {acq_date}, updating timestamp only")
            try:
                payload = {
                    "tile_id": tile_id,
                    "acquisition_date": acq_date,
                    "collection": MPC_COLLECTION.upper(),
                    "updated_at": datetime.datetime.utcnow().isoformat() + "Z"
                }
                supabase.table("satellite_tiles").upsert(
                    payload, 
                    on_conflict="tile_id,acquisition_date,collection"
                ).execute()
                logging.info(f"✅ Timestamp updated for {tile_id} {acq_date}")
                return True
            except Exception as e:
                logging.error(f"❌ Failed to update timestamp: {e}")
                return False
        
        # Download only missing files
        assets = scene.get("assets", {})
        red_url = _signed_asset_url(assets, "red", "B04")
        nir_url = _signed_asset_url(assets, "nir", "B08")

        if not red_url or not nir_url:
            logging.warning(f"❌ Missing red/nir URLs for {tile_id} {acq_date}")
            return False

        red_local = None
        nir_local = None
        red_b2 = f"b2://{B2_BUCKET_NAME}/{paths['red']}"
        nir_b2 = f"b2://{B2_BUCKET_NAME}/{paths['nir']}"
        need_cleanup_red = False
        need_cleanup_nir = False
        
        # Download Red if not exists
        if not exists["red"]:
            logging.info(f"📥 Downloading Red band for {tile_id} {acq_date}")
            red_local, red_b2, red_size = download_band(red_url, paths["red"])
            file_sizes["red"] = red_size
            need_cleanup_red = True
            if not red_local:
                logging.error(f"❌ Failed to download Red for {tile_id}")
                return False
        else:
            logging.info(f"✅ Red band already exists, skipping download")
        
        # Download NIR if not exists
        if not exists["nir"]:
            logging.info(f"📥 Downloading NIR band for {tile_id} {acq_date}")
            nir_local, nir_b2, nir_size = download_band(nir_url, paths["nir"])
            file_sizes["nir"] = nir_size
            need_cleanup_nir = True
            if not nir_local:
                logging.error(f"❌ Failed to download NIR for {tile_id}")
                if need_cleanup_red and red_local:
                    try: os.remove(red_local)
                    except: pass
                return False
        else:
            logging.info(f"✅ NIR band already exists, skipping download")
        
        # Compute NDVI if not exists
        ndvi_b2 = f"b2://{B2_BUCKET_NAME}/{paths['ndvi']}"
        stats = None
        
        if not exists["ndvi"]:
            # Need bands for NDVI computation
            
            # If files weren't just downloaded, get them from B2
            if not red_local:
                if exists["red"]:
                    logging.info(f"📥 Downloading Red from B2 for NDVI computation")
                    red_local = download_from_b2(paths["red"])
                    need_cleanup_red = True
                else:
                    logging.error(f"❌ Red band not available")
                    return False
            
            if not nir_local:
                if exists["nir"]:
                    logging.info(f"📥 Downloading NIR from B2 for NDVI computation")
                    nir_local = download_from_b2(paths["nir"])
                    need_cleanup_nir = True
                else:
                    logging.error(f"❌ NIR band not available")
                    if need_cleanup_red and red_local:
                        try: os.remove(red_local)
                        except: pass
                    return False
            
            if red_local and nir_local:
                logging.info(f"🧮 Computing NDVI for {tile_id} {acq_date}")
                ndvi_b2, stats, ndvi_size = compute_and_upload_ndvi(tile_id, acq_date, red_local, nir_local)
                file_sizes["ndvi"] = ndvi_size
                
                # Cleanup temporary files
                if need_cleanup_red and red_local:
                    try: 
                        os.remove(red_local)
                        logging.debug(f"🧹 Cleaned up temp Red file")
                    except: 
                        pass
                if need_cleanup_nir and nir_local:
                    try: 
                        os.remove(nir_local)
                        logging.debug(f"🧹 Cleaned up temp NIR file")
                    except: 
                        pass
            else:
                logging.error(f"❌ Failed to get bands for NDVI computation")
                return False
        else:
            logging.info(f"✅ NDVI already exists, skipping computation")
        
        # Calculate total file size in MB
        total_size_mb = None
        if file_sizes["red"] and file_sizes["nir"] and file_sizes.get("ndvi"):
            total_size_mb = (file_sizes["red"] + file_sizes["nir"] + file_sizes["ndvi"]) / (1024 * 1024)
        
    # Prepare DB payload with all relevant fields
        now = datetime.datetime.utcnow().isoformat() + "Z"
        
        payload = {
            "tile_id": tile_id,
            "acquisition_date": acq_date,
            "collection": MPC_COLLECTION.upper(),
            "processing_level": "L2A",
            "cloud_cover": float(cloud_cover) if cloud_cover is not None else None,
            
            # Paths
            "red_band_path": red_b2,
            "nir_band_path": nir_b2,
            "ndvi_path": ndvi_b2,
            
            # File sizes - convert to native int
            "file_size_mb": round(total_size_mb, 2) if total_size_mb else None,
            "red_band_size_bytes": int(file_sizes.get("red")) if file_sizes.get("red") else None,
            "nir_band_size_bytes": int(file_sizes.get("nir")) if file_sizes.get("nir") else None,
            "ndvi_size_bytes": int(file_sizes.get("ndvi")) if file_sizes.get("ndvi") else None,
            
            # Resolution
            "resolution": "10m",
            
            # Status and timestamps
            "status": "ready",
            "updated_at": now,
            "processing_completed_at": now,
            "ndvi_calculation_timestamp": now,
            
            # Processing metadata
            "api_source": "planetary_computer",
            "processing_method": "cog_streaming",
            "actual_download_status": "downloaded",
            "processing_stage": "completed",
            
            # Foreign keys
            "country_id": country_id,
            "mgrs_tile_id": mgrs_tile_id,
            
            # Geometry
            "bbox": json.dumps(bbox) if bbox else None,
        }
        
        # Add created_at only for new records
        if not db_exists:
            payload["created_at"] = now
        elif original_created_at:
            payload["created_at"] = original_created_at
        
        # Add NDVI statistics if available - CONVERT ALL NUMPY TYPES TO PYTHON NATIVE TYPES
        if stats:
            payload.update({
                "ndvi_min": str(stats.get("ndvi_min")) if stats.get("ndvi_min") is not None else None,
                "ndvi_max": str(stats.get("ndvi_max")) if stats.get("ndvi_max") is not None else None,
                "ndvi_mean": str(stats.get("ndvi_mean")) if stats.get("ndvi_mean") is not None else None,
                "ndvi_std_dev": str(stats.get("ndvi_std_dev")) if stats.get("ndvi_std_dev") is not None else None,
                "vegetation_coverage_percent": str(stats.get("vegetation_coverage_percent")) if stats.get("vegetation_coverage_percent") is not None else None,
                "data_completeness_percent": str(stats.get("data_completeness_percent")) if stats.get("data_completeness_percent") is not None else None,
                # CRITICAL FIX: Convert numpy int64 to Python int
                "pixel_count": int(stats.get("pixel_count")) if stats.get("pixel_count") is not None else None,
                "valid_pixel_count": int(stats.get("valid_pixel_count")) if stats.get("valid_pixel_count") is not None else None,
                "vegetation_health_score": str(stats.get("vegetation_health_score")) if stats.get("vegetation_health_score") is not None else None,
            })
        # Save to DB
        logging.info(f"💾 Saving record to database for {tile_id} {acq_date}")
        logging.info(f"📋 Payload has {len(payload)} fields")
        
        try:
            resp = supabase.table("satellite_tiles").upsert(
                payload, 
                on_conflict="tile_id,acquisition_date,collection"
            ).execute()
            
            if resp.data:
                record_id = resp.data[0].get('id', 'unknown')
                logging.info(f"✅ Successfully saved {tile_id} {acq_date} (record id: {record_id})")
            else:
                logging.warning(f"⚠️ Upsert returned no data for {tile_id} {acq_date}")
            
        except Exception as db_err:
            logging.error(f"❌ Database operation failed: {db_err}")
            logging.error(traceback.format_exc())
            return False

        return True

    except Exception as e:
        logging.error(f"❌ process_tile error for {tile.get('tile_id')}: {e}")
        logging.error(traceback.format_exc())
        return False

def main(cloud_cover=20, lookback_days=5):
    """Main entry point"""
    global CLOUD_COVER, LOOKBACK_DAYS
    CLOUD_COVER = int(cloud_cover)
    LOOKBACK_DAYS = int(lookback_days)

    logging.info(f"🚀 Starting tile processing (cloud_cover<={CLOUD_COVER}%, lookback={LOOKBACK_DAYS} days)")
    logging.info(f"💾 Memory optimization: DOWNSAMPLE_FACTOR={DOWNSAMPLE_FACTOR}")
    
    processed = 0
    tiles = fetch_agri_tiles()
    
    if not tiles:
        logging.warning("⚠️ No tiles fetched from database")
        return 0
    
    logging.info(f"📋 Processing {len(tiles)} tiles...")
    
    for i, t in enumerate(tiles, 1):
        tile_id = t.get('tile_id', 'unknown')
        logging.info(f"🔄 [{i}/{len(tiles)}] Processing: {tile_id}")
        if process_tile(t):
            processed += 1
            logging.info(f"✅ [{i}/{len(tiles)}] Success: {tile_id}")
        else:
            logging.info(f"⏭️  [{i}/{len(tiles)}] Skipped: {tile_id}")
    
    logging.info(f"✨ Finished: processed {processed}/{len(tiles)} tiles successfully")
    return processed

if __name__ == "__main__":
    cc = int(os.environ.get("RUN_CLOUD_COVER", CLOUD_COVER))
    lb = int(os.environ.get("RUN_LOOKBACK_DAYS", LOOKBACK_DAYS))
    main(cc, lb)
