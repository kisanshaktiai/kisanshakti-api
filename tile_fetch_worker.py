import os, requests, json, datetime, tempfile, logging, traceback
from supabase import create_client
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from requests.adapters import HTTPAdapter, Retry
from shapely import wkb, wkt
from shapely.geometry import mapping
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
    """Check if file exists in B2 bucket"""
    try:
        bucket.get_file_info_by_name(b2_path)
        return True
    except Exception:
        return False

def get_b2_paths(tile_id, acq_date):
    """Generate B2 paths for red, nir, and ndvi"""
    return {
        "red": f"{B2_PREFIX}raw/{tile_id}/{acq_date}/B04.tif",
        "nir": f"{B2_PREFIX}raw/{tile_id}/{acq_date}/B08.tif",
        "ndvi": f"{B2_PREFIX}ndvi/{tile_id}/{acq_date}/ndvi.tif"
    }

def check_existing_files(tile_id, acq_date):
    """Check which files already exist in B2"""
    paths = get_b2_paths(tile_id, acq_date)
    exists = {
        "red": check_b2_file_exists(paths["red"]),
        "nir": check_b2_file_exists(paths["nir"]),
        "ndvi": check_b2_file_exists(paths["ndvi"])
    }
    logging.info(f"📂 Existing files in B2: Red={exists['red']}, NIR={exists['nir']}, NDVI={exists['ndvi']}")
    return exists, paths

def download_band(url, b2_path):
    """Download band → compress → upload → return compressed local + b2_uri"""
    raw_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    compressed_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    try:
        # Download raw file with streaming to minimize memory
        logging.info(f"📥 Downloading from {url[:80]}...")
        r = session.get(url, stream=True, timeout=120)
        if not r.ok:
            logging.error(f"❌ Download failed: {r.status_code}")
            return None, None
        
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

        # Upload compressed version
        logging.info(f"☁️  Uploading to B2: {b2_path}")
        bucket.upload_local_file(local_file=compressed_tmp.name, file_name=b2_path)
        logging.info(f"✅ Upload complete")

        return compressed_tmp.name, f"b2://{B2_BUCKET_NAME}/{b2_path}"

    except Exception as e:
        logging.error(f"❌ Download/compress/upload failed: {e}\n{traceback.format_exc()}")
        return None, None
    finally:
        try: 
            os.remove(raw_tmp.name)
        except: 
            pass

def _record_exists(tile_id, acq_date):
    try:
        resp = supabase.table("satellite_tiles") \
            .select("id,status") \
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

# ---------- NDVI Calculation ----------
def compute_and_upload_ndvi(tile_id, acq_date, red_local, nir_local):
    """Compute NDVI with minimal memory footprint and upload to B2"""
    ndvi_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    try:
        logging.info(f"🧮 Computing NDVI for {tile_id} {acq_date}")
        
        with rasterio.open(red_local) as rsrc, rasterio.open(nir_local) as nsrc:
            meta = rsrc.meta.copy()
            height = rsrc.height
            width = rsrc.width
            
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
            total_pixels = height * width
            stats = {
                "ndvi_min": None,
                "ndvi_max": None,
                "ndvi_mean": None,
                "ndvi_std_dev": None,
                "vegetation_coverage_percent": None,
                "data_completeness_percent": None
            }
            
            if valid_pixels > 0:
                mean = sum_ndvi / valid_pixels
                variance = (sum_sq_ndvi / valid_pixels) - (mean ** 2)
                std_dev = np.sqrt(max(0, variance))
                
                stats = {
                    "ndvi_min": float(min_val),
                    "ndvi_max": float(max_val),
                    "ndvi_mean": float(mean),
                    "ndvi_std_dev": float(std_dev),
                    "vegetation_coverage_percent": float(veg_pixels / valid_pixels * 100.0),
                    "data_completeness_percent": float(valid_pixels / total_pixels * 100.0)
                }
                logging.info(f"📈 NDVI stats: min={stats['ndvi_min']:.3f}, max={stats['ndvi_max']:.3f}, mean={stats['ndvi_mean']:.3f}")
            
            # Write NDVI
            meta.update(dtype=rasterio.float32, count=1, compress="LZW", tiled=True, blockxsize=256, blockysize=256)
            with rasterio.open(ndvi_tmp.name, "w", **meta) as dst:
                dst.write(ndvi_full.astype(rasterio.float32), 1)
            
            # Free memory
            del ndvi_full

        ndvi_b2_path = f"{B2_PREFIX}ndvi/{tile_id}/{acq_date}/ndvi.tif"
        logging.info(f"☁️  Uploading NDVI to B2: {ndvi_b2_path}")
        bucket.upload_local_file(local_file=ndvi_tmp.name, file_name=ndvi_b2_path)
        logging.info(f"✅ NDVI uploaded successfully")
        
        return f"b2://{B2_BUCKET_NAME}/{ndvi_b2_path}", stats
        
    except Exception as e:
        logging.error(f"❌ NDVI computation failed: {e}\n{traceback.format_exc()}")
        return None, None
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
        
        # Check if files already exist in B2
        exists, paths = check_existing_files(tile_id, acq_date)
        
        # Check DB record
        db_exists, row = _record_exists(tile_id, acq_date)
        
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
        
        # Download Red if not exists
        if not exists["red"]:
            logging.info(f"📥 Downloading Red band for {tile_id} {acq_date}")
            red_local, red_b2 = download_band(red_url, paths["red"])
            if not red_local:
                logging.error(f"❌ Failed to download Red for {tile_id}")
                return False
        else:
            logging.info(f"✅ Red band already exists, skipping download")
        
        # Download NIR if not exists
        if not exists["nir"]:
            logging.info(f"📥 Downloading NIR band for {tile_id} {acq_date}")
            nir_local, nir_b2 = download_band(nir_url, paths["nir"])
            if not nir_local:
                logging.error(f"❌ Failed to download NIR for {tile_id}")
                if red_local:
                    try: os.remove(red_local)
                    except: pass
                return False
        else:
            logging.info(f"✅ NIR band already exists, skipping download")
        
        # Compute NDVI if not exists or if we downloaded new bands
        ndvi_b2 = f"b2://{B2_BUCKET_NAME}/{paths['ndvi']}"
        stats = None
        
        if not exists["ndvi"] or red_local or nir_local:
            # Need to download bands temporarily if they exist in B2 but not locally
            temp_files_to_cleanup = []
            
            if not red_local:
                logging.info(f"📥 Temporarily downloading Red for NDVI computation")
                red_local, _ = download_band(red_url, paths["red"])
                temp_files_to_cleanup.append(red_local)
            
            if not nir_local:
                logging.info(f"📥 Temporarily downloading NIR for NDVI computation")
                nir_local, _ = download_band(nir_url, paths["nir"])
                temp_files_to_cleanup.append(nir_local)
            
            if red_local and nir_local:
                logging.info(f"🧮 Computing NDVI for {tile_id} {acq_date}")
                ndvi_b2, stats = compute_and_upload_ndvi(tile_id, acq_date, red_local, nir_local)
                
                # Cleanup
                for f in [red_local, nir_local]:
                    try: 
                        if f:
                            os.remove(f)
                    except: 
                        pass
            else:
                logging.error(f"❌ Failed to get bands for NDVI computation")
                return False
        else:
            logging.info(f"✅ NDVI already exists, skipping computation")
        
        # Prepare DB payload
        payload = {
            "tile_id": tile_id,
            "acquisition_date": acq_date,
            "collection": MPC_COLLECTION.upper(),
            "red_band_path": red_b2,
            "nir_band_path": nir_b2,
            "ndvi_path": ndvi_b2,
            "status": "ready",
            "api_source": "planetary_computer",
            "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "processing_method": "cog_streaming",
            "actual_download_status": "downloaded",
            "processing_stage": "completed",
            "ndvi_calculation_timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }
        
        if country_id:
            payload["country_id"] = country_id
        if mgrs_tile_id:
            payload["mgrs_tile_id"] = mgrs_tile_id
        if cloud_cover is not None:
            payload["cloud_cover"] = float(cloud_cover)
        if stats:
            payload.update(stats)

        # Save to DB
        logging.info(f"💾 Saving record to database for {tile_id} {acq_date}")
        try:
            resp = supabase.table("satellite_tiles").upsert(
                payload, 
                on_conflict="tile_id,acquisition_date,collection"
            ).execute()
            
            if resp.data:
                logging.info(f"✅ Successfully saved {tile_id} {acq_date}")
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
