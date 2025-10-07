import os, requests, json, datetime, tempfile, logging, traceback
from supabase import create_client
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from requests.adapters import HTTPAdapter, Retry
from shapely import wkb, wkt
from shapely.geometry import mapping
import planetary_computer as pc
import rasterio
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

def download_band(url, b2_path):
    """Download band → compress → upload → return compressed local + b2_uri"""
    raw_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    compressed_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    try:
        # Download raw file
        r = session.get(url, stream=True, timeout=120)
        if not r.ok:
            return None, None
        with open(raw_tmp.name, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)

        # Compress raw → compressed_tmp
        with rasterio.open(raw_tmp.name) as src:
            data = src.read()
            meta = src.meta.copy()
            meta.update(
                compress="LZW",
                tiled=True,
                blockxsize=512,
                blockysize=512
            )
            with rasterio.open(compressed_tmp.name, "w", **meta) as dst:
                dst.write(data)

        # Upload compressed version
        logging.info(f"Uploading compressed COG to B2: {B2_BUCKET_NAME}/{b2_path}")
        bucket.upload_local_file(local_file=compressed_tmp.name, file_name=b2_path)

        return compressed_tmp.name, f"b2://{B2_BUCKET_NAME}/{b2_path}"

    except Exception as e:
        logging.error(f"Download/compress/upload failed: {e}\n{traceback.format_exc()}")
        return None, None
    finally:
        try: os.remove(raw_tmp.name)
        except: pass

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
    """Compute NDVI and upload to B2, return B2 URI and stats"""
    ndvi_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    try:
        logging.info(f"📂 Opening Red band: {red_local}")
        logging.info(f"📂 Opening NIR band: {nir_local}")
        
        with rasterio.open(red_local) as rsrc, rasterio.open(nir_local) as nsrc:
            red = rsrc.read(1).astype("float32")
            nir = nsrc.read(1).astype("float32")
            meta = rsrc.meta.copy()
        
        logging.info(f"✅ Bands loaded. Shape: {red.shape}")

        np.seterr(divide="ignore", invalid="ignore")
        ndvi = (nir - red) / (nir + red)
        ndvi = np.where((nir + red) == 0, np.nan, ndvi)

        valid = ~np.isnan(ndvi)
        logging.info(f"📊 Valid pixels: {valid.sum()}/{ndvi.size} ({valid.sum()/ndvi.size*100:.1f}%)")
        
        stats = {
            "ndvi_min": None,
            "ndvi_max": None,
            "ndvi_mean": None,
            "ndvi_std_dev": None,
            "vegetation_coverage_percent": None,
            "data_completeness_percent": None
        }
        
        if valid.sum() > 0:
            stats = {
                "ndvi_min": float(np.nanmin(ndvi)),
                "ndvi_max": float(np.nanmax(ndvi)),
                "ndvi_mean": float(np.nanmean(ndvi)),
                "ndvi_std_dev": float(np.nanstd(ndvi)),
                "vegetation_coverage_percent": float((ndvi > 0.3).sum() / valid.sum() * 100.0),
                "data_completeness_percent": float(valid.sum() / ndvi.size * 100.0)
            }
            logging.info(f"📈 NDVI stats: min={stats['ndvi_min']:.3f}, max={stats['ndvi_max']:.3f}, mean={stats['ndvi_mean']:.3f}")

        logging.info(f"💾 Writing NDVI to temp file: {ndvi_tmp.name}")
        meta.update(dtype=rasterio.float32, count=1, compress="LZW", tiled=True)
        with rasterio.open(ndvi_tmp.name, "w", **meta) as dst:
            dst.write(ndvi.astype(rasterio.float32), 1)

        ndvi_b2_path = f"{B2_PREFIX}ndvi/{tile_id}/{acq_date}/ndvi.tif"
        logging.info(f"☁️  Uploading NDVI to B2: {B2_BUCKET_NAME}/{ndvi_b2_path}")
        bucket.upload_local_file(local_file=ndvi_tmp.name, file_name=ndvi_b2_path)
        logging.info(f"✅ NDVI uploaded successfully")
        
        return f"b2://{B2_BUCKET_NAME}/{ndvi_b2_path}", stats
    except Exception as e:
        logging.error(f"❌ NDVI computation failed: {e}\n{traceback.format_exc()}")
        return None, None
    finally:
        try: 
            os.remove(ndvi_tmp.name)
            logging.debug(f"🧹 Removed temp NDVI file: {ndvi_tmp.name}")
        except Exception as e:
            logging.warning(f"⚠️ Could not remove temp NDVI file: {e}")

# ---------------- Main process ----------------
def process_tile(tile):
    """Process a single tile: fetch scene, download bands, compute NDVI, save to DB"""
    try:
        tile_id = tile["tile_id"]
        geom_value = tile["geometry"]
        country_id = tile.get("country_id")  # Get country_id from mgrs_tiles
        mgrs_tile_id = tile.get("id")  # Get mgrs_tiles.id

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
        
        exists, row = _record_exists(tile_id, acq_date)
        if exists and row and row.get("status") in ("downloaded", "ready"):
            logging.info(f"⏩ Skipping {tile_id} {acq_date}, already in DB with status={row.get('status')}")
            return False

        assets = scene.get("assets", {})
        red_url = _signed_asset_url(assets, "red", "B04")
        nir_url = _signed_asset_url(assets, "nir", "B08")

        if not red_url or not nir_url:
            logging.warning(f"❌ Missing red/nir URLs for {tile_id} {acq_date}")
            return False

        # Download + compress + upload Red/NIR
        logging.info(f"📥 Downloading Red band for {tile_id} {acq_date}")
        red_local, red_b2 = download_band(red_url, f"{B2_PREFIX}raw/{tile_id}/{acq_date}/B04.tif")
        
        logging.info(f"📥 Downloading NIR band for {tile_id} {acq_date}")
        nir_local, nir_b2 = download_band(nir_url, f"{B2_PREFIX}raw/{tile_id}/{acq_date}/B08.tif")
        
        if not red_local or not nir_local:
            logging.error(f"❌ Failed to download/compress Red or NIR for {tile_id}")
            return False

        # Compute NDVI
        logging.info(f"🧮 Computing NDVI for {tile_id} {acq_date}")
        try:
            ndvi_b2, stats = compute_and_upload_ndvi(tile_id, acq_date, red_local, nir_local)
            logging.info(f"📊 NDVI computation complete. B2 path: {ndvi_b2}, Stats: {stats}")
        except Exception as ndvi_err:
            logging.error(f"❌ NDVI computation exception for {tile_id}: {ndvi_err}")
            logging.error(traceback.format_exc())
            return False
        
        # Cleanup temp Red/NIR after NDVI is done
        try:
            os.remove(red_local)
            os.remove(nir_local)
            logging.info(f"🧹 Cleaned up temp files for {tile_id}")
        except Exception as cleanup_err:
            logging.warning(f"⚠️ Cleanup warning: {cleanup_err}")

        if not ndvi_b2 or stats is None:
            logging.error(f"❌ Failed NDVI computation for {tile_id} - ndvi_b2: {ndvi_b2}, stats: {stats}")
            return False
        
        logging.info(f"✅ NDVI ready for {tile_id} {acq_date}")

        # Prepare payload with all required fields including country_id
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
        
        # Add country_id and mgrs_tile_id (CRITICAL for foreign key constraint)
        if country_id:
            payload["country_id"] = country_id
        if mgrs_tile_id:
            payload["mgrs_tile_id"] = mgrs_tile_id
            
        # Add cloud_cover if available
        if cloud_cover is not None:
            payload["cloud_cover"] = float(cloud_cover)
        
        # Add stats only if they exist
        if stats:
            payload.update(stats)

        # Insert/Update record in Supabase
        logging.info(f"💾 Saving record to database for {tile_id} {acq_date}")
        logging.debug(f"Payload keys: {list(payload.keys())}")
        
        try:
            # Use upsert with proper conflict handling
            resp = supabase.table("satellite_tiles") \
                .upsert(payload, on_conflict="tile_id,acquisition_date,collection") \
                .execute()
            
            if resp.data:
                logging.info(f"✅ Successfully saved {tile_id} {acq_date} (record id: {resp.data[0].get('id')})")
            else:
                logging.warning(f"⚠️ Upsert returned no data for {tile_id} {acq_date}")
            
            logging.debug(f"📦 Supabase response: {resp}")
            
        except Exception as db_err:
            error_msg = str(db_err)
            logging.error(f"❌ Database operation failed for {tile_id} {acq_date}")
            logging.error(f"Error: {error_msg}")
            
            # Log specific constraint violations
            if "foreign key" in error_msg.lower():
                logging.error(f"⚠️ Foreign key constraint violation - country_id: {country_id}, mgrs_tile_id: {mgrs_tile_id}")
            if "unique" in error_msg.lower():
                logging.error(f"⚠️ Unique constraint violation - may be duplicate record")
            
            logging.error(f"Payload: {json.dumps(payload, indent=2, default=str)}")
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
