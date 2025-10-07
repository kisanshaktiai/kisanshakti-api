import os, requests, json, datetime, tempfile, logging, traceback
from supabase import create_client
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from requests.adapters import HTTPAdapter, Retry
from shapely import wkb, wkt
from shapely.geometry import mapping  # shapely -> GeoJSON dict
import planetary_computer as pc
import rasterio
import numpy as np

# ---------------- Config ----------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

B2_APP_KEY_ID = os.environ.get("B2_KEY_ID")
B2_APP_KEY = os.environ.get("B2_APP_KEY")
B2_BUCKET_NAME = os.environ.get("B2_BUCKET_RAW", "kisanshakti-ndvi-tiles")
B2_PREFIX = os.environ.get("B2_PREFIX", "tiles/")   # optional prefix

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
    """Get MGRS tiles where is_agri=True"""
    try:
        resp = supabase.table("mgrs_tiles") \
            .select("tile_id, geometry") \
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

def compress_and_upload(local_path, b2_key):
    """Compress a raster to LZW COG and upload to B2"""
    compressed_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    try:
        with rasterio.open(local_path) as src:
            data = src.read()
            meta = src.meta.copy()
            meta.update(
                compress="LZW",        # lossless compression
                tiled=True,            # cloud-optimized layout
                blockxsize=512,
                blockysize=512
            )
            with rasterio.open(compressed_tmp.name, "w", **meta) as dst:
                dst.write(data)

        logging.info(f"Uploading compressed COG to B2: {B2_BUCKET_NAME}/{b2_key}")
        bucket.upload_local_file(local_file=compressed_tmp.name, file_name=b2_key)
        return f"b2://{B2_BUCKET_NAME}/{b2_key}"
    except Exception as e:
        logging.error(f"Compression/upload failed: {e}\n{traceback.format_exc()}")
        return None
    finally:
        try: os.remove(compressed_tmp.name)
        except: pass

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

        # Return path to compressed local file (safe for rasterio open)
        return compressed_tmp.name, f"b2://{B2_BUCKET_NAME}/{b2_path}"

    except Exception as e:
        logging.error(f"Download/compress/upload failed: {e}\n{traceback.format_exc()}")
        return None, None
    finally:
        try: os.remove(raw_tmp.name)
        except: pass
        # ⚠️ do NOT delete compressed_tmp.name, because compute_and_upload_ndvi still needs it

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
    except:
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
    ndvi_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    try:
        with rasterio.open(red_local) as rsrc, rasterio.open(nir_local) as nsrc:
            red = rsrc.read(1).astype("float32")
            nir = nsrc.read(1).astype("float32")
            meta = rsrc.meta.copy()

        np.seterr(divide="ignore", invalid="ignore")
        ndvi = (nir - red) / (nir + red)
        ndvi = np.where((nir + red) == 0, np.nan, ndvi)

        valid = ~np.isnan(ndvi)
        stats = {}
        if valid.sum() > 0:
            stats = {
                "ndvi_min": float(np.nanmin(ndvi)),
                "ndvi_max": float(np.nanmax(ndvi)),
                "ndvi_mean": float(np.nanmean(ndvi)),
                "ndvi_std_dev": float(np.nanstd(ndvi)),
                "vegetation_coverage_percent": float((ndvi > 0.3).sum() / valid.sum() * 100.0),
                "data_completeness_percent": float(valid.sum() / ndvi.size * 100.0)
            }

        meta.update(dtype=rasterio.float32, count=1, compress="LZW", tiled=True)
        with rasterio.open(ndvi_tmp.name, "w", **meta) as dst:
            dst.write(ndvi.astype(rasterio.float32), 1)

        ndvi_b2_path = f"{B2_PREFIX}ndvi/{tile_id}/{acq_date}/ndvi.tif"
        bucket.upload_local_file(local_file=ndvi_tmp.name, file_name=ndvi_b2_path)
        return f"b2://{B2_BUCKET_NAME}/{ndvi_b2_path}", stats
    finally:
        try: os.remove(ndvi_tmp.name)
        except: pass

# ---------------- Main process ----------------
def process_tile(tile):
    try:
        tile_id = tile["tile_id"]
        geom_value = tile["geometry"]

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

        # download + compress + upload Red/NIR
        red_local, red_b2 = download_band(red_url, f"{B2_PREFIX}raw/{tile_id}/{acq_date}/B04.tif")
        nir_local, nir_b2 = download_band(nir_url, f"{B2_PREFIX}raw/{tile_id}/{acq_date}/B08.tif")
        if not red_local or not nir_local:
            logging.error(f"❌ Failed to download/compress Red or NIR for {tile_id}")
            return False

        # compute NDVI (compressed)
        ndvi_b2, stats = compute_and_upload_ndvi(tile_id, acq_date, red_local, nir_local)
        if not ndvi_b2:
            logging.error(f"❌ Failed NDVI computation for {tile_id}")
            return False

        # cleanup temp Red/NIR after NDVI is done
        try:
            os.remove(red_local)
            os.remove(nir_local)
        except Exception:
            pass

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
            **stats
        }

        try:
            resp = supabase.table("satellite_tiles") \
                .upsert(payload, on_conflict=["tile_id", "acquisition_date", "collection"]) \
                .execute()
            logging.info(f"✅ Upserted {tile_id} {acq_date}, payload={payload}")
            logging.info(f"📦 Supabase raw response: {resp}")
        except Exception as db_err:
            logging.error(f"❌ Supabase upsert failed for {tile_id} {acq_date}: {db_err}\nPayload={payload}")

        return True

    except Exception as e:
        logging.error(f"process_tile error for {tile.get('tile_id')}: {e}\n{traceback.format_exc()}")
        return False

def main(cloud_cover=20, lookback_days=5):
    global CLOUD_COVER, LOOKBACK_DAYS
    CLOUD_COVER = int(cloud_cover)
    LOOKBACK_DAYS = int(lookback_days)

    processed = 0
    tiles = fetch_agri_tiles()
    for t in tiles:
        if process_tile(t):
            processed += 1
    logging.info(f"Finished: processed {processed}/{len(tiles)} tiles")
    return processed

if __name__ == "__main__":
    cc = int(os.environ.get("RUN_CLOUD_COVER", CLOUD_COVER))
    lb = int(os.environ.get("RUN_LOOKBACK_DAYS", LOOKBACK_DAYS))
    main(cc, lb)




