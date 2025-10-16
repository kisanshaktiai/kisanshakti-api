#-------------------------------------------------
# Tiles feting worker V1.0.0
#------------------------------------
# Update On 16/10/2025 due to mgrs_tiles table chnage 

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
    """Get MGRS tiles where is_agri=True, including geojson_geometry"""
    try:
        resp = supabase.table("mgrs_tiles") \
            .select("tile_id, geojson_geometry, geometry, country_id, id") \
            .eq("is_agri", True) \
            .execute()
        tiles = resp.data or []
        logging.info(f"Fetched {len(tiles)} agri tiles")
        return tiles
    except Exception as e:
        logging.error(f"Failed to fetch agri tiles: {e}")
        return []

def decode_geom_to_geojson(geom_value):
    """Decode geometry (WKB/WKT/JSON string) into GeoJSON dict"""
    try:
        if geom_value is None:
            return None
        if isinstance(geom_value, dict) and "type" in geom_value:
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

def get_tile_geometry(tile):
    """
    Returns GeoJSON geometry dict from mgrs_tiles row.
    Priority: geojson_geometry (jsonb) > geometry (WKB/WKT).
    """
    try:
        if tile.get("geojson_geometry"):
            if isinstance(tile["geojson_geometry"], str):
                return json.loads(tile["geojson_geometry"])
            return tile["geojson_geometry"]
        return decode_geom_to_geojson(tile.get("geometry"))
    except Exception as e:
        logging.error(f"Failed to parse geometry for tile {tile.get('tile_id')}: {e}")
        return None

def extract_bbox(geom_json):
    """Extract bounding box polygon from GeoJSON geometry"""
    try:
        if not geom_json:
            return None
        geom = shape(geom_json)
        minx, miny, maxx, maxy = geom.bounds
        return {
            "type": "Polygon",
            "coordinates": [[
                [minx, miny], [maxx, miny],
                [maxx, maxy], [minx, maxy],
                [minx, miny]
            ]]
        }
    except Exception as e:
        logging.error(f"Failed to extract bbox: {e}")
        return None

def query_mpc(tile_geom, start_date, end_date):
    """Query Microsoft Planetary Computer STAC API"""
    try:
        if not tile_geom:
            return []
        body = {
            "collections": [MPC_COLLECTION],
            "intersects": tile_geom,
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

# ---------- (rest of helpers unchanged) ----------
# All other functions: check_b2_file_exists, download_band, compute_and_upload_ndvi, etc.
# remain exactly the same as your previous version.

# ---------------- Main process ----------------
def process_tile(tile):
    """Process a single tile: fetch scene, download bands if needed, compute NDVI, save to DB"""
    try:
        tile_id = tile["tile_id"]
        geom_json = get_tile_geometry(tile)   # ✅ use geojson_geometry
        country_id = tile.get("country_id")
        mgrs_tile_id = tile.get("id")

        today = datetime.date.today()
        start_date = (today - datetime.timedelta(days=LOOKBACK_DAYS)).isoformat()
        end_date = today.isoformat()

        scenes = query_mpc(geom_json, start_date, end_date)
        if not scenes:
            logging.info(f"🔍 No scenes for {tile_id} in {start_date}..{end_date}")
            return False

        scene = pick_best_scene(scenes)
        if not scene:
            logging.info(f"⚠️ No valid scene after sorting for {tile_id}")
            return False

        acq_date = scene["properties"]["datetime"].split("T")[0]
        cloud_cover = scene["properties"].get("eo:cloud_cover")

        # ✅ bbox from geojson_geometry
        bbox = extract_bbox(geom_json)

        # rest of process_tile logic unchanged — checks, downloads, NDVI, upsert...
        # ...
        # just ensure the final payload includes:
        # "bbox": json.dumps(bbox) if bbox else None
        # before upserting into satellite_tiles

        # (insert your unchanged code from your previous version here)
        # -- from check_existing_files() onwards --

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
