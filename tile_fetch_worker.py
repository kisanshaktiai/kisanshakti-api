#-------------------------------------------------
# Tiles Fetching Worker v1.1.0
#-------------------------------------------------
# Update: 18/10/2025
# - Improved Sentinel-2 fetch reliability
# - Dynamic cloud_cover & lookback_days from frontend
# - Added automatic L1C fallback if L2A not found
# - Extended retry logic & better STAC logging
#-------------------------------------------------

import os, requests, json, datetime, tempfile, logging, traceback
from supabase import create_client
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from requests.adapters import HTTPAdapter, Retry
from shapely import wkb, wkt
from shapely.geometry import mapping, shape

# --------------------------------------------------
# ENV + CONFIG
# --------------------------------------------------

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

B2_APP_KEY_ID = os.environ.get("B2_KEY_ID")
B2_APP_KEY = os.environ.get("B2_APP_KEY")
B2_BUCKET_NAME = os.environ.get("B2_BUCKET_RAW", "kisanshakti-ndvi-tiles")
B2_PREFIX = os.environ.get("B2_PREFIX", "tiles/")

MPC_STAC = os.environ.get("MPC_STAC_BASE", "https://planetarycomputer.microsoft.com/api/stac/v1/search")
DEFAULT_COLLECTIONS = ["sentinel-2-l2a", "sentinel-2-l1c"]

DOWNSAMPLE_FACTOR = int(os.environ.get("DOWNSAMPLE_FACTOR", "4"))

# --------------------------------------------------
# LOGGING SETUP
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --------------------------------------------------
# CLIENTS
# --------------------------------------------------

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

if not B2_APP_KEY_ID or not B2_APP_KEY:
    raise RuntimeError("Missing B2_KEY_ID or B2_APP_KEY")

info = InMemoryAccountInfo()
b2_api = B2Api(info)
b2_api.authorize_account("production", B2_APP_KEY_ID, B2_APP_KEY)
bucket = b2_api.get_bucket_by_name(B2_BUCKET_NAME)

session = requests.Session()
retries = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def fetch_agri_tiles():
    """Fetch MGRS tiles that contain agricultural lands."""
    try:
        resp = supabase.table("mgrs_tiles") \
            .select("tile_id, geojson_geometry, country_id, id") \
            .eq("is_agri", True) \
            .eq("is_land_contain", True) \
            .execute()
        tiles = resp.data or []
        logging.info(f"Fetched {len(tiles)} active agricultural tiles containing lands.")
        return tiles
    except Exception as e:
        logging.error(f"Failed to fetch active tiles: {e}")
        return []


def decode_geom_to_geojson(geom_value):
    """Convert geometry field to GeoJSON dict."""
    try:
        if geom_value is None:
            return None
        if isinstance(geom_value, dict) and "type" in geom_value:
            return geom_value
        if isinstance(geom_value, (bytes, bytearray)):
            return mapping(wkb.loads(geom_value))
        if isinstance(geom_value, str):
            s = geom_value.strip()
            if s.startswith("{"):
                return json.loads(s)
            try:
                return mapping(wkt.loads(s))
            except:
                return mapping(wkb.loads(bytes.fromhex(s)))
        return None
    except Exception as e:
        logging.error(f"decode_geom_to_geojson failed: {e}")
        return None


def get_tile_geometry(tile):
    """Get usable GeoJSON geometry from mgrs_tiles row."""
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
    """Compute bounding box polygon for a geometry."""
    try:
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


def query_mpc(tile_geom, start_date, end_date, cloud_cover):
    """Query Microsoft Planetary Computer STAC with fallback to L1C."""
    if not tile_geom:
        return []
    for coll in DEFAULT_COLLECTIONS:
        try:
            body = {
                "collections": [coll],
                "intersects": tile_geom,
                "datetime": f"{start_date}/{end_date}",
                "query": {"eo:cloud_cover": {"lt": cloud_cover}},
                "limit": 10
            }
            logging.info(f"STAC query: {coll}, {start_date}->{end_date}, cloud<{cloud_cover}")
            resp = session.post(MPC_STAC, json=body, timeout=45)
            if not resp.ok:
                logging.warning(f"{coll} STAC returned {resp.status_code}")
                continue
            features = resp.json().get("features", [])
            if features:
                return features
        except Exception as e:
            logging.warning(f"Query failed for {coll}: {e}")
            continue
    return []


def pick_best_scene(scenes):
    """Pick scene with lowest cloud cover, latest acquisition."""
    if not scenes:
        return None
    try:
        return sorted(
            scenes,
            key=lambda s: (
                s["properties"].get("eo:cloud_cover", 100),
                -datetime.datetime.fromisoformat(
                    s["properties"]["datetime"].replace("Z", "+00:00")
                ).timestamp()
            )
        )[0]
    except Exception as e:
        logging.error(f"pick_best_scene failed: {e}")
        return None


# --------------------------------------------------
# TILE PROCESSOR
# --------------------------------------------------

def process_tile(tile, cloud_cover, lookback_days):
    """Process one tile for Sentinel imagery and NDVI generation."""
    try:
        tile_id = tile["tile_id"]
        geom_json = get_tile_geometry(tile)
        if not geom_json:
            logging.warning(f"No valid geometry for {tile_id}")
            return False

        today = datetime.date.today()
        start_date = (today - datetime.timedelta(days=lookback_days)).isoformat()
        end_date = today.isoformat()

        scenes = query_mpc(geom_json, start_date, end_date, cloud_cover)

        # retry logic with longer window if nothing found
        if not scenes:
            logging.info(f"🔁 Retrying {tile_id} with extended 45-day window...")
            retry_start = (today - datetime.timedelta(days=45)).isoformat()
            scenes = query_mpc(geom_json, retry_start, end_date, min(cloud_cover + 20, 95))

        if not scenes:
            logging.info(f"🔍 No scenes for {tile_id} in {start_date}..{end_date}")
            return False

        scene = pick_best_scene(scenes)
        if not scene:
            logging.info(f"⚠️ No valid scene selected for {tile_id}")
            return False

        acq_date = scene["properties"]["datetime"].split("T")[0]
        cc = scene["properties"].get("eo:cloud_cover")

        logging.info(f"📦 Using scene {scene['id']} (cloud={cc}%) for {tile_id}")

        bbox = extract_bbox(geom_json)
        # (Insert your NDVI download + processing + upload logic here)

        return True

    except Exception as e:
        logging.error(f"❌ process_tile error for {tile.get('tile_id')}: {e}")
        logging.error(traceback.format_exc())
        return False


# --------------------------------------------------
# MAIN ENTRY POINT
# --------------------------------------------------

def main(cloud_cover: int, lookback_days: int):
    """Main function triggered by backend or cron."""
    logging.info(f"🚀 Starting tile worker (cloud<{cloud_cover}%, lookback={lookback_days} days)")
    logging.info(f"💾 DOWNSAMPLE_FACTOR={DOWNSAMPLE_FACTOR}")

    tiles = fetch_agri_tiles()
    if not tiles:
        logging.warning("⚠️ No tiles fetched from database")
        return 0

    processed = 0
    logging.info(f"📋 Processing {len(tiles)} tiles...")
    for i, t in enumerate(tiles, start=1):
        tile_id = t.get("tile_id")
        logging.info(f"🔄 [{i}/{len(tiles)}] Processing: {tile_id}")
        if process_tile(t, cloud_cover, lookback_days):
            processed += 1
            logging.info(f"✅ [{i}/{len(tiles)}] Success: {tile_id}")
        else:
            logging.info(f"⏭️  [{i}/{len(tiles)}] Skipped: {tile_id}")

    logging.info(f"✨ Finished: processed {processed}/{len(tiles)} tiles successfully")
    return processed


if __name__ == "__main__":
    # Frontend-sent values
    cloud_cover = int(os.environ.get("RUN_CLOUD_COVER", "40"))  # frontend value
    lookback_days = int(os.environ.get("RUN_LOOKBACK_DAYS", "30"))  # frontend value
    main(cloud_cover, lookback_days)
