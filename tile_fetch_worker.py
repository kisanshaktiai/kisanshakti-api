import os, requests, json, datetime, tempfile, logging, traceback
from supabase import create_client
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from requests.adapters import HTTPAdapter, Retry
from shapely import wkb
from shapely.geometry import mapping  # shapely -> GeoJSON dict

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
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

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
        resp = supabase.table("mgrs_tiles").select("tile_id, geometry").eq("is_agri", True).execute()
        return resp.data or []
    except Exception as e:
        logging.error(f"Failed to fetch agri tiles: {e}")
        return []


def decode_geom_to_geojson(geom_value):
    """Handle geometry from Supabase (either WKB hex string or GeoJSON dict)"""
    try:
        if isinstance(geom_value, str):  # WKB hex
            geom = wkb.loads(bytes.fromhex(geom_value))
            return mapping(geom)
        elif isinstance(geom_value, dict):  # already GeoJSON
            return geom_value
        else:
            logging.error(f"Unexpected geometry type: {type(geom_value)}")
            return None
    except Exception as e:
        logging.error(f"Failed to decode geometry: {e}\n{traceback.format_exc()}")
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
        resp = session.post(MPC_STAC, json=body, timeout=30)
        resp.raise_for_status()
        return resp.json().get("features", [])
    except Exception as e:
        logging.error(f"MPC query failed: {e}")
        return []


def download_to_b2(url, b2_path):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    try:
        r = session.get(url, stream=True, timeout=60)
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                tmp.write(chunk)
        tmp.close()
        bucket.upload_local_file(local_file=tmp.name, file_name=b2_path)
        return f"b2://{B2_BUCKET_NAME}/{b2_path}"
    except Exception as e:
        logging.error(f"Download/upload failed for {url}: {e}")
        return None
    finally:
        try: os.remove(tmp.name)
        except: pass


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
    except Exception as e:
        logging.error(f"Scene sorting failed: {e}")
        return None


def process_tile(tile):
    tile_id = tile["tile_id"]
    geom_value = tile["geometry"]

    today = datetime.date.today()
    start_date = (today - datetime.timedelta(days=LOOKBACK_DAYS)).isoformat()
    end_date = today.isoformat()

    scenes = query_mpc(geom_value, start_date, end_date)
    if not scenes:
        logging.info(f"No scenes for {tile_id}")
        return

    scene = pick_best_scene(scenes)
    if not scene:
        return

    acq_date = scene["properties"]["datetime"].split("T")[0]
    assets = scene.get("assets", {})
    red_url = assets.get("B04", {}).get("href")
    nir_url = assets.get("B08", {}).get("href")
    if not red_url or not nir_url:
        return

    red_b2 = download_to_b2(red_url, f"{B2_PREFIX}raw/{tile_id}/{acq_date}/B04.tif")
    nir_b2 = download_to_b2(nir_url, f"{B2_PREFIX}raw/{tile_id}/{acq_date}/B08.tif")

    if red_b2 and nir_b2:
        supabase.table("satellite_tiles").upsert({
            "tile_id": tile_id,
            "acquisition_date": acq_date,
            "collection": MPC_COLLECTION.upper(),
            "red_band_path": red_b2,
            "nir_band_path": nir_b2,
            "status": "downloaded",
            "api_source": "planetary_computer"
        }, on_conflict=["tile_id", "acquisition_date", "collection"]).execute()
        logging.info(f"✅ Stored {tile_id} {acq_date}")


def main(cloud_cover=20, lookback_days=5):
    global CLOUD_COVER, LOOKBACK_DAYS
    CLOUD_COVER = cloud_cover
    LOOKBACK_DAYS = lookback_days

    for t in fetch_agri_tiles():
        process_tile(t)


if __name__ == "__main__":
    main()

