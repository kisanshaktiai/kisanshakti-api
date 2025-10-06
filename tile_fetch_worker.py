import os, requests, json, datetime, tempfile, logging, traceback
from supabase import create_client
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from requests.adapters import HTTPAdapter, Retry

# ---------------- Config ----------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
B2_APP_KEY_ID = os.environ.get("B2_KEY_ID")
B2_APP_KEY = os.environ.get("B2_APP_KEY")
B2_BUCKET_NAME = os.environ.get("B2_BUCKET_RAW", "ksai-sat-raw")
MPC_STAC = os.environ.get("MPC_STAC_BASE", "https://planetarycomputer.microsoft.com/api/stac/v1/search")
MPC_COLLECTION = os.environ.get("MPC_COLLECTION", "sentinel-2-l2a")
CLOUD_COVER = int(os.environ.get("DEFAULT_CLOUD_COVER_MAX", "30"))
LOOKBACK_DAYS = int(os.environ.get("MAX_SCENE_LOOKBACK_DAYS", "5"))

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------- Clients ----------------
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    logging.critical(f"Supabase client init failed: {e}")
    raise

try:
    info = InMemoryAccountInfo()
    b2_api = B2Api(info)
    b2_api.authorize_account("production", B2_APP_KEY_ID, B2_APP_KEY)
    bucket = b2_api.get_bucket_by_name(B2_BUCKET_NAME)
except Exception as e:
    logging.critical(f"Backblaze B2 init failed: {e}")
    raise

session = requests.Session()
retries = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))


# ---------------- Helpers ----------------
def fetch_agri_tiles():
    """Get MGRS tiles where is_agri=True"""
    try:
        resp = supabase.table("mgrs_tiles").select("tile_id, geometry").eq("is_agri", True).execute()
        if not resp.data:
            logging.warning("No agri tiles found.")
            return []
        return resp.data
    except Exception as e:
        logging.error(f"Failed to fetch agri tiles: {e}")
        return []


def query_mpc(tile_geom, start_date, end_date):
    """Query MPC for Sentinel-2 scenes intersecting tile geometry"""
    try:
        geom_json = json.loads(tile_geom) if isinstance(tile_geom, str) else tile_geom
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
        logging.error(f"MPC query failed: {e}\n{traceback.format_exc()}")
        return []


def download_to_b2(url, b2_path):
    """Download a COG band and upload to B2"""
    tmp = tempfile.NamedTemporaryFile(delete=False)
    try:
        r = session.get(url, stream=True, timeout=60)
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                tmp.write(chunk)
        tmp.close()

        bucket.upload_local_file(local_file=tmp.name, file_name=b2_path)
        logging.info(f"Uploaded to B2: {b2_path}")
        return f"b2://{B2_BUCKET_NAME}/{b2_path}"
    except Exception as e:
        logging.error(f"Download/upload failed for {url}: {e}\n{traceback.format_exc()}")
        return None
    finally:
        try:
            os.remove(tmp.name)
        except:
            pass


def pick_best_scene(scenes):
    """Choose most recent, lowest cloud cover scene"""
    try:
        if not scenes:
            return None
        scenes_sorted = sorted(
            scenes,
            key=lambda s: (
                s["properties"].get("eo:cloud_cover", 100),
                -datetime.datetime.fromisoformat(
                    s["properties"]["datetime"].replace("Z", "+00:00")
                ).timestamp()
            )
        )
        return scenes_sorted[0]
    except Exception as e:
        logging.error(f"Scene sorting failed: {e}\n{traceback.format_exc()}")
        return None


def process_tile(tile):
    """Process a single tile: query MPC, download bands, store in Supabase"""
    tile_id = tile["tile_id"]
    geom = tile["geometry"]

    today = datetime.date.today()
    start_date = (today - datetime.timedelta(days=LOOKBACK_DAYS)).isoformat()
    end_date = today.isoformat()

    try:
        scenes = query_mpc(geom, start_date, end_date)
        if not scenes:
            logging.info(f"No scenes found for tile {tile_id}")
            return

        scene = pick_best_scene(scenes)
        if not scene:
            logging.warning(f"No valid scene selected for tile {tile_id}")
            return

        acq_date = scene["properties"]["datetime"].split("T")[0]
        cloud = scene["properties"]["eo:cloud_cover"]

        assets = scene.get("assets", {})
        red_url = assets.get("B04", {}).get("href")
        nir_url = assets.get("B08", {}).get("href")
        if not red_url or not nir_url:
            logging.warning(f"Missing bands for {tile_id} on {acq_date}")
            return

        red_b2 = download_to_b2(red_url, f"tiles/raw/{tile_id}/{acq_date}/B04.tif")
        nir_b2 = download_to_b2(nir_url, f"tiles/raw/{tile_id}/{acq_date}/B08.tif")

        if not red_b2 or not nir_b2:
            logging.error(f"Failed to store bands for tile {tile_id}")
            return

        # Insert into satellite_tiles
        supabase.table("satellite_tiles").upsert({
            "tile_id": tile_id,
            "acquisition_date": acq_date,
            "collection": MPC_COLLECTION.upper(),
            "cloud_cover": cloud,
            "red_band_path": red_b2,
            "nir_band_path": nir_b2,
            "status": "downloaded",
            "api_source": "planetary_computer",
            "copernicus_red_band_url": red_url,
            "copernicus_nir_band_url": nir_url,
            "copernicus_download_attempted_at": datetime.datetime.utcnow().isoformat()
        }, on_conflict=["tile_id", "acquisition_date", "collection"]).execute()

        logging.info(f"✅ Tile {tile_id} {acq_date} stored successfully.")
    except Exception as e:
        logging.error(f"Tile {tile_id} failed: {e}\n{traceback.format_exc()}")


def main():
    tiles = fetch_agri_tiles()
    if not tiles:
        logging.info("No agricultural tiles to process.")
        return

    for t in tiles:
        process_tile(t)


if __name__ == "__main__":
    main()
