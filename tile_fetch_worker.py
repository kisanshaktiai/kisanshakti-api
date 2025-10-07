import os, requests, json, datetime, tempfile, logging, traceback
from supabase import create_client
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from requests.adapters import HTTPAdapter, Retry
from shapely import wkb, wkt
from shapely.geometry import mapping  # shapely -> GeoJSON dict
import planetary_computer as pc

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
        # Keep it simple; we will decode various formats in Python
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
    """Handle geometry from Supabase: GeoJSON dict, WKB hex (str), bytes (WKB), or WKT string"""
    try:
        if geom_value is None:
            logging.warning("Geometry is None")
            return None

        # Case 1: already GeoJSON dict
        if isinstance(geom_value, dict) and "type" in geom_value and "coordinates" in geom_value:
            return geom_value

        # Case 2: bytes -> WKB
        if isinstance(geom_value, (bytes, bytearray)):
            geom = wkb.loads(geom_value)
            gj = mapping(geom)
            return gj

        # Case 3: string -> could be WKB hex OR WKT OR JSON string
        if isinstance(geom_value, str):
            s = geom_value.strip()
            preview = s[:60].replace("\n", "")
            logging.debug(f"Geometry string preview: {preview}...")

            # Try JSON-GeoJSON string
            if s.startswith("{") and s.endswith("}"):
                try:
                    d = json.loads(s)
                    if isinstance(d, dict) and "type" in d and "coordinates" in d:
                        return d
                except Exception:
                    pass

            # Try WKT
            try:
                geom = wkt.loads(s)  # e.g., "POLYGON((...))"
                return mapping(geom)
            except Exception:
                pass

            # Try WKB hex
            try:
                geom = wkb.loads(bytes.fromhex(s))
                return mapping(geom)
            except Exception:
                pass

            logging.error("Failed to parse geometry string as JSON/WKT/WKB.")
            return None

        logging.error(f"Unexpected geometry type: {type(geom_value)}")
        return None
    except Exception as e:
        logging.error(f"Failed to decode geometry: {e}\n{traceback.format_exc()}")
        return None

def query_mpc(tile_geom, start_date, end_date):
    try:
        geom_json = decode_geom_to_geojson(tile_geom)
        if not geom_json:
            logging.warning("Skipping MPC query: geometry decode failed.")
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
            logging.error(f"STAC HTTP {resp.status_code}: {resp.text}")
            return []
        out = resp.json()
        feats = out.get("features", [])
        logging.info(f"STAC returned {len(feats)} features")
        return feats
    except Exception as e:
        logging.error(f"MPC query failed: {e}\n{traceback.format_exc()}")
        return []

def _signed_asset_url(assets, primary_key, fallback_key=None):
    """Get a signed Planetary Computer asset URL for a given key."""
    href = None
    if primary_key in assets and "href" in assets[primary_key]:
        href = assets[primary_key]["href"]
    elif fallback_key and fallback_key in assets and "href" in assets[fallback_key]:
        href = assets[fallback_key]["href"]
    if not href:
        return None
    try:
        # Planetary Computer assets often require signing
        return pc.sign(href)
    except Exception as e:
        logging.warning(f"Failed to sign asset URL, using raw href. {e}")
        return href

def download_to_b2(url, b2_path):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    try:
        logging.info(f"Downloading: {url}")
        r = session.get(url, stream=True, timeout=120)
        if not r.ok:
            logging.error(f"Download HTTP {r.status_code}: {r.text[:300]}")
            r.raise_for_status()
        size = 0
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                tmp.write(chunk)
                size += len(chunk)
        tmp.close()
        logging.info(f"Uploading to B2: {B2_BUCKET_NAME}/{b2_path} (bytes={size})")
        bucket.upload_local_file(local_file=tmp.name, file_name=b2_path)
        return f"b2://{B2_BUCKET_NAME}/{b2_path}"
    except Exception as e:
        logging.error(f"Download/upload failed for {url}: {e}\n{traceback.format_exc()}")
        return None
    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass

def _record_exists(tile_id, acq_date):
    try:
        resp = supabase.table("satellite_tiles") \
            .select("id,status,red_band_path,nir_band_path") \
            .eq("tile_id", tile_id) \
            .eq("acquisition_date", acq_date) \
            .eq("collection", MPC_COLLECTION.upper()) \
            .limit(1).execute()
        rows = resp.data or []
        if not rows:
            return False, None
        return True, rows[0]
    except Exception as e:
        logging.warning(f"existence check failed: {e}")
        return False, None

def pick_best_scene(scenes):
    try:
        return sorted(
            scenes,
            key=lambda s: (
                s.get("properties", {}).get("eo:cloud_cover", 100.0),
                -datetime.datetime.fromisoformat(
                    s["properties"]["datetime"].replace("Z", "+00:00")
                ).timestamp()
            )
        )[0] if scenes else None
    except Exception as e:
        logging.error(f"Scene sorting failed: {e}")
        return None

def process_tile(tile):
    try:
        tile_id = tile["tile_id"]
        geom_value = tile["geometry"]

        today = datetime.date.today()
        start_date = (today - datetime.timedelta(days=LOOKBACK_DAYS)).isoformat()
        end_date = today.isoformat()

        scenes = query_mpc(geom_value, start_date, end_date)
        if not scenes:
            logging.info(f"🔍 No scenes for {tile_id} in window {start_date}..{end_date}")
            return False

        scene = pick_best_scene(scenes)
        if not scene:
            logging.info(f"⚠️ No valid scene after sorting for {tile_id}")
            return False

        acq_date = scene["properties"]["datetime"].split("T")[0]
        exists, row = _record_exists(tile_id, acq_date)
        if exists and row and row.get("status") in ("downloaded", "ready"):
            logging.info(f"⏩ Skipping existing {tile_id} {acq_date} (status={row.get('status')})")
            return False

        assets = scene.get("assets", {})
        # Try red/nir first, fallback to band IDs
        red_url = _signed_asset_url(assets, "red", "B04")
        nir_url = _signed_asset_url(assets, "nir", "B08")

        if not red_url or not nir_url:
            logging.warning(f"❌ Missing red/nir URLs for {tile_id} {acq_date}")
            return False

        red_b2 = download_to_b2(red_url, f"{B2_PREFIX}raw/{tile_id}/{acq_date}/B04.tif")
        nir_b2 = download_to_b2(nir_url, f"{B2_PREFIX}raw/{tile_id}/{acq_date}/B08.tif")

        if not red_b2 or not nir_b2:
            logging.error(f"❌ Upload to B2 failed for {tile_id} {acq_date}")
            return False

        payload = {
            "tile_id": tile_id,
            "acquisition_date": acq_date,
            "collection": MPC_COLLECTION.upper(),
            "red_band_path": red_b2,
            "nir_band_path": nir_b2,
            "status": "downloaded",
            "api_source": "planetary_computer",
            "updated_at": datetime.datetime.utcnow().isoformat() + "Z"
        }

        supabase.table("satellite_tiles") \
            .upsert(payload, on_conflict="tile_id,acquisition_date,collection") \
            .execute()

        logging.info(f"✅ Stored {tile_id} {acq_date} -> red={red_b2} nir={nir_b2}")
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
        ok = process_tile(t)
        if ok:
            processed += 1
    logging.info(f"Finished: processed {processed}/{len(tiles)} tiles")
    return processed

if __name__ == "__main__":
    # Example: drive it with env or defaults
    cc = int(os.environ.get("RUN_CLOUD_COVER", CLOUD_COVER))
    lb = int(os.environ.get("RUN_LOOKBACK_DAYS", LOOKBACK_DAYS))
    main(cc, lb)
