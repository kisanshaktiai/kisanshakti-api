# ──────────────────────────────────────────────────────────────────────────────
# File: tile_fetch_worker.py
# Version: v2025.10.29 — NDVI Stable Release v4.2.0
# Author: Amarsinh Patil
# Purpose:
#   Fetch Sentinel-2 tiles via Microsoft Planetary Computer (MPC),
#   download RED/NIR bands, compute accurate NDVI, compress GeoTIFF,
#   upload to Backblaze B2, and update Supabase `satellite_tiles` table.
#
# Notes:
#   ✅ Structure and reliability same as Version 4.0.0
#   ✅ Adds accurate NDVI computation (float32, masked, clipped)
#   ✅ Uses internal DEFLATE compression (COG-style tiling)
#   ✅ Safe bbox_geom (SRID=4326)
#   ✅ Fully compatible with ndvi_land_worker.py and ndvi_land_api.py
# ──────────────────────────────────────────────────────────────────────────────

import os, requests, json, datetime, tempfile, logging, traceback
from supabase import create_client
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from requests.adapters import HTTPAdapter, Retry
from shapely import wkb, wkt
from shapely.geometry import mapping, shape
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

# Retry / download controls
RETRY_TOTAL = int(os.environ.get("HTTP_RETRIES", "3"))
DOWNSAMPLE_FACTOR = int(os.environ.get("DOWNSAMPLE_FACTOR", "1"))

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tile-fetch-worker-v4.2")

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

# requests session
session = requests.Session()
retries = Retry(total=RETRY_TOTAL, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

# ---------------- Helpers ----------------
def fetch_agri_tiles():
    """Fetch MGRS tiles from Supabase where is_agri=True"""
    try:
        resp = supabase.table("mgrs_tiles").select("tile_id, geometry, country_id, id").eq("is_agri", True).execute()
        tiles = resp.data or []
        logger.info(f"Fetched {len(tiles)} agri tiles")
        return tiles
    except Exception as e:
        logger.error(f"Failed to fetch agri tiles: {e}")
        return []

def decode_geom_to_geojson(geom_value):
    """Decode geometry to valid GeoJSON dict"""
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
                try: return json.loads(s)
                except: pass
            try: return mapping(wkt.loads(s))
            except: pass
            try: return mapping(wkb.loads(bytes.fromhex(s)))
            except: pass
        return None
    except Exception as e:
        logger.error(f"decode_geom_to_geojson failed: {e}")
        return None

def extract_bbox(geom_json):
    """Get bbox polygon from a GeoJSON geometry"""
    try:
        if not geom_json: return None
        geom = shape(geom_json)
        xmin, ymin, xmax, ymax = geom.bounds
        return {
            "type": "Polygon",
            "coordinates": [[
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax],
                [xmin, ymin]
            ]]
        }
    except Exception as e:
        logger.error(f"Failed to extract bbox: {e}")
        return None

def _signed_asset_url(assets, key):
    """Return signed URL for given asset key"""
    href = assets.get(key, {}).get("href")
    if not href:
        return None
    try:
        return pc.sign(href)
    except Exception:
        return href

def check_b2_file_exists(b2_path):
    """Check if file exists in B2"""
    try:
        info = bucket.get_file_info_by_name(b2_path)
        return True, info.size if hasattr(info, "size") else None
    except Exception:
        return False, None

def get_b2_paths(tile_id, acq_date):
    """Standardize B2 paths"""
    return {
        "red": f"{B2_PREFIX}raw/{tile_id}/{acq_date}/B04.tif",
        "nir": f"{B2_PREFIX}raw/{tile_id}/{acq_date}/B08.tif",
        "ndvi": f"{B2_PREFIX}ndvi/{tile_id}/{acq_date}/ndvi.tif"
    }

def check_existing_files(tile_id, acq_date):
    """Check if raw and NDVI files already exist in B2"""
    paths = get_b2_paths(tile_id, acq_date)
    red_exists, red_size = check_b2_file_exists(paths["red"])
    nir_exists, nir_size = check_b2_file_exists(paths["nir"])
    ndvi_exists, ndvi_size = check_b2_file_exists(paths["ndvi"])
    logger.info(f"📂 Existing files in B2: Red={red_exists}, NIR={nir_exists}, NDVI={ndvi_exists}")
    return {"red": red_exists, "nir": nir_exists, "ndvi": ndvi_exists}, paths, {"red": red_size, "nir": nir_size, "ndvi": ndvi_size}

def download_file_stream(url, local_path, timeout=60):
    """Download file via streaming"""
    try:
        with session.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"Download failed: {url} - {e}")
        return False

def upload_file_to_b2(local_file_path, b2_dest_path):
    """Upload file to Backblaze B2"""
    try:
        with open(local_file_path, "rb") as fh:
            data = fh.read()
        bucket.upload_bytes(data, b2_dest_path)
        logger.info(f"☁️ Uploaded to B2: {b2_dest_path} ({len(data)/1024/1024:.2f} MB)")
        return True
    except Exception as e:
        logger.error(f"Upload failed for {b2_dest_path}: {e}")
        return False

# ---------------- NDVI Calculation ----------------
def compute_ndvi_from_files(red_path, nir_path, out_ndvi_path):
    """
    Compute NDVI = (NIR - RED) / (NIR + RED)
    Applies clipping, nodata masking, and internal compression.
    """
    try:
        with rasterio.open(red_path) as red, rasterio.open(nir_path) as nir:
            r = red.read(1).astype("float32")
            n = nir.read(1).astype("float32")

            # Safe NDVI calculation
            np.seterr(divide="ignore", invalid="ignore")
            ndvi = (n - r) / (n + r)
            ndvi = np.clip(ndvi, -1, 1)
            ndvi[(r == 0) & (n == 0)] = -1.0  # nodata

            profile = red.profile.copy()
            profile.update({
                "dtype": "float32",
                "count": 1,
                "compress": "deflate",
                "predictor": 2,
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "driver": "GTiff",
                "nodata": -1.0
            })

            with rasterio.open(out_ndvi_path, "w", **profile) as dst:
                dst.write(ndvi, 1)

        logger.info(f"✅ NDVI GeoTIFF written: {out_ndvi_path}")
        return True
    except Exception as e:
        logger.error(f"compute_ndvi_from_files failed: {e}\n{traceback.format_exc()}")
        return False

# ---------------- Core Processing ----------------
def process_tile(tile):
    """Process a single MGRS tile"""
    try:
        tile_id = tile["tile_id"]
        geom_value = tile["geometry"]
        country_id = tile.get("country_id")
        mgrs_tile_id = tile.get("id")

        today = datetime.date.today()
        start_date = (today - datetime.timedelta(days=LOOKBACK_DAYS)).isoformat()
        end_date = today.isoformat()

        # STAC search
        geom_json = decode_geom_to_geojson(geom_value)
        if not geom_json:
            logger.warning(f"Skipping {tile_id} — invalid geometry")
            return False

        body = {
            "collections": [MPC_COLLECTION],
            "intersects": geom_json,
            "datetime": f"{start_date}/{end_date}",
            "query": {"eo:cloud_cover": {"lt": CLOUD_COVER}},
        }

        resp = session.post(MPC_STAC, json=body, timeout=45)
        if not resp.ok:
            logger.error(f"STAC query failed for {tile_id}: {resp.status_code}")
            return False

        features = resp.json().get("features", [])
        if not features:
            logger.info(f"No Sentinel-2 scenes found for {tile_id}")
            return False

        # Pick best (lowest cloud, latest)
        scene = sorted(features, key=lambda s: (
            s["properties"].get("eo:cloud_cover", 100),
            -datetime.datetime.fromisoformat(s["properties"]["datetime"].replace("Z", "+00:00")).timestamp()
        ))[0]

        acq_date = scene["properties"]["datetime"].split("T")[0]
        cloud_cover = scene["properties"].get("eo:cloud_cover")

        # Asset URLs
        assets = scene.get("assets", {})
        red_url = _signed_asset_url(assets, "B04")
        nir_url = _signed_asset_url(assets, "B08")
        if not red_url or not nir_url:
            logger.error(f"Missing B04/B08 assets for {tile_id}")
            return False

        exists, paths, sizes = check_existing_files(tile_id, acq_date)

        with tempfile.TemporaryDirectory() as tmp:
            red_path = os.path.join(tmp, "B04.tif")
            nir_path = os.path.join(tmp, "B08.tif")
            ndvi_path = os.path.join(tmp, "ndvi.tif")

            if not download_file_stream(red_url, red_path):
                return False
            if not download_file_stream(nir_url, nir_path):
                return False

            if not compute_ndvi_from_files(red_path, nir_path, ndvi_path):
                return False

            upload_file_to_b2(red_path, paths["red"])
            upload_file_to_b2(nir_path, paths["nir"])
            upload_file_to_b2(ndvi_path, paths["ndvi"])

        # bbox / bbox_geom
        bbox = extract_bbox(geom_json)
        bbox_geom_wkt = None
        if bbox:
            try:
                geom_obj = shape(bbox)
                if geom_obj.is_valid:
                    bbox_geom_wkt = f"SRID=4326;{geom_obj.wkt}"
            except Exception as e:
                logger.warning(f"bbox_geom error for {tile_id}: {e}")

        now = datetime.datetime.utcnow().isoformat() + "Z"
        payload = {
            "tile_id": tile_id,
            "acquisition_date": acq_date,
            "collection": MPC_COLLECTION.upper(),
            "processing_level": "L2A",
            "cloud_cover": cloud_cover,
            "red_band_path": paths["red"],
            "nir_band_path": paths["nir"],
            "ndvi_path": paths["ndvi"],
            "file_size_mb": sizes.get("ndvi") or 0,
            "resolution": "10m",
            "status": "ready",
            "updated_at": now,
            "processing_completed_at": now,
            "ndvi_calculation_timestamp": now,
            "api_source": "planetary_computer",
            "processing_method": "computed_from_bands",
            "actual_download_status": "downloaded",
            "processing_stage": "completed",
            "country_id": country_id,
            "mgrs_tile_id": mgrs_tile_id,
            "bbox": bbox,
            "bbox_geom": bbox_geom_wkt,
        }

        supabase.table("satellite_tiles").upsert(payload, on_conflict="tile_id,acquisition_date,collection").execute()
        logger.info(f"✅ Updated satellite_tiles for {tile_id} ({acq_date})")

        return True

    except Exception as e:
        logger.error(f"❌ process_tile failed for {tile.get('tile_id')}: {e}\n{traceback.format_exc()}")
        return False

# ---------------- Main Entry ----------------
def main(cloud_cover=20, lookback_days=5):
    global CLOUD_COVER, LOOKBACK_DAYS
    CLOUD_COVER, LOOKBACK_DAYS = int(cloud_cover), int(lookback_days)
    logger.info(f"🚀 Starting tile fetch worker (cloud_cover<={CLOUD_COVER}%, lookback={LOOKBACK_DAYS} days)")
    tiles = fetch_agri_tiles()
    if not tiles:
        logger.warning("⚠️ No agri tiles found")
        return 0

    success = 0
    for i, t in enumerate(tiles, 1):
        tid = t.get("tile_id")
        logger.info(f"🔄 [{i}/{len(tiles)}] Processing {tid}")
        if process_tile(t):
            success += 1
            logger.info(f"✅ Done {tid}")
        else:
            logger.info(f"⏭️ Skipped {tid}")
    logger.info(f"✨ Completed {success}/{len(tiles)} successfully.")
    return success

if __name__ == "__main__":
    cc = int(os.environ.get("RUN_CLOUD_COVER", CLOUD_COVER))
    lb = int(os.environ.get("RUN_LOOKBACK_DAYS", LOOKBACK_DAYS))
    main(cc, lb)
    logger.info("✅ NDVI Tile Fetch Worker finished cleanly.")
