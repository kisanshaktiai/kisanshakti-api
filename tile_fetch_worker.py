#-------------------------------------------------
# Tiles Fetching & Processing Worker v1.3.1
#-------------------------------------------------
# - Fixes 409 (private MPC asset) using SAS signing API
# - Fixes Supabase upsert constraint (42P10 error)
# - Fully compatible with satellite_tiles schema
# - Tested on Sentinel-2 L2A / L1C MPC assets
#-------------------------------------------------

import os, json, datetime, tempfile, logging, traceback
import requests
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask
from shapely.geometry import shape
from supabase import create_client
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from requests.adapters import HTTPAdapter, Retry

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

MPC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
MPC_SIGN_API = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"

DEFAULT_COLLECTIONS = ["sentinel-2-l2a", "sentinel-2-l1c"]

DOWNSAMPLE_FACTOR = int(os.environ.get("DOWNSAMPLE_FACTOR", "4"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --------------------------------------------------
# CLIENTS
# --------------------------------------------------
session = requests.Session()
retries = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

info = InMemoryAccountInfo()
b2_api = B2Api(info)
b2_api.authorize_account("production", B2_APP_KEY_ID, B2_APP_KEY)
bucket = b2_api.get_bucket_by_name(B2_BUCKET_NAME)

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def fetch_agri_tiles():
    try:
        resp = supabase.table("mgrs_tiles") \
            .select("id, tile_id, geojson_geometry, country_id") \
            .eq("is_agri", True).eq("is_land_contain", True).execute()
        tiles = resp.data or []
        logging.info(f"Fetched {len(tiles)} active agricultural tiles containing lands.")
        return tiles
    except Exception as e:
        logging.error(f"Failed to fetch tiles: {e}")
        return []


def sign_mpc_item(scene):
    """Sign MPC asset URLs with SAS tokens."""
    try:
        resp = session.post(MPC_SIGN_API, json=scene, timeout=30)
        if resp.ok:
            signed = resp.json()
            logging.info(f"🔏 Signed MPC assets for {scene.get('id')}")
            return signed
        else:
            logging.warning(f"Signing failed ({resp.status_code}): {resp.text}")
            return scene
    except Exception as e:
        logging.error(f"sign_mpc_item error: {e}")
        return scene


def query_mpc(tile_geom, start_date, end_date, cloud_cover):
    """Query Planetary Computer STAC for Sentinel scenes."""
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
            if resp.ok:
                feats = resp.json().get("features", [])
                if feats:
                    return feats
        except Exception as e:
            logging.warning(f"STAC query failed for {coll}: {e}")
    return []


def pick_best_scene(scenes):
    try:
        return sorted(
            scenes,
            key=lambda s: (
                s["properties"].get("eo:cloud_cover", 100),
                -datetime.datetime.fromisoformat(
                    s["properties"]["datetime"].replace("Z", "+00:00")
                ).timestamp(),
            ),
        )[0]
    except Exception as e:
        logging.error(f"pick_best_scene failed: {e}")
        return None


def download_asset(url, dest):
    """Download file with retries."""
    try:
        with session.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                    f.write(chunk)
        logging.info(f"Downloaded asset -> {dest}")
    except Exception as e:
        raise RuntimeError(f"Download failed: {e}")


def compute_ndvi(red_path, nir_path):
    """Compute NDVI and return path + stats."""
    with rasterio.open(red_path) as rred, rasterio.open(nir_path) as rnir:
        red = rred.read(1).astype("float32")
        nir = rnir.read(1).astype("float32")
        ndvi = np.where((nir + red) != 0, (nir - red) / (nir + red), np.nan)
        stats = {
            "ndvi_min": float(np.nanmin(ndvi)),
            "ndvi_max": float(np.nanmax(ndvi)),
            "ndvi_mean": float(np.nanmean(ndvi)),
            "ndvi_std_dev": float(np.nanstd(ndvi)),
            "valid_pixel_count": int(np.count_nonzero(~np.isnan(ndvi))),
            "pixel_count": int(ndvi.size),
        }
        stats["data_completeness_percent"] = round(
            (stats["valid_pixel_count"] / stats["pixel_count"]) * 100, 2
        )
        out = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
        meta = rred.meta.copy()
        meta.update({"count": 1, "dtype": "float32", "driver": "GTiff", "nodata": np.nan})
        with rasterio.open(out, "w", **meta) as dst:
            dst.write(ndvi, 1)
        logging.info(f"NDVI written -> {out}")
        return out, stats


def upload_to_b2(local_path, dest_key):
    b2_key = os.path.join(B2_PREFIX, dest_key).replace("\\", "/")
    with open(local_path, "rb") as fh:
        size = os.path.getsize(local_path)
        bucket.upload_bytes(fh.read(), b2_key)
    return f"b2://{B2_BUCKET_NAME}/{b2_key}", size


def upsert_satellite_tile(record):
    try:
        resp = supabase.table("satellite_tiles").upsert(
            record, on_conflict="tile_id,acquisition_date,collection"
        ).execute()
        logging.info(
            f"📤 Upserted satellite_tiles for tile={record.get('tile_id')} date={record.get('acquisition_date')}"
        )
        return resp
    except Exception as e:
        logging.error(f"upsert_satellite_tile error: {e}")
        logging.error(traceback.format_exc())
        raise


# --------------------------------------------------
# MAIN TILE PROCESSOR
# --------------------------------------------------

def process_tile(tile, cloud_cover, lookback_days):
    tile_id = tile.get("tile_id")
    geom = json.loads(tile["geojson_geometry"])
    today = datetime.date.today()
    start = (today - datetime.timedelta(days=lookback_days)).isoformat()
    end = today.isoformat()

    scenes = query_mpc(geom, start, end, cloud_cover)
    if not scenes:
        logging.warning(f"No scenes for {tile_id}")
        return False

    scene = pick_best_scene(scenes)
    if not scene:
        logging.warning(f"No valid scene for {tile_id}")
        return False

    scene = sign_mpc_item(scene)
    scene_id = scene["id"]
    acq_date = scene["properties"]["datetime"].split("T")[0]
    cc = scene["properties"].get("eo:cloud_cover")

    assets = scene.get("assets", {})
    red = assets.get("B04", {}).get("href") or assets.get("red", {}).get("href")
    nir = assets.get("B08", {}).get("href") or assets.get("nir", {}).get("href")

    if not red or not nir:
        logging.error(f"Missing red/nir for {scene_id}")
        return False

    # download assets
    rtmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
    ntmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
    try:
        download_asset(red, rtmp)
        download_asset(nir, ntmp)
    except Exception as e:
        logging.error(f"❌ Asset download failed for {tile_id}: {e}")
        rec = {
            "tile_id": tile_id,
            "acquisition_date": acq_date,
            "collection": scene.get("collection", "SENTINEL-2"),
            "status": "failed",
            "error_message": str(e),
            "processing_stage": "download",
            "cloud_cover": cc,
            "mgrs_tile_id": tile.get("id"),
            "updated_at": datetime.datetime.utcnow().isoformat(),
        }
        upsert_satellite_tile(rec)
        return False

    # compute NDVI
    ndvi_path, stats = compute_ndvi(rtmp, ntmp)

    # upload
    r_key = f"{tile_id}/{acq_date}/red.tif"
    n_key = f"{tile_id}/{acq_date}/nir.tif"
    ndvi_key = f"{tile_id}/{acq_date}/ndvi.tif"
    red_b2, rs = upload_to_b2(rtmp, r_key)
    nir_b2, ns = upload_to_b2(ntmp, n_key)
    ndvi_b2, ds = upload_to_b2(ndvi_path, ndvi_key)

    record = {
        "tile_id": tile_id,
        "acquisition_date": acq_date,
        "collection": scene.get("collection", "SENTINEL-2"),
        "cloud_cover": cc,
        "ndvi_path": ndvi_b2,
        "red_band_path": red_b2,
        "nir_band_path": nir_b2,
        "file_size_mb": round((rs + ns + ds) / (1024 * 1024), 2),
        "status": "processed",
        "processing_completed_at": datetime.datetime.utcnow().isoformat(),
        "updated_at": datetime.datetime.utcnow().isoformat(),
        "processing_stage": "complete",
        "mgrs_tile_id": tile["id"],
        "ndvi_min": stats["ndvi_min"],
        "ndvi_max": stats["ndvi_max"],
        "ndvi_mean": stats["ndvi_mean"],
        "ndvi_std_dev": stats["ndvi_std_dev"],
        "pixel_count": stats["pixel_count"],
        "valid_pixel_count": stats["valid_pixel_count"],
        "data_completeness_percent": stats["data_completeness_percent"],
        "actual_download_status": "downloaded",
        "api_source": "planetary_computer",
    }

    upsert_satellite_tile(record)
    logging.info(f"✅ Completed tile {tile_id}")
    return True


# --------------------------------------------------
# MAIN ENTRY
# --------------------------------------------------

def main(cloud_cover, lookback_days):
    logging.info(f"Starting tile worker (cloud<{cloud_cover}, lookback={lookback_days})")
    tiles = fetch_agri_tiles()
    processed = 0
    for i, t in enumerate(tiles, start=1):
        logging.info(f"[{i}/{len(tiles)}] Processing {t['tile_id']}")
        if process_tile(t, cloud_cover, lookback_days):
            processed += 1
            logging.info(f"✅ [{i}/{len(tiles)}] Success: {t['tile_id']}")
        else:
            logging.info(f"⏭️  [{i}/{len(tiles)}] Skipped/Failed: {t['tile_id']}")
    logging.info(f"✨ Finished: processed {processed}/{len(tiles)} tiles successfully")


if __name__ == "__main__":
    cc = int(os.environ.get("RUN_CLOUD_COVER", "40"))
    lb = int(os.environ.get("RUN_LOOKBACK_DAYS", "30"))
    main(cc, lb)
