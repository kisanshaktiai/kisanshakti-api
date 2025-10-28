# ──────────────────────────────────────────────────────────────────────────────
# File: tile_fetch_worker.py
# Version: v2025.10.28 — Geometry Safe + NDVI build + B2 upload Version 4.1.0
# Author: Amarsinh Patil (updated)
# Purpose:
#   Download B04/B08 (via STAC / Planetary Computer), compute NDVI, upload NDVI TIFF to B2,
#   update satellite_tiles table with paths (ndvi_path, red_band_path, nir_band_path) and bbox/bbox_geom.
# ──────────────────────────────────────────────────────────────────────────────

import os, requests, json, datetime, tempfile, logging, traceback, io
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

# controls
DOWNSAMPLE_FACTOR = int(os.environ.get("DOWNSAMPLE_FACTOR", "1"))
RETRY_TOTAL = int(os.environ.get("HTTP_RETRIES", "3"))

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tile_fetch_worker")

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
retries = Retry(total=RETRY_TOTAL, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

# ---------------- Helpers ----------------
def fetch_agri_tiles():
    try:
        resp = supabase.table("mgrs_tiles").select("tile_id, geometry, country_id, id").eq("is_agri", True).execute()
        tiles = resp.data or []
        logger.info(f"Fetched {len(tiles)} agri tiles")
        return tiles
    except Exception as e:
        logger.error(f"Failed to fetch agri tiles: {e}")
        return []

def decode_geom_to_geojson(geom_value):
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
        logger.error(f"decode_geom_to_geojson failed: {e}\n{traceback.format_exc()}")
        return None

def extract_bbox(geom_json):
    try:
        if not geom_json:
            return None
        geom = shape(geom_json)
        bounds = geom.bounds
        return {
            "type": "Polygon",
            "coordinates": [[
                [bounds[0], bounds[1]],
                [bounds[2], bounds[1]],
                [bounds[2], bounds[3]],
                [bounds[0], bounds[3]],
                [bounds[0], bounds[1]]
            ]]
        }
    except Exception as e:
        logger.error(f"Failed to extract bbox: {e}")
        return None

def _signed_asset_url(assets, primary_key, fallback_key=None):
    href = None
    if primary_key in assets and "href" in assets[primary_key]:
        href = assets[primary_key]["href"]
    elif fallback_key and fallback_key in assets and "href" in assets[fallback_key]:
        href = assets[fallback_key]["href"]
    if not href:
        return None
    try:
        return pc.sign(href)
    except Exception:
        return href

def check_b2_file_exists(b2_path):
    try:
        info = bucket.get_file_info_by_name(b2_path)
        return True, info.size if hasattr(info, "size") else None
    except Exception:
        return False, None

def get_b2_paths(tile_id, acq_date):
    # normalized path WITHOUT protocol prefix so worker can build signed url with same string
    return {
        "red": f"{B2_PREFIX}raw/{tile_id}/{acq_date}/B04.tif",
        "nir": f"{B2_PREFIX}raw/{tile_id}/{acq_date}/B08.tif",
        "ndvi": f"{B2_PREFIX}ndvi/{tile_id}/{acq_date}/ndvi.tif",
    }

def check_existing_files(tile_id, acq_date):
    paths = get_b2_paths(tile_id, acq_date)
    red_exists, red_size = check_b2_file_exists(paths["red"])
    nir_exists, nir_size = check_b2_file_exists(paths["nir"])
    ndvi_exists, ndvi_size = check_b2_file_exists(paths["ndvi"])
    exists = {"red": red_exists, "nir": nir_exists, "ndvi": ndvi_exists}
    sizes = {"red": red_size, "nir": nir_size, "ndvi": ndvi_size}
    logger.info(f"📂 Existing files in B2: Red={red_exists}, NIR={nir_exists}, NDVI={ndvi_exists}")
    return exists, paths, sizes

# ---------------- I/O helpers ----------------
def download_file_stream(url, local_path, timeout=60):
    """Download a remote file by URL to local_path using streaming."""
    logger.debug(f"Downloading {url} -> {local_path}")
    try:
        with session.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception as e:
        logger.error(f"Download failed {url}: {e}")
        return False

def upload_file_to_b2(local_file_path, b2_dest_path):
    """
    Upload a local file to B2 under the given destination path (relative path in bucket).
    We return True on success.
    """
    # b2 dest path should be e.g. "tiles/ndvi/43QCU/2025-10-05/ndvi.tif"
    try:
        # Read bytes and upload (upload_bytes exists in many b2sdk versions)
        with open(local_file_path, "rb") as fh:
            data = fh.read()
        try:
            bucket.upload_bytes(data, b2_dest_path)
        except AttributeError:
            # fallback: try upload_local_file if available
            try:
                bucket.upload_local_file(local_file_path, file_name=b2_dest_path)
            except Exception as e:
                logger.error(f"B2 upload fallback failed: {e}")
                return False
        logger.info(f"Uploaded to B2: {b2_dest_path} ({len(data)/1024:.1f} KB)")
        return True
    except Exception as e:
        logger.error(f"B2 upload failed for {local_file_path} -> {b2_dest_path}: {e}")
        return False

# ---------------- NDVI compute ----------------
def compute_ndvi_from_files(red_path, nir_path, out_ndvi_path):
    """
    Read red and nir local files, compute NDVI, write GeoTIFF using red's metadata.
    Returns True on success.
    """
    try:
        with rasterio.open(red_path) as red_src, rasterio.open(nir_path) as nir_src:
            # sanity: ensure alignment/shape match (we assume same)
            if red_src.width != nir_src.width or red_src.height != nir_src.height:
                logger.warning("Red/NIR shapes differ — attempting to read and resample NIR to red grid")
                # Resample NIR to red grid
                data_nir = nir_src.read(
                    1,
                    out_shape=(red_src.height, red_src.width),
                    resampling=rasterio.enums.Resampling.bilinear
                )
                transform = red_src.transform
            else:
                data_nir = nir_src.read(1)
                transform = red_src.transform
            data_red = red_src.read(1).astype("float32")
            data_nir = data_nir.astype("float32")

            # safe NDVI calc
            np.seterr(divide="ignore", invalid="ignore")
            ndvi = (data_nir - data_red) / (data_nir + data_red)
            ndvi = np.clip(ndvi, -1, 1)
            # set nodata for pixels where both bands are zero
            mask_no_data = (data_nir == 0) & (data_red == 0)
            ndvi[mask_no_data] = -1.0

            profile = red_src.profile.copy()
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

            # Write NDVI
            with rasterio.open(out_ndvi_path, "w", **profile) as dst:
                dst.write(ndvi.astype("float32"), 1)

        logger.info(f"NDVI written: {out_ndvi_path}")
        return True
    except Exception as e:
        logger.error(f"compute_ndvi_from_files failed: {e}\n{traceback.format_exc()}")
        return False

# ---------------- Main tile processing ----------------
def process_tile(tile):
    try:
        tile_id = tile["tile_id"]
        geom_value = tile["geometry"]
        country_id = tile.get("country_id")
        mgrs_tile_id = tile.get("id")

        today = datetime.date.today()
        start_date = (today - datetime.timedelta(days=LOOKBACK_DAYS)).isoformat()
        end_date = today.isoformat()

        # query STAC
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
        logger.info(f"STAC query for {tile_id}: {start_date}->{end_date}, cloud<{CLOUD_COVER}")
        resp = session.post(MPC_STAC, json=body, timeout=45)
        if not resp.ok:
            logger.error(f"STAC query failed: {resp.status_code} {resp.text}")
            return False
        features = resp.json().get("features", [])
        if not features:
            logger.info(f"No scenes found for {tile_id} in {start_date}..{end_date}")
            return False

        # pick best scene
        scene = sorted(features, key=lambda s: (s["properties"].get("eo:cloud_cover", 100),
                                              -datetime.datetime.fromisoformat(
                                                  s["properties"]["datetime"].replace("Z", "+00:00")).timestamp()))[0]
        acq_date = scene["properties"]["datetime"].split("T")[0]
        cloud_cover = scene["properties"].get("eo:cloud_cover")

        # asset urls (signed)
        assets = scene.get("assets", {})
        red_url = _signed_asset_url(assets, "B04", fallback_key="B04")
        nir_url = _signed_asset_url(assets, "B08", fallback_key="B08")
        if not red_url or not nir_url:
            logger.error(f"No B04/B08 assets for scene {acq_date} / tile {tile_id}")
            return False

        # Make sure B2 paths remove leading slash
        exists, paths, file_sizes = check_existing_files(tile_id, acq_date)

        # Build local temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            red_local = os.path.join(tmpdir, "B04.tif")
            nir_local = os.path.join(tmpdir, "B08.tif")
            ndvi_local = os.path.join(tmpdir, "ndvi.tif")

            # Download red/nir (only if not already in B2)
            # If they already exist in B2, you may prefer to fetch from B2 instead — but here we download from PC (signed) for consistency
            if not download_file_stream(red_url, red_local):
                logger.error("Failed to download red band")
                return False
            if not download_file_stream(nir_url, nir_local):
                logger.error("Failed to download nir band")
                return False

            # compute NDVI
            if not compute_ndvi_from_files(red_local, nir_local, ndvi_local):
                logger.error("NDVI computation failed")
                return False

            # upload files to B2 under correct paths
            red_b2_path = paths["red"]
            nir_b2_path = paths["nir"]
            ndvi_b2_path = paths["ndvi"]  # tiles/ndvi/{tile}/{date}/ndvi.tif

            # Upload red/nir if not present in B2
            if not exists["red"]:
                if not upload_file_to_b2(red_local, red_b2_path):
                    logger.warning(f"Failed to upload RED to B2: {red_b2_path}")
            else:
                logger.info("Red file already present in B2 — skipping upload")

            if not exists["nir"]:
                if not upload_file_to_b2(nir_local, nir_b2_path):
                    logger.warning(f"Failed to upload NIR to B2: {nir_b2_path}")
            else:
                logger.info("NIR file already present in B2 — skipping upload")

            # Always upload NDVI (overwrite)
            if not upload_file_to_b2(ndvi_local, ndvi_b2_path):
                logger.error("Failed to upload NDVI to B2")
                return False

        # Extract bbox / bbox_geom safe
        bbox = extract_bbox(geom_json)
        bbox_geom_wkt = None
        try:
            if bbox:
                geom_obj = shape(bbox)
                if geom_obj.is_valid:
                    bbox_geom_wkt = f"SRID=4326;{geom_obj.wkt}"
                    logger.info(f"✅ bbox_geom generated for {tile_id}")
        except Exception as e:
            logger.warning(f"⚠️ bbox_geom generation failed for {tile_id}: {e}")

        now_iso = datetime.datetime.utcnow().isoformat() + "Z"
        payload = {
            "tile_id": tile_id,
            "acquisition_date": acq_date,
            "collection": MPC_COLLECTION.upper(),
            "processing_level": "L2A",
            "cloud_cover": float(cloud_cover) if cloud_cover is not None else None,
            # store relative paths (no b2://) so downstream code can sign them using same prefix
            "red_band_path": f"{paths['red']}",
            "nir_band_path": f"{paths['nir']}",
            "ndvi_path": f"{paths['ndvi']}",
            "file_size_mb": file_sizes.get("ndvi") or 0,
            "resolution": "10m",
            "status": "ready",
            "updated_at": now_iso,
            "processing_completed_at": now_iso,
            "ndvi_calculation_timestamp": now_iso,
            "api_source": "planetary_computer",
            "processing_method": "computed_from_bands",
            "actual_download_status": "downloaded",
            "processing_stage": "completed",
            "country_id": country_id,
            "mgrs_tile_id": mgrs_tile_id,
            "bbox": bbox if bbox else None,
            "bbox_geom": bbox_geom_wkt
        }

        resp = supabase.table("satellite_tiles").upsert(
            payload, on_conflict="tile_id,acquisition_date,collection"
        ).execute()

        if resp.data:
            record_id = resp.data[0].get("id", "unknown")
            logger.info(f"✅ Saved {tile_id} {acq_date} (record id: {record_id})")
        else:
            logger.warning(f"⚠️ Upsert returned no data for {tile_id}")

        return True

    except Exception as e:
        logger.error(f"❌ process_tile error for {tile.get('tile_id')}: {e}")
        logger.error(traceback.format_exc())
        return False

# ---------------- Main Entry ----------------
def main(cloud_cover=20, lookback_days=5):
    global CLOUD_COVER, LOOKBACK_DAYS
    CLOUD_COVER = int(cloud_cover)
    LOOKBACK_DAYS = int(lookback_days)
    logger.info(f"🚀 Starting tile processing (cloud_cover<={CLOUD_COVER}%, lookback={LOOKBACK_DAYS} days)")
    processed = 0
    tiles = fetch_agri_tiles()
    if not tiles:
        logger.warning("⚠️ No tiles fetched from database")
        return 0
    for i, t in enumerate(tiles, 1):
        tile_id = t.get("tile_id", "unknown")
        logger.info(f"🔄 [{i}/{len(tiles)}] Processing: {tile_id}")
        if process_tile(t):
            processed += 1
            logger.info(f"✅ [{i}/{len(tiles)}] Success: {tile_id}")
        else:
            logger.info(f"⏭️  [{i}/{len(tiles)}] Skipped: {tile_id}")
    logger.info(f"✨ Finished: processed {processed}/{len(tiles)} tiles successfully")
    return processed

if __name__ == "__main__":
    cc = int(os.environ.get("RUN_CLOUD_COVER", CLOUD_COVER))
    lb = int(os.environ.get("RUN_LOOKBACK_DAYS", LOOKBACK_DAYS))
    main(cc, lb)
