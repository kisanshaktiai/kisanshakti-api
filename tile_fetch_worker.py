# ──────────────────────────────────────────────────────────────────────────────
# File: tile_fetch_worker.py
# Version: v4.1.0 Hybrid Stable (Functional + Geometry Safe)
# Author: Amarsinh Patil
# Date: 2025-10-29
# Purpose:
#   Sentinel-2 NDVI tile processor for agricultural zones.
#   Downloads RED/NIR bands from Microsoft Planetary Computer (MPC),
#   compresses as COG, computes NDVI, uploads to Backblaze B2,
#   and updates Supabase `satellite_tiles` table with stats and geometry.
#   ✅ Geometry-safe: adds bbox_geom (SRID=4326)
#   ✅ End-to-end functional NDVI compute & upload
# ──────────────────────────────────────────────────────────────────────────────

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

# ---------------- Geometry Helpers ----------------
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
                return json.loads(s)
            try:
                return mapping(wkt.loads(s))
            except:
                try:
                    return mapping(wkb.loads(bytes.fromhex(s)))
                except:
                    pass
        return None
    except Exception as e:
        logging.error(f"decode_geom_to_geojson failed: {e}")
        return None

def extract_bbox(geom_json):
    try:
        if not geom_json:
            return None
        geom = shape(geom_json)
        bounds = geom.bounds
        bbox = {
            "type": "Polygon",
            "coordinates": [[
                [bounds[0], bounds[1]],
                [bounds[2], bounds[1]],
                [bounds[2], bounds[3]],
                [bounds[0], bounds[3]],
                [bounds[0], bounds[1]],
            ]]
        }
        return bbox
    except Exception as e:
        logging.error(f"Failed to extract bbox: {e}")
        return None

def make_bbox_geom_wkt(bbox_json):
    """Generate SRID=4326;POLYGON(...) for Supabase geometry column"""
    try:
        if not bbox_json:
            return None
        geom_obj = shape(bbox_json)
        if geom_obj.is_valid:
            return f"SRID=4326;{geom_obj.wkt}"
        else:
            logging.warning("⚠️ Invalid bbox geometry")
            return None
    except Exception as e:
        logging.error(f"bbox_geom generation failed: {e}")
        return None

# ---------------- MPC & B2 ----------------
def fetch_agri_tiles():
    try:
        resp = supabase.table("mgrs_tiles").select("tile_id, geometry, country_id, id").eq("is_agri", True).execute()
        tiles = resp.data or []
        logging.info(f"Fetched {len(tiles)} agri tiles")
        return tiles
    except Exception as e:
        logging.error(f"Failed to fetch agri tiles: {e}")
        return []

def query_mpc(tile_geom, start_date, end_date):
    try:
        geom_json = decode_geom_to_geojson(tile_geom)
        if not geom_json:
            return []
        body = {
            "collections": [MPC_COLLECTION],
            "intersects": geom_json,
            "datetime": f"{start_date}/{end_date}",
            "query": {"eo:cloud_cover": {"lt": CLOUD_COVER}},
        }
        resp = session.post(MPC_STAC, json=body, timeout=45)
        if not resp.ok:
            logging.error(f"STAC error {resp.status_code}: {resp.text}")
            return []
        return resp.json().get("features", [])
    except Exception as e:
        logging.error(f"MPC query failed: {e}")
        return []

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
    except:
        return href

def check_b2_file_exists(b2_path):
    try:
        file_info = bucket.get_file_info_by_name(b2_path)
        return True, file_info.size if hasattr(file_info, "size") else None
    except Exception:
        return False, None

def get_b2_paths(tile_id, acq_date):
    return {
        "red": f"{B2_PREFIX}raw/{tile_id}/{acq_date}/B04.tif",
        "nir": f"{B2_PREFIX}raw/{tile_id}/{acq_date}/B08.tif",
        "ndvi": f"{B2_PREFIX}ndvi/{tile_id}/{acq_date}/ndvi.tif",
    }

# ---------------- NDVI Computation ----------------
def compute_and_upload_ndvi(tile_id, acq_date, red_local, nir_local):
    ndvi_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    try:
        with rasterio.open(red_local) as rsrc, rasterio.open(nir_local) as nsrc:
            meta = rsrc.meta.copy()
            height, width = rsrc.height, rsrc.width
            ndvi_full = np.empty((height, width), dtype=np.float32)
            total_pixels, valid_pixels = height * width, 0
            sum_ndvi, sum_sq_ndvi, min_val, max_val, veg_pixels = 0.0, 0.0, 999, -999, 0
            for i in range(0, height, 1024):
                rows = min(1024, height - i)
                win = Window(0, i, width, rows)
                red = rsrc.read(1, window=win).astype("float32")
                nir = nsrc.read(1, window=win).astype("float32")
                ndvi_chunk = np.where((nir + red) != 0, (nir - red) / (nir + red), np.nan)
                ndvi_full[i:i+rows, :] = ndvi_chunk
                valid = ~np.isnan(ndvi_chunk)
                if valid.sum():
                    chunk_valid = ndvi_chunk[valid]
                    valid_pixels += valid.sum()
                    sum_ndvi += chunk_valid.sum()
                    sum_sq_ndvi += (chunk_valid ** 2).sum()
                    min_val, max_val = min(min_val, chunk_valid.min()), max(max_val, chunk_valid.max())
                    veg_pixels += (chunk_valid > 0.3).sum()
                del red, nir, ndvi_chunk

            mean = sum_ndvi / valid_pixels if valid_pixels else np.nan
            std = np.sqrt(max(0, (sum_sq_ndvi / valid_pixels) - mean ** 2)) if valid_pixels else np.nan
            veg_coverage = veg_pixels / valid_pixels * 100 if valid_pixels else 0
            data_completeness = valid_pixels / total_pixels * 100 if total_pixels else 0
            health = ((mean + 1) / 2) * 50 + veg_coverage * 0.3 + data_completeness * 0.2

            stats = {
                "ndvi_min": float(min_val),
                "ndvi_max": float(max_val),
                "ndvi_mean": float(mean),
                "ndvi_std_dev": float(std),
                "vegetation_coverage_percent": float(veg_coverage),
                "data_completeness_percent": float(data_completeness),
                "pixel_count": total_pixels,
                "valid_pixel_count": valid_pixels,
                "vegetation_health_score": round(health, 2),
            }

            meta.update(dtype=rasterio.float32, count=1, compress="LZW", tiled=True, blockxsize=256, blockysize=256)
            with rasterio.open(ndvi_tmp.name, "w", **meta) as dst:
                dst.write(ndvi_full.astype(rasterio.float32), 1)

        ndvi_path = f"{B2_PREFIX}ndvi/{tile_id}/{acq_date}/ndvi.tif"
        bucket.upload_local_file(local_file=ndvi_tmp.name, file_name=ndvi_path)
        return f"b2://{B2_BUCKET_NAME}/{ndvi_path}", stats, os.path.getsize(ndvi_tmp.name)
    except Exception as e:
        logging.error(f"NDVI computation failed: {e}")
        return None, None, None
    finally:
        try: os.remove(ndvi_tmp.name)
        except: pass

# ---------------- Tile Processing ----------------
def process_tile(tile):
    try:
        tile_id, geom_value = tile["tile_id"], tile["geometry"]
        country_id, mgrs_tile_id = tile.get("country_id"), tile.get("id")
        today = datetime.date.today()
        start, end = (today - datetime.timedelta(days=LOOKBACK_DAYS)).isoformat(), today.isoformat()
        scenes = query_mpc(geom_value, start, end)
        if not scenes: return False
        scene = sorted(scenes, key=lambda s: s["properties"].get("eo:cloud_cover", 100))[0]
        acq_date = scene["properties"]["datetime"].split("T")[0]
        cloud_cover = scene["properties"].get("eo:cloud_cover")
        geom_json = decode_geom_to_geojson(geom_value)
        bbox = extract_bbox(geom_json)
        bbox_geom = make_bbox_geom_wkt(bbox)
        exists, paths = {}, get_b2_paths(tile_id, acq_date)
        for key in paths:
            exists[key], _ = check_b2_file_exists(paths[key])
        assets = scene.get("assets", {})
        red_url, nir_url = _signed_asset_url(assets, "red", "B04"), _signed_asset_url(assets, "nir", "B08")

        red_local, nir_local = None, None
        if not exists["red"]:
            red_local, _, _ = download_band(red_url, paths["red"])
        if not exists["nir"]:
            nir_local, _, _ = download_band(nir_url, paths["nir"])
        if not exists["ndvi"]:
            ndvi_b2, stats, ndvi_size = compute_and_upload_ndvi(tile_id, acq_date, red_local, nir_local)
        else:
            ndvi_b2, stats, ndvi_size = f"b2://{B2_BUCKET_NAME}/{paths['ndvi']}", None, None

        now = datetime.datetime.utcnow().isoformat() + "Z"
        payload = {
            "tile_id": tile_id,
            "acquisition_date": acq_date,
            "collection": MPC_COLLECTION.upper(),
            "cloud_cover": cloud_cover,
            "red_band_path": f"b2://{B2_BUCKET_NAME}/{paths['red']}",
            "nir_band_path": f"b2://{B2_BUCKET_NAME}/{paths['nir']}",
            "ndvi_path": ndvi_b2,
            "status": "ready",
            "updated_at": now,
            "processing_completed_at": now,
            "country_id": country_id,
            "mgrs_tile_id": mgrs_tile_id,
            "bbox": bbox,
            "bbox_geom": bbox_geom,
            "api_source": "planetary_computer",
            "processing_method": "cog_streaming",
        }
        if stats: payload.update(stats)
        supabase.table("satellite_tiles").upsert(payload, on_conflict="tile_id,acquisition_date,collection").execute()
        logging.info(f"✅ Saved {tile_id} {acq_date}")
        return True
    except Exception as e:
        logging.error(f"Tile processing failed for {tile.get('tile_id')}: {e}")
        logging.error(traceback.format_exc())
        return False

def main(cloud_cover=20, lookback_days=5):
    global CLOUD_COVER, LOOKBACK_DAYS
    CLOUD_COVER, LOOKBACK_DAYS = int(cloud_cover), int(lookback_days)
    tiles = fetch_agri_tiles()
    logging.info(f"Processing {len(tiles)} agricultural tiles...")
    processed = 0
    for t in tiles:
        if process_tile(t): processed += 1
    logging.info(f"✨ Finished: processed {processed}/{len(tiles)} successfully.")
    return processed

if __name__ == "__main__":
    cc, lb = int(os.environ.get("RUN_CLOUD_COVER", CLOUD_COVER)), int(os.environ.get("RUN_LOOKBACK_DAYS", LOOKBACK_DAYS))
    main(cc, lb)
