"""
tile_fetch_worker_v1.6.1
------------------------
Production-ready MPC → B2 → Supabase pipeline.
Safely computes NDVI for agricultural MGRS tiles and records metadata
into the satellite_tiles table.

Fixes in this version:
✅ NDVI stats stored as numeric(5,3)
✅ JSON serialization (bbox dict, no double encoding)
✅ Int64 / float32 → float casting
✅ Added safe_float() helper for uniform precision
✅ Aligned with geojson_geometry usage
✅ Compatible with triggers and constraints in satellite_tiles
✅ Fixed B2SDK stream flush warnings
✅ Fixed NDVI division RuntimeWarning
✅ Improved temp file cleanup
"""

import os, json, datetime, tempfile, logging, traceback, requests, numpy as np, warnings
import rasterio
from rasterio.windows import Window
from shapely import wkb, wkt
from shapely.geometry import mapping, shape
from supabase import create_client
import planetary_computer as pc
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from requests.adapters import HTTPAdapter, Retry

# Suppress B2SDK stream warnings
warnings.filterwarnings('ignore', message='I/O operation on closed file')

# ------------- CONFIG -------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
B2_APP_KEY_ID = os.getenv("B2_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
B2_BUCKET_NAME = os.getenv("B2_BUCKET_RAW", "kisanshakti-ndvi-tiles")
B2_PREFIX = os.getenv("B2_PREFIX", "tiles/")
MPC_COLLECTION = os.getenv("MPC_COLLECTION", "sentinel-2-l2a")
CLOUD_COVER = int(os.getenv("DEFAULT_CLOUD_COVER_MAX", "20"))
LOOKBACK_DAYS = int(os.getenv("MAX_SCENE_LOOKBACK_DAYS", "5"))
DOWNSAMPLE_FACTOR = int(os.getenv("DOWNSAMPLE_FACTOR", "4"))

# ------------- LOGGING -------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tile_worker")

# ------------- CLIENTS -------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

info = InMemoryAccountInfo()
b2_api = B2Api(info)
b2_api.authorize_account("production", B2_APP_KEY_ID, B2_APP_KEY)
bucket = b2_api.get_bucket_by_name(B2_BUCKET_NAME)

session = requests.Session()
retries = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))


# ---------------- HELPER FUNCTIONS ----------------

def safe_float(value, decimals=None):
    """Safely cast value to float and round."""
    if value is None:
        return None
    try:
        f = float(value)
        return round(f, decimals) if decimals is not None else f
    except Exception:
        return None


def fetch_agri_tiles():
    """Fetch agricultural tiles with both geometry and geojson_geometry."""
    try:
        resp = supabase.table("mgrs_tiles").select(
            "id, tile_id, geometry, geojson_geometry, country_id"
        ).eq("is_agri", True).eq("is_land_contain", True).execute()
        tiles = resp.data or []
        logger.info(f"Fetched {len(tiles)} agri tiles.")
        return tiles
    except Exception as e:
        logger.error(f"Failed to fetch agri tiles: {e}")
        return []


def decode_geom_to_geojson(tile):
    """Prefer geojson_geometry for MPC queries; fallback to geometry."""
    geom = tile.get("geojson_geometry") or tile.get("geometry")
    try:
        if geom is None:
            return None
        if isinstance(geom, dict) and "type" in geom:
            return geom
        if isinstance(geom, (bytes, bytearray)):
            return mapping(wkb.loads(geom))
        if isinstance(geom, str):
            s = geom.strip()
            if s.startswith("{") and s.endswith("}"):
                return json.loads(s)
            try:
                return mapping(wkt.loads(s))
            except:
                return mapping(wkb.loads(bytes.fromhex(s)))
        return None
    except Exception as e:
        logger.error(f"decode_geom_to_geojson failed: {e}")
        return None


def extract_bbox(geom_json):
    """Get bounding box as GeoJSON polygon."""
    try:
        if not geom_json:
            return None
        geom = shape(geom_json)
        minx, miny, maxx, maxy = geom.bounds
        return {
            "type": "Polygon",
            "coordinates": [[
                [minx, miny],
                [maxx, miny],
                [maxx, maxy],
                [minx, maxy],
                [minx, miny]
            ]]
        }
    except Exception as e:
        logger.error(f"extract_bbox failed: {e}")
        return None


def query_mpc(tile_geom, start_date, end_date):
    """Query MPC STAC for given geometry and date range."""
    try:
        body = {
            "collections": [MPC_COLLECTION],
            "intersects": tile_geom,
            "datetime": f"{start_date}/{end_date}",
            "query": {"eo:cloud_cover": {"lt": CLOUD_COVER}}
        }
        url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
        logger.info(f"🛰️ Querying MPC: {MPC_COLLECTION} {start_date}->{end_date}, cloud<{CLOUD_COVER}%")
        r = session.post(url, json=body, timeout=60)
        if not r.ok:
            logger.error(f"MPC query error {r.status_code}: {r.text}")
            return []
        data = r.json()
        return data.get("features", [])
    except Exception as e:
        logger.error(f"query_mpc failed: {e}\n{traceback.format_exc()}")
        return []


def pick_best_scene(scenes):
    """Choose scene with lowest cloud cover, latest date."""
    try:
        if not scenes:
            return None
        return sorted(
            scenes,
            key=lambda s: (
                s["properties"].get("eo:cloud_cover", 100),
                -datetime.datetime.fromisoformat(
                    s["properties"]["datetime"].replace("Z", "+00:00")
                ).timestamp()
            )
        )[0]
    except Exception:
        return None


def get_b2_path(tile_id, date, subdir, name):
    return f"{B2_PREFIX}{subdir}/{tile_id}/{date}/{name}"


def upload_to_b2(local_file, b2_path):
    """Upload file to B2 and return size in bytes."""
    try:
        # Get size before upload
        size = os.path.getsize(local_file)
        
        # Upload with metadata
        file_info = {'src': 'tile_worker', 'version': 'v1.6.1'}
        bucket.upload_local_file(
            local_file=local_file, 
            file_name=b2_path,
            file_infos=file_info
        )
        
        logger.info(f"✅ Uploaded {b2_path} ({size/1024/1024:.2f}MB)")
        return size
    except Exception as e:
        logger.error(f"B2 upload failed: {e}")
        return None


def compute_ndvi(red_path, nir_path, tile_id, date):
    """Compute NDVI and return stats + NDVI temp file path."""
    ndvi_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
    try:
        with rasterio.open(red_path) as rsrc, rasterio.open(nir_path) as nsrc:
            meta = rsrc.meta.copy()
            height, width = rsrc.height, rsrc.width
            total_pixels = height * width
            chunk = 1024
            ndvi_full = np.empty((height, width), dtype=np.float32)

            valid, veg_pixels = 0, 0
            sum_val, sum_sq, minv, maxv = 0.0, 0.0, np.inf, -np.inf

            for i in range(0, height, chunk):
                rows = min(chunk, height - i)
                window = Window(0, i, width, rows)
                red = rsrc.read(1, window=window).astype("float32")
                nir = nsrc.read(1, window=window).astype("float32")
                
                # Fix: Use np.errstate to suppress division warnings
                with np.errstate(divide='ignore', invalid='ignore'):
                    denominator = nir + red
                    ndvi = np.where(denominator != 0, (nir - red) / denominator, np.nan)
                
                ndvi_full[i:i+rows, :] = ndvi

                valid_mask = ~np.isnan(ndvi)
                count = valid_mask.sum()
                if count:
                    vals = ndvi[valid_mask]
                    valid += count
                    sum_val += vals.sum()
                    sum_sq += (vals ** 2).sum()
                    minv, maxv = min(minv, vals.min()), max(maxv, vals.max())
                    veg_pixels += (vals > 0.3).sum()

            mean = sum_val / valid if valid else np.nan
            var = (sum_sq / valid) - (mean ** 2) if valid else np.nan
            std = np.sqrt(max(0, var)) if valid else np.nan
            veg_cov = veg_pixels / valid * 100 if valid else 0
            data_comp = valid / total_pixels * 100 if valid else 0
            health = ((mean + 1) / 2 * 100) * 0.5 + veg_cov * 0.3 + data_comp * 0.2

            stats = {
                "ndvi_min": safe_float(minv, 3),
                "ndvi_max": safe_float(maxv, 3),
                "ndvi_mean": safe_float(mean, 3),
                "ndvi_std_dev": safe_float(std, 3),
                "vegetation_coverage_percent": safe_float(veg_cov, 2),
                "data_completeness_percent": safe_float(data_comp, 2),
                "pixel_count": int(total_pixels),
                "valid_pixel_count": int(valid),
                "vegetation_health_score": safe_float(health, 2),
            }

            meta.update(dtype=rasterio.float32, count=1, compress="LZW", tiled=True, blockxsize=256, blockysize=256)
            with rasterio.open(ndvi_tmp.name, "w", **meta) as dst:
                dst.write(ndvi_full.astype(rasterio.float32), 1)

        return ndvi_tmp.name, stats
    except Exception as e:
        logger.error(f"NDVI computation failed: {e}")
        if ndvi_tmp and os.path.exists(ndvi_tmp.name):
            try:
                os.unlink(ndvi_tmp.name)
            except:
                pass
        return None, None


# ---------------- MAIN PROCESS ----------------

def process_tile(tile):
    red_tmp = None
    nir_tmp = None
    ndvi_file = None
    
    try:
        tile_id = tile["tile_id"]
        geom_json = decode_geom_to_geojson(tile)
        if not geom_json:
            logger.warning(f"No valid geometry for {tile_id}")
            return False

        bbox = extract_bbox(geom_json)
        today = datetime.date.today()
        start_date = (today - datetime.timedelta(days=LOOKBACK_DAYS)).isoformat()
        end_date = today.isoformat()

        scenes = query_mpc(geom_json, start_date, end_date)
        if not scenes:
            logger.warning(f"No scenes found for {tile_id}")
            return False

        scene = pick_best_scene(scenes)
        if not scene:
            logger.warning(f"No valid scene after filtering for {tile_id}")
            return False

        acq_date = scene["properties"]["datetime"].split("T")[0]
        cloud = scene["properties"].get("eo:cloud_cover")

        assets = scene["assets"]
        red_url = pc.sign(assets.get("red", assets.get("B04"))["href"])
        nir_url = pc.sign(assets.get("nir", assets.get("B08"))["href"])

        red_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
        nir_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")

        # Download red band
        with session.get(red_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(red_tmp.name, "wb") as f:
                for chunk in r.iter_content(chunk_size=512 * 1024):
                    if chunk:
                        f.write(chunk)
        
        # Explicitly close before upload
        red_tmp.close()

        # Download nir band
        with session.get(nir_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(nir_tmp.name, "wb") as f:
                for chunk in r.iter_content(chunk_size=512 * 1024):
                    if chunk:
                        f.write(chunk)
        
        nir_tmp.close()

        red_size = upload_to_b2(red_tmp.name, get_b2_path(tile_id, acq_date, "raw", "B04.tif"))
        nir_size = upload_to_b2(nir_tmp.name, get_b2_path(tile_id, acq_date, "raw", "B08.tif"))

        ndvi_file, stats = compute_ndvi(red_tmp.name, nir_tmp.name, tile_id, acq_date)
        ndvi_size = upload_to_b2(ndvi_file, get_b2_path(tile_id, acq_date, "ndvi", "ndvi.tif")) if ndvi_file else None

        total_size_mb = (red_size + nir_size + (ndvi_size or 0)) / (1024 * 1024)
        now = datetime.datetime.utcnow().isoformat() + "Z"

        payload = {
            "tile_id": tile_id,
            "acquisition_date": acq_date,
            "collection": MPC_COLLECTION.upper(),
            "cloud_cover": safe_float(cloud, 2),
            "processing_level": "L2A",
            "file_size_mb": safe_float(total_size_mb, 2),
            "red_band_size_bytes": red_size,
            "nir_band_size_bytes": nir_size,
            "ndvi_size_bytes": ndvi_size,
            "red_band_path": f"b2://{B2_BUCKET_NAME}/{get_b2_path(tile_id, acq_date, 'raw', 'B04.tif')}",
            "nir_band_path": f"b2://{B2_BUCKET_NAME}/{get_b2_path(tile_id, acq_date, 'raw', 'B08.tif')}",
            "ndvi_path": f"b2://{B2_BUCKET_NAME}/{get_b2_path(tile_id, acq_date, 'ndvi', 'ndvi.tif')}",
            "status": "ready",
            "resolution": "R10m",
            "processing_method": "cog_streaming",
            "api_source": "planetary_computer",
            "updated_at": now,
            "processing_completed_at": now,
            "ndvi_calculation_timestamp": now,
            "bbox": bbox if isinstance(bbox, dict) else None,
            "country_id": tile.get("country_id"),
            "mgrs_tile_id": tile.get("id"),
            "actual_download_status": "downloaded",
            "processing_stage": "completed",
        }

        if stats:
            payload.update({
                "ndvi_min": safe_float(stats["ndvi_min"], 3),
                "ndvi_max": safe_float(stats["ndvi_max"], 3),
                "ndvi_mean": safe_float(stats["ndvi_mean"], 3),
                "ndvi_std_dev": safe_float(stats["ndvi_std_dev"], 3),
                "vegetation_coverage_percent": safe_float(stats["vegetation_coverage_percent"], 2),
                "data_completeness_percent": safe_float(stats["data_completeness_percent"], 2),
                "pixel_count": int(stats["pixel_count"]),
                "valid_pixel_count": int(stats["valid_pixel_count"]),
                "vegetation_health_score": safe_float(stats["vegetation_health_score"], 2),
            })

        supabase.table("satellite_tiles").upsert(
            payload, on_conflict="tile_id,acquisition_date,collection"
        ).execute()

        logger.info(f"✅ Inserted {tile_id} ({acq_date})")
        return True

    except Exception as e:
        logger.error(f"process_tile failed for {tile.get('tile_id')}: {e}\n{traceback.format_exc()}")
        return False
    
    finally:
        # Cleanup temp files
        for tmp_file in [red_tmp, nir_tmp]:
            if tmp_file and hasattr(tmp_file, 'name'):
                try:
                    if os.path.exists(tmp_file.name):
                        os.unlink(tmp_file.name)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file: {e}")
        
        if ndvi_file and os.path.exists(ndvi_file):
            try:
                os.unlink(ndvi_file)
            except Exception as e:
                logger.warning(f"Failed to cleanup NDVI file: {e}")


def main(cloud_cover=100, lookback_days=90):
    global CLOUD_COVER, LOOKBACK_DAYS
    CLOUD_COVER, LOOKBACK_DAYS = cloud_cover, lookback_days
    logger.info(f"Running worker with cloud_cover={cloud_cover}, lookback_days={lookback_days}")
    tiles = fetch_agri_tiles()
    processed = 0
    for i, t in enumerate(tiles, 1):
        logger.info(f"🔄 [{i}/{len(tiles)}] Processing {t['tile_id']}")
        if process_tile(t):
            processed += 1
    logger.info(f"✨ Completed: {processed}/{len(tiles)} succeeded.")


if __name__ == "__main__":
    cc = int(os.getenv("RUN_CLOUD_COVER", CLOUD_COVER))
    lb = int(os.getenv("RUN_LOOKBACK_DAYS", LOOKBACK_DAYS))
    main(cc, lb)
