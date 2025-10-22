# ──────────────────────────────────────────────────────────────────────────────
# File: tile_fetch_worker.py
# Version: 2.0.0
# Author: Amarsinh Patil
# Purpose:
#   NDVI tile processor for Sentinel-2 scenes.
#   Downloads RED/NIR bands from Microsoft Planetary Computer (MPC),
#   compresses as COG, computes NDVI, uploads to Backblaze B2 bucket
#   (kisanshakti-ndvi-tiles), and updates Supabase satellite_tiles table.
# ──────────────────────────────────────────────────────────────────────────────

import os
import json
import math
import time
import tempfile
import logging
import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import rasterio
from rasterio.windows import Window
from rasterio.shutil import copy as rio_copy
from shapely import wkb, wkt
from shapely.geometry import shape, mapping

import planetary_computer as pc
from supabase import create_client
from b2sdk.v2 import InMemoryAccountInfo, B2Api, B2Error
from requests.adapters import HTTPAdapter, Retry

# ------------------------------- CONFIG --------------------------------------

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

B2_KEY_ID = os.getenv("B2_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
B2_BUCKET_NAME = "kisanshakti-ndvi-tiles"
B2_PREFIX = "tiles/"

MPC_COLLECTION = os.getenv("MPC_COLLECTION", "sentinel-2-l2a")
PROCESSING_LEVEL = os.getenv("PROCESSING_LEVEL", "L2A")
RESOLUTION_LABEL = os.getenv("RESOLUTION", "R10m")

CLOUD_COVER_DEFAULT = int(os.getenv("DEFAULT_CLOUD_COVER_MAX", "80"))
LOOKBACK_DAYS_DEFAULT = int(os.getenv("MAX_SCENE_LOOKBACK_DAYS", "90"))
BLOCK_ROWS = int(os.getenv("BLOCK_ROWS", "1024"))

TIFF_COMPRESSION = os.getenv("TIFF_COMPRESSION", "ZSTD")
ZSTD_LEVEL = int(os.getenv("ZSTD_LEVEL", "9"))
B2_UPLOAD_RETRIES = int(os.getenv("B2_UPLOAD_RETRIES", "3"))

# --------------------------- CLIENTS & LOGGING -------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("tile_worker")

_supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

_b2_info = InMemoryAccountInfo()
_b2_api = B2Api(_b2_info)
_b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
_bucket = _b2_api.get_bucket_by_name(B2_BUCKET_NAME)

_session = requests.Session()
_session.mount(
    "https://",
    HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])),
)

# ---------------------------- UTILITY HELPERS --------------------------------

def _safe_float(v, d=None):
    try:
        if v is None:
            return None
        f = float(v)
        return round(f, d) if d is not None else f
    except Exception:
        return None

def _to_int(v):
    try:
        return int(v) if v is not None else None
    except Exception:
        return None

def _decode_geom_to_geojson(row) -> Optional[Dict]:
    g = row.get("geojson_geometry") or row.get("geometry")
    if not g:
        return None
    try:
        if isinstance(g, dict):
            return g
        if isinstance(g, (bytes, bytearray)):
            return mapping(wkb.loads(g))
        if isinstance(g, str):
            s = g.strip()
            if s.startswith("{"):
                return json.loads(s)
            try:
                return mapping(wkt.loads(s))
            except Exception:
                return mapping(wkb.loads(bytes.fromhex(s)))
    except Exception as e:
        logger.warning("decode geom failed: %s", e)
    return None

def _extract_bbox(geom: Dict) -> Dict:
    s = shape(geom)
    minx, miny, maxx, maxy = s.bounds
    return {
        "type": "Polygon",
        "coordinates": [[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]],
    }

def _b2_key(tile_id: str, acq_date: str, subdir: str, name: str) -> str:
    return f"{B2_PREFIX}{subdir}/{tile_id}/{acq_date}/{name}"

# -------------------------- SUPABASE STAGE TRACKER ---------------------------

def _is_valid_date(s: Optional[str]) -> bool:
    if not s:
        return False
    try:
        datetime.date.fromisoformat(s)
        return True
    except Exception:
        return False

def _update_stage(tile_id: str, acq_date: Optional[str], stage: str,
                  status: str = "processing", extra: Optional[Dict] = None):
    payload = {
        "tile_id": tile_id,
        "collection": MPC_COLLECTION.upper(),
        "status": status,
        "processing_stage": stage,
        "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
    }
    if _is_valid_date(acq_date):
        payload["acquisition_date"] = acq_date
    if extra:
        payload.update(extra)
    try:
        _supabase.table("satellite_tiles").upsert(
            payload, on_conflict="tile_id,acquisition_date,collection"
        ).execute()
        logger.info(f"📦 Stage update: {tile_id} - {stage}")
    except Exception as e:
        logger.warning(f"Could not update stage {stage} for {tile_id}: {e}")

# ------------------------------ B2 HELPERS -----------------------------------

def _ensure_b2_auth():
    try:
        _b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
    except Exception as e:
        logger.error(f"B2 re-auth failed: {e}")
        time.sleep(2)

def _b2_exists(name: str) -> Tuple[bool, Optional[int]]:
    try:
        info = _bucket.get_file_info_by_name(name)
        return True, int(getattr(info, "size", 0))
    except Exception:
        return False, None

def _b2_upload(local_path: str, b2_name: str) -> Optional[int]:
    for attempt in range(1, B2_UPLOAD_RETRIES + 1):
        try:
            if not os.path.exists(local_path):
                raise RuntimeError("local file missing")
            size = os.path.getsize(local_path)
            if size < 1024:
                raise RuntimeError(f"file too small ({size} bytes)")
            _bucket.upload_local_file(local_path, b2_name)
            logger.info(f"⬆️  Uploaded to B2: {b2_name} ({size/1024/1024:.2f} MB)")
            exists, _ = _b2_exists(b2_name)
            if not exists:
                raise RuntimeError("Verification failed after upload")
            return size
        except B2Error as be:
            logger.warning(f"B2 upload error on attempt {attempt}: {be}")
            _ensure_b2_auth()
        except Exception as e:
            logger.warning(f"Upload attempt {attempt} failed for {b2_name}: {e}")
            if attempt == B2_UPLOAD_RETRIES:
                logger.error(f"❌ Giving up on {b2_name}")
            time.sleep(2 ** attempt)
    return None

# -------------------------- NDVI COMPUTATION ---------------------------------

def _compute_ndvi_stream(red_path: str, nir_path: str, out_path: str) -> Dict:
    with rasterio.open(red_path) as rsrc:
        profile = rsrc.profile.copy()
        width, height = rsrc.width, rsrc.height

    profile.update(dtype=rasterio.float32, count=1, tiled=True,
                   blockxsize=512, blockysize=512, compress=TIFF_COMPRESSION)
    if TIFF_COMPRESSION.upper() == "ZSTD":
        profile.update(zstd_level=ZSTD_LEVEL)

    stats = {"pixel_count": 0, "valid_pixel_count": 0,
             "vegetation_health_score": None, "ndvi_mean": None,
             "ndvi_std_dev": None, "ndvi_min": None, "ndvi_max": None,
             "vegetation_coverage_percent": None,
             "data_completeness_percent": None}

    total_pix = valid_pix = veg_pix = 0
    s_sum = s_sq = 0.0
    minv, maxv = float("inf"), float("-inf")

    with rasterio.open(red_path) as rsrc, rasterio.open(nir_path) as nsrc, rasterio.open(out_path, "w", **profile) as dst:
        for row in range(0, height, BLOCK_ROWS):
            nrows = min(BLOCK_ROWS, height - row)
            win = Window(0, row, width, nrows)
            red = rsrc.read(1, window=win, out_dtype="float32")
            nir = nsrc.read(1, window=win, out_dtype="float32")
            with np.errstate(divide="ignore", invalid="ignore"):
                denom = nir + red
                ndvi = np.where(denom != 0, (nir - red) / denom, np.nan)
            dst.write(ndvi.astype("float32"), 1, window=win)
            total_pix += red.size
            mask = np.isfinite(ndvi)
            count = int(mask.sum())
            if count:
                vals = ndvi[mask]
                valid_pix += count
                s_sum += float(vals.sum())
                s_sq += float((vals ** 2).sum())
                minv = min(minv, float(vals.min()))
                maxv = max(maxv, float(vals.max()))
                veg_pix += int((vals > 0.3).sum())

    if valid_pix > 0:
        mean = s_sum / valid_pix
        std = math.sqrt(max(0.0, (s_sq / valid_pix) - (mean ** 2)))
        veg_cov = veg_pix / valid_pix * 100.0
        data_comp = valid_pix / total_pix * 100.0 if total_pix else 0.0
        health = ((mean + 1) / 2 * 100.0) * 0.4 + veg_cov * 0.4 + data_comp * 0.2
        stats.update({
            "ndvi_min": _safe_float(minv, 3),
            "ndvi_max": _safe_float(maxv, 3),
            "ndvi_mean": _safe_float(mean, 3),
            "ndvi_std_dev": _safe_float(std, 3),
            "vegetation_coverage_percent": _safe_float(veg_cov, 2),
            "data_completeness_percent": _safe_float(data_comp, 2),
            "pixel_count": _to_int(total_pix),
            "valid_pixel_count": _to_int(valid_pix),
            "vegetation_health_score": _safe_float(health, 2),
        })
    return stats

# --------------------------- MAIN PIPELINE -----------------------------------

def _download_asset(url: str, out_path: str):
    headers = {"Accept": "image/tiff, application/octet-stream"}
    with _session.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(1024 * 512):
                f.write(chunk)
    logger.info(f"✅ Downloaded {os.path.basename(out_path)} ({os.path.getsize(out_path)/1024/1024:.2f} MB)")

# (continues below next message: scene search + main())
# --------------------------- SCENE SEARCH ------------------------------------

def _search_latest_scene(geom: Dict, cloud_cover: int, lookback_days: int) -> Optional[Dict]:
    """Search MPC STAC for newest scene intersecting geom within lookback_days and cloud_cover threshold."""
    today = datetime.date.today()
    start_date = (today - datetime.timedelta(days=lookback_days)).isoformat()
    end_date = today.isoformat()
    payload = {
        "collections": [MPC_COLLECTION],
        "intersects": geom,
        "datetime": f"{start_date}/{end_date}",
        "query": {"eo:cloud_cover": {"lt": cloud_cover}},
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
        "limit": 50,
    }
    try:
        r = _session.post("https://planetarycomputer.microsoft.com/api/stac/v1/search", json=payload, timeout=60)
        r.raise_for_status()
        feats = r.json().get("features", []) or []
        if not feats:
            return None
        # Prefer newest datetime, and within same date prefer lower cloud cover
        feats.sort(key=lambda f: (f["properties"].get("datetime", ""), -float(f["properties"].get("eo:cloud_cover", 999))), reverse=True)
        return feats[0]
    except Exception as e:
        logger.warning("MPC search failed: %s", e)
        return None

# ---------------------------- SUPABASE TILE QUERY ----------------------------

def _get_tiles(filter_tile_ids: Optional[List[str]], limit: Optional[int]) -> List[Dict]:
    q = (_supabase.table("mgrs_tiles")
         .select("id,tile_id,geometry,geojson_geometry,country_id")
         .eq("is_agri", True)
         .eq("is_land_contain", True)
         .order("last_ndvi_update", desc=True))
    if filter_tile_ids:
        q = q.in_("tile_id", filter_tile_ids)
    if limit:
        q = q.limit(limit)
    resp = q.execute()
    data = resp.data if hasattr(resp, "data") else []
    logger.info("Fetched %d candidate tiles", len(data))
    return data

# -------------------------- PROCESS SINGLE TILE ------------------------------

def _process_one(tile: Dict, cloud_cover: int, lookback_days: int, force: bool) -> bool:
    tile_id = tile.get("tile_id")
    geom = _decode_geom_to_geojson(tile)
    if not geom:
        logger.warning("Tile %s has no geometry; skipping", tile_id)
        return False

    # Stage: searching (no acquisition_date yet)
    _update_stage(tile_id, None, "searching_scene")

    scene = _search_latest_scene(geom, cloud_cover, lookback_days)
    if not scene:
        logger.info("No scene found for %s in lookback window", tile_id)
        _update_stage(tile_id, None, "no_scene_found", "pending")
        return False

    acq_date = scene["properties"]["datetime"].split("T")[0]
    cloud = scene["properties"].get("eo:cloud_cover")
    latest_have = None
    try:
        # try to fetch latest acquisition we already have (supabase)
        resp = _supabase.table("satellite_tiles").select("acquisition_date").eq("tile_id", tile_id).eq("collection", MPC_COLLECTION.upper()).order("acquisition_date", desc=True).limit(1).execute()
        if getattr(resp, "data", None):
            latest_have = resp.data[0].get("acquisition_date")
    except Exception:
        pass

    if latest_have and acq_date <= latest_have and not force:
        logger.info("Skipping %s: latest have %s >= scene %s", tile_id, latest_have, acq_date)
        _update_stage(tile_id, acq_date, "skipped_up_to_date", "ready")
        return False

    assets = scene.get("assets", {})
    red_href = (assets.get("red") or assets.get("B04") or {}).get("href")
    nir_href = (assets.get("nir") or assets.get("B08") or {}).get("href")
    if not red_href or not nir_href:
        logger.warning("Missing RED/NIR assets for %s %s", tile_id, acq_date)
        _update_stage(tile_id, acq_date, "missing_assets", "failed", {"error_message": "missing red/nir assets"})
        return False

    # Signed URLs from PC
    try:
        red_url = pc.sign(red_href)
        nir_url = pc.sign(nir_href)
    except Exception as e:
        logger.error("PC signing failed for %s: %s", tile_id, e)
        _update_stage(tile_id, acq_date, "signing_failed", "failed", {"error_message": str(e)})
        return False

    red_key = _b2_key(tile_id, acq_date, "raw", "B04.tif")
    nir_key = _b2_key(tile_id, acq_date, "raw", "B08.tif")
    ndvi_key = _b2_key(tile_id, acq_date, "ndvi", "ndvi.tif")

    # Local temp files
    red_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
    nir_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
    ndvi_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name

    red_cog = None
    nir_cog = None

    try:
        # Stage: downloading
        _update_stage(tile_id, acq_date, "downloading")
        _download_asset(red_url, red_tmp)
        _download_asset(nir_url, nir_tmp)

        # verify downloads
        _verify_tif(red_tmp)
        _verify_tif(nir_tmp)
        red_size_local = os.path.getsize(red_tmp)
        nir_size_local = os.path.getsize(nir_tmp)
        _update_stage(tile_id, acq_date, "downloaded", extra={
            "actual_download_status": "completed",
            "red_band_size_bytes": red_size_local,
            "nir_band_size_bytes": nir_size_local
        })

        # Stage: compress to COG (use rio_copy as COG writer)
        _update_stage(tile_id, acq_date, "compressing")
        red_cog = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
        nir_cog = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name
        try:
            rio_copy(red_tmp, red_cog, driver="COG", compress=TIFF_COMPRESSION)
            rio_copy(nir_tmp, nir_cog, driver="COG", compress=TIFF_COMPRESSION)
        except Exception as e:
            # If rio_copy with paths fails, attempt open-copy-write fallback
            logger.warning("COG direct copy failed: %s — attempting fallback", e)
            with rasterio.open(red_tmp) as src:
                rio_copy(src, red_cog, driver="COG", compress=TIFF_COMPRESSION)
            with rasterio.open(nir_tmp) as src:
                rio_copy(src, nir_cog, driver="COG", compress=TIFF_COMPRESSION)

        _update_stage(tile_id, acq_date, "compressed")

        # Stage: compute NDVI (from COGs to keep parity)
        _update_stage(tile_id, acq_date, "computing_ndvi")
        stats = _compute_ndvi_stream(red_cog, nir_cog, ndvi_tmp)

        # Stage: uploading
        _update_stage(tile_id, acq_date, "uploading")
        red_size = _b2_upload(red_cog, red_key)
        nir_size = _b2_upload(nir_cog, nir_key)
        ndvi_size = _b2_upload(ndvi_tmp, ndvi_key)

        if red_size is None or nir_size is None or ndvi_size is None:
            raise RuntimeError("One or more uploads failed")

        total_mb = _safe_float((red_size + nir_size + ndvi_size) / (1024 * 1024), 2)
        now_iso = datetime.datetime.utcnow().isoformat() + "Z"
        bbox = _extract_bbox(geom)

        payload = {
            "tile_id": tile_id,
            "acquisition_date": acq_date,
            "collection": MPC_COLLECTION.upper(),
            "status": "ready",
            "processing_stage": "completed",
            "processing_completed_at": now_iso,
            "red_band_path": f"b2://{B2_BUCKET_NAME}/{red_key}",
            "nir_band_path": f"b2://{B2_BUCKET_NAME}/{nir_key}",
            "ndvi_path": f"b2://{B2_BUCKET_NAME}/{ndvi_key}",
            "file_size_mb": total_mb,
            "red_band_size_bytes": red_size,
            "nir_band_size_bytes": nir_size,
            "ndvi_size_bytes": ndvi_size,
            "bbox": json.dumps(bbox),
            "mgrs_tile_id": tile.get("id"),
            "country_id": tile.get("country_id"),
            "cloud_cover": _safe_float(cloud, 2),
            "processing_method": "cog_streaming",
            "api_source": "planetary_computer",
            "processing_level": PROCESSING_LEVEL,
            "resolution": RESOLUTION_LABEL,
            "updated_at": now_iso,
            "ndvi_calculation_timestamp": now_iso,
            "processing_completed_at": now_iso,
        }
        payload.update(stats)

        # Upsert final record
        resp = _supabase.table("satellite_tiles").upsert(payload, on_conflict="tile_id,acquisition_date,collection").execute()
        logger.info("Upsert response: %s", getattr(resp, "_http_response", resp))
        logger.info("✅ Successfully processed %s %s", tile_id, acq_date)
        return True

    except Exception as e:
        logger.error("❌ Processing failed for %s %s: %s", tile_id, acq_date if 'acq_date' in locals() else "unknown", e)
        # Mark failed in DB
        try:
            _update_stage(tile_id, acq_date if _is_valid_date(locals().get("acq_date", None)) else None, "failed", "failed", {"error_message": str(e)})
        except Exception:
            logger.warning("Could not upsert failure row for %s", tile_id)
        return False

    finally:
        # cleanup local files
        for p in [red_tmp, nir_tmp, ndvi_tmp, red_cog, nir_cog]:
            try:
                if p and os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass

# ------------------------------- ENTRyPOINT ----------------------------------

def main(
    cloud_cover: int = CLOUD_COVER_DEFAULT,
    lookback_days: int = LOOKBACK_DAYS_DEFAULT,
    filter_tile_ids: Optional[List[str]] = None,
    max_tiles: Optional[int] = None,
    force: bool = False,
) -> Dict:
    tiles = _get_tiles(filter_tile_ids, max_tiles)
    if not tiles:
        logger.info("No tiles to process.")
        return {"processed": 0, "total": 0}

    processed = 0
    total = len(tiles)
    for i, t in enumerate(tiles, start=1):
        logger.info("🔄 [%d/%d] %s", i, total, t.get("tile_id"))
        ok = _process_one(t, cloud_cover, lookback_days, force)
        if ok:
            processed += 1

    logger.info("✨ Completed: %d/%d tiles", processed, total)
    return {"processed": processed, "total": total}

if __name__ == "__main__":
    cc = int(os.getenv("RUN_CLOUD_COVER", str(CLOUD_COVER_DEFAULT)))
    lb = int(os.getenv("RUN_LOOKBACK_DAYS", str(LOOKBACK_DAYS_DEFAULT)))
    # Optionally support env var to force a single tile id: RUN_TILE_ID=43QCU
    run_tile = os.getenv("RUN_TILE_ID")
    if run_tile:
        tiles = [{"tile_id": run_tile, "geometry": None, "geojson_geometry": None, "id": None}]
        # Use _get_tiles if you want full query; here we just call process one
        _process_one(tiles[0], cc, lb, force=True)
    else:
        main(cc, lb)
