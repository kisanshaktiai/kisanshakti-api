# ──────────────────────────────────────────────────────────────────────────────
# File: worker.py  (NDVI processor)
# Version: 1.8.1
# Purpose: Fetch Sentinel-2 scenes from Microsoft Planetary Computer for
#          MGRS polygons in `mgrs_tiles`, compute NDVI, upload artifacts to B2,
#          and upsert metadata into `satellite_tiles` with dedup by acquisition
#          date per (tile_id, collection).
# Design goals: low memory (≤512 MB), resumable, idempotent, scalable.
# Changes in this file:
# - removed `from __future__ import annotations`
# - added more defensive checks, explicit supabase response logging
# - ensure we upsert a 'ready' record even if NDVI already exists in B2
# - improved B2 existence logic to tolerate API quirks
# ──────────────────────────────────────────────────────────────────────────────

import os
import io
import json
import math
import time
import tempfile
import logging
import datetime
from typing import Dict, List, Optional, Tuple
from pystac_client import Client

import numpy as np
import requests
import rasterio
from rasterio.windows import Window
from rasterio.shutil import copy as rio_copy
from shapely import wkb, wkt
from shapely.geometry import shape, mapping

import planetary_computer as pc
from supabase import create_client
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from requests.adapters import HTTPAdapter, Retry

# ------------------------------- Config --------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
B2_KEY_ID = os.getenv("B2_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
B2_BUCKET_NAME = os.getenv("B2_BUCKET_RAW", "kisanshakti-ndvi-tiles")
B2_PREFIX = os.getenv("B2_PREFIX", "tiles/")
MPC_COLLECTION = os.getenv("MPC_COLLECTION", "sentinel-2-l2a")
PROCESSING_LEVEL = os.getenv("PROCESSING_LEVEL", "L2A")
RESOLUTION_LABEL = os.getenv("RESOLUTION", "R10m")  # label only

# Limits / defaults
CLOUD_COVER_DEFAULT = int(os.getenv("DEFAULT_CLOUD_COVER_MAX", "80"))
LOOKBACK_DAYS_DEFAULT = int(os.getenv("MAX_SCENE_LOOKBACK_DAYS", "90"))
BLOCK_ROWS = int(os.getenv("BLOCK_ROWS", "1024"))  # streaming rows at a time

# Compression (use ZSTD when available via GDAL, fallback to LZW)
TIFF_COMPRESSION = os.getenv("TIFF_COMPRESSION", "ZSTD")  # "ZSTD" or "LZW"
ZSTD_LEVEL = int(os.getenv("ZSTD_LEVEL", "9"))

# Upload strategy
B2_UPLOAD_RETRIES = int(os.getenv("B2_UPLOAD_RETRIES", "3"))

# ---------------------------- Clients & Logging ------------------------------
logger = logging.getLogger("tile_worker")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

# ------------------------------ Helpers -------------------------------------

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
        if v is None:
            return None
        return int(v)
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
        "coordinates": [
            [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]
        ],
    }


def _b2_key(tile_id: str, acq_date: str, subdir: str, name: str) -> str:
    return f"{B2_PREFIX}{subdir}/{tile_id}/{acq_date}/{name}"


def _b2_exists(name: str) -> Tuple[bool, Optional[int]]:
    try:
        info = _bucket.get_file_info_by_name(name)
        size = int(getattr(info, "size", 0))
        return True, size
    except Exception as e:
        # Be verbose for debugging B2 quirks
        logger.debug("B2 exists check failed for %s: %s", name, e)
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
            logger.info("⬆️  Uploaded %s (%.2f MB)", b2_name, size / 1024 / 1024)
            return size
        except Exception as e:
            logger.warning("upload attempt %d failed for %s: %s", attempt, b2_name, e)
            if attempt == B2_UPLOAD_RETRIES:
                logger.error("❌ upload failed for %s", b2_name)
            time.sleep(2 ** attempt)
    return None


def _verify_tif(path: str) -> None:
    if not os.path.exists(path):
        raise RuntimeError(f"missing file {path}")
    size = os.path.getsize(path)
    if size < 1024:
        raise RuntimeError(f"too small {size} bytes")
    try:
        with rasterio.open(path) as src:
            _ = src.count
    except Exception as e:
        logger.error("❌ rasterio failed to open %s: %s", path, e)
        raise


# ----------------------- Supabase convenience --------------------------------

def _get_tiles(filter_tile_ids: Optional[List[str]], limit: Optional[int]) -> List[Dict]:
    q = (
        _supabase.table("mgrs_tiles")
        .select("id,tile_id,geometry,geojson_geometry,country_id")
        .eq("is_agri", True)
        .eq("is_land_contain", True)
        .order("last_ndvi_update", desc=True)
    )
    if filter_tile_ids:
        q = q.in_("tile_id", filter_tile_ids)
    if limit:
        q = q.limit(limit)
    resp = q.execute()
    data = resp.data if hasattr(resp, "data") else []
    logger.info("Fetched %d candidate tiles", len(data))
    return data


def _get_latest_acq_date(tile_id: str, collection: str) -> Optional[str]:
    """Return the latest acquisition_date we already have in satellite_tiles for a tile_id+collection."""
    try:
        resp = (
            _supabase.table("satellite_tiles")
            .select("acquisition_date")
            .eq("tile_id", tile_id)
            .eq("collection", collection.upper())
            .order("acquisition_date", desc=True)
            .limit(1)
            .execute()
        )
        if resp.data:
            return resp.data[0].get("acquisition_date")
    except Exception as e:
        logger.warning("Could not fetch latest acquisition for %s: %s", tile_id, e)
    return None


# --------------------------- MPC scene search --------------------------------

def _search_latest_scene(geom: Dict, cloud_cover: int, lookback_days: int) -> Optional[Dict]:
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
    r = _session.post(
        "https://planetarycomputer.microsoft.com/api/stac/v1/search",
        json=payload,
        timeout=60,
    )
    r.raise_for_status()
    feats = r.json().get("features", [])
    if not feats:
        return None
    # Prefer newest by datetime; secondary sort by cloud (lowest)
    feats.sort(key=lambda f: (f["properties"].get("datetime", ""), -float(f["properties"].get("eo:cloud_cover", 1000))), reverse=True)
    return feats[0]


# ---------------------------- NDVI pipeline ----------------------------------

def _compute_ndvi_stream(red_path: str, nir_path: str, out_path: str) -> Dict:
    """Stream rows to keep memory low and write compressed COG/GeoTIFF NDVI."""
    _verify_tif(red_path)
    _verify_tif(nir_path)

    with rasterio.open(red_path) as rsrc:
        profile = rsrc.profile.copy()
        width, height = rsrc.width, rsrc.height

    # Output profile: float32, tiled, compressed
    profile.update(
        dtype=rasterio.float32,
        count=1,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        compress=TIFF_COMPRESSION,
    )
    if TIFF_COMPRESSION.upper() == "ZSTD":
        profile.update(zstd_level=ZSTD_LEVEL)

    stats = {
        "ndvi_min": None,
        "ndvi_max": None,
        "ndvi_mean": None,
        "ndvi_std_dev": None,
        "vegetation_coverage_percent": 0.0,
        "data_completeness_percent": 0.0,
        "pixel_count": 0,
        "valid_pixel_count": 0,
        "vegetation_health_score": None,
    }

    total_pix = 0
    valid_pix = 0
    veg_pix = 0
    s_sum = 0.0
    s_sq = 0.0
    minv = float("inf")
    maxv = float("-inf")

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
        health = ((mean + 1) / 2 * 100.0) * 0.5 + veg_cov * 0.3 + data_comp * 0.2
        stats.update(
            {
                "ndvi_min": _safe_float(minv, 3),
                "ndvi_max": _safe_float(maxv, 3),
                "ndvi_mean": _safe_float(mean, 3),
                "ndvi_std_dev": _safe_float(std, 3),
                "vegetation_coverage_percent": _safe_float(veg_cov, 2),
                "data_completeness_percent": _safe_float(data_comp, 2),
                "pixel_count": _to_int(total_pix),
                "valid_pixel_count": _to_int(valid_pix),
                "vegetation_health_score": _safe_float(health, 2),
            }
        )
    else:
        stats.update({"pixel_count": _to_int(total_pix), "valid_pixel_count": 0})

    return stats


# --------------------------- Tile processing ---------------------------------

def _download_asset(url: str, out_path: str) -> None:
    headers = {"Accept": "image/tiff, application/octet-stream"}
    with _session.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(1024 * 512):
                f.write(chunk)
    logger.info("✅ Downloaded %s (%.2f MB)", out_path, os.path.getsize(out_path)/1024/1024)



def _process_one(tile: Dict, cloud_cover: int, lookback_days: int, force: bool) -> bool:
    tile_id = tile.get("tile_id")
    geom = _decode_geom_to_geojson(tile)
    if not geom:
        logger.warning("Tile %s has no geometry, skipping", tile_id)
        return False

    # 1) Find newest scene within window
    scene = _search_latest_scene(geom, cloud_cover, lookback_days)
    if not scene:
        logger.info("No scenes for %s in lookback window", tile_id)
        return False

    acq_date = scene["properties"]["datetime"].split("T")[0]
    cloud = scene["properties"].get("eo:cloud_cover")

    latest_have = _get_latest_acq_date(tile_id, MPC_COLLECTION)
    if latest_have and acq_date <= latest_have and not force:
        logger.info("Skipping %s: latest have %s >= scene %s", tile_id, latest_have, acq_date)
        return False

    assets = scene.get("assets", {})
    red_asset = assets.get("red") or assets.get("B04")
    nir_asset = assets.get("nir") or assets.get("B08")
    if not red_asset or not nir_asset:
        logger.warning("Missing red/nir assets for %s scene %s", tile_id, acq_date)
        return False

    red_href = red_asset.get("href")
    nir_href = nir_asset.get("href")
    if not red_href or not nir_href:
        logger.warning("Missing href in red/nir assets for %s", tile_id)
        return False

    red_url = pc.sign(red_href)
    nir_url = pc.sign(nir_href)

    # 2) Prep paths
    red_key = _b2_key(tile_id, acq_date, "raw", "B04.tif")
    nir_key = _b2_key(tile_id, acq_date, "raw", "B08.tif")
    ndvi_key = _b2_key(tile_id, acq_date, "ndvi", "ndvi.tif")

    # If NDVI already exists in B2 and record exists for that date, skip (idempotent)
    exists_ndvi, _ = _b2_exists(ndvi_key)
    # Ensure we still upsert a metadata row if NDVI exists but DB row missing

    red_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif"); red_tmp.close()
    nir_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif"); nir_tmp.close()
    ndvi_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif"); ndvi_tmp.close()

    try:
        if exists_ndvi and not force:
            logger.info("NDVI already in B2 for %s %s, attempting DB upsert if missing", tile_id, acq_date)
            # try to upsert metadata referencing existing NDVI path
            now_iso = datetime.datetime.datetime.utcnow().isoformat() + "Z"
            bbox = _extract_bbox(geom)
            payload = {
                "tile_id": tile_id,
                "acquisition_date": acq_date,
                "collection": MPC_COLLECTION.upper(),
                "ndvi_path": f"b2://{B2_BUCKET_NAME}/{ndvi_key}",
                "status": "ready",
                "updated_at": now_iso,
                "processing_completed_at": now_iso,
                "ndvi_calculation_timestamp": now_iso,
                "bbox": json.dumps(bbox),
                "mgrs_tile_id": tile.get("id"),
                "country_id": tile.get("country_id"),
            }
            resp = _supabase.table("satellite_tiles").upsert(payload, on_conflict="tile_id,acquisition_date,collection").execute()
            logger.info("Upsert response (ndvi exists): %s", getattr(resp, "_http_response", resp))
            return True

        logger.info("↓ Downloading RED for %s %s", tile_id, acq_date)
        _download_asset(red_url, red_tmp.name)
        logger.info("↓ Downloading NIR for %s %s", tile_id, acq_date)
        _download_asset(nir_url, nir_tmp.name)

        _verify_tif(red_tmp.name)
        _verify_tif(nir_tmp.name)

        # 4) Compute NDVI streamed
        stats = _compute_ndvi_stream(red_tmp.name, nir_tmp.name, ndvi_tmp.name)

        # 5) Upload artifacts
        red_size = _b2_upload(red_tmp.name, red_key)
        nir_size = _b2_upload(nir_tmp.name, nir_key)
        ndvi_size = _b2_upload(ndvi_tmp.name, ndvi_key)

        total_mb = None
        if red_size is not None and nir_size is not None and ndvi_size is not None:
            total_mb = _safe_float((red_size + nir_size + ndvi_size) / (1024 * 1024), 2)

        now_iso = datetime.datetime.datetime.utcnow().isoformat() + "Z"
        bbox = _extract_bbox(geom)

        payload = {
            "tile_id": tile_id,
            "acquisition_date": acq_date,
            "collection": MPC_COLLECTION.upper(),
            "cloud_cover": _safe_float(cloud, 2),
            "file_size_mb": total_mb,
            "red_band_path": f"b2://{B2_BUCKET_NAME}/{red_key}",
            "nir_band_path": f"b2://{B2_BUCKET_NAME}/{nir_key}",
            "ndvi_path": f"b2://{B2_BUCKET_NAME}/{ndvi_key}",
            "status": "ready",
            "processing_method": "cog_streaming",
            "api_source": "planetary_computer",
            "updated_at": now_iso,
            "processing_completed_at": now_iso,
            "ndvi_calculation_timestamp": now_iso,
            "bbox": json.dumps(bbox),
            "country_id": tile.get("country_id"),
            "mgrs_tile_id": tile.get("id"),
            "processing_level": PROCESSING_LEVEL,
            "resolution": RESOLUTION_LABEL,
        }
        payload.update(stats)

        resp = _supabase.table("satellite_tiles").upsert(payload, on_conflict="tile_id,acquisition_date,collection").execute()
        logger.info("Upsert response: %s", getattr(resp, "_http_response", resp))

        logger.info("✅ Processed %s %s", tile_id, acq_date)
        return True
    except Exception as e:
        logger.error("❌ Failed %s %s: %s", tile_id, acq_date, e)
        # Insert or update error row for observability
        try:
            _supabase.table("satellite_tiles").upsert(
                {
                    "tile_id": tile_id,
                    "acquisition_date": acq_date,
                    "collection": MPC_COLLECTION.upper(),
                    "status": "failed",
                    "error_message": str(e),
                    "updated_at": datetime.datetime.datetime.utcnow().isoformat() + "Z",
                },
                on_conflict="tile_id,acquisition_date,collection",
            ).execute()
        except Exception:
            logger.warning("Could not upsert failure row for %s %s", tile_id, acq_date)
        return False
    finally:
        for p in (red_tmp.name, nir_tmp.name, ndvi_tmp.name):
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass


# ------------------------------- Entrypoint ----------------------------------

def main(
    cloud_cover: int = CLOUD_COVER_DEFAULT,
    lookback_days: int = LOOKBACK_DAYS_DEFAULT,
    filter_tile_ids: Optional[List[str]] = None,
    max_tiles: Optional[int] = None,
    force: bool = False,
) -> Dict:
    tiles = _get_tiles(filter_tile_ids, max_tiles)
    if not tiles:
        return {"processed": 0, "total": 0}

    processed = 0
    for i, t in enumerate(tiles, start=1):
        logger.info("🔄 [%d/%d] %s", i, len(tiles), t.get("tile_id"))
        ok = _process_one(t, cloud_cover=cloud_cover, lookback_days=lookback_days, force=force)
        if ok:
            processed += 1
    stats = {"processed": processed, "total": len(tiles)}
    logger.info("✨ Completed: %d/%d tiles", processed, len(tiles))
    return stats


if __name__ == "__main__":
    cc = int(os.getenv("RUN_CLOUD_COVER", str(CLOUD_COVER_DEFAULT)))
    lb = int(os.getenv("RUN_LOOKBACK_DAYS", str(LOOKBACK_DAYS_DEFAULT)))
    main(cc, lb)

