# ndvi_land_worker.py
import io
import os
import json
import datetime
import logging
import traceback
import numpy as np
import requests
import rasterio
from rasterio.mask import mask
from rasterio.io import MemoryFile
from rasterio.warp import transform_geom
from shapely.geometry import shape, mapping
from supabase import create_client

# === CONFIG / CLIENTS ===
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
B2_BUCKET_URL = os.getenv("B2_BUCKET_URL")  # e.g. https://f002.backblazeb2.com/file/your-bucket
# Create supabase client here so worker can run standalone
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ndvi-worker")

# ---------------- Utility functions ----------------
def download_tile_from_b2(tile_path: str) -> io.BytesIO:
    """Download GeoTIFF tile from Backblaze B2 bucket - tile_path is the path stored in DB"""
    if not tile_path:
        raise ValueError("No tile_path provided")
    url = f"{B2_BUCKET_URL.rstrip('/')}/{tile_path.lstrip('/')}"
    logger.info(f"Downloading tile from: {url}")
    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        raise Exception(f"Failed to download tile: {url} (Status: {resp.status_code})")
    return io.BytesIO(resp.content)

def open_raster_from_bytesio(buf: io.BytesIO):
    """Open a rasterio dataset from bytes (MemoryFile)"""
    buf.seek(0)
    memfile = MemoryFile(buf.read())
    return memfile.open()

def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    np.seterr(divide='ignore', invalid='ignore')
    denom = nir.astype(float) + red.astype(float)
    ndvi = np.where(denom == 0, np.nan, (nir - red) / denom)
    return np.clip(ndvi, -1, 1)

def calculate_vegetation_indices(red: np.ndarray, nir: np.ndarray, blue: np.ndarray = None) -> dict:
    indices = {"ndvi": calculate_ndvi(red, nir)}
    if blue is not None:
        np.seterr(divide='ignore', invalid='ignore')
        evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
        indices["evi"] = np.clip(evi, -1, 1)
    # SAVI
    L = 0.5
    savi = ((nir - red) / (nir + red + L)) * (1 + L)
    indices["savi"] = np.clip(savi, -1, 1)
    return indices

def generate_ndvi_visualization(ndvi_array: np.ndarray, output_size=(512, 512)) -> bytes:
    import matplotlib.pyplot as plt
    ndvi_clean = np.nan_to_num(ndvi_array, nan=0.0)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.imshow(ndvi_clean, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def calculate_statistics(array: np.ndarray) -> dict:
    valid = array[~np.isnan(array)]
    if valid.size == 0:
        return {
            "mean": None, "min": None, "max": None, "std": None,
            "median": None, "q25": None, "q75": None,
            "valid_pixels": 0, "total_pixels": int(array.size),
            "coverage_percentage": 0.0
        }
    return {
        "mean": float(np.mean(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "std": float(np.std(valid)),
        "median": float(np.median(valid)),
        "q25": float(np.percentile(valid, 25)),
        "q75": float(np.percentile(valid, 75)),
        "valid_pixels": int(valid.size),
        "total_pixels": int(array.size),
        "coverage_percentage": float(valid.size / array.size * 100)
    }

def resolve_tile_tif_path(tile: dict) -> str:
    """
    Prefer combined NDVI/product if present, else per-product, else nir_band_path, then b2_path.
    The tile dict is expected to contain any of these keys based on your ingestion.
    """
    if not tile:
        return None
    if tile.get("ndvi_path"):
        return tile["ndvi_path"]
    if tile.get("product_path"):
        return tile["product_path"]
    if tile.get("combined_path"):
        return tile["combined_path"]
    # prefer a path that likely contains required bands
    if tile.get("nir_band_path"):
        return tile["nir_band_path"]
    return tile.get("b2_path")

# ---------------- Core processing ----------------
def process_farmer_land(land: dict, tile: dict) -> bool:
    """
    Process NDVI for a single land using tile reference.
    Uses boundary_polygon_old as the authoritative land polygon.
    """
    land_id = land.get("id")
    farmer_id = land.get("farmer_id")
    tenant_id = land.get("tenant_id")
    try:
        logger.info(f"Processing land {land_id} with tile {tile.get('id') or tile.get('tile_id')}")

        # --- Geometry: use boundary_polygon_old only ---
        raw = land.get("boundary_polygon_old")
        if not raw:
            raise Exception("Missing boundary_polygon_old for land")
        if isinstance(raw, str):
            try:
                geom_json = json.loads(raw)
            except Exception as exc:
                raise Exception(f"Invalid JSON in boundary_polygon_old: {exc}")
        else:
            geom_json = raw
        land_geom = shape(geom_json)

        # --- Resolve tile file path & download ---
        tif_path = resolve_tile_tif_path(tile)
        if not tif_path:
            raise Exception("No usable tile path found in satellite_tiles row")
        tif_bytes = download_tile_from_b2(tif_path)

        # --- Open raster with MemoryFile and clip ---
        with open_raster_from_bytesio(tif_bytes) as src:
            if src.crs is None:
                raise Exception("Source raster has no CRS")
            src_crs_str = src.crs.to_string() if hasattr(src.crs, "to_string") else str(src.crs)
            land_geom_transformed = transform_geom("EPSG:4326", src_crs_str, mapping(land_geom))
            clipped, out_transform = mask(src, [land_geom_transformed], crop=True, all_touched=True, nodata=np.nan)
            num_bands = src.count
            logger.debug(f"Raster opened: bands={num_bands}, width={src.width}, height={src.height}, crs={src_crs_str}")
            # If single-band NDVI product is present
            if num_bands == 1:
                ndvi = clipped[0].astype(float)
                red = None; nir = None; blue = None
            else:
                # Determine band indices heuristically. Prefer explicit indexes from tile if present.
                try:
                    red_index = int(tile.get("red_band_index", 4)) - 1
                    nir_index = int(tile.get("nir_band_index", 8)) - 1
                    blue_index = int(tile.get("blue_band_index", 2)) - 1
                except Exception:
                    red_index = 3; nir_index = 7; blue_index = 1
                # clamp safe selection
                bands_count = clipped.shape[0]
                if red_index >= bands_count: red_index = 0
                if nir_index >= bands_count: nir_index = min(bands_count - 1, nir_index)
                if blue_index >= bands_count: blue_index = 0
                try:
                    red = clipped[red_index].astype(float)
                    nir = clipped[nir_index].astype(float)
                    blue = clipped[blue_index].astype(float) if bands_count > blue_index else None
                except Exception:
                    # fallback: pick first 2 bands as red/nir
                    red = clipped[0].astype(float)
                    nir = clipped[1].astype(float) if bands_count > 1 else clipped[0].astype(float)
                    blue = clipped[2].astype(float) if bands_count > 2 else None
                ndvi = calculate_ndvi(red, nir)

        # --- Calculate stats ---
        ndvi_stats = calculate_statistics(ndvi)
        if ndvi_stats["valid_pixels"] == 0:
            raise Exception("No valid NDVI pixels after clipping")

        # --- Visualization ---
        viz_bytes = generate_ndvi_visualization(ndvi)
        acquisition = tile.get("acquisition_date") or datetime.datetime.utcnow().date().isoformat()
        viz_filename = f"{land_id}_{acquisition}.png"
        viz_path = f"ndvi-thumbnails/{viz_filename}"

        # upload thumbnail - supabase.storage.from_.upload accepts fileobj or bytes depending on client
        upload_resp = supabase.storage.from_("ndvi-thumbnails").upload(viz_path, io.BytesIO(viz_bytes), {"content-type": "image/png", "upsert": "true"})
        # check for error property depending on supabase client implementation
        try:
            if hasattr(upload_resp, "error") and upload_resp.error:
                raise Exception(f"Storage upload failed: {upload_resp.error}")
            if isinstance(upload_resp, dict) and upload_resp.get("error"):
                raise Exception(f"Storage upload failed: {upload_resp.get('error')}")
        except Exception:
            # log but continue -- still construct viz_url even if upload failed
            logger.warning("Storage upload returned unexpected response: %s", upload_resp)

        viz_url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/public/ndvi-thumbnails/{viz_path}"

        # --- Prepare metadata & DB writes ---
        metadata = {
            "tile_id": tile.get("tile_id") or tile.get("id"),
            "resolution": tile.get("resolution", 10),
            "cloud_cover": tile.get("cloud_cover"),
            "processing_timestamp": datetime.datetime.utcnow().isoformat(),
            "statistics": ndvi_stats
        }

        if "evi" in locals() or "savi" in locals():
            # add if computed (we computed savi always)
            metadata["savi_stats"] = calculate_statistics((locals().get("savi") if "savi" in locals() else ndvi))

        # Upsert into ndvi_micro_tiles
        upsert_payload = {
            "land_id": land_id,
            "farmer_id": farmer_id,
            "tenant_id": tenant_id,
            "bbox": mapping(land_geom),
            "acquisition_date": acquisition,
            "cloud_cover": tile.get("cloud_cover"),
            "ndvi_mean": ndvi_stats["mean"],
            "ndvi_min": ndvi_stats["min"],
            "ndvi_max": ndvi_stats["max"],
            "ndvi_std_dev": ndvi_stats["std"],
            "ndvi_thumbnail_url": viz_url,
            "resolution_meters": metadata["resolution"],
            "metadata": metadata
        }
        res_upsert = supabase.table("ndvi_micro_tiles").upsert(upsert_payload, on_conflict="land_id,acquisition_date").execute()
        if getattr(res_upsert, "error", None):
            raise Exception(f"ndvi_micro_tiles upsert error: {res_upsert.error}")

        # Insert compact record into ndvi_data (frontend uses this)
        insert_ndvi = {
            "land_id": land_id,
            "farmer_id": farmer_id,
            "tenant_id": tenant_id,
            "date": acquisition,
            "ndvi_mean": ndvi_stats["mean"],
            "stats": ndvi_stats,
            "thumbnail_url": viz_url,
            "tile_id": metadata["tile_id"],
            "created_at": datetime.datetime.utcnow().isoformat()
        }
        res_ndvi = supabase.table("ndvi_data").insert(insert_ndvi).execute()
        if getattr(res_ndvi, "error", None):
            raise Exception(f"ndvi_data insert error: {res_ndvi.error}")

        # Update lands summary fields
        res_update = supabase.table("lands").update({
            "last_ndvi_calculation": acquisition,
            "last_ndvi_value": ndvi_stats["mean"],
            "updated_at": datetime.datetime.utcnow().isoformat()
        }).eq("id", land_id).execute()
        if getattr(res_update, "error", None):
            logger.warning("lands update returned error: %s", getattr(res_update, "error", None))

        logger.info(f"✅ Processed land {land_id}, mean NDVI {ndvi_stats['mean']:.4f}")
        return True

    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("❌ Failed processing land %s: %s", land_id, e)
        # Write a processing log row
        try:
            supabase.table("ndvi_processing_logs").insert({
                "satellite_tile_id": tile.get("id") or tile.get("tile_id"),
                "processing_step": "ndvi_calculation",
                "step_status": "failed",
                "completed_at": datetime.datetime.utcnow().isoformat(),
                "error_message": str(e),
                "traceback": tb,
                "metadata": {"land_id": land_id, "tenant_id": tenant_id}
            }).execute()
        except Exception as inner_e:
            logger.error("Failed to write ndvi_processing_logs: %s", inner_e)
        return False

# ---------------- Queue processing ----------------
def process_queue(limit: int = 10):
    """
    Process NDVI requests in ndvi_request_queue (queued).
    Each request contains land_ids, tile_id (optional), date range, cloud_coverage.
    """
    logger.info("Checking NDVI request queue")
    rq = supabase.table("ndvi_request_queue").select("*").eq("status", "queued").limit(limit).execute()
    requests_data = getattr(rq, "data", []) or []
    if not requests_data:
        logger.info("No queued requests")
        return

    for req in requests_data:
        req_id = req.get("id")
        logger.info("Processing queue request %s", req_id)
        try:
            supabase.table("ndvi_request_queue").update({"status": "processing", "started_at": datetime.datetime.utcnow().isoformat()}).eq("id", req_id).execute()

            land_ids = req.get("land_ids") or []
            if not land_ids:
                supabase.table("ndvi_request_queue").update({"status": "failed", "error_message": "Empty land_ids"}).eq("id", req_id).execute()
                continue

            # Fetch lands with tenant/farmer isolation
            lands_res = supabase.table("lands").select("id, farmer_id, tenant_id, boundary_polygon_old, name").in_("id", land_ids).execute()
            lands_list = getattr(lands_res, "data", []) or []
            if not lands_list:
                supabase.table("ndvi_request_queue").update({"status": "failed", "error_message": "No lands found"}).eq("id", req_id).execute()
                continue

            # Determine candidate tiles.
            # If request provided tile_id, prefer that tile for all lands; else pick latest tile within date range and cloud threshold
            tile_id_req = req.get("tile_id")
            date_from = req.get("date_from")
            date_to = req.get("date_to")
            cloud_threshold = req.get("cloud_coverage", 20)

            tiles_query = supabase.table("satellite_tiles").select("*").order("acquisition_date", desc=True)
            if tile_id_req:
                tiles_query = tiles_query.eq("tile_id", tile_id_req)
            else:
                if date_from:
                    tiles_query = tiles_query.gte("acquisition_date", date_from)
                if date_to:
                    tiles_query = tiles_query.lte("acquisition_date", date_to)
                tiles_query = tiles_query.lte("cloud_cover", cloud_threshold)

            tiles_res = tiles_query.execute()
            tiles_list = getattr(tiles_res, "data", []) or []
            if not tiles_list:
                supabase.table("ndvi_request_queue").update({"status": "failed", "error_message": "No tiles found"}).eq("id", req_id).execute()
                continue

            # Use the latest tile (first) as default. If you have tile mapping per land, you can extend this.
            latest_tile = tiles_list[0]
            success_count = 0
            skipped_count = 0

            for land in lands_list:
                # skip if boundary_polygon_old missing/null
                if not land.get("boundary_polygon_old"):
                    logger.warning("Skipping land %s - missing boundary_polygon_old", land.get("id"))
                    skipped_count += 1
                    continue

                ok = process_farmer_land(land, latest_tile)
                if ok:
                    success_count += 1

            supabase.table("ndvi_request_queue").update({
                "status": "completed",
                "completed_at": datetime.datetime.utcnow().isoformat(),
                "processed_count": success_count,
                "skipped_count": skipped_count
            }).eq("id", req_id).execute()

        except Exception as e:
            tb = traceback.format_exc()
            logger.exception("Error processing request %s: %s", req_id, e)
            supabase.table("ndvi_request_queue").update({
                "status": "failed",
                "error_message": str(e),
                "completed_at": datetime.datetime.utcnow().isoformat()
            }).eq("id", req_id).execute()

# ---------------- Ad-hoc runner ----------------
def main(limit: int = None, use_queue: bool = True):
    """
    Entry: if use_queue True - process queued requests, otherwise process latest N lands using latest tile.
    """
    logger.info("NDVI worker started (limit=%s, use_queue=%s)", limit, use_queue)
    if use_queue:
        process_queue(limit or 10)
        return

    # non-queue mode: process latest lands
    lands_res = supabase.table("lands").select("id, farmer_id, tenant_id, boundary_polygon_old").limit(limit or 50).execute()
    lands_list = getattr(lands_res, "data", []) or []
    if not lands_list:
        logger.info("No lands to process in non-queue mode")
        return

    tiles_res = supabase.table("satellite_tiles").select("*").order("acquisition_date", desc=True).limit(1).execute()
    tiles_list = getattr(tiles_res, "data", []) or []
    if not tiles_list:
        logger.info("No tiles available")
        return
    latest_tile = tiles_list[0]

    for land in lands_list:
        process_farmer_land(land, latest_tile)
