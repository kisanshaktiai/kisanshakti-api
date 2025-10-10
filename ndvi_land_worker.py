# --- changes for download/open and save to ndvi_data + tenant filtering ---
import io
import json
import datetime
import logging
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.io import MemoryFile
from rasterio.warp import transform_geom
from shapely.geometry import shape, mapping

# assume supabase client already created as 'supabase' and B2_BUCKET_URL set

def download_tile_from_b2(tile_path: str) -> io.BytesIO:
    """Download GeoTIFF tile from Backblaze B2 bucket - tile_path is the path stored in DB"""
    if not tile_path:
        raise ValueError("No tile_path provided")
    url = f"{B2_BUCKET_URL.rstrip('/')}/{tile_path.lstrip('/')}"
    logging.info(f"Downloading tile from: {url}")
    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        raise Exception(f"Failed to download tile: {url} (Status: {resp.status_code})")
    return io.BytesIO(resp.content)

def open_raster_from_bytesio(buf: io.BytesIO):
    """Open a rasterio dataset from bytes (MemoryFile)"""
    buf.seek(0)
    memfile = MemoryFile(buf.read())
    return memfile.open()

def resolve_tile_tif_path(tile: dict) -> str:
    """
    Prefer NDVI product if present, else red/nir combination.
    This function returns a single path to open a multi-band GeoTIFF (recommended).
    """
    # Prefer precomputed NDVI product path if available
    if tile.get("ndvi_path"):
        return tile["ndvi_path"]
    # fall back to combined product path (if your sentinel L2A combined exists)
    if tile.get("product_path"):
        return tile["product_path"]
    # if you only have per-band paths, prefer a path that indicates combined tif
    # else attempt to use nir_band_path (some setups have 3-band file)
    if tile.get("nir_band_path"):
        return tile["nir_band_path"]
    # last-resort: b2_path if it exists in some deployments
    return tile.get("b2_path")  # keep as last fallback (may be None)

def process_farmer_land(land: dict, tile: dict) -> bool:
    land_id = land["id"]
    farmer_id = land.get("farmer_id")
    tenant_id = land.get("tenant_id")
    try:
        logging.info(f"Processing land {land_id} with tile {tile.get('id') or tile.get('tile_id')}")
        # load land geometry (support stringified json)
        boundary_data = land.get("boundary") or land.get("boundary_polygon_old")
        if not boundary_data:
            raise Exception("No boundary found for land")
        if isinstance(boundary_data, str):
            boundary_geojson = json.loads(boundary_data)
        else:
            boundary_geojson = boundary_data
        land_geom = shape(boundary_geojson)

        # Resolve tile path and download
        tif_path = resolve_tile_tif_path(tile)
        if not tif_path:
            raise Exception("No usable tile path found in satellite_tiles row")
        tif_bytes = download_tile_from_b2(tif_path)

        # Open with rasterio MemoryFile
        with open_raster_from_bytesio(tif_bytes) as src:
            # transform land geometry to source CRS
            land_geom_transformed = transform_geom("EPSG:4326", src.crs, mapping(land_geom))
            clipped, transform = mask(src, [land_geom_transformed], crop=True, all_touched=True, nodata=np.nan)
            # Note: rasterio returns shape (bands, rows, cols)
            num_bands = src.count

            # heuristics for identifying red/nir/blue bands in the file:
            # If precomputed NDVI product exists, it may be single band; else use band indices:
            if num_bands == 1:
                # precomputed NDVI product
                ndvi = clipped[0].astype(float)
                red = None
                nir = None
                blue = None
            else:
                # attempt sentinel-2 band mapping inside a combined product:
                # common order for some stacks: [B1,B2,B3,B4,B5,B6,B7,B8,...]
                # But we can't assume; we try to use index hints present in tile row:
                # If tile contains band indices, prefer them; else assume red=band 3 (index2), nir=band 7 (index6)
                try:
                    # prefer explicit index fields if provided (e.g., tile["red_band_index"])
                    red_index = int(tile.get("red_band_index", 4)) - 1   # user may store 4 for band 4
                    nir_index = int(tile.get("nir_band_index", 8)) - 1
                    blue_index = int(tile.get("blue_band_index", 2)) - 1
                except Exception:
                    red_index = 3  # 4th band (0-based)
                    nir_index = 7  # 8th band (0-based)
                    blue_index = 1
                # clamp indices
                red = clipped[red_index] if red_index < clipped.shape[0] else clipped[0]
                nir = clipped[nir_index] if nir_index < clipped.shape[0] else clipped[0]
                blue = clipped[blue_index] if blue_index < clipped.shape[0] else None

                ndvi = calculate_ndvi(red, nir)

        # Stats
        ndvi_stats = calculate_statistics(ndvi)
        if ndvi_stats["valid_pixels"] == 0:
            raise Exception("No valid NDVI pixels after clipping")

        # Visualization (png bytes)
        viz_bytes = generate_ndvi_visualization(ndvi)
        viz_filename = f"{land_id}_{tile.get('acquisition_date','unknown')}.png"
        viz_path = f"ndvi-thumbnails/{viz_filename}"

        # Upload thumbnail to Supabase storage (ensure bucket exists)
        # supabase.storage.from_("ndvi-thumbnails").upload expects file path and bytes or fileobj
        supabase.storage.from_("ndvi-thumbnails").upload(viz_path, viz_bytes, {"content-type": "image/png", "upsert": "true"})
        viz_url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/public/ndvi-thumbnails/{viz_path}"

        # metadata
        metadata = {
            "tile_id": tile.get("tile_id") or tile.get("id"),
            "resolution": tile.get("resolution", 10),
            "cloud_cover": tile.get("cloud_cover"),
            "processing_timestamp": datetime.datetime.utcnow().isoformat(),
            "statistics": ndvi_stats
        }

        # Upsert to ndvi_micro_tiles (keep existing behaviour)
        supabase.table("ndvi_micro_tiles").upsert({
            "land_id": land_id,
            "farmer_id": farmer_id,
            "tenant_id": tenant_id,
            "bbox": mapping(land_geom),
            "acquisition_date": tile.get("acquisition_date"),
            "cloud_cover": tile.get("cloud_cover"),
            "ndvi_mean": ndvi_stats["mean"],
            "ndvi_min": ndvi_stats["min"],
            "ndvi_max": ndvi_stats["max"],
            "ndvi_std_dev": ndvi_stats["std"],
            "ndvi_thumbnail_url": viz_url,
            "resolution_meters": metadata["resolution"]
        }, on_conflict="land_id,acquisition_date").execute()

        # ALSO insert a record in ndvi_data (the table your frontend reads)
        # Keep ndvi_data schema entry compact: store date, land_id, ndvi_mean, stats, thumbnail url, tile reference.
        supabase.table("ndvi_data").insert({
            "land_id": land_id,
            "farmer_id": farmer_id,
            "tenant_id": tenant_id,
            "date": tile.get("acquisition_date"),
            "ndvi_mean": ndvi_stats["mean"],
            "stats": ndvi_stats,
            "thumbnail_url": viz_url,
            "tile_id": metadata["tile_id"],
            "created_at": datetime.datetime.utcnow().isoformat()
        }).execute()

        # Update lands summary fields
        supabase.table("lands").update({
            "last_ndvi_calculation": tile.get("acquisition_date"),
            "last_ndvi_value": ndvi_stats["mean"],
            "updated_at": datetime.datetime.utcnow().isoformat()
        }).eq("id", land_id).execute()

        logging.info(f"Processed land {land_id}, mean NDVI {ndvi_stats['mean']:.4f}")
        return True

    except Exception as e:
        logging.exception(f"Failed processing land {land_id}: {e}")
        # logging into processing logs table
        supabase.table("ndvi_processing_logs").insert({
            "satellite_tile_id": tile.get("id") or tile.get("tile_id"),
            "processing_step": "ndvi_calculation",
            "step_status": "failed",
            "completed_at": datetime.datetime.utcnow().isoformat(),
            "error_message": str(e),
            "metadata": {"land_id": land_id, "tenant_id": tenant_id}
        }).execute()
        return False
