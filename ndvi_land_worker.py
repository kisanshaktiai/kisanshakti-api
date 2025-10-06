import os, logging, tempfile, datetime
from supabase import create_client
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from rio_tiler.io import COGReader
from rio_tiler.colormap import cmap
from PIL import Image
import io

# ---------------- Config ----------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
B2_APP_KEY_ID = os.environ.get("B2_KEY_ID")
B2_APP_KEY = os.environ.get("B2_APP_KEY")
B2_BUCKET_NAME = os.environ.get("B2_BUCKET_PROCESSED", "ksai-sat-processed")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Backblaze B2
info = InMemoryAccountInfo()
b2_api = B2Api(info)
b2_api.authorize_account("production", B2_APP_KEY_ID, B2_APP_KEY)
bucket = b2_api.get_bucket_by_name(B2_BUCKET_NAME)


# ---------------- Utility Functions ----------------
def upload_bytes_to_b2(data_bytes, b2_path):
    """Upload raw bytes to B2 and return path"""
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(data_bytes)
            tmp.flush()
            bucket.upload_local_file(local_file=tmp.name, file_name=b2_path)
        return f"b2://{B2_BUCKET_NAME}/{b2_path}"
    except Exception as e:
        logging.error(f"Upload failed for {b2_path}: {e}")
        return None
    finally:
        try:
            os.remove(tmp.name)
        except:
            pass


def generate_png_tile(ndvi_tif, land_id, acq_date, z, x, y):
    """Generate single PNG tile for given z/x/y"""
    try:
        with COGReader(ndvi_tif) as cog:
            arr, mask = cog.tile(x, y, z)

            # Apply colormap (viridis works well for NDVI)
            cmapped = cmap.get("viridis")(arr, mask=mask)

            img = Image.fromarray((cmapped * 255).astype("uint8"), mode="RGBA")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)

            b2_path = f"tiles/processed/{land_id}/{acq_date}/{z}/{x}/{y}.png"
            return upload_bytes_to_b2(buf.getvalue(), b2_path)
    except Exception as e:
        logging.error(f"Tile {z}/{x}/{y} generation failed for {land_id}: {e}")
        return None


def generate_tiles(ndvi_tif, land_id, acq_date, min_zoom=12, max_zoom=18):
    """Generate tile pyramid and upload to B2"""
    uploaded_tiles = []
    try:
        with COGReader(ndvi_tif) as cog:
            bbox = cog.bounds

        for z in range(min_zoom, max_zoom + 1):
            try:
                # Compute x/y ranges for bounding box
                from mercantile import tiles
                tile_list = list(tiles(*bbox, [z]))
                for t in tile_list:
                    url = generate_png_tile(ndvi_tif, land_id, acq_date, z, t.x, t.y)
                    if url:
                        uploaded_tiles.append(url)
            except Exception as e:
                logging.error(f"Zoom {z} failed for {land_id}: {e}")
    except Exception as e:
        logging.error(f"Tile pyramid generation failed: {e}")

    logging.info(f"Generated {len(uploaded_tiles)} tiles for {land_id}")
    return uploaded_tiles


# ---------------- Main Worker Logic ----------------
def process_land_ndvi(land_id, ndvi_tif, acq_date):
    """
    Process NDVI GeoTIFF → Generate PNG tiles → Upload to B2 → Save metadata
    """
    try:
        logging.info(f"Processing NDVI for land {land_id} on {acq_date}")

        # Generate tiles
        tiles = generate_tiles(ndvi_tif, land_id, acq_date)

        if not tiles:
            logging.warning(f"No tiles generated for land {land_id}")
            return False

        # Store metadata in Supabase
        supabase.table("ndvi_micro_tiles").upsert({
            "land_id": land_id,
            "acquisition_date": acq_date,
            "ndvi_thumbnail_url": tiles[0],  # use first tile as thumbnail
            "thumbnail_size_kb": os.path.getsize(ndvi_tif) // 1024 if os.path.exists(ndvi_tif) else None,
            "statistics_only": False,
            "expires_at": (datetime.datetime.utcnow() + datetime.timedelta(days=180)).isoformat(),
        }, on_conflict=["land_id", "acquisition_date"]).execute()

        logging.info(f"NDVI tiles metadata stored for {land_id}")
        return True

    except Exception as e:
        logging.error(f"NDVI processing failed for {land_id}: {e}")
        return False
