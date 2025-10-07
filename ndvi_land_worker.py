import os, logging, tempfile, datetime, io
from supabase import create_client
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from rio_tiler.io import COGReader
from rio_tiler.colormap import cmap
from PIL import Image
from mercantile import tiles

# ---------------- Config ----------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")

# Backblaze B2 (farmer app only needs READ)
B2_APP_KEY_ID = os.environ.get("B2_KEY_ID")        # matches env
B2_APP_KEY = os.environ.get("B2_APP_KEY")          # ✅ fixed name
B2_BUCKET_NAME = os.environ.get("B2_BUCKET")       # matches env
B2_PREFIX = os.environ.get("B2_PREFIX", "tiles/processed/")

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------- Clients ----------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

info = InMemoryAccountInfo()
b2_api = B2Api(info)
b2_api.authorize_account("production", B2_APP_KEY_ID, B2_APP_KEY)
bucket = b2_api.get_bucket_by_name(B2_BUCKET_NAME)


def upload_bytes_to_b2(data_bytes, b2_path):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(data_bytes)
            tmp.flush()
            bucket.upload_local_file(local_file=tmp.name, file_name=b2_path)
        return f"b2://{B2_BUCKET_NAME}/{b2_path}"
    except Exception as e:
        logging.error(f"Upload failed {b2_path}: {e}")
        return None
    finally:
        try: os.remove(tmp.name)
        except: pass


def generate_png_tile(ndvi_tif, land_id, acq_date, z, x, y):
    try:
        with COGReader(ndvi_tif) as cog:
            arr, mask = cog.tile(x, y, z)
            cmapped = cmap.get("viridis")(arr, mask=mask)
            img = Image.fromarray((cmapped * 255).astype("uint8"), mode="RGBA")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            return upload_bytes_to_b2(buf.getvalue(), f"tiles/processed/{land_id}/{acq_date}/{z}/{x}/{y}.png")
    except Exception as e:
        logging.error(f"Tile {z}/{x}/{y} failed: {e}")
        return None


def generate_tiles(ndvi_tif, land_id, acq_date, min_zoom=12, max_zoom=18):
    urls = []
    try:
        with COGReader(ndvi_tif) as cog:
            bbox = cog.bounds
        for z in range(min_zoom, max_zoom + 1):
            for t in tiles(*bbox, [z]):
                url = generate_png_tile(ndvi_tif, land_id, acq_date, z, t.x, t.y)
                if url:
                    urls.append(url)
    except Exception as e:
        logging.error(f"Tile pyramid failed: {e}")
    logging.info(f"Generated {len(urls)} tiles for land {land_id}")
    return urls


def process_land_ndvi(land_id, ndvi_tif, acq_date):
    try:
        urls = generate_tiles(ndvi_tif, land_id, acq_date)
        if not urls:
            return False
        supabase.table("ndvi_micro_tiles").upsert({
            "land_id": land_id,
            "acquisition_date": acq_date,
            "ndvi_thumbnail_url": urls[0],
            "expires_at": (datetime.datetime.utcnow() + datetime.timedelta(days=180)).isoformat()
        }, on_conflict=["land_id", "acquisition_date"]).execute()
        logging.info(f"✅ NDVI stored for land {land_id}")
        return True
    except Exception as e:
        logging.error(f"NDVI failed for {land_id}: {e}")
        return False
