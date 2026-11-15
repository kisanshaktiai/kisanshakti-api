"""Native-grid parcel aggregation. No fertility inference or soil_health writes.

Cell identity comes from the returned affine grid, never rounded GPS positions.
Quantiles are retained per cell; their spatial averages are NOT parcel quantiles.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone

import requests
from pyproj import CRS, Transformer
from rasterio.io import MemoryFile
from shapely import normalize
from shapely.geometry import Polygon, box, mapping, shape
from shapely.ops import transform

from .ingest import ISRIC_URL, SOILGRIDS, SOILGRIDS_CRS, SOILGRIDS_PROJ, finite_value

ALGORITHM = "parcel-native-area-v1"
DEPTHS = ("0-5cm", "5-15cm", "15-30cm")
STATISTICS = ("mean", "Q0.05", "Q0.5", "Q0.95")
MAX_CELLS = 4096
MAX_BYTES = 8_000_000


def geometry(geojson):
    """Strict EPSG:4326 GeoJSON; holes retained, no automatic polygon repair."""
    if not isinstance(geojson, dict) or "crs" in geojson:
        raise ValueError("Expected longitude/latitude GeoJSON without a CRS override")
    geom = shape(geojson)
    if geom.geom_type not in ("Polygon", "MultiPolygon") or geom.is_empty or not geom.is_valid:
        raise ValueError("Invalid parcel topology")
    west, south, east, north = geom.bounds
    if not all(math.isfinite(v) for v in geom.bounds) or not (-180 <= west < east <= 180 and -90 < south < north < 90):
        raise ValueError("Invalid longitude/latitude bounds")
    if east - west > 1 or north - south > 1:
        raise ValueError("Parcel exceeds bounded-subset limits or crosses antimeridian")
    projected = transform(Transformer.from_crs(4326, SOILGRIDS_PROJ, always_xy=True).transform, geom)
    if not projected.is_valid or not 1 <= projected.area <= 100_000_000:
        raise ValueError("Invalid projected parcel area")
    fingerprint = hashlib.sha256(normalize(geom).wkb).hexdigest()
    return projected, fingerprint


def cells_from_tiff(blob, parcel, prop):
    """Intersect actual native pixels with the parcel in equal-area projection."""
    if len(blob) > MAX_BYTES or not blob.startswith((b"II*\x00", b"MM\x00*")):
        raise ValueError("Invalid or oversized GeoTIFF")
    _, unit, divisor, low, high = SOILGRIDS[prop]
    with MemoryFile(blob) as mem, mem.open() as ds:
        if ds.count != 1 or not ds.crs or ds.width * ds.height > MAX_CELLS:
            raise ValueError("Invalid raster dimensions")
        if not CRS(ds.crs).equals(CRS(SOILGRIDS_PROJ), ignore_axis_order=True):
            raise ValueError("Raster is not on the SoilGrids equal-area grid")
        a = ds.transform
        if a.b or a.d or abs(a.a - 250) > .01 or abs(a.e + 250) > .01:
            raise ValueError("Resampled or non-native raster rejected")
        if any(abs(v / 250 - round(v / 250)) > 1e-6 for v in (a.c, a.f)):
            raise ValueError("Raster origin is not aligned to verified native grid")
        if not box(*ds.bounds).buffer(.01).covers(parcel):
            raise ValueError("Raster does not cover the complete parcel")
        band = ds.read(1, masked=True)
        if ds.nodata is None and not band.any():
            raise ValueError("All-zero raster without nodata metadata")
        result = []
        inverse = Transformer.from_crs(SOILGRIDS_PROJ, 4326, always_xy=True)
        # Boustrophedon traversal gives reproducible zigzag representative positions.
        for row in range(ds.height):
            cols = range(ds.width) if row % 2 == 0 else range(ds.width - 1, -1, -1)
            for col in cols:
                left, top = a * (col, row)
                pixel = box(left, top - 250, left + 250, top)
                overlap = parcel.intersection(pixel)
                if overlap.is_empty or overlap.area <= 1e-8:
                    continue
                representative = overlap.representative_point()
                lon, lat = inverse.transform(representative.x, representative.y)
                raw = band[row, col]
                valid = not bool(getattr(raw, "mask", False))
                value = finite_value(float(raw) / divisor, low, high) if valid else None
                if prop == "bdod" and value == 0:
                    value = None
                result.append({"pixel_id": f"igh:{left:.3f}:{top:.3f}:250",
                    "area_m2": overlap.area, "lon": lon, "lat": lat,
                    "raw_value": float(raw) if valid and math.isfinite(float(raw)) else None,
                    "value": value, "unit": unit, "native_divisor": divisor})
        if not result or abs(sum(c["area_m2"] for c in result) / parcel.area - 1) > 1e-6:
            raise ValueError("Incomplete native pixel coverage")
        return result


def aggregate(cells, parcel_area):
    if len({c["pixel_id"] for c in cells}) != len(cells):
        raise ValueError("Duplicate pixels must not be double counted")
    if not math.isfinite(parcel_area) or parcel_area <= 0:
        raise ValueError("Invalid area")
    valid = [c for c in cells if c["value"] is not None]
    area = sum(c["area_m2"] for c in valid)
    mean = sum(c["area_m2"] * c["value"] for c in valid) / area if area else None
    return {"mean": mean, "min": min((c["value"] for c in valid), default=None),
        "max": max((c["value"] for c in valid), default=None),
        "spatial_std": math.sqrt(sum(c["area_m2"] * (c["value"] - mean)**2 for c in valid) / area) if len(valid) > 1 else None,
        "distinct_pixels": len(cells), "valid_pixels": len(valid),
        "rejected_pixels": len(cells) - len(valid), "generated_points": len(cells),
        "coverage_fraction": min(1.0, area / parcel_area),
        "quality": "complete" if len(valid) == len(cells) else "partial" if valid else "unavailable"}


def get_bounded(session, params, sleep=time.sleep):
    """Bound bytes, time and retries; never log URLs containing parcel coordinates."""
    for attempt in range(3):
        try:
            with session.get(ISRIC_URL, params=params, timeout=(10, 60), stream=True) as response:
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    if attempt == 2:
                        response.raise_for_status()
                    retry = response.headers.get("Retry-After", "")
                    delay = min(30, max(1, int(retry))) if retry.isdigit() else 2 ** (attempt + 1)
                    sleep(delay)
                    continue
                response.raise_for_status()
                chunks, size = [], 0
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise ValueError("Provider response exceeds size cap")
                    chunks.append(chunk)
                return b"".join(chunks)
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 2:
                raise
            sleep(2 ** (attempt + 1))
    raise RuntimeError("Provider retries exhausted")


def fetch_parcel(geojson, *, source_version, session=None, properties=None, depths=DEPTHS,
                 statistics=STATISTICS, progress=None):
    """WCS bounded native subsets are authoritative for area weights.

    REST point ingestion remains in ingest.py for diagnostics. We never silently
    fall back to a centroid when a native raster is unavailable.
    """
    if not source_version or not isinstance(source_version, str):
        raise ValueError("An operator-verified source version is required")
    properties = tuple(properties or SOILGRIDS)
    if not set(properties) <= SOILGRIDS.keys() or not set(depths) <= set(DEPTHS):
        raise ValueError("Unknown property or depth")
    if "mean" not in statistics or not set(statistics) <= set(STATISTICS):
        raise ValueError("Mean prediction is required")
    if not depths or len(set(depths)) != len(depths) or len(set(statistics)) != len(statistics) or len(set(properties)) != len(properties):
        raise ValueError("Empty or duplicate layer selections")
    parcel, fingerprint = geometry(geojson)
    west, south, east, north = parcel.bounds
    # DescribeCoverage phh2o_0-5cm_mean verified 2026-11-15: outer origin
    # (-19949750,8361000), steps (250,-250). Unsnapped WCS subsets resample.
    west, south = math.floor(west / 250) * 250, math.floor(south / 250) * 250
    east, north = math.ceil(east / 250) * 250, math.ceil(north / 250) * 250
    if math.ceil((east-west)/250+2) * math.ceil((north-south)/250+2) > MAX_CELLS:
        raise ValueError("Parcel bounding raster exceeds request cap")
    session = session or requests.Session()
    output, warnings = {}, []
    for prop in properties:
        output[prop] = {}
        for depth in depths:
            evidence, summaries = {}, {}
            for statistic in statistics:
                coverage = f"{prop}_{depth}_{statistic}"
                params = [("map", f"/map/{prop}.map"), ("SERVICE", "WCS"),
                    ("VERSION", "2.0.1"), ("REQUEST", "GetCoverage"),
                    ("COVERAGEID", coverage), ("FORMAT", "GEOTIFF_INT16"),
                    ("SUBSET", f"X({west},{east})"),
                    ("SUBSET", f"Y({south},{north})"),
                    ("SUBSETTINGCRS", SOILGRIDS_CRS), ("OUTPUTCRS", SOILGRIDS_CRS)]
                blob = get_bounded(session, params)
                cells = cells_from_tiff(blob, parcel, prop)
                evidence[statistic] = {"coverage_id": coverage,
                    "sha256": hashlib.sha256(blob).hexdigest(), "cells": cells}
                summaries[statistic] = aggregate(cells, parcel.area)
                if progress:
                    progress(coverage)
            mean = summaries.pop("mean")
            output[prop][depth] = {**mean, "unit": SOILGRIDS[prop][1],
                "prediction_quantile_spatial_averages": summaries,
                "quantile_note": "Spatial averages of pixel quantiles; not a parcel confidence interval",
                "evidence": evidence}
    for depth in depths:
        if all(p in output for p in ("sand", "silt", "clay")):
            vals = [output[p][depth]["mean"] for p in ("sand", "silt", "clay")]
            if all(v is not None for v in vals) and abs(sum(vals)-100) > 2:
                warnings.append(f"texture_sum_inconsistent:{depth}")
    summary = {}
    if set(depths) == set(DEPTHS):
        for prop in properties:
            # pH is logarithmic; do not invent a blended laboratory pH.
            if prop == "phh2o":
                continue
            values = [output[prop][d] for d in DEPTHS]
            summary[prop] = {"value": sum(v["mean"] * w for v, w in zip(values, (5,10,15))) / 30
                if all(v["quality"] == "complete" for v in values) else None,
                "unit": SOILGRIDS[prop][1], "method": "thickness_weighted_concentration_not_stock"}
    available = sum(v["valid_pixels"] for p in output.values() for v in p.values())
    complete = all(v["quality"] == "complete" for p in output.values() for v in p.values())
    return {"algorithm_version": ALGORITHM, "source": "ISRIC_SOILGRIDS", "source_version": source_version,
        "modelled_not_measured": True, "geometry_fingerprint": fingerprint,
        "retrieved_at": datetime.now(timezone.utc).isoformat(), "area_m2": parcel.area,
        "requested_properties": list(properties), "requested_depths": list(depths),
        "requested_statistics": list(statistics),
        "method": "area_weighted_native_raster", "properties": output, "summary_0_30cm": summary,
        "quality": "complete" if complete else "partial" if available else "unavailable", "warnings": warnings}
