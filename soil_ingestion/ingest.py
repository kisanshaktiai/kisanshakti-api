"""KisanShakti verified-source soil ingestion (never writes soil_health).

Install: pip install -r soil_ingestion/requirements.txt
Dry run: python -m soil_ingestion.ingest --land-id UUID --provider power
Write:   SOIL_INGEST_WRITE_ENABLED=yes python -m soil_ingestion.ingest --land-id UUID --provider power --write

Only the protected soil_source_observations table receives writes. No fake NPK,
no available-N interpretation of SoilGrids total nitrogen, no fake lab results.
"""
from __future__ import annotations
import argparse
from datetime import datetime, date, timezone
import hashlib
import io
import json
import math
import os
import re
import sys
import uuid
from urllib.parse import urlparse

import requests

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
ISRIC_URL = "https://maps.isric.org/mapserv"
APPEEARS_URL = "https://appeears.earthdatacloud.nasa.gov/api"
SOILGRIDS_CRS = "http://www.opengis.net/def/crs/EPSG/0/152160"
SOILGRIDS_PROJ = "+proj=igh +datum=WGS84 +units=m +no_defs +type=crs"
SOILGRIDS_DEPTH = {"0-5cm": (0, 5), "5-15cm": (5, 15), "15-30cm": (15, 30),
                    "30-60cm": (30, 60), "60-100cm": (60, 100), "100-200cm": (100, 200)}
# Property: canonical code, unit, native divisor, min, max. Values are *predictions*.
SOILGRIDS = {
    "phh2o": ("soil_ph", "pH", 10, 2, 12),
    "soc": ("soil_organic_carbon", "%", 100, 0, 35),
    "nitrogen": ("soil_total_nitrogen", "g/kg", 100, 0, 100),
    "clay": ("soil_clay", "%", 10, 0, 100),
    "sand": ("soil_sand", "%", 10, 0, 100),
    "silt": ("soil_silt", "%", 10, 0, 100),
    "bdod": ("soil_bulk_density", "g/cm3", 100, 0, 3),
    "cec": ("soil_cec", "cmol(+)/kg", 10, 0, 200),
}
WETNESS = {
    "GWETTOP": "soil_wetness_surface_fraction",
    "GWETROOT": "soil_wetness_rootzone_fraction",
    "GWETPROF": "soil_wetness_profile_fraction",
}
SMAP = {
    "Geophysical_Data_sm_surface": ("soil_moisture_surface", 0, 5),
    "Geophysical_Data_sm_rootzone": ("soil_moisture_rootzone", 0, 100),
}
ALLOWED = {
    "ISRIC_SOILGRIDS": {v[0]: v[1] for v in SOILGRIDS.values()},
    "NASA_POWER": {v: "fraction" for v in WETNESS.values()},
    "NASA_SMAP_L4_APPEEARS": {v[0]: "m3/m3" for v in SMAP.values()},
}

def valid_coords(lat, lon):
    for v, low, high in ((lat, -90, 90), (lon, -180, 180)):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not low <= v <= high:
            raise ValueError("Invalid location coordinates")
    return float(lat), float(lon)

def finite_value(v, low, high):
    if isinstance(v, bool):
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v in (-999, -9999, -32768) or not low <= v <= high:
        return None
    return v

def record(provider, property_code, unit, value, *, depth_top_cm=None, depth_bottom_cm=None,
           observed_at=None, statistic, reference, metadata):
    if provider not in ALLOWED or ALLOWED[provider].get(property_code) != unit:
        raise ValueError("Invalid source/property/unit combination")
    if finite_value(value, 0 if property_code != "soil_ph" else 2,
                    12 if property_code == "soil_ph" else
                    200 if property_code == "soil_cec" else
                    100 if property_code in ("soil_total_nitrogen","soil_clay","soil_sand","soil_silt") else
                    35 if property_code == "soil_organic_carbon" else
                    3 if property_code == "soil_bulk_density" else 1) is None:
        raise ValueError("Out-of-range or missing data")
    return dict(provider=provider, property_code=property_code, unit=unit,
                value=round(float(value), 6), depth_top_cm=depth_top_cm,
                depth_bottom_cm=depth_bottom_cm, observed_at=observed_at,
                statistic=statistic, source_reference=reference, source_metadata=metadata)

def soilgrids_tiff_pixel(blob, lat, lon):
    """Read a masked WCS raster at a GPS point; all-zero grid fails closed."""
    from rasterio.io import MemoryFile
    from rasterio.warp import transform
    if not blob.startswith((b"II*\x00", b"MM\x00*")):
        raise ValueError("WCS returned non-TIFF response")
    with MemoryFile(blob) as mem, mem.open() as ds:
        if ds.count != 1 or not ds.crs:
            raise ValueError("Unusable WCS raster structure")
        band = ds.read(1, masked=True)
        if band.count() == 0:
            raise ValueError("WCS has no valid pixels")
        if ds.nodata is None and not band.any():
            raise ValueError("WCS all-zero raster without nodata metadata")
        x, y = transform("EPSG:4326", ds.crs, [lon], [lat])
        if not (ds.bounds.left <= x[0] < ds.bounds.right and ds.bounds.bottom < y[0] <= ds.bounds.top):
            raise ValueError("WCS raster does not cover location")
        sample = next(ds.sample([(x[0], y[0])], masked=True))[0]
        if bool(getattr(sample, "mask", False)):
            raise ValueError("WCS point is nodata")
        return float(sample)

def get_soilgrids(session, lat, lon, depth="0-5cm", statistics=("Q0.05", "Q0.5", "Q0.95"),
                  properties=("phh2o", "soc", "nitrogen", "clay", "sand", "silt", "bdod", "cec")):
    from pyproj import Transformer
    lat, lon = valid_coords(lat, lon)
    if depth not in SOILGRIDS_DEPTH or not set(statistics).issubset({"Q0.05","Q0.5","Q0.95","mean"}):
        raise ValueError("Invalid SoilGrids depth/statistic")
    if not set(properties).issubset(SOILGRIDS):
        raise ValueError("Unknown SoilGrids property")
    tr = Transformer.from_crs("EPSG:4326", SOILGRIDS_PROJ, always_xy=True)
    x, y = tr.transform(lon, lat)
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("Cannot project SoilGrids location")
    out = []
    top, bottom = SOILGRIDS_DEPTH[depth]
    for layer in properties:
        code, unit, divisor, low, high = SOILGRIDS[layer]
        for stat in statistics:
            cov = f"{layer}_{depth}_{stat}"
            params = [("map",f"/map/{layer}.map"),("SERVICE","WCS"),
                      ("VERSION","2.0.1"),("REQUEST","GetCoverage"),
                      ("COVERAGEID",cov),("FORMAT","GEOTIFF_INT16"),
                      ("SUBSET",f"X({x-500:.3f},{x+500:.3f})"),
                      ("SUBSET",f"Y({y-500:.3f},{y+500:.3f})"),
                      ("SUBSETTINGCRS",SOILGRIDS_CRS),("OUTPUTCRS",SOILGRIDS_CRS)]
            res = session.get(ISRIC_URL, params=params, timeout=90)
            res.raise_for_status()
            raw = soilgrids_tiff_pixel(res.content, lat, lon)
            value = finite_value(raw / divisor, low, high)
            if value is None:
                raise ValueError(f"Invalid SoilGrids raster value for {cov}: {raw}")
            out.append(record("ISRIC_SOILGRIDS",code,unit,value,depth_top_cm=top,
                depth_bottom_cm=bottom,statistic=stat,reference=res.url,
                metadata={"coverage_id":cov,"raw_value":raw,"native_divisor":divisor,
                          "modelled_not_measured":True,"nominal_grid_m":250,
                          "soilgrids_depth":depth,"license":"CC-BY-4.0"}))
    return out

def parse_power(data):
    parameters = data.get("properties",{}).get("parameter",{})
    if not isinstance(parameters,dict):
        raise ValueError("NASA POWER response missing parameters")
    out = []
    for key,code in WETNESS.items():
        if key not in parameters or not isinstance(parameters[key],dict):
            raise ValueError("Missing requested POWER parameter "+key)
        for day,v in parameters[key].items():
            if not re.fullmatch(r"\d{8}",str(day)):
                raise ValueError("POWER returned invalid date")
            current = date.fromisoformat(f"{day[:4]}-{day[4:6]}-{day[6:]}")
            n = finite_value(v,0,1)
            if n is None:
                continue
            out.append(record("NASA_POWER",code,"fraction",n,
                observed_at=current.isoformat()+"T00:00:00+00:00",statistic="daily",
                reference=POWER_URL,metadata={"power_parameter":key,
                  "modelled_wetness_not_volumetric_water":True,
                  "product_attribution":"POWER_grid_not_field_sensor"}))
    return out

def get_power(session,lat,lon,start,end):
    lat,lon=valid_coords(lat,lon)
    a,b=date.fromisoformat(start),date.fromisoformat(end)
    if a>b or b>datetime.now(timezone.utc).date() or (b-a).days>31:
        raise ValueError("POWER date range must be up to 32 days and not in future")
    p={"latitude":lat,"longitude":lon,"start":a.strftime("%Y%m%d"),
       "end":b.strftime("%Y%m%d"),"parameters":",".join(WETNESS),
       "community":"AG","format":"JSON","time-standard":"UTC"}
    res=session.get(POWER_URL,params=p,timeout=90)
    res.raise_for_status()
    out=parse_power(res.json())
    for rec in out:
        rec["source_reference"]=res.url
    return out

def appeears_layers(session,product="SPL4SMGP.008"):
    if product!="SPL4SMGP.008":
        raise ValueError("Unreviewed SMAP product")
    res=session.get(f"{APPEEARS_URL}/product/{product}",timeout=45)
    res.raise_for_status()
    details=res.json()
    layers={k:k for k in SMAP if isinstance(details.get(k),dict) and details[k].get("Available") is not False}
    if len(layers)!=2:
        raise RuntimeError("Required SMAP layers missing")
    return layers

def appeears_token(session):
    token=os.getenv("APPEEARS_TOKEN")
    if token:
        return token
    username,password=os.getenv("EARTHDATA_USERNAME"),os.getenv("EARTHDATA_PASSWORD")
    if not username or not password:
        raise RuntimeError("Provide APPEEARS_TOKEN or EARTHDATA_USERNAME/PASSWORD securely")
    r=session.post(f"{APPEEARS_URL}/login",auth=(username,password),timeout=45)
    r.raise_for_status()
    return r.json()["token"]

def appeears_submit(session,lat,lon,start,end):
    lat,lon=valid_coords(lat,lon)
    a,b=date.fromisoformat(start),date.fromisoformat(end)
    if a>b or b>datetime.now(timezone.utc).date() or (b-a).days>31:
        raise ValueError("AppEEARS date range invalid")
    layers=appeears_layers(session)
    task={"task_type":"point","task_name":f"soil-sm-{a:%Y%m%d}-{b:%Y%m%d}",
          "params":{"dates":[{"startDate":a.strftime("%m-%d-%Y"),"endDate":b.strftime("%m-%d-%Y")}],
            "layers":[{"product":"SPL4SMGP.008","layer":k} for k in layers],
            "coordinates":[{"id":"soil","latitude":lat,"longitude":lon}]}}
    res=session.post(f"{APPEEARS_URL}/task",json=task,
                     headers={"Authorization":"Bearer "+appeears_token(session)},timeout=60)
    res.raise_for_status()
    task_id=res.json().get("task_id")
    if not task_id or not re.fullmatch(r"[0-9a-fA-F-]{36}",task_id):
        raise ValueError("Missing valid NASA task ID")
    return {"task_id":task_id,"product":"SPL4SMGP.008","lat":lat,"lon":lon}

def parse_smap_csv(blob, task_id):
    import csv
    reader=csv.DictReader(io.StringIO(blob.decode("utf-8-sig")))
    fields=reader.fieldnames or []
    date_col=next((x for x in fields if x.lower().strip() in ("date","datetime","date_time","time")),None)
    if date_col is None:
        raise ValueError("SMAP CSV date column missing")
    matches={k:next((x for x in fields if x==k or x.endswith("_"+k)),None) for k in SMAP}
    if not all(matches.values()):
        raise ValueError("SMAP CSV layer column missing")
    out=[]
    for item in reader:
        timestamp=item[date_col].strip()
        try:
            dt=datetime.fromisoformat(timestamp.replace("Z","+00:00"))
            if dt.tzinfo is None:
                dt=dt.replace(tzinfo=timezone.utc)
        except ValueError as ex:
            raise ValueError(f"Bad AppEEARS date: {timestamp}") from ex
        for field,col in matches.items():
            n=finite_value(item.get(col),0,1)
            if n is None:
                continue
            code,top,bottom=SMAP[field]
            out.append(record("NASA_SMAP_L4_APPEEARS",code,"m3/m3",n,
                depth_top_cm=top,depth_bottom_cm=bottom,observed_at=dt.astimezone(timezone.utc).isoformat(),
                statistic="3hourly",reference=f"{APPEEARS_URL}/bundle/{task_id}",
                metadata={"task_id":task_id,"layer":field,"product":"SPL4SMGP.008",
                          "modelled_not_field_measured":True,"nominal_grid_m":9000}))
    return out

def appeears_fetch(session,task_id,expected_lat,expected_lon):
    """Never attach a finished NASA task to another land's coordinates."""
    expected_lat, expected_lon = valid_coords(expected_lat, expected_lon)
    if not re.fullmatch(r"[0-9a-fA-F-]{36}",task_id):
        raise ValueError("Invalid task ID")
    headers={"Authorization":"Bearer "+appeears_token(session)}
    task=session.get(f"{APPEEARS_URL}/task/{task_id}",headers=headers,timeout=45)
    task.raise_for_status()
    task_data=task.json()
    coords=task_data.get("params",{}).get("coordinates",[])
    if not isinstance(coords,list) or len(coords)!=1:
        raise ValueError("NASA task has no uniquely verifiable target coordinate")
    point=coords[0]
    try:
        returned_lat,returned_lon=valid_coords(float(point["latitude"]),float(point["longitude"]))
    except (KeyError,TypeError,ValueError) as ex:
        raise ValueError("NASA task coordinates unavailable") from ex
    if abs(returned_lat-expected_lat)>0.000001 or abs(returned_lon-expected_lon)>0.000001:
        raise PermissionError("NASA task coordinates differ from the registered land")
    status=task_data.get("status")
    if status!="done":
        return {"status":status,"records":[]}
    res=session.get(f"{APPEEARS_URL}/bundle/{task_id}",headers=headers,timeout=45)
    res.raise_for_status()
    entries=res.json().get("files",[])
    out=[]
    for item in entries:
        if not item.get("file_name","").lower().endswith(".csv"):
            continue
        if item.get("file_size",0)>20_000_000:
            raise ValueError("SMAP CSV exceeds size cap")
        data=session.get(f"{APPEEARS_URL}/bundle/{task_id}/{item['file_id']}",headers=headers,timeout=90)
        data.raise_for_status()
        if len(data.content)>20_000_000:
            raise ValueError("SMAP CSV exceeds size cap")
        digest=item.get("sha256")
        if digest and hashlib.sha256(data.content).hexdigest()!=digest:
            raise ValueError("NASA bundle checksum mismatch")
        out+=parse_smap_csv(data.content,task_id)
    if not out:
        raise ValueError("No usable SMAP CSV values")
    return {"status":"done","records":out}

def validate_for_store(rec):
    if rec.get("provider") not in ALLOWED:
        raise ValueError("Unknown provider")
    if ALLOWED[rec["provider"]].get(rec.get("property_code")) != rec.get("unit"):
        raise ValueError("Invalid property or unit for provider")
    provider=rec["provider"]
    if provider=="ISRIC_SOILGRIDS" and (rec.get("observed_at") is not None or
        rec.get("source_metadata",{}).get("coverage_id") is None):
        raise ValueError("Bad static SoilGrids provenance")
    if provider!="ISRIC_SOILGRIDS" and not rec.get("observed_at"):
        raise ValueError("Missing temporal observation timestamp")
    v=finite_value(rec.get("value"),0,200)
    if v is None:
        raise ValueError("Missing data")
    record(provider,rec["property_code"],rec["unit"],v,statistic=rec["statistic"],
           observed_at=rec.get("observed_at"),reference=rec["source_reference"],
           metadata=rec["source_metadata"],depth_top_cm=rec.get("depth_top_cm"),
           depth_bottom_cm=rec.get("depth_bottom_cm"))

def load_land(session, url, key, land_id):
    uuid.UUID(land_id)
    base=url.rstrip("/")+"/rest/v1"
    headers={"apikey":key,"Authorization":"Bearer "+key}
    r=session.get(base+"/lands",params={"select":"id,tenant_id,center_lat,center_lon",
                                          "id":"eq."+land_id},
                  headers=headers,timeout=40)
    r.raise_for_status()
    lands=r.json()
    if not isinstance(lands,list) or len(lands)!=1:
        raise ValueError("Land not found")
    land=lands[0]
    if not land.get("tenant_id"):
        raise ValueError("Land has no tenant")
    if land.get("center_lat") is None or land.get("center_lon") is None:
        raise ValueError("No verified land centroid; do not guess from area")
    lat,lon=valid_coords(float(land["center_lat"]),float(land["center_lon"]))
    return land,lat,lon

def upsert(session, url, key, land, lat, lon, records, write):
    if not records:
        raise ValueError("No verified records to store")
    for rec in records:
        validate_for_store(rec)
    if not write:
        return {"mode":"dry_run","records":len(records),"writes":0}
    if os.getenv("SOIL_INGEST_WRITE_ENABLED")!="yes":
        raise PermissionError("SOIL_INGEST_WRITE_ENABLED=yes required")
    if not key or not url:
        raise ValueError("Missing Supabase credentials")
    rows=[]
    for rec in records:
        stable=[land["id"],lat,lon,rec["provider"],rec["property_code"],
                rec.get("depth_top_cm"),rec.get("depth_bottom_cm"),
                rec.get("observed_at"),rec["statistic"],
                rec["source_metadata"].get("coverage_id") or
                rec["source_metadata"].get("power_parameter") or
                rec["source_metadata"].get("layer")]
        digest=hashlib.sha256(json.dumps(stable,separators=(",",":"),sort_keys=True).encode()).hexdigest()
        rows.append(dict(record_key=digest,land_id=land["id"],tenant_id=land["tenant_id"],
          requested_lat=lat,requested_lon=lon,**rec))
    headers={"apikey":key,"Authorization":"Bearer "+key,"Content-Type":"application/json",
             "Prefer":"resolution=merge-duplicates,return=minimal"}
    endpoint=url.rstrip("/")+"/rest/v1/soil_source_observations"
    for idx in range(0,len(rows),100):
        r=session.post(endpoint,params={"on_conflict":"record_key"},
                       json=rows[idx:idx+100],headers=headers,timeout=60)
        r.raise_for_status()
    return {"mode":"staging_upsert","writes":len(rows),"table":"soil_source_observations"}

def main(argv=None):
    p=argparse.ArgumentParser(description="KisanShakti soil source ingestion")
    p.add_argument("--land-id",help="Real registered land UUID; required for fetch and upsert")
    p.add_argument("--provider",choices=("soilgrids","power","smap-submit","smap-fetch"),required=True)
    p.add_argument("--start",default="2025-01-01")
    p.add_argument("--end",default="2025-01-07")
    p.add_argument("--depth",default="0-5cm",choices=tuple(SOILGRIDS_DEPTH))
    p.add_argument("--task-id")
    p.add_argument("--out",default="soil_ingest_report.json")
    p.add_argument("--write",action="store_true")
    args=p.parse_args(argv)
    if args.write and os.getenv("SOIL_INGEST_WRITE_ENABLED")!="yes":
        p.error("Writing disabled; set SOIL_INGEST_WRITE_ENABLED=yes")
    url,key=os.getenv("SUPABASE_URL"),os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key or not url.startswith("https://"):
        p.error("A trusted server-side SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")
    session=requests.Session()
    session.headers["User-Agent"]="KisanShakti soil source research importer/1.0"
    if not args.land_id:
        p.error("--land-id required")
    land,lat,lon=load_land(session,url,key,args.land_id)
    if args.provider=="soilgrids":
        rows=get_soilgrids(session,lat,lon,depth=args.depth)
    elif args.provider=="power":
        rows=get_power(session,lat,lon,args.start,args.end)
    elif args.provider=="smap-submit":
        if args.write:
            p.error("--write is not supported for task submission")
        task=appeears_submit(session,lat,lon,args.start,args.end)
        result={"status":"task_submitted","task":task}
        with open(args.out,"w",encoding="utf-8") as file:
            json.dump(result,file,indent=2)
        print(json.dumps(result))
        return 0
    else:
        if not args.task_id:
            p.error("--task-id is required for smap-fetch")
        result=appeears_fetch(session,args.task_id,lat,lon)
        if result["status"]!="done":
            print(json.dumps({"status":result["status"],"writes":0}))
            return 2
        rows=result["records"]
    status=upsert(session,url,key,land,lat,lon,rows,args.write)
    result={"status":status,"land_id":land["id"],"location":{"lat":lat,"lon":lon},
            "retrieved_at":datetime.now(timezone.utc).isoformat(),"records":rows}
    with open(args.out,"w",encoding="utf-8") as file:
        json.dump(result,file,indent=2)
    print(json.dumps({"status":status,"output":args.out}))
    return 0

if __name__=="__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"INGEST ERROR: {type(e).__name__}: {e}",file=sys.stderr)
        sys.exit(1)
