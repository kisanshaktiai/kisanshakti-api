"""TEMPORARY read-only diagnostic for official ISRIC endpoints."""
import json
from urllib.parse import urlencode
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from pyproj import Transformer
from rasterio.io import MemoryFile

POINTS={"pune_city":(18.5204,73.8567),"kolhapur_sample":(16.75,74.2),
        "sangli_sample":(16.9,74.6),"rural_maharashtra":(19.215,75.45)}
BASE="https://rest.isric.org/soilgrids/v2.0"
WCS="https://maps.isric.org/mapserv"
crs="http://www.opengis.net/def/crs/EPSG/0/152160"
tr=Transformer.from_crs("EPSG:4326","+proj=igh +datum=WGS84 +units=m +no_defs +type=crs",always_xy=True)

def get(url,timeout=14):
    try:
        with urlopen(Request(url,headers={"User-Agent":"KisanShakti SoilGrids validation/1.0","Accept":"application/json, image/tiff"}),timeout=timeout) as r:
            return r.status,r.headers.get("Content-Type"),r.read(2_000_000)
    except HTTPError as e:
        return e.code,e.headers.get("Content-Type"),e.read(3500)

for name,(lat,lon) in POINTS.items():
    for kind,path in [("properties","properties/query"),("classification","classification/query")]:
        params={"lon":lon,"lat":lat}
        if kind=="properties":params.update({"property":"phh2o","depth":"0-5cm","value":"Q0.5"})
        else:params["number_classes"]=3
        u=BASE+"/"+path+"?"+urlencode(params)
        try:
            status,ct,blob=get(u)
            info={"status":status,"type":ct,"bytes":len(blob)}
            if status==200:
                j=json.loads(blob); info.update({"json_keys":list(j),"preview":str(j)[:350]})
            else:info["error_body"]=blob[:550].decode(errors="replace")
            print(json.dumps({"point":name,"method":kind,**info}),flush=True)
        except Exception as e: print(json.dumps({"point":name,"method":kind,"error":str(e)[:300]}),flush=True)
    x,y=tr.transform(lon,lat)
    params=[("map","/map/phh2o.map"),("SERVICE","WCS"),("VERSION","2.0.1"),
        ("REQUEST","GetCoverage"),("COVERAGEID","phh2o_0-5cm_Q0.5"),("FORMAT","GEOTIFF_INT16"),
        ("SUBSET",f"X({x-500:.2f},{x+500:.2f})"),("SUBSET",f"Y({y-500:.2f},{y+500:.2f})"),
        ("SUBSETTINGCRS",crs),("OUTPUTCRS",crs)]
    try:
        status,ct,blob=get(WCS+"?"+urlencode(params),timeout=35)
        info={"status":status,"type":ct,"bytes":len(blob)}
        if status==200 and ct and "tiff" in ct.lower():
            with MemoryFile(blob) as mem,mem.open() as ds:
                band=ds.read(1,masked=True)
                sample=next(ds.sample([(x,y)],masked=True))[0]
                info.update({"pixel":None if bool(getattr(sample,"mask",False)) else int(sample),
                    "nodata":ds.nodata,"min":int(band.min()),"max":int(band.max()),
                    "valid_count":int(band.count()),"array_shape":list(band.shape)})
        else:info["error_body"]=blob[:600].decode(errors="replace")
        print(json.dumps({"point":name,"method":"wcs",**info}),flush=True)
    except Exception as e:print(json.dumps({"point":name,"method":"wcs","error":str(e)[:300]}),flush=True)
