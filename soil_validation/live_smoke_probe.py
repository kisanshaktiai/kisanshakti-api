"""Read-only live provider probe. No credentials. No database writes."""
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from xml.etree import ElementTree as ET
import json

def request(url):
    try:
        with urlopen(Request(url,headers={"User-Agent":"KisanShaktiSoilValidation/1.0"}),timeout=40) as response:
            return response.status, response.headers.get("content-type",""),response.read(1500000)
    except HTTPError as e:
        return e.code, e.headers.get("content-type",""), e.read(8000)

def main():
    base_power="https://power.larc.nasa.gov/api/temporal/daily/point"
    base={"longitude":73.8567,"latitude":18.5204,"start":"20261001","end":"20261007","format":"JSON"}
    endpoints={
      "soilgrids_capabilities":"https://maps.isric.org/mapserv?"+urlencode({"map":"/map/phh2o.map","SERVICE":"WCS","VERSION":"2.0.0","REQUEST":"GetCapabilities"}),
      "power_HY_SFMC_RZMC":base_power+"?"+urlencode({**base,"parameters":"SFMC,RZMC","community":"HY"}),
      "power_AG_SFMC_RZMC_2026":base_power+"?"+urlencode({**base,"parameters":"SFMC,RZMC","community":"AG"}),
      "power_AG_SFMC_RZMC_2024":base_power+"?"+urlencode({**base,"start":"20240101","end":"20240107","parameters":"SFMC,RZMC","community":"AG"}),
      "power_AG_SFMC_RZMC_2025":base_power+"?"+urlencode({**base,"start":"20250101","end":"20250107","parameters":"SFMC,RZMC","community":"AG"}),
      "power_HY_SFMC":base_power+"?"+urlencode({**base,"parameters":"SFMC","community":"HY"}),
      "power_HY_T2M":base_power+"?"+urlencode({**base,"parameters":"T2M","community":"HY"}),
      "power_AG_GWET_2025":base_power+"?"+urlencode({**base,"start":"20250101","end":"20250107","parameters":"GWETTOP,GWETROOT","community":"AG"}),
      "soilgrids_getcoverage":"https://maps.isric.org/mapserv?"+urlencode([
        ("map","/map/phh2o.map"),("SERVICE","WCS"),("VERSION","2.0.1"),
        ("REQUEST","GetCoverage"),("COVERAGEID","phh2o_0-5cm_Q0.5"),
        ("FORMAT","GEOTIFF_INT16"),("SUBSET","X(7968349,7969349)"),
        ("SUBSET","Y(2061181,2062181)"),
        ("SUBSETTINGCRS","http://www.opengis.net/def/crs/EPSG/0/152160"),
        ("OUTPUTCRS","http://www.opengis.net/def/crs/EPSG/0/152160")]),
      "appeears_product":"https://appeears.earthdatacloud.nasa.gov/api/product/SPL4SMGP.008"
    }
    results={}
    for name,url in endpoints.items():
        try:
            status,ctype,blob=request(url)
            d={"http_status":status,"bytes":len(blob),"content_type":ctype}
            if name=="soilgrids_getcoverage":
                d["looks_like_tiff"]=blob[:4] in (b"II*\x00",b"MM\x00*")
                if d["looks_like_tiff"]:
                    from io import BytesIO
                    from rasterio.io import MemoryFile
                    from pyproj import Transformer
                    with MemoryFile(blob) as mem:
                        with mem.open() as ds:
                            x,y=Transformer.from_crs("EPSG:4326",ds.crs,always_xy=True).transform(73.8567,18.5204)
                            sample=list(ds.sample([(x,y)],masked=True))[0][0]
                            d["pixel_masked"]=bool(getattr(sample,"mask",False))
                            d["point_projected"]=[round(x,2),round(y,2)]
                            d["raster_bounds"]=list(ds.bounds)
                            d["in_raster"]=ds.bounds.left <= x < ds.bounds.right and ds.bounds.bottom < y <= ds.bounds.top
                            d["nodata"]=ds.nodata
                            d["min_max"]=[float(ds.read(1).min()),float(ds.read(1).max())]
                            if not d["pixel_masked"]:
                                d["raw_ph"]=float(sample)
                                d["estimated_ph"]=float(sample)/10.0
                            d["crs"]=str(ds.crs)
                else:d["provider_error"]=blob[:350].decode(errors="replace")
            elif name=="soilgrids_capabilities" and status==200:
                root=ET.fromstring(blob)
                identifiers=[e.text for e in root.iter() if e.tag.rsplit("}",1)[-1]=="CoverageId"]
                d["coverages"]=len(identifiers)
                d["has_ph_median"]="phh2o_0-5cm_Q0.5" in identifiers
            elif name=="appeears_product" and status==200:
                data=json.loads(blob)
                d["layer_matches"]=[k for k in data if k.lower().endswith(("sm_surface","sm_rootzone"))]
            else:
                try:
                    data=json.loads(blob)
                    if status==200:
                        params=data.get("properties",{}).get("parameter",{})
                        d["parameter_names"]=list(params)
                        d["valid_counts"]={k:sum(isinstance(v,(float,int)) and 0<=v<=1 for v in values.values()) for k,values in params.items()}
                        d["parameter_samples"]={k:list(values.items())[:2] for k,values in params.items()}
                        d["header"]={k:v for k,v in data.get("header",{}).items() if k in ("title","source","fill_value")}
                    else:
                        d["provider_error"]=str(data)[:1600]
                except Exception:
                    d["provider_error"]=blob.decode(errors="replace")[:1200]
            results[name]=d
        except Exception as e:
            results[name]={"error":f"{type(e).__name__}: {e}"}
    print(json.dumps(results,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
