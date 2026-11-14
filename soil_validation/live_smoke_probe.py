"""No credentials, no database writes. Test public provider availability."""
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET
import json

def check_provider(name, url):
    with urlopen(Request(url, headers={"User-Agent": "KisanShaktiSoilValidation/1.0"}), timeout=40) as response:
        status=response.status
        content=response.read(2_000_000)
        ctype=response.headers.get("content-type", "")
    result={"http_status":status, "bytes":len(content), "type":ctype}
    if name=="soilgrids":
        root=ET.fromstring(content)
        names=[e.text for e in root.iter() if e.tag.rsplit("}",1)[-1]=="CoverageId"]
        result["count"]=len(names)
        result["has_median"]="phh2o_0-5cm_Q0.5" in names
        result["ok"]=status==200 and result["has_median"]
    elif name=="power":
        data=json.loads(content)
        params=data.get("properties",{}).get("parameter",{})
        result["available"]={k:sum(isinstance(v,(int,float)) and 0<=v<=1 for v in p.values()) for k,p in params.items()}
        result["ok"]=status==200 and all(result["available"].get(k,0)>0 for k in ("SFMC","RZMC"))
    else:
        data=json.loads(content)
        matches=[k for k in data if k.lower().endswith(("sm_surface","sm_rootzone"))]
        result["layer_matches"]=matches
        result["ok"]=status==200 and any(x.lower().endswith("sm_surface") for x in matches) and any(x.lower().endswith("sm_rootzone") for x in matches)
    return result

def main():
    endpoints={
      "soilgrids": "https://maps.isric.org/mapserv?"+urlencode({"map":"/map/phh2o.map","SERVICE":"WCS","VERSION":"2.0.0","REQUEST":"GetCapabilities"}),
      "power": "https://power.larc.nasa.gov/api/temporal/daily/point?"+urlencode({"parameters":"SFMC,RZMC","community":"HY","longitude":73.8567,"latitude":18.5204,"start":"20261001","end":"20261007","format":"JSON"}),
      "appeears": "https://appeears.earthdatacloud.nasa.gov/api/product/SPL4SMGP.008"
    }
    results={}
    for name,url in endpoints.items():
        try: results[name]=check_provider(name,url)
        except Exception as e: results[name]={"ok":False,"error":str(e)}
    print(json.dumps(results, indent=2))
    return 0 if all(r.get("ok") for r in results.values()) else 1

if __name__=="__main__":raise SystemExit(main())
