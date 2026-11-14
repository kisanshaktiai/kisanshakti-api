"""One read-only live SoilGrids property request; illustrative coordinates."""
import json
import requests
from soil_ingestion.ingest import get_soilgrids_rest, get_soilgrids
session=requests.Session()
session.headers["User-Agent"]="KisanShakti-SoilGrids-Validation/1.0"
lat,lon=16.75,74.2
try:
    rows=get_soilgrids_rest(session,lat,lon)
    print(json.dumps({"transport":"REST","num_records":len(rows),
      "values":[{"p":x["property_code"],"stat":x["statistic"],"value":x["value"],"unit":x["unit"]}
                for x in rows]},indent=2),flush=True)
except Exception as e:
    print(json.dumps({"transport":"REST","error":str(e)[:300]}),flush=True)
try:
    rows=get_soilgrids(session,lat,lon,properties=("phh2o",),statistics=("Q0.5",))
    print(json.dumps({"transport":"WCS","num_records":len(rows),
      "value":rows[0]["value"] if rows else None},indent=2),flush=True)
except Exception as e:
    print(json.dumps({"transport":"WCS","error":str(e)[:300]}),flush=True)
