import json
from datetime import datetime
import uuid
from unittest.mock import Mock
import pytest
from soil_ingestion.ingest import (
    valid_coords,finite_value,record,parse_power,get_power,
    soilgrids_tiff_pixel,get_soilgrids,parse_smap_csv,
    appeears_layers,appeears_submit,load_land,upsert,validate_for_store,
)

class Response:
    def __init__(self,data=None,content=b"",url="https://example.org"):
        self._data=data
        self.content=content
        self.url=url
    def raise_for_status(self):pass
    def json(self):return self._data

class Session:
    def __init__(self,responses):
        self.responses=list(responses)
        self.calls=[]
    def get(self,url,**kwargs):
        self.calls.append(("get",url,kwargs))
        return self.responses.pop(0)
    def post(self,url,**kwargs):
        self.calls.append(("post",url,kwargs))
        return self.responses.pop(0)

def sample_record(provider="NASA_POWER",property_code="soil_wetness_surface_fraction",unit="fraction",value=0.65,meta=None):
    return record(provider,property_code,unit,value,statistic="daily",
        observed_at="2025-01-01T00:00:00+00:00",reference="https://power.larc.nasa.gov/",
        metadata=meta or {"power_parameter":"GWETTOP"})

@pytest.mark.parametrize("lat,lon",[(91,77),(10,190),(float("nan"),70),(True,70)])
def test_invalid_coordinates(lat,lon):
    with pytest.raises(ValueError):valid_coords(lat,lon)

def test_valid_ranges_and_missing():
    assert finite_value(-999,0,1) is None
    assert finite_value(1.2,0,1) is None
    assert finite_value("0.65",0,1)==0.65

def test_provider_units_restricted():
    with pytest.raises(ValueError):record("NASA_POWER","soil_moisture_surface","m3/m3",0.3,
          statistic="daily",reference="https://example.org",metadata={})

def test_power_missing_not_fabricated():
    p={"properties":{"parameter":{
      "GWETTOP":{"20250101":0.66,"20250102":-999},
      "GWETROOT":{"20250101":0.65,"20250102":-999},
      "GWETPROF":{"20250101":0.54,"20250102":-999}}}}
    result=parse_power(p)
    assert len(result)==3
    assert all(x["unit"]=="fraction" for x in result)
    assert all("moisture" not in x["property_code"] for x in result)
    assert result[0]["observed_at"]=="2025-01-01T00:00:00+00:00"

def test_power_request_uses_working_ag_community():
    data={"properties":{"parameter":{k:{"20250101":0.5} for k in
        ("GWETTOP","GWETROOT","GWETPROF")}}}
    s=Session([Response(data)])
    rows=get_power(s,18.5,73.8,"2025-01-01","2025-01-01")
    assert len(rows)==3
    params=s.calls[0][2]["params"]
    assert params["community"]=="AG"
    assert "GWETTOP" in params["parameters"]

def test_power_rejects_future_date():
    with pytest.raises(ValueError):get_power(Session([]),18.5,73.8,"2029-01-01","2029-01-02")

def tiff(raw=65,nodata=-32768):
    import numpy as np
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin
    with MemoryFile() as mem:
        with mem.open(driver="GTiff",height=10,width=10,count=1,
            dtype="int16",crs="EPSG:4326",nodata=nodata,
            transform=from_origin(73.85,18.53,.001,.001)) as ds:
            ds.write(np.full((10,10),raw,dtype="int16"),1)
        return mem.read()

def test_soilgrids_pixel_and_ph_conversion():
    assert soilgrids_tiff_pixel(tiff(),18.525,73.855)==65
    s=Session([Response(content=tiff(),url="https://maps.isric.org/test")])
    r=get_soilgrids(s,18.525,73.855,statistics=("Q0.5",),properties=("phh2o",))
    assert len(r)==1
    assert r[0]["value"]==6.5 and r[0]["unit"]=="pH"
    assert r[0]["observed_at"] is None
    assert r[0]["source_metadata"]["coverage_id"]=="phh2o_0-5cm_Q0.5"

def test_all_zero_wcs_fail_closed():
    with pytest.raises(ValueError,match="all-zero"):
        soilgrids_tiff_pixel(tiff(0,nodata=None),18.525,73.855)

def test_nodata_soilgrids_fail_closed():
    with pytest.raises(ValueError,match="no valid pixels"):
        soilgrids_tiff_pixel(tiff(-32768),18.525,73.855)

def test_wcs_xml_not_tiff():
    with pytest.raises(ValueError,match="non-TIFF"):
        soilgrids_tiff_pixel(b"<ExceptionReport/>",18.525,73.855)

def test_smap_discover_required_layers():
    data={k:{"Available":True} for k in ("Geophysical_Data_sm_surface","Geophysical_Data_sm_rootzone")}
    assert len(appeears_layers(Session([Response(data)])))==2
    with pytest.raises(RuntimeError):
        appeears_layers(Session([Response({"Geophysical_Data_sm_surface":{"Available":True}})]))

def test_smap_parse_time_and_units():
    blob=b"Date,Geophysical_Data_sm_surface,Geophysical_Data_sm_rootzone\n2025-01-01T03:00:00Z,0.2,0.4\n"
    rows=parse_smap_csv(blob,"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    assert len(rows)==2 and rows[0]["unit"]=="m3/m3"
    assert rows[0]["observed_at"]=="2025-01-01T03:00:00+00:00"

def test_smap_missing_columns_rejected():
    with pytest.raises(ValueError,match="layer column missing"):
        parse_smap_csv(b"Date,wrong\n2025-01-01,0.2\n","task")

def test_store_no_writes_without_flag(monkeypatch):
    monkeypatch.delenv("SOIL_INGEST_WRITE_ENABLED",raising=False)
    land={"id":"00000000-0000-0000-0000-000000000001",
          "tenant_id":"00000000-0000-0000-0000-000000000002"}
    session=Session([])
    data=[sample_record()]
    assert upsert(session,"https://example.supabase.co","key",land,18.5,73.8,data,False)["writes"]==0
    assert session.calls==[]
    with pytest.raises(PermissionError):
        upsert(session,"https://example.supabase.co","key",land,18.5,73.8,data,True)
    assert session.calls==[]

def test_upsert_is_idempotent_and_scoped(monkeypatch):
    monkeypatch.setenv("SOIL_INGEST_WRITE_ENABLED","yes")
    land={"id":"00000000-0000-0000-0000-000000000001",
          "tenant_id":"00000000-0000-0000-0000-000000000002"}
    s=Session([Response()])
    result=upsert(s,"https://example.supabase.co","test",land,18.5,73.8,[sample_record()],True)
    assert result["writes"]==1
    call=s.calls[0]
    assert call[2]["params"]["on_conflict"]=="record_key"
    assert call[2]["json"][0]["tenant_id"]==land["tenant_id"]
    assert call[2]["json"][0]["requested_lat"]==18.5
    assert len(call[2]["json"][0]["record_key"])==64
    assert call[1].endswith("/soil_source_observations")

def test_load_land_not_guessing_missing_centroid():
    s=Session([Response([{"id":"00000000-0000-0000-0000-000000000001",
                          "tenant_id":"00000000-0000-0000-0000-000000000002",
                          "center_lat":None,"center_lon":None}])])
    with pytest.raises(ValueError,match="No verified land centroid"):
        load_land(s,"https://example.supabase.co","key","00000000-0000-0000-0000-000000000001")

def test_no_static_observation_timestamp():
    r=record("ISRIC_SOILGRIDS","soil_ph","pH",6.5,statistic="Q0.5",
          reference="https://maps.isric.org",metadata={"coverage_id":"phh2o_0-5cm_Q0.5"})
    validate_for_store(r)
    r["observed_at"]="2025-01-01T00:00:00Z"
    with pytest.raises(ValueError,match="Bad static"):validate_for_store(r)

def test_no_available_npk_outputs():
    from soil_ingestion.ingest import ALLOWED
    assert not any("available" in key for values in ALLOWED.values() for key in values)

def test_smap_task_cannot_be_reassigned_to_another_field(monkeypatch):
    from soil_ingestion.ingest import appeears_fetch
    monkeypatch.setenv("APPEEARS_TOKEN","fake-local-token")
    task={"params":{"coordinates":[{"latitude":18.525,"longitude":73.855}]},"status":"done"}
    session=Session([Response(task)])
    with pytest.raises(PermissionError,match="differ"):
        appeears_fetch(session,"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",19.0,73.855)
    assert len(session.calls)==1

def test_smap_task_requires_provenance_coordinates(monkeypatch):
    from soil_ingestion.ingest import appeears_fetch
    monkeypatch.setenv("APPEEARS_TOKEN","fake-local-token")
    session=Session([Response({"status":"done","params":{}})])
    with pytest.raises(ValueError,match="verifiable target coordinate"):
        appeears_fetch(session,"aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",18.525,73.855)
    assert len(session.calls)==1

def test_nasa_power_preserves_underlying_model_source_metadata():
    data={"header":{"sources":["MERRA2","GEOSIT"],"time_standard":"UTC",
                    "api":{"version":"v2.10.3"}},
          "properties":{"parameter":{
              "GWETTOP":{"20261001":0.66},
              "GWETROOT":{"20261001":0.65},
              "GWETPROF":{"20261001":0.67}}}}
    rows=parse_power(data)
    assert len(rows)==3
    assert all(r["source_metadata"]["source_products"]==["MERRA2","GEOSIT"] for r in rows)
    assert all(r["source_metadata"]["time_standard"]=="UTC" for r in rows)
    assert all(r["source_metadata"]["api_version"]=="v2.10.3" for r in rows)

def test_nasa_power_unexpected_fill_value_refused():
    data={"header":{"fill_value":-888},"properties":{"parameter":{
      "GWETTOP":{"20261001":-888},
      "GWETROOT":{"20261001":-888},
      "GWETPROF":{"20261001":-888}}}}
    with pytest.raises(ValueError,match="fill-value"):
        parse_power(data)
