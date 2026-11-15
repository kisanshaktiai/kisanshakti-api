import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import box, Polygon

from soil_ingestion.parcel import aggregate, cells_from_tiff, geometry, get_bounded
from soil_ingestion.ingest import SOILGRIDS_PROJ


def tiff(values, nodata=-32768, resolution=250):
    data = np.array(values, dtype='int16')
    with MemoryFile() as mem:
        with mem.open(driver='GTiff', height=data.shape[0], width=data.shape[1],
                      count=1, dtype='int16', crs=SOILGRIDS_PROJ,
                      transform=from_origin(0, 500, resolution, resolution), nodata=nodata) as ds:
            ds.write(data, 1)
        return mem.read()


def test_small_field_one_pixel():
    field = box(10, 260, 100, 480)
    cells = cells_from_tiff(tiff([[60,80],[70,90]]), field, 'phh2o')
    assert len(cells) == 1
    result = aggregate(cells, field.area)
    assert result['mean'] == 6
    assert result['coverage_fraction'] == 1
    assert result['spatial_std'] is None


def test_unequal_area_mean_manual():
    # 250*200 m2 at pH 6 plus 50*200 m2 at pH 8: (6*5+8)/6.
    field = box(0,300,300,500)
    cells = cells_from_tiff(tiff([[60,80],[70,90]]), field, 'phh2o')
    result = aggregate(cells, field.area)
    assert result['mean'] == pytest.approx(38/6)
    assert result['distinct_pixels'] == 2
    assert result['min'] == 6 and result['max'] == 8


def test_holes_irregular_and_reproducible_zigzag():
    field = Polygon([(0,0),(500,0),(500,500),(0,500)], holes=[[(10,10),(200,10),(200,200),(10,200)]])
    blob = tiff([[60,70],[80,90]])
    cells = cells_from_tiff(blob, field, 'phh2o')
    assert [c['raw_value'] for c in cells] == [60,70,90,80]
    assert sum(c['area_m2'] for c in cells) == field.area
    assert cells == cells_from_tiff(blob, field, 'phh2o')


@pytest.mark.parametrize('bad', [-32768, 0, 200])
def test_invalid_ph_preserves_missing_coverage(bad):
    field = box(0,250,500,500)
    cells = cells_from_tiff(tiff([[60,bad]]), field, 'phh2o')
    result = aggregate(cells, field.area)
    assert result['mean'] == 6
    assert result['coverage_fraction'] == .5
    assert result['quality'] == 'partial'
    assert result['rejected_pixels'] == 1


@pytest.mark.parametrize('prop,raw,expected', [('soc',150,1.5),('nitrogen',150,1.5),('bdod',135,1.35),('cec',230,23),('clay',400,40)])
def test_official_units(prop,raw,expected):
    assert cells_from_tiff(tiff([[raw]]), box(0,250,250,500), prop)[0]['value'] == expected


def test_reject_duplicate_pixels():
    c = cells_from_tiff(tiff([[60]]), box(0,250,250,500), 'phh2o')
    with pytest.raises(ValueError,match='Duplicate'):
        aggregate(c+c,62500)


def test_reject_resampling_and_incomplete_raster():
    with pytest.raises(ValueError,match='non-native'):
        cells_from_tiff(tiff([[60]],resolution=500),box(0,250,250,500),'phh2o')
    with pytest.raises(ValueError,match='complete parcel'):
        cells_from_tiff(tiff([[60]]),box(0,0,500,500),'phh2o')


def test_missing_all_zero_fail_closed():
    with pytest.raises(ValueError,match='All-zero'):
        cells_from_tiff(tiff([[0]],nodata=None),box(0,250,250,500),'soc')


def test_geometry_canonical_fingerprint():
    p = {'type':'Polygon','coordinates':[[[74,16],[74.001,16],[74.001,16.001],[74,16.001],[74,16]]]}
    a, key = geometry(p)
    p['coordinates'][0].reverse()
    assert geometry(p)[1] == key
    assert a.area > 0


@pytest.mark.parametrize('p', [
    {'type':'Point','coordinates':[74,16]},
    {'type':'Polygon','coordinates':[[[74,16],[74.001,16.001],[74.001,16],[74,16.001],[74,16]]]},
    {'type':'Polygon','coordinates':[[[190,16],[191,16],[191,17],[190,17],[190,16]]]},
])
def test_invalid_geometry(p):
    with pytest.raises(ValueError): geometry(p)


def test_retry_after_bounded():
    class Response:
        status_code = 429
        headers = {'Retry-After':'99999'}
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def raise_for_status(self): raise RuntimeError('rate limited')
    class Session:
        def get(self,*args,**kwargs): return Response()
    sleeps = []
    with pytest.raises(RuntimeError,match='rate limited'):
        get_bounded(Session(),[],sleeps.append)
    assert sleeps == [30,30]
