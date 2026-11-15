from datetime import datetime, timezone
import pytest
from soil_ingestion.cache import cache_decision, six_months_after

NOW=datetime(2026,8,31,tzinfo=timezone.utc)
REPORT={'quality':'complete','geometry_fingerprint':'g','source_version':'s','algorithm_version':'a','retrieved_at':'2026-01-01T00:00:00+00:00'}


def decide(report=REPORT,**changes):
    args=dict(geometry_fingerprint='g',source_version='s',algorithm_version='a',now=NOW)
    args.update(changes)
    return cache_decision(report,**args)


def test_six_calendar_months_clamps_month_end():
    assert six_months_after(NOW)==datetime(2027,2,28,tzinfo=timezone.utc)


def test_new_cycle_or_elapsed_months_do_not_invent_new_measurement():
    d=decide()
    assert d['action']=='cached'
    assert d['last_successful_fetch_at']==REPORT['retrieved_at']
    assert d['last_checked_at']==NOW.isoformat()


@pytest.mark.parametrize('key',['geometry_fingerprint','source_version','algorithm_version'])
def test_changed_inputs_invalidate(key):
    assert decide(**{key:'new'})['action']=='fetch'


def test_partial_baseline_requires_retry():
    assert decide({**REPORT,'quality':'partial'})['action']=='fetch'


def test_source_unknown_requires_check():
    assert decide(source_version=None)['action']=='source_check_required'


def test_first_fetch():
    assert decide(None)['action']=='fetch'
