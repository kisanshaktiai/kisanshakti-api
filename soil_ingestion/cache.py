"""Static model eligibility: a check date is not a soil measurement date."""
from calendar import monthrange
from datetime import datetime


def six_months_after(value: datetime) -> datetime:
    month = value.month + 6
    year = value.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    return value.replace(year=year, month=month, day=min(value.day, monthrange(year, month)[1]))


def cache_decision(report, *, geometry_fingerprint, source_version, algorithm_version, now):
    """Caller must check the source release before extending eligibility.

    A new crop cycle is a reason to call this function, not a reason to refetch
    an unchanged static map. Partial results are not silently certified complete.
    """
    if now.tzinfo is None:
        raise ValueError('Timezone-aware check time required')
    if not source_version:
        return {'action':'source_check_required', 'reason':'source_version_unknown'}
    if not report or report.get('quality') != 'complete':
        return {'action':'fetch', 'reason':'no_complete_baseline'}
    for key,value in [('geometry_fingerprint',geometry_fingerprint),('source_version',source_version),('algorithm_version',algorithm_version)]:
        if report.get(key) != value:
            return {'action':'fetch', 'reason':key+'_changed'}
    return {'action':'cached', 'reason':'unchanged_static_source',
            'last_checked_at':now.isoformat(), 'next_eligible_check_at':six_months_after(now).isoformat(),
            'last_successful_fetch_at':report['retrieved_at']}
