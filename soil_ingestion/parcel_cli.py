"""Read-only parcel diagnostics. Never connects to Supabase or writes farmer data."""
import argparse
import json
from pathlib import Path

from .parcel import fetch_parcel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', required=True, help='EPSG:4326 Polygon/MultiPolygon GeoJSON file')
    parser.add_argument('--source-version', required=True, help='Operator-verified release identifier; use unverified only for diagnostics')
    parser.add_argument('--out', required=True)
    parser.add_argument('--property', action='append', dest='properties')
    args = parser.parse_args()
    report = fetch_parcel(json.loads(Path(args.geometry).read_text()), source_version=args.source_version,
                          properties=args.properties)
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'quality':report['quality'], 'output':args.out}))
    return 0 if report['quality']=='complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
