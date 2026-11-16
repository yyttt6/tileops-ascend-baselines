"""Read-only access to the frozen 2026-11-16 baseline refresh. No Torch/NPU imports."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / 'catalog/baseline-over3-refresh-20261116'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('command', choices=['summary', 'case', 'list', 'verify'])
parser.add_argument('--ordinal', type=int)
parser.add_argument('--library')
parser.add_argument('--op')
args = parser.parse_args()

if args.command == 'summary':
    print((CATALOG / 'summary.json').read_text())
elif args.command == 'verify':
    manifest = json.loads((ROOT / 'reports/baseline-over3-campaign-20261110/EXPORT_MANIFEST.json').read_text())
    bad = []
    for row in manifest['files']:
        path = ROOT / row['path']
        digest = hashlib.sha256()
        if not path.is_file():
            bad.append(row['path'])
            continue
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != row['sha256']:
            bad.append(row['path'])
    cases = json.loads((CATALOG / 'cases.json').read_text())['rows']
    assert len(cases) == 2119 and {r['ordinal'] for r in cases} == set(range(2119))
    assert len({tuple(r['key']) for r in cases}) == 2119
    assert all(r['qualified_routes'] and any(q['device_us'] is not None for q in r['qualified_routes']) for r in cases)
    print(json.dumps({'files_checked': len(manifest['files']), 'cases': len(cases), 'mismatches': bad}, indent=2))
    raise SystemExit(bool(bad))
else:
    cases = json.loads((CATALOG / 'cases.json').read_text())['rows']
    cases = [r for r in cases if (args.ordinal is None or r['ordinal'] == args.ordinal)
             and (not args.library or r['key'][0] == args.library)
             and (not args.op or r['key'][1] == args.op)]
    if args.command == 'case':
        if args.ordinal is None:
            parser.error('case requires --ordinal')
        if not cases:
            parser.error('case not found')
        print(json.dumps(cases[0], ensure_ascii=False, indent=2))
    else:
        for row in cases:
            print('\t'.join([str(row['ordinal']), *row['key'], str(len(row['qualified_routes']))]))
