"""Read-only baseline delivery reconciliation. Standard library only; no device access."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / 'catalog/baseline-reconciliation-20261116'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['summary', 'case', 'pending', 'verify', 'check-handoff'])
    parser.add_argument('--op')
    parser.add_argument('--case')
    parser.add_argument('--focused', action='store_true')
    args = parser.parse_args()
    plan = json.loads((CATALOG / 'release-plan.json').read_text())
    cases = plan['quant_cases']
    if args.command == 'summary':
        print(json.dumps(plan['summary'], ensure_ascii=False, indent=2))
    elif args.command == 'verify':
        manifest = json.loads((CATALOG / 'source-manifest.json').read_text())
        errors = []
        for row in manifest:
            path = ROOT / row['path']
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                errors.append(row['path'])
        if len(cases) != 71 or len({(x['op'], x['case']) for x in cases}) != 71:
            errors.append('71 unique exact cases required')
        for row in cases:
            registry = json.loads((ROOT / row['registry_path']).read_text())
            candidate = registry[int(row['registry_pointer'].lstrip('/'))]
            for field, expected in [('op', row['op']), ('case', row['case']),
                                    ('variant', row['historical_selected_variant']),
                                    ('parameters', row['parameters']),
                                    ('result_sha256', row['selected_result_sha256']),
                                    ('latency_us', row['historical_selected_device_us'])]:
                if candidate[field] != expected:
                    errors.append(f"{row['op']}/{row['case']}: registry {field}")
        print(json.dumps({'cases': len(cases), 'source_files': len(manifest), 'errors': errors}, indent=2))
        return bool(errors)
    else:
        cases = [r for r in cases if (not args.op or r['op'] == args.op)
                 and (not args.case or r['case'] == args.case)
                 and (not args.focused or r['in_focused_over3_63'])]
        if args.command == 'case':
            if not args.op or not args.case:
                parser.error('case requires --op and --case')
            print(json.dumps(cases, ensure_ascii=False, indent=2))
            return int(not cases)
        pending = [r for r in cases if r['adoption_status'] == 'historical_optimization_requires_producer_adoption_closure']
        print(json.dumps({'pending_count': len(pending), 'cases': [
            {'op': r['op'], 'case': r['case'], 'variant': r['historical_selected_variant'],
             'adoption_status': r['adoption_status']} for r in pending]}, ensure_ascii=False, indent=2))
        if args.command == 'check-handoff':
            # A successful artifact verify is not a successful adoption/publication check.
            return 2 if pending else 0
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
