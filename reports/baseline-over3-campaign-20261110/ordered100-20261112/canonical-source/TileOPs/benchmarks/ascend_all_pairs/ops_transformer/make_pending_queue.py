"""CPU-only exact-manifest queue for registered cases still marked untested."""
import argparse
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/root/dyq/workspace')
E = ROOT / '.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_transformer'
parser = argparse.ArgumentParser()
parser.add_argument('label', help='New immutable output directory label, e.g. pending_r15')
args = parser.parse_args()
if not args.label.replace('_', '').replace('-', '').isalnum():
    parser.error('Use letters, digits, hyphens or underscores in the output label.')
coverage_path = E / 'coverage.json'
coverage = json.loads(coverage_path.read_text())
inventory = json.loads((E.parent / 'inventory.json').read_text())
lookup = {(c['op'], c['case_id']): c for c in inventory['cases']}
selected = []
for op in coverage['operators']:
    if not op['registered']:
        continue
    for case in op['cases']:
        if case['status'] != 'supported_untested':
            continue
        key = (op['op'], case['case_id'])
        item = lookup[key]
        size = sum(math.prod(t[0]) for t in item.get('tensors', {}).values())
        selected.append((key, size))
# Visit each operation's smallest untested case before its large continuations.
groups = collections.defaultdict(list)
for key, size in selected:
    groups[key[0]].append((size, key))
first, rest = [], []
for values in groups.values():
    values.sort()
    first.append(values[0]); rest.extend(values[1:])
queue = []
for _, (op, cid) in sorted(first) + sorted(rest):
    queue.append(['--op', op, '--case-id', cid, '--library', 'ops-transformer',
                  '--output', str(E / args.label / f'{op}__{cid}.json')])
destination = E / f'{args.label}_queue.json'
if destination.exists():
    raise SystemExit(f'Refusing to overwrite existing queue snapshot: {destination}')
destination.write_text(json.dumps(queue, indent=2) + '\n')
summary = {'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'schema': 'ops-transformer-pending-queue/v1', 'queue': str(destination),
           'cases': len(queue), 'operators': len(groups),
           'coverage_sha256': hashlib.sha256(coverage_path.read_bytes()).hexdigest(),
           'selection': 'registered + supported_untested only; excludes existing failures and verified receipts',
           'guard': 'npu-guard run --owner allpairs/ops_transformer --wait 3600 --timeout 900 -- bash TileOPs/benchmarks/ascend_all_pairs/ops_transformer/run.sh --queue QUEUE --seconds 700',
           'note': '900 seconds only for original JIT compilation. Rebuild ownership/coverage after each run before generating another snapshot. Existing per-case outputs are immutable resume keys.'}
(E / f'{args.label}_queue_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary))
