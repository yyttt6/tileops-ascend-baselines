"""CPU-only preservation of logs flushed after a guarded checker exits."""
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/root/dyq/workspace')
E = ROOT / '.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_transformer'
pid = int(sys.argv[1])
if Path(f'/proc/{pid}').exists():
    raise SystemExit('Wait for the original checker to exit before tail recovery.')
hashes, conflicts = {}, set()
receipts = []
for file in E.rglob('*.json'):
    if file.name in ('coverage.json', 'progress.json') or file.name.endswith('.ownership.json'):
        continue
    try:
        row = json.loads(file.read_text())
    except (ValueError, OSError):
        continue
    if not isinstance(row, dict) or row.get('runtime_pid') != pid or row.get('library') != 'ops-transformer':
        continue
    receipts.append(str(file))
    for path, digest in row.get('source_sha256', {}).items():
        if path in hashes and hashes[path] != digest:
            conflicts.add(path)
        hashes[path] = digest
for path in conflicts:
    hashes.pop(path)
target = E / 'runtime_tail' / str(pid)
target.parent.mkdir(exist_ok=True)
events = []
sources = []
for file in (ROOT / '.cache/all-pairs/ops_transformer/logs').rglob(f'*{pid}*.log'):
    raw = file.read_bytes()
    sources.append({'file': str(file), 'sha256': hashlib.sha256(raw).hexdigest()})
    for line, text in enumerate(raw.decode(errors='replace').splitlines(), 1):
        if any(marker in text for marker in ('StreamLaunchKernel', 'LogTaskDetails', '[ERROR]')):
            events.append({'file': str(file), 'line': line, 'text': text,
                           'captured_at': datetime.datetime.now(datetime.timezone.utc).isoformat()})
output = target.with_suffix('.native.jsonl')
if output.exists():
    raise SystemExit('Tail evidence already exists; refusing to overwrite.')
output.write_text(''.join(json.dumps(event) + '\n' for event in events))
record = {'schema': 'native-post-exit-tail/v1', 'runtime_pid': pid, 'source_sha256': hashes,
          'source_hash_conflicts_excluded': sorted(conflicts), 'corroborating_receipts': receipts,
          'raw_log_sources': sources, 'capture_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
          'purpose': 'Preserve delayed native log flush; no numerical receipt or execution claim.'}
target.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
print(len(events), 'post-exit events preserved')
