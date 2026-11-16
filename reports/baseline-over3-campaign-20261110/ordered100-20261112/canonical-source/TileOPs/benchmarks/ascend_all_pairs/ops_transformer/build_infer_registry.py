"""CPU-only build of missing graph metadata with the original CANN registry ABI.

Produces a staged library; installation is a separate step when no provider
process is running. No numerical kernels or schedules are compiled or edited.
"""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/root/dyq/workspace')
EVIDENCE = ROOT / '.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_transformer'
CACHE = ROOT / '.cache/all-pairs/ops_transformer'
base = json.loads((EVIDENCE / 'infer_shim_build.json').read_text())
command = base['command'][:base['command'].index('-shared')] + [
    '-shared', base['source'], '-L/usr/local/Ascend/cann-9.2.0-beta.2/lib64',
    '-Wl,--whole-archive', '-lrt2_registry', '-Wl,--no-whole-archive',
    '-lregister', '-lgraph', '-lexe_graph', '-lascendalog',
    '-o', str(CACHE / 'libcust_opsproto_metadata_r4.so'),
]
with (EVIDENCE / 'infer_registry_r4_build.log').open('w') as stream:
    rc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode
record = {
    'schema': 'original-shape-metadata-deployment/v1',
    'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'command': command, 'source': base['source'],
    'source_sha256': hashlib.sha256(Path(base['source']).read_bytes()).hexdigest(),
    'returncode': rc, 'installed': False,
    'reason': 'Link original CANN rt2_registry archive to export registry entry points; metadata only.',
}
if rc == 0:
    record['binary_sha256'] = hashlib.sha256(Path(command[-1]).read_bytes()).hexdigest()
(EVIDENCE / 'infer_registry_r4_build.json').write_text(json.dumps(record, indent=2) + '\n')
raise SystemExit(rc)
