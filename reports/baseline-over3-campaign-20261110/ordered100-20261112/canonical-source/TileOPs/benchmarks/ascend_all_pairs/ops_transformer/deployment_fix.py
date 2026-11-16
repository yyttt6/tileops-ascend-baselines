"""Compatibility-only removal of -Werror from isolated generated wrappers.

Do not run while a checker is using this package. Original vendor sources and
kernel/math/schedule/compiler optimization flags are never changed.
"""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path('/root/dyq/workspace')
CACHE = ROOT/'.cache/all-pairs/ops_transformer'
EVIDENCE = ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_transformer'


def main():
    changes = []
    for path in sorted((CACHE/'install/vendors').glob('*/op_impl/ai_core/tbe/*/dynamic/*.py')):
        original = path.read_bytes()
        changed = original.replace(b", '-Werror'", b'').replace(b"'-Werror', ", b'')
        if original == changed:
            continue
        before = hashlib.sha256(original).hexdigest()
        snapshot = CACHE/'source_snapshots'/before/path.name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists() and snapshot.read_bytes() != original:
            raise RuntimeError('Content-addressed snapshot mismatch')
        if not snapshot.exists():
            snapshot.write_bytes(original)
        path.write_bytes(changed)
        changes.append({'file': str(path.relative_to(ROOT)), 'before_sha256': before,
                        'after_sha256': hashlib.sha256(changed).hexdigest(),
                        'original_snapshot': str(snapshot)})
    receipt = {'schema': 'ops-transformer-deployment-compatibility/v1',
               'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'reason': 'CANN 9.2 beta SDK headers emit callee-function compiler warnings; generated -Werror promotes them to fatal compilation errors.',
               'change': 'Remove only warning-as-error flag in generated isolated Python wrappers; preserve all kernel bodies and schedule flags.',
               'changes': changes}
    if changes:
        target = EVIDENCE/('deployment_compat_'+datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S')+'.json')
        target.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'changed_wrappers': len(changes)}))


if __name__ == '__main__':
    main()
