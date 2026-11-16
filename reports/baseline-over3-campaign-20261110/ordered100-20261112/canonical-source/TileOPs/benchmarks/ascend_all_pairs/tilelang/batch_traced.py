"""Foreground bounded batch. Caller must hold npu-guard for the entire process."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import signal
import hashlib
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT.parent)]
sys.dont_write_bytecode = True
os.environ['TILELANG_CACHE_DIR'] = str(ROOT.parent / '.cache/all-pairs/tilelang/jit')
os.environ['TMPDIR'] = str(ROOT.parent / '.cache/all-pairs/tilelang/tmp')
from scripts import ascend_classic_check as checker
checker.require_guard()
p = argparse.ArgumentParser()
p.add_argument('--queue', type=Path, required=True)
p.add_argument('--seconds', type=int, default=240)
p.add_argument('--case-seconds', type=int, default=240)
args = p.parse_args()
from benchmarks.ascend_all_pairs.tilelang import PROVIDERS
from benchmarks.ascend_all_pairs.tilelang.diagnostics import DIAGNOSTIC_PROVIDERS
from benchmarks.ascend_all_pairs.tilelang import complex_check, reference_check
reference_check.install()
from benchmarks.ops import bench_ascend_classics
def build_candidate(case):
    candidate=(DIAGNOSTIC_PROVIDERS | PROVIDERS)[case._call.signature.name](case)
    return reference_check.annotate(complex_check.annotate(candidate,case),case)
bench_ascend_classics.candidate_factories = lambda case: {'tilelang': lambda: build_candidate(case)}
from benchmarks.ascend_all_pairs.tilelang import trace
trace.install()
# Keep exact own adapter revisions available after later, unrelated adapters evolve.
snapshots=ROOT.parent/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/tilelang/source_snapshots'
snapshots.mkdir(exist_ok=True)
for module in Path(__file__).parent.glob('*.py'):
    raw=module.read_bytes();destination=snapshots/(hashlib.sha256(raw).hexdigest()+'.py')
    if not destination.exists():destination.write_bytes(raw)
def expired(signum,frame):
    raise TimeoutError('Full manifest case exceeded bounded foreground allowance; partial launches are not verification.')
signal.signal(signal.SIGALRM,expired)
started = time.monotonic()
batch_state=snapshots.parent/'batch_state.json'
completed=0
def record_state(**details):
    state={'schema':'tilelang-batch-state/v1','queue':str(args.queue),
           'updated_at':datetime.now(timezone.utc).isoformat(),'completed_in_batch':completed,
           'elapsed_seconds':time.monotonic()-started,**details}
    temporary=batch_state.with_suffix('.pending.json')
    temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(batch_state)
record_state(phase='started')
for row in json.loads(args.queue.read_text()):
    if time.monotonic() - started > args.seconds-20:
        break
    output = ROOT.parent / row['output']
    if output.exists():
        continue
    remaining=args.seconds-(time.monotonic()-started)
    if remaining < row.get('estimated_seconds',20)+5:
        continue  # Leave this full case unattempted for a reservation with enough time.
    sys.argv = ['check', '--op', row['op'], '--case-id', row['case_id'], '--library', 'tilelang', '--output', str(output)]
    record_state(phase='case_running',op=row['op'],case_id=row['case_id'],receipt=str(output))
    trace.begin()
    signal.alarm(max(1,min(args.case_seconds,int(args.seconds-(time.monotonic()-started)))))
    try:
        checker.main()
    finally:
        signal.alarm(0)
    attribution = output.with_suffix('.attribution.json')
    assert not attribution.exists()
    selected=(DIAGNOSTIC_PROVIDERS | PROVIDERS)[row['op']]
    attribution_data=trace.result() | {'provider_module':'benchmarks.ascend_all_pairs.tilelang',
        'execution_environment':{'ASCEND_LAUNCH_BLOCKING':os.environ.get('ASCEND_LAUNCH_BLOCKING')},
        'runner_sources':{str(path.relative_to(ROOT.parent)):hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__),Path(trace.__file__),
                                       Path(complex_check.__file__),Path(reference_check.__file__))},
        'provider_entry_sha256':hashlib.sha256((Path(__file__).parent/'__init__.py').read_bytes()).hexdigest(),
        'factory_module':selected.__module__,'factory_qualname':selected.__qualname__}
    if complex_check._metadata is not None:
        attribution_data['input_preservation_compatibility']=complex_check._metadata
    if reference_check._metadata is not None:
        attribution_data['reference_compatibility']=reference_check._metadata
    attribution.write_text(json.dumps(attribution_data, indent=2)+'\n')
    completed+=1
    receipt=json.loads(output.read_text())
    record_state(phase='case_complete',op=row['op'],case_id=row['case_id'],receipt=str(output),
                 status=receipt.get('status'),stage=receipt.get('stage'),
                 native_successful_calls=attribution_data['total_successful_calls'])
    if receipt.get('status')=='device_sync_failed':
        record_state(phase='aborted_device_fault',op=row['op'],case_id=row['case_id'],receipt=str(output))
        raise SystemExit(1)
    # The checker synchronizes each case. Garbage collection frees old inputs.
    import gc
    import torch
    gc.collect()
    torch.npu.empty_cache()
record_state(phase='finished')
