"""Exact-evidence device capture and isolated CPU replay; original APIs only.

Device phase is always wholly guarded. CPU phase has no device imports, RNG or
candidate execution. Exact contracts without controls/probes and the original
MoE defined-row validator are replayed; other special-property cases stay pending.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
PACKAGE = Path(__file__).parent
OUT = ROOT / '.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/tilelang_continuation'
CACHE = ROOT / '.cache/all-pairs/tilelang_continuation'
sys.path[:0] = [str(ROOT/'TileOPs'), str(ROOT/'TileOPs/src'), str(ROOT)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, default=str); stream.write('\n')


def sources(paths):
    return {str(Path(p).resolve()): sha(p) for p in paths if Path(p).is_file()}


def move(torch, value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, tuple): return tuple(move(torch, x) for x in value)
    if isinstance(value, list): return [move(torch, x) for x in value]
    if isinstance(value, dict): return {k: move(torch, v) for k, v in value.items()}
    return value


def metadata(torch, value):
    if isinstance(value, torch.Tensor):
        return {'tensor': True, 'device': str(value.device), 'dtype': str(value.dtype),
                'shape': list(value.shape), 'stride': list(value.stride()),
                'layout': str(value.layout), 'storage_offset': value.storage_offset()}
    if isinstance(value, (tuple, list)):
        return {'container': type(value).__name__, 'items': [metadata(torch, x) for x in value]}
    if isinstance(value, dict): return {k: metadata(torch, v) for k, v in value.items()}
    return value


def contract(torch, case):
    from workloads.numerics import Exact, Custom
    evidence = case.verification
    if (type(evidence) is Custom and case._call.signature.name == 'MoEGroupedGemmFwdOp'
            and not evidence.controls and evidence.probe is None):
        return {'kind': evidence.kind, 'reason': evidence.reason, 'controls': [], 'probe': None,
                'validator_qualname': evidence.validator.__qualname__,
                'validator_source_sha256': hashlib.sha256(inspect.getsource(evidence.validator).encode()).hexdigest()}
    if type(evidence) is not Exact or evidence.controls:
        raise ValueError('Original contract requires controls/custom probes; minimal exact capture does not admit it.')
    if case._call.signature.name in ('AddFwdOp', 'FP8QuantFwdOp'):
        raise ValueError('Original extra edge/property checks require the complete classic checker.')
    return asdict(evidence)


def capture_case(torch, row, run_dir):
    from benchmarks.api import cases
    from workloads.device import set_run_device
    from tools.baseline_protocol import seed_case
    from benchmarks.ascend_all_pairs.tilelang import PROVIDERS
    from benchmarks.ascend_all_pairs.tilelang import continuation_trace as trace
    started = time.monotonic()
    record = {'schema': 'tilelang-continuation-capture/v1', 'op': row['op'], 'case_id': row['case_id'],
              'tag': run_dir.name, 'status': 'started', 'original_input_generation_device': 'npu',
              'started_at': datetime.now(timezone.utc).isoformat(), 'performance_measured': False,
              'provider_sources': sources(PACKAGE.glob('*.py'))}
    run_dir.mkdir(parents=True, exist_ok=False)
    write(run_dir/'case_start.json', record)
    packet = {}
    candidate = None
    case = None
    stage = 'case'
    sync_failed = False
    def synchronize():
        nonlocal sync_failed
        try:
            torch.npu.synchronize()
        except Exception:
            sync_failed = True
            raise
    trace.begin()
    try:
        set_run_device('npu')
        record['input_seed'] = seed_case(torch, row['op'], row['case_id'])
        matches = [c for c in cases(row['op']) if c.id == row['case_id']]
        assert len(matches) == 1
        case = matches[0]
        inputs = case.inputs  # Original NPU generator, no replacement RNG.
        record['input_metadata'] = metadata(torch, inputs)
        record['evidence'] = contract(torch, case)
        record['case_params'] = case.params
        record['declared_outputs'] = list(case._call.signature.outputs)
        oracle_paths = [inspect.getsourcefile(type(case.workload)), inspect.getsourcefile(case.reference),
                        ROOT/'TileOPs/workloads/numerics.py', ROOT/'TileOPs/workloads/workload_base.py',
                        ROOT/'TileOPs/benchmarks/api.py', ROOT/'TileOPs/benchmarks/_cases.py',
                        ROOT/'tools/baseline_protocol.py']
        oracle_paths += list((ROOT/'TileOPs/src/tileops/manifest').rglob('*.py'))
        inventory_path = OUT.parent/'inventory.json'
        inventory = json.loads(inventory_path.read_text())
        manifest = next(r['manifest'] for r in inventory['operators'] if r['op'] == row['op'])
        oracle_paths.append(ROOT/'TileOPs'/manifest)
        record['manifest'] = manifest
        record['runtime_versions'] = {'torch': torch.__version__,
                                      'torch_file': torch.__file__,
                                      'torch_npu_file': sys.modules['torch_npu'].__file__}
        record['oracle_sources'] = sources(oracle_paths)
        record['provider_sources'] = sources(PACKAGE.glob('*.py'))
        packet['inputs'] = move(torch, inputs)
        stage = 'build'
        write(run_dir/'stage_build.json', record)
        candidate = PROVIDERS[row['op']](case)
        impl = candidate.implementation
        if impl.noncomparable_reason or impl.args is not None or impl.reset is not None:
            raise ValueError('Private argument/reset/noncomparable implementations require original full runner.')
        record.update(api=candidate.api, adaptation=candidate.adaptation, limitations=candidate.limitations)
        packet['after_build_inputs'] = move(torch, inputs)
        record['after_build_metadata'] = metadata(torch, inputs)
        # Original verifier restores inputs after the reference; preserve its inputs
        # independently before any restore can hide a mutation.
        stage = 'reference'
        write(run_dir/'stage_reference.json', dict(record, candidate_sources=dict(candidate.source_sha256)))
        if row['op'] == 'AdaptiveMaxPool2dFwdOp':
            record['reference_device'] = 'cpu_replay_of_exact_original_npu_inputs'
        else:
            reference = case.reference(*inputs)
            synchronize()
            packet['reference_output'] = move(torch, reference)
            record['reference_output_metadata'] = metadata(torch, reference)
            packet['after_reference_inputs'] = move(torch, inputs)
            record['after_reference_metadata'] = metadata(torch, inputs)
            record['reference_device'] = 'npu_original_manifest'
            del reference
        stage = 'candidate'
        write(run_dir/'stage_candidate.json', dict(record, candidate_sources=dict(candidate.source_sha256)))
        actual = impl.run(*inputs)
        synchronize()
        packet['candidate_output'] = move(torch, actual)
        record['candidate_output_metadata'] = metadata(torch, actual)
        packet['after_candidate_inputs'] = move(torch, inputs)
        record['after_candidate_metadata'] = metadata(torch, inputs)
        record['candidate_sources'] = dict(candidate.source_sha256)  # Include hashes populated lazily in run.
        record['status'] = 'captured_uncompared'
    except Exception as exc:
        record.update(status='capture_failed', stage=stage, error_type=type(exc).__name__,
                      error=str(exc), traceback=traceback.format_exc())
    finally:
        # Once synchronization fails, perform no further device work or cleanup.
        if sync_failed:
            record.update(status='device_sync_failed', device_synchronized=False)
        else:
            try:
                synchronize()
                record['device_synchronized'] = True
            except Exception as exc:
                record.update(status='device_sync_failed', sync_error=str(exc), device_synchronized=False)
        record['native'] = trace.result()
        if candidate is not None:
            record['candidate_sources'] = dict(candidate.source_sha256)
        if record.get('device_synchronized'):
            if candidate is not None and candidate.cleanup is not None:
                candidate.cleanup()
            if case is not None:
                case._release()
        record['wall_seconds'] = time.monotonic() - started
        record['finished_at'] = datetime.now(timezone.utc).isoformat()
        if packet and record.get('device_synchronized'):
            file = run_dir/'packet.pt'
            torch.save(packet, file)
            record['packet'] = str(file)
            # Content hashing and comparisons occur after guard release.
        write(run_dir/'capture.json', record)
        print(json.dumps({k: record.get(k) for k in ('op','case_id','status','stage','wall_seconds','error')}), flush=True)
    return record


def device(args):
    started = time.monotonic()
    from scripts.ascend_classic_check import require_guard
    require_guard()
    os.environ['TILELANG_CACHE_DIR'] = str(ROOT/'.cache/all-pairs/tilelang/jit')
    os.environ['TMPDIR'] = str(CACHE/'tmp')
    (CACHE/'tmp').mkdir(exist_ok=True)
    import torch
    import torch_npu
    from benchmarks.ascend_all_pairs.tilelang import continuation_trace
    continuation_trace.install()
    results = []
    for row in json.loads(Path(args.queue).read_text()):
        if results and time.monotonic() - started >= 60:
            break
        path = CACHE/'runs'/row['tag']
        if path.exists():
            continue
        record = capture_case(torch, row, path)
        results.append({k: record.get(k) for k in ('op','case_id','tag','status')})
        if record['status'] == 'device_sync_failed':
            return 78
        gc.collect()
        torch.npu.empty_cache()
    torch.npu.synchronize()
    return 0


def isolated_torch():
    assert sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
    os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
    import importlib.abc
    class BlockDevice(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in ('torch_npu','tilelang','triton','acl','torch_cuda'):
                raise ImportError('CPU replay blocks device import: ' + fullname)
    sys.meta_path.insert(0, BlockDevice())
    sys.path.append(str(ROOT/'.venv/lib/python3.12/site-packages'))
    import torch
    torch.set_num_threads(4)
    return torch


def assert_metadata(actual, expected, *, output=False):
    if isinstance(actual, dict) and actual.get('tensor'):
        fields = ('dtype','shape','layout') if output else ('device','dtype','shape','layout')
        for field in fields:
            assert actual[field] == expected[field], (field, actual, expected)
        if output:
            assert actual['device'].startswith('npu'), actual
            if expected['device'].startswith('npu'):
                assert actual['device'] == expected['device'], (actual, expected)
    elif isinstance(actual, dict) and 'container' in actual:
        assert actual['container'] == expected['container']
        for a, b in zip(actual['items'], expected['items'], strict=True):
            assert_metadata(a, b, output=output)
    else:
        assert actual == expected


def compare(args):
    torch = isolated_torch()
    from benchmarks.api import cases
    from workloads.numerics import compare_outputs, _unchanged, _unique
    run_dir = Path(args.run)
    record = json.loads((run_dir/'capture.json').read_text())
    result = {'schema': 'tilelang-continuation-verification/v1', 'op': record['op'],
              'case_id': record['case_id'], 'tag': record['tag'],
              'capture_sha256': sha(run_dir/'capture.json'), 'status': 'unchecked',
              'cpu_isolation': {'isolated': True, 'no_site': True, 'no_bytecode': True,
                                'torch_backend_autoload': '0', 'device_imports_blocked': True},
              'native_layer': record['native']['layer'], 'direct_driver_load_launch_observed': False}
    guard = None
    try:
        matching_guards = []
        for terminal in (OUT/'reservations').glob('*.result.json'):
            queue = terminal.with_name(terminal.name.replace('.result.json', '.queue.json'))
            if any(r['tag'] == record['tag'] for r in json.loads(queue.read_text())):
                matching_guards.append((terminal, json.loads(terminal.read_text())))
        assert len(matching_guards) == 1, 'Exactly one terminal guard result must own this capture.'
        terminal, guard = matching_guards[0]
        result['guard_result'] = str(terminal)
        result['guard_result_sha256'] = sha(terminal)
        result['guard_exit'] = guard['guard_exit']
        assert guard['guard_exit'] == 0, 'Nonzero terminal guard invalidates numerical acceptance.'
        assert record['status'] == 'captured_uncompared', record.get('error')
        assert record['device_synchronized']
        for scope in ('provider_sources','oracle_sources','candidate_sources'):
            for name, expected in record[scope].items():
                assert sha(ROOT/name) == expected, ('source_drift', name)
        packet_path = Path(record['packet'])
        result['packet_sha256'] = sha(packet_path)
        packet = torch.load(packet_path, weights_only=True, map_location='cpu')
        case = next(c for c in cases(record['op']) if c.id == record['case_id'])
        case.__dict__['inputs'] = packet['inputs']
        assert json.loads(json.dumps(case.params, default=str)) == record['case_params']
        assert json.loads(json.dumps(contract(torch, case))) == record['evidence']
        # Original zero-tolerance/equal-NaN comparator. The original device and
        # tensor attributes are separately checked before the transport copies.
        pristine = _unique(packet['inputs'])
        stages = ['build', 'candidate'] + (['reference'] if 'after_reference_inputs' in packet else [])
        for stage in stages:
            assert_metadata(record['after_'+stage+'_metadata'], record['input_metadata'])
            assert _unchanged(_unique(packet['after_'+stage+'_inputs']), pristine), 'input mutated by ' + stage
        if record['reference_device'].startswith('cpu_replay'):
            expected = case.reference(*packet['inputs'])
            # Check the replay's own mutation before comparing any output.
            assert _unchanged(_unique(packet['inputs']), _unique(packet['after_build_inputs']))
            expected_meta = metadata(torch, expected)
            result['cpu_reference'] = 'unchanged Case.reference on captured original NPU data; no CPU RNG'
        else:
            expected = packet['reference_output']
            expected_meta = record['reference_output_metadata']
        assert_metadata(record['candidate_output_metadata'], expected_meta, output=True)
        check = compare_outputs(packet['candidate_output'], expected, case.verification)
        result['verification'] = asdict(check)
        assert check.checked_outputs and check.checked_outputs == check.total_outputs
        assert record['native']['successful_calls'] > 0
        result.update(status='verified', input_preservation='original comparator on complete copies plus original metadata',
                      full_original_outputs=True, input_seed=record['input_seed'], evidence=record['evidence'])
    except Exception as exc:
        result.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        if guard is not None and guard['guard_exit'] != 0:
            result['status'] = 'guard_invalidated'
    result['finished_at'] = datetime.now(timezone.utc).isoformat()
    write(OUT/'runs'/(record['tag']+'.json'), result)
    print(json.dumps({k: result.get(k) for k in ('op','case_id','status','error')}), flush=True)
    return int(result['status'] != 'verified')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('device','compare'))
    parser.add_argument('--queue')
    parser.add_argument('--run')
    args = parser.parse_args()
    return device(args) if args.phase == 'device' else compare(args)


if __name__ == '__main__':
    raise SystemExit(main())
