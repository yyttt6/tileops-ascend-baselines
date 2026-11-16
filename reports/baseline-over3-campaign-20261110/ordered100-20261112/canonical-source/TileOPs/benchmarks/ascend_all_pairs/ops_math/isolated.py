"""Foreground native execution isolated from the unchanged Torch verifier OPP."""
import json
import os
import subprocess
import sys
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path
from .specs import ROOT, CACHE, SOURCE, VENDOR, SPECS, files


def require_guard():
    state = json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
    if state.get('phase') != 'running' or state.get('child', {}).get('pid') != os.getpgrp():
        raise RuntimeError('A foreground process in the host npu-guard group is required.')


@lru_cache(maxsize=None)
def declaration(needs_norm):
    from benchmarks.ascend_baselines import source_hashes
    own = Path(__file__).parent
    names = ('__init__.py', 'isolated.py', 'native_child.py', 'graph.py', 'specs.py',
             'composition.py', 'composition_specs.py', 'semantic_limits.py', 'layernorm.py',
             'running_stats.py', 'weighted.py', 'pooling.py', 'pool_indices.py',
             'mean_pooling.py', 'dequant.py', 'distance.py', 'topk_select.py')
    selected = [own / name for name in names]
    selected += [own / 'validated_abs/graph.py', own / 'validated_abs/specs.py']
    selected += [f for op in SPECS for f in files(op)]
    for folder in ('reduce_mean_with_count', 'cdist', 'arg_max_with_value', 'top_k_v2'):
        selected += [q for part in ('op_host', 'op_kernel', 'op_graph')
                     for q in (SOURCE / 'math' / folder / part).rglob('*')
                     if q.is_file() and q.suffix in ('.cpp', '.h', '.py')]
    selected += [q for q in VENDOR.rglob('*') if q.is_file() and q.suffix in ('.so', '.json', '.py', '.cpp', '.h')]
    if needs_norm:
        selected += [SOURCE / 'examples/fast_kernel_launch_example/csrc/layer_norm/dav-2201/layer_norm.asc',
                     CACHE / 'layernorm/libops_math_layernorm.so', CACHE / 'layernorm/receipt.json']
    return source_hashes(*sorted(set(selected)))


def make(case, op):
    require_guard()
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    full_vendor = CACHE / 'install/vendors/allpairs_math_math'
    configured = {Path(p).resolve() for p in os.environ.get('ASCEND_CUSTOM_OPP_PATH', '').split(':') if p}
    if configured.intersection((VENDOR.resolve(), full_vendor.resolve())):
        raise RuntimeError('Verifier must start without the expanded ops-math OPP; use ops_math/run_cases.py.')
    needs_norm = op in ('LayerNormFwdOp', 'GroupNormFwdOp', 'InstanceNormFwdOp',
                       'AdaLayerNormFwdOp', 'AdaLayerNormZeroFwdOp', 'FusedAddLayerNormFwdOp')
    needs_norm |= op == 'BatchNormFwdOp' and case.params.get('running_mean') is None
    hashes = dict(declaration(needs_norm))

    def map_tensors(value, fn):
        if isinstance(value, torch.Tensor):
            return fn(value)
        if isinstance(value, tuple):
            return tuple(map_tensors(v, fn) for v in value)
        if isinstance(value, list):
            return [map_tensors(v, fn) for v in value]
        if isinstance(value, dict):
            return {k: map_tensors(v, fn) for k, v in value.items()}
        return value

    def run(*args):
        require_guard()
        device = next(x.device for x in args if isinstance(x, torch.Tensor))
        transfer = CACHE / 'native-transfers'
        transfer.mkdir(exist_ok=True)
        records = Path(os.environ.get('OPS_MATH_NATIVE_CHILD_RECORDS', str(CACHE / 'native-child-records')))
        records.mkdir(parents=True, exist_ok=True)
        ident = uuid.uuid4().hex
        record = records / (ident + '.json')
        with tempfile.TemporaryDirectory(prefix=ident + '-', dir=transfer) as temporary:
            temporary = Path(temporary)
            source, target = temporary / 'input.pt', temporary / 'output.pt'
            torch.save({'op': op, 'case_id': case.id, 'params': case.params,
                        'inputs': map_tensors(args, lambda t: t.detach().cpu())}, source)
            env = dict(os.environ)
            paths = [p for p in env.get('ASCEND_CUSTOM_OPP_PATH', '').split(':')
                     if p and Path(p).resolve() not in (VENDOR.resolve(), full_vendor.resolve())]
            env['ASCEND_CUSTOM_OPP_PATH'] = ':'.join((str(VENDOR), *paths))
            env['PYTHONDONTWRITEBYTECODE'] = '1'
            command = [sys.executable, '-B', str(Path(__file__).with_name('native_child.py')),
                       '--input', str(source), '--output', str(target), '--record', str(record)]
            # subprocess.run waits in this process group; no new reservation,
            # session, daemon, background escape or detached native work.
            with record.with_suffix('.log').open('w') as log:
                result = subprocess.run(command, env=env, cwd=CACHE, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                detail = json.loads(record.read_text()) if record.exists() else {}
                if result.returncode == 78:
                    os.environ['OPS_MATH_CHILD_DEVICE_FAILED'] = '1'
                if detail.get('error_type') == 'UnsupportedCase':
                    raise UnsupportedCase(detail.get('error'))
                raise RuntimeError(f'Original native child exit {result.returncode}: {detail.get("error", record.with_suffix(".log"))}')
            completed = json.loads(record.read_text())
            if completed.get('status') != 'native_execution_complete':
                raise RuntimeError('Native child has no complete synchronized output record.')
            values = torch.load(target, map_location='cpu', weights_only=False)
            return map_tensors(values, lambda t: t.to(device))

    return Candidate(Implementation(run), 'ops-math::original native APIs in isolated foreground process for ' + op,
                     'Original inputs copied through host storage to an isolated same-guard native process. All original library math is unchanged; complete outputs return to the original device for the unchanged checker. Transfers and process startup are timed. Caller inputs are never exposed for native mutation.',
                     hashes, timing_mode='host_roundtrip',
                     limitations=('Native child records and every graph/fast-launch primitive require ownership evidence; CPU transfer does not compute outputs.',))
