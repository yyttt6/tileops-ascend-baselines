"""Three verified exact cases on server2's original FlashAttn deployment."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os

ROOT = Path('/root/dyq/workspace/reports/baseline-single-agent-server2-20261027/nsa_compressed')
CASES = frozenset(('prefill-4x1k-float16', 'prefill-8x1k-float16', 'prefill-ragged-bfloat16'))

def _load(name):
    path = ROOT / (name + '.py')
    spec = importlib.util.spec_from_file_location('_single_server2_nsa_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def build(case):
    guard = json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
    if guard.get('owner') != 'baseline-single-server2/nsa-compressed' or guard.get('phase') != 'running' or guard.get('child', {}).get('pid') != os.getpgrp():
        raise RuntimeError('Use server2 npu-guard owner baseline-single-server2/nsa-compressed with its original vendor environment.')
    assert case.id in CASES and case._call.signature.name == 'NSACompressedVarlenFwdOp'
    receipt = json.loads((ROOT/case.id/'checker.json').read_text())
    assert receipt['status'] == 'verified'
    assert json.loads(json.dumps(case.params)) == receipt['case_snapshot']['parameters']
    pins = json.loads((ROOT/'PROVIDER_PINS.json').read_text())
    for path, digest in pins.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Original baseline dependency changed: ' + path)
    adapter, api = _load('adapter'), _load('direct_api')
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    import torch
    def run(q, k, v, offsets, chunk_offsets, token_indices):
        return adapter.compressed(torch, api.flash_attn, api.flash_attn_metadata,
            q, k, v, offsets, chunk_offsets, token_indices,
            bs=case.params['bs'], scale=case.params['scale'])
    return Candidate(Implementation(run), 'Original FlashAttn and FlashAttnMetadata',
        'One full original API call after prefix-batch packing; copies, host metadata transfer, zero-prefix output and LSE output conversion are inside run.',
        dict(pins), timing_mode='host_sync',
        limitations=('Original server2 custom OPP environment and guard owner required; correctness only, no performance measurement.',))

def wrap(previous):
    def dispatch(case):
        if case.id in CASES:
            return build(case)
        return previous(case)
    return dispatch
