"""Exact DSA source-race reconciliation; unchanged frozen whole-API binding."""
from pathlib import Path
import hashlib, types, json
ROOT = Path(__file__).resolve().parents[4] / 'reports/baseline-integration-700-20261026/sgl_dsa_source_reconcile1'
KEYS = frozenset(['mla-d512-h128-kv64k-bfloat16'])
PARAMETERS = {'q': [[1, 4096, 128, 512], 'bfloat16'], 'kv': [[1, 65536, 1, 512], 'bfloat16'], 'indices': [[1, 4096, 1, 1024], 'int32'], 'o': [[1, 4096, 128, 512], 'bfloat16'], 'dim_tail': 0, 'stride_kv': 1, 'q_start_index_s': 61440, 'sm_scale': None, 'is_causal': True}
PINS = {'python/sgl_kernel_npu/sgl_kernel_npu/__init__.py': '9111ad5e698dc0e4a93523c910c27aac6171b59855ce45230b2a935a19535f2a', 'python/sgl_kernel_npu/sgl_kernel_npu/attention/decode_attention.py': 'e60bac9a781e135ef8eb9a7366fb4aeef2fd0c87e6611eff655307a546dab2f2', 'python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu.so': 'c8026d311e1bf56f8bff8a03a1908b420efc421c45980885ded6d69b51c24a6d'}
ADAPTER_SHA256 = 'bb4b09f8f840241f808c6a5df82bd3122038ec0e9d8e1e70a3247f870c152e59'
def build(case):
    import benchmarks.ascend_baselines as base
    if case.id not in KEYS or json.loads(json.dumps(case.params)) != PARAMETERS:
        raise base.UnsupportedCase('Outside canonical DSA source reconciliation exact key/parameters')
    for relative, expected in PINS.items():
        path = ROOT/'libraries/sgl-kernel-npu'/relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Pinned original SGL dependency changed: '+str(path))
    path = ROOT/'dsa_adapter.py'
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != ADAPTER_SHA256:
        raise RuntimeError('Frozen DSA binding changed')
    saved = base.LIBRARIES
    base.LIBRARIES = ROOT/'libraries'
    try:
        # Execute only the just-hashed bytes, independent of generic module/bytecode caches.
        module = types.ModuleType('_baseline700_sgl_dsa_source_reconcile1_frozen')
        module.__file__ = str(path)
        exec(compile(source, str(path), 'exec'), module.__dict__)
        return module.build(case)
    finally:
        base.LIBRARIES = saved
