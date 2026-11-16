"""Exact original aclStft route; native system AiCPU dependency disclosed."""
from pathlib import Path
import hashlib, json, os, types
_KEYS = frozenset([('FFTC2CFwdOp', 'scientific-1m-complex128'), ('FFTC2CFwdOp', 'throughput-4k-complex128')])
_P = Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/fft_remaining_c128_2')
_VENDOR = '/root/dyq/workspace/.cache/ascend-classic/native-pilot-20261020/ops-math/install/vendors/classicpilot_math'
def _load():
    data=(_P/'adapter.py').read_bytes()
    if hashlib.sha256(data).hexdigest()!='44aa9a2147096990d928dc5d7cd7e1593987880a4cb6e8da51a65da55222eff1': raise RuntimeError('Frozen adapter changed')
    m=types.ModuleType('fft_remaining_c128_2_frozen');m.__file__=str(_P/'adapter.py');exec(compile(data,m.__file__,'exec'),m.__dict__);return m
def make(case):
    from benchmarks.ascend_baselines import UnsupportedCase
    if (case._call.signature.name,case.id)not in _KEYS: raise UnsupportedCase('No exact receipt')
    if os.environ.get('ASCEND_CLASSIC_MATH_VENDOR')!=_VENDOR or os.environ.get('ASCEND_CUSTOM_OPP_PATH')!=_VENDOR: raise UnsupportedCase('Select recorded classicpilot_math before runtime initialization')
    data=(_P/'REGISTER_DEPENDENCIES.json').read_bytes()
    if hashlib.sha256(data).hexdigest()!='fe03201d1edd0493edbcae8ba16cb348d9f0e9c8934522a2ac643455a2788ad4': raise RuntimeError('Dependency manifest changed')
    for relative,h in json.loads(data).items():
        if hashlib.sha256((Path('/root/dyq/workspace')/relative).read_bytes()).hexdigest()!=h: raise RuntimeError('Dependency changed: '+relative)
    return _load().make(case)
