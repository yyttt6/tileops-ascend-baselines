"""Exact original aclStft route; native system AiCPU dependency disclosed."""
from pathlib import Path
import hashlib, json, os, types
_KEY = ('FFTC2CFwdOp', 'latency-4k-complex64')
_P = Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/fft_window_first1')
_VENDOR = '/root/dyq/workspace/.cache/ascend-classic/native-pilot-20261020/ops-math/install/vendors/classicpilot_math'
def _load():
    data=(_P/'adapter.py').read_bytes()
    if hashlib.sha256(data).hexdigest()!='191e3f5030015a00958ea5e340530be18c586389a27a79d4b5aa2221849e1cf0': raise RuntimeError('Frozen adapter changed')
    m=types.ModuleType('fft_window_first1_frozen');m.__file__=str(_P/'adapter.py');exec(compile(data,m.__file__,'exec'),m.__dict__);return m
def make(case):
    from benchmarks.ascend_baselines import UnsupportedCase
    if (case._call.signature.name,case.id)!=_KEY: raise UnsupportedCase('No exact receipt')
    if os.environ.get('ASCEND_CLASSIC_MATH_VENDOR')!=_VENDOR or os.environ.get('ASCEND_CUSTOM_OPP_PATH')!=_VENDOR: raise UnsupportedCase('Select recorded classicpilot_math before runtime initialization')
    data=(_P/'REGISTER_DEPENDENCIES.json').read_bytes()
    if hashlib.sha256(data).hexdigest()!='475e012541e3c0e581a4471640d6de50bce0fec45a0332999dc46269441b2b6e': raise RuntimeError('Dependency manifest changed')
    for relative,h in json.loads(data).items():
        if hashlib.sha256((Path('/root/dyq/workspace')/relative).read_bytes()).hexdigest()!=h: raise RuntimeError('Dependency changed: '+relative)
    return _load().make(case)
