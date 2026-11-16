"""Exact original aclStft route; native system AiCPU dependency disclosed."""
from pathlib import Path
import hashlib, json, os, types
_KEYS = frozenset([('FFTC2CFwdOp', 'fnet-512-complex64'), ('FFTC2CFwdOp', 'longconv-16k-complex64'), ('FFTC2CFwdOp', 'longconv-64k-complex64'), ('FFTC2CFwdOp', 'shortconv-64-complex64'), ('FFTC2CFwdOp', 'spectrogram-1k-b300-complex64'), ('FFTC2CFwdOp', 'spectrogram-1k-complex64'), ('FFTC2CFwdOp', 'throughput-4k-complex64')])
_P = Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/fft_remaining_c64_7')
_VENDOR = '/root/dyq/workspace/.cache/ascend-classic/native-pilot-20261020/ops-math/install/vendors/classicpilot_math'
def _load():
    data=(_P/'adapter.py').read_bytes()
    if hashlib.sha256(data).hexdigest()!='351f7992b7ea7de56cb5ec906dde29015dd61b9628bb7c5d1bd0442d1369186e': raise RuntimeError('Frozen adapter changed')
    m=types.ModuleType('fft_remaining_c64_7_frozen');m.__file__=str(_P/'adapter.py');exec(compile(data,m.__file__,'exec'),m.__dict__);return m
def make(case):
    from benchmarks.ascend_baselines import UnsupportedCase
    if (case._call.signature.name,case.id)not in _KEYS: raise UnsupportedCase('No exact receipt')
    if os.environ.get('ASCEND_CLASSIC_MATH_VENDOR')!=_VENDOR or os.environ.get('ASCEND_CUSTOM_OPP_PATH')!=_VENDOR: raise UnsupportedCase('Select recorded classicpilot_math before runtime initialization')
    data=(_P/'REGISTER_DEPENDENCIES.json').read_bytes()
    if hashlib.sha256(data).hexdigest()!='b1515b5dfc648c7db6d44cc9edbe1f38b878d464831ce700d46ed2c5c1e6e363': raise RuntimeError('Dependency manifest changed')
    for relative,h in json.loads(data).items():
        if hashlib.sha256((Path('/root/dyq/workspace')/relative).read_bytes()).hexdigest()!=h: raise RuntimeError('Dependency changed: '+relative)
    return _load().make(case)
