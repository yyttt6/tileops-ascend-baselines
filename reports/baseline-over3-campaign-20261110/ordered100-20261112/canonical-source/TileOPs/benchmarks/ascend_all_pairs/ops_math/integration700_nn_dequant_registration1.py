"""One reviewed original Mul route; no diagnostic validator replacement."""
import builtins
import hashlib
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path('/root/dyq/workspace')
PACKET = ROOT / 'reports/baseline-integration-700-20261026/nn_dequant_registration1'
KEY = ('ops-math', 'INT8DequantPerChannelFwdOp', 'llama-70b-mlp-down-bfloat16')
PARAMS = {'q': [[8192, 28672], 'int8'], 'scale': [[8192], 'float32'],
          'x': [[8192, 28672], 'bfloat16'], 'out_dtype': 'bfloat16'}
VENDOR = ROOT / '.cache/all-pairs/ops_math/graph-runtime/vendors/allpairs_math_math'
PINS_SHA256 = '7c054c06e141432bfdadf681170caa93d54cfa09c2174664aa9265f8c33ddc7c'
PREFIX = 'benchmarks.ascend_all_pairs.ops_math'


def matches(case, baseline='ops-math'):
    if (baseline, case._call.signature.name, case.id) != KEY:
        return False
    # JSON normalizes tuple/list shape representations, but retains numeric types.
    try:
        return json.dumps(case.params, sort_keys=True, allow_nan=False) == json.dumps(PARAMS, sort_keys=True)
    except (TypeError, ValueError):
        return False


def _checked(path, digest):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise RuntimeError('Frozen dependency changed: ' + str(path))
    return data


def verify_pins():
    pins = json.loads(_checked(PACKET / 'PINS.json', PINS_SHA256))
    # Return the exact verified bytes; never reopen Python source for execution.
    data = {path: _checked(path, digest) for path, digest in pins.items()}
    return pins, data


def _load(pins, data):
    private = {}
    original_import = builtins.__import__
    def private_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == PREFIX + '.specs' or (name == 'specs' and level == 1):
            return private['specs']
        if name == PREFIX + '.graph':
            return private['graph']
        return original_import(name, globals, locals, fromlist, level)
    for name, path in (
        ('specs', ROOT / 'TileOPs/benchmarks/ascend_all_pairs/ops_math/specs.py'),
        ('graph', ROOT / 'TileOPs/benchmarks/ascend_all_pairs/ops_math/graph.py'),
        ('adapter', PACKET / 'adapter.py'),
    ):
        mod = ModuleType('_dequant_verified_' + name)
        mod.__file__ = str(path)
        mod.__package__ = PREFIX
        mod.__dict__['__builtins__'] = dict(vars(builtins), __import__=private_import)
        exec(compile(data[str(path)], str(path), 'exec'), mod.__dict__)
        private[name] = mod
    # Metadata-only replacement, private graph instance: no whole-vendor crawl/cache.
    private['graph'].deployment_hashes = lambda: {
        str(Path(p).relative_to(ROOT)): h for p, h in pins.items() if str(VENDOR) in p
    }
    return private


def _require_runtime():
    state = json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
    if state.get('phase') != 'running' or state.get('child', {}).get('pid') != os.getpgrp():
        raise RuntimeError('Run the entire foreground process inside host npu-guard.')
    if os.environ.get('ASCEND_CUSTOM_OPP_PATH') != str(VENDOR):
        raise RuntimeError('Select only the reviewed allpairs_math_math graph vendor before runtime import.')


def _loaded_vendor_pins(pins, require=False):
    expected = {str(Path(p).resolve()): h for p,h in pins.items() if p.endswith('.so')}
    observed = set()
    names = {Path(p).name for p in expected}
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6: continue
        path = fields[5]
        if Path(path.removesuffix(' (deleted)')).name not in names: continue
        if path.endswith(' (deleted)'): raise RuntimeError('Deleted vendor library mapping')
        resolved = str(Path(path).resolve())
        if resolved not in expected: raise RuntimeError('Unexpected loaded vendor library: ' + path)
        stat = Path(resolved).stat()
        major, minor = (int(v, 16) for v in fields[3].split(':'))
        if (stat.st_ino, os.major(stat.st_dev), os.minor(stat.st_dev)) != (int(fields[4]), major, minor):
            raise RuntimeError('Mapped vendor inode differs from pinned file: ' + path)
        _checked(resolved, expected[resolved]); observed.add(resolved)
    if require and observed != set(expected):
        raise RuntimeError('Reviewed vendor libraries were not all observed in this process')


def make(case, baseline='ops-math'):
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    if not matches(case, baseline):
        raise UnsupportedCase('No reviewed exact dequant key and complete canonical parameters')
    pins, _ = verify_pins()
    # Freeze parameter values: caller mutations cannot alter the lazy construction.
    frozen_case = SimpleNamespace(params=json.loads(json.dumps(PARAMS)),
                                  _call=SimpleNamespace(signature=SimpleNamespace(name=KEY[1])), id=KEY[2])
    child = None
    def run(*args):
        nonlocal child
        _require_runtime()  # Before torch imports, ACL loading or compilation.
        if child is None:
            current_pins, data = verify_pins()
            _loaded_vendor_pins(current_pins)
            child = _load(current_pins, data)['adapter'].make(frozen_case)
        result = child.implementation.run(*args)
        _loaded_vendor_pins(pins, require=True)
        return result
    def cleanup():
        if child is not None:
            _require_runtime()
            if child.cleanup is not None: child.cleanup()
    # Implementation's public protocol, without importing benchmarks.api/torch.
    impl = SimpleNamespace(run=run, args=None, reset=None, noncomparable_reason=None)
    return Candidate(impl, 'ops-math::Mul (symmetric INT8 dequantization)',
        'Original complete Mul graph; lossless INT8-to-FP32, scale[:,None], original BF16 output cast; lazy guarded construction.',
        {str(Path(p).relative_to(ROOT)): h for p,h in pins.items()},
        limitations=('Original full checker may hit IsFinite infrastructure failure in max_abs metrics; no diagnostic monkeypatch is installed.',),
        cleanup=cleanup)
