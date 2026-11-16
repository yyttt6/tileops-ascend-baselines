"""Unregistered, CPU-lazy exact-case provider proposal; fresh guarded process only.

Device proof belongs to run_isnan.py + the identical element_adapter.py.
This packaging layer has CPU routing checks, not an additional device execution.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types

HERE = Path(__file__).resolve().parent
PARAMS = {'input': [[4096, 4096], 'float32'], 'output': [[4096, 4096], 'bool']}


def matches(op, case):
    return op == 'IsnanFwdOp' and case.id == '16M-float32'


def validate_case(op, case):
    if not matches(op, case):
        raise ValueError('Outside exact Isnan scope')
    if json.loads(json.dumps(case.params)) != PARAMS:
        raise ValueError('Manifest parameters changed')


def inspect_pins():
    pins = json.loads((HERE / 'provider_pins.json').read_text())
    for name, expected in pins.items():
        path = Path(name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Isnan deployment changed: ' + name)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _prepare():
    meta = json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
    if meta.get('phase') != 'running' or meta.get('child', {}).get('pid') != os.getpgrp():
        raise RuntimeError('Entire provider process requires npu-guard')
    if any(n.split('.')[0] in ('tilelang', 'tvm', 'cython_wrapper') for n in sys.modules):
        raise RuntimeError('Use a fresh process before TileLang/TVM imports')
    inspect_pins()
    P = HERE
    W = Path('/opt/tileops-all-pairs/workspace')
    V = W / 'ascend_baselines/tilelang-ascend'
    C = W / '.cache/all-pairs/remote_tla_compiler'
    R = P / 'runtime'
    sdk = W / '.cache/all-pairs/remote_native_deepgemm_deepep/sdk-920b2'
    for name in ('ASCEND_NPU_ARCH', 'TVM_HOME', 'TVM_IMPORT_PYTHON_PATH'):
        os.environ.pop(name, None)
    os.environ.update(TL_PLATFORM='A5', TL_ROOT=str(V), TL_TEMPLATE_PATH=str(V/'src'),
                      TILELANG_LIBRARY_PATH=str(C/'handoff/lib'), TVM_LIBRARY_PATH=str(C/'handoff/lib'),
                      SETUPTOOLS_USE_DISTUTILS='local', PYTHONDONTWRITEBYTECODE='1',
                      ASCEND_HOME_PATH=str(sdk), ASCEND_TOOLKIT_HOME=str(sdk),
                      BISHENG_HOME=str(sdk/'tools/bisheng_compiler'))
    root = P / 'provider_runs'
    root.mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='isnan-', dir=root))
    for name in ('TILELANG_CACHE_DIR', 'TMPDIR', 'XDG_CACHE_HOME', 'TLA_CYTHON_CACHE_DIR',
                 'ASCEND_WORK_PATH', 'ASCEND_CACHE_PATH', 'ASCEND_PROCESS_LOG_PATH',
                 'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR'):
        directory = out / name
        directory.mkdir()
        os.environ[name] = str(directory)
    tempfile.tempdir = os.environ['TMPDIR']
    sys.dont_write_bytecode = True
    os.environ['PATH'] = str(sdk/'tools/bisheng_compiler/bin') + ':' + str(C/'handoff/bin') + ':' + os.environ['PATH']
    sys.path[:0] = [str(R), str(V/'3rdparty/tvm/python'), str(C/'python'),
                    str(W/'.cache/all-pairs/remote_native_tla/python-deps'),
                    str(P/'frozen'), str(W/'TileOPs/src'), str(W)]
    import setuptools
    name = '_baseline700_isnan_private'
    package = types.ModuleType(name)
    package.__path__ = [str(P/'private_tla')]
    sys.modules[name] = package
    deployment = _load(name+'.deployment', P/'private_tla/deployment.py')
    native = _load(name+'.native', P/'private_tla/native.py')
    deployment.install_cache_hook(R, Path(os.environ['TLA_CYTHON_CACHE_DIR']))
    import tilelang
    if Path(tilelang.__file__).resolve() != (R/'tilelang/__init__.py').resolve():
        raise RuntimeError('Wrong TileLang runtime')
    native.VENDOR = V
    native.runtime = lambda: tilelang
    native.CACHE = out
    adapter = _load(name+'.adapter', P/'element_adapter.py')
    return native, tilelang, adapter, out


def factory(op, case):
    validate_case(op, case)
    native, tilelang, adapter, out = _prepare()
    return adapter.factory(case, op, native, tilelang, out)


def install(providers):
    prior=providers.get("IsnanFwdOp")
    def route(case):
        if matches("IsnanFwdOp",case): return factory("IsnanFwdOp",case)
        if prior is not None: return prior(case)
        raise ValueError("No registered exact IsnanFwdOp route")
    providers["IsnanFwdOp"]=route
