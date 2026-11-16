"""Bind returned TileLang calls to actual runtime modules; no driver claim."""
import hashlib
import json
from pathlib import Path

_records = {}
_loaded = {}
_current = {}


def _module_binding(adapter):
    module = getattr(adapter, 'executable', None)
    observed = _loaded.get(id(module))
    return observed[1] if observed is not None and observed[0] is module else None


def install():
    from tilelang.jit.kernel import JITKernel
    from tilelang import tvm
    original_load = tvm.runtime.load_module
    def load(path, *args, **kwargs):
        module = original_load(path, *args, **kwargs)
        file = Path(path)
        # Retain the object itself, so an unrelated module cannot reuse its id.
        _loaded[id(module)] = (module, {'path': str(file),
                                      'sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
                                      'returned_runtime_module_id': id(module)})
        return module
    tvm.runtime.load_module = load
    original = JITKernel.__call__
    def call(kernel, *args, **kwargs):
        key = id(kernel)
        if key not in _records:
            source = kernel.get_kernel_source()
            adapter = kernel.adapter
            libpath = getattr(adapter, 'libpath', None)
            record = {'generated_source': source, 'kernel_sha256': hashlib.sha256(source.encode()).hexdigest(),
                      'execution_backend': kernel.execution_backend, 'target': str(kernel.target),
                      'adapter_class': type(adapter).__module__ + '.' + type(adapter).__qualname__,
                      'adapter_libpath': str(libpath) if libpath else None,
                      'adapter_library_sha256': hashlib.sha256(Path(libpath).read_bytes()).hexdigest()
                        if libpath and Path(libpath).is_file() else None,
                      'runtime_module_load': _module_binding(adapter),
                      'direct_driver_load_launch_observed': False}
            # Keep the actual object alive to prevent id reuse across cases.
            _records[key] = (kernel, record)
        if key not in _current:
            _current[key] = dict(_records[key][1], successful_calls=0)
        result = original(kernel, *args, **kwargs)
        if _current[key]['runtime_module_load'] is None:
            _current[key]['runtime_module_load'] = _module_binding(kernel.adapter)
        _current[key]['successful_calls'] += 1
        count = _current[key]['successful_calls']
        if count == 1 or count % 100000 == 0:
            print(json.dumps({'native_call_observation': True,
                              'kernel_sha256': _current[key]['kernel_sha256'],
                              'successful_calls_for_this_kernel': count,
                              'runtime_module_load': _current[key]['runtime_module_load'],
                              'direct_driver_load_launch_observed': False}), flush=True)
        return result
    JITKernel.__call__ = call


def begin():
    _current.clear()


def result():
    return {'schema': 'tilelang-continuation-runtime-observation/v1',
            'layer': 'Returned original JITKernel calls and, when available, actual TVM runtime.load_module result identity; device synchronization recorded separately.',
            'kernels': list(_current.values()),
            'successful_calls': sum(r['successful_calls'] for r in _current.values())}
