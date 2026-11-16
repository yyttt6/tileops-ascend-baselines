"""Observe original TileLang JITKernel calls without changing launch arguments."""
import hashlib
import weakref
from pathlib import Path

_CURRENT={}
_DESCRIPTIONS=weakref.WeakKeyDictionary()

def install():
    from tilelang.jit.kernel import JITKernel
    original=JITKernel.__call__
    def traced(kernel,*args,**kwargs):
        identity=kernel
        if identity not in _DESCRIPTIONS:
            source=kernel.get_kernel_source()
            raw=source.encode() if isinstance(source,str) else bytes(source)
            lib=getattr(kernel.adapter,'libpath',None)
            _DESCRIPTIONS[identity]={'kernel_sha256':hashlib.sha256(raw).hexdigest(),
                'target':str(kernel.target),'execution_backend':kernel.execution_backend,
                'adapter_class':type(kernel.adapter).__module__+'.'+type(kernel.adapter).__qualname__,
                'library':str(lib) if lib else None,
                'library_sha256':hashlib.sha256(Path(lib).read_bytes()).hexdigest() if lib and Path(lib).is_file() else None,
                'has_ascend_entry':any(v in source for v in ('__global__','__aicore__','__simt_vf__','__vector__'))}
        description=_DESCRIPTIONS[identity]
        key=description['kernel_sha256']
        if key not in _CURRENT:_CURRENT[key]=description|{'successful_calls':0}
        result=original(kernel,*args,**kwargs)
        _CURRENT[key]['successful_calls']+=1
        return result
    JITKernel.__call__=traced

def begin():
    _CURRENT.clear()

def result():
    return {'schema':'tilelang-native-call-attribution/v1','observation':'JITKernel.__call__ delegates unmodified to original; successful host launches, checker synchronizes device before receipt.',
            'total_successful_calls':sum(x['successful_calls'] for x in _CURRENT.values()),'kernels':list(_CURRENT.values())}
