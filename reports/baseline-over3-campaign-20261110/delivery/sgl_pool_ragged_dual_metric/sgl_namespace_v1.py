"""Load independent original Triton modules without unrelated C++ extension registry."""
import sys,types,importlib.machinery,pathlib
from sgl_layout_v1 import ROOT

def initialize():
 if 'sgl_kernel_npu' in sys.modules:raise RuntimeError('Unexpected existing SGL module; do not replace it')
 pkg=types.ModuleType('sgl_kernel_npu');pkg.__path__=[str(ROOT/'sgl_kernel_npu')]
 pkg.__file__=__file__;pkg.__package__='sgl_kernel_npu'
 pkg.__spec__=importlib.machinery.ModuleSpec('sgl_kernel_npu',loader=None,is_package=True)
 pkg.__spec__.submodule_search_locations=pkg.__path__
 sys.modules['sgl_kernel_npu']=pkg

def bindings():
 out={}
 for name,m in list(sys.modules.items()):
  if name.startswith('sgl_kernel_npu.') and getattr(m,'__file__',None):
   f=pathlib.Path(m.__file__).resolve();assert f.is_relative_to(ROOT.resolve()),(name,str(f));out[name]=str(f)
 return {'package_shim':__file__,'original_triton_modules':out,'cpp_extension_used':False,'reason':'Independent Triton APIs use torch/triton/device properties only; unrelated C++ package initializer is not needed'}
