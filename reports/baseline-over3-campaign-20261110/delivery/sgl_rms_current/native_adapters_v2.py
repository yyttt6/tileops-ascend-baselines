import pathlib,sys,types,importlib,hashlib,importlib.util
ROOT=pathlib.Path(__file__).parent
def factory(op,params,fixed,args,*,case):
 assert op=='RMSNormFwdOp' and not fixed
 sys.path.append('/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_triton_runtime/overlay/site-packages')
 import provider as p
 package=types.ModuleType('sgl_kernel_npu');package.__path__=[str(ROOT/'sgl-source/sgl_kernel_npu')];sys.modules['sgl_kernel_npu']=package
 p.LIBRARIES=ROOT
 def local_function(library,pythonpath,module,function,source):
  assert library=='sgl-kernel-npu'
  return getattr(importlib.import_module(module),function)
 p.local_function=local_function
 p.source_hashes=lambda *paths:{}
 p.Candidate=lambda impl,*a,**kw:impl.run
 fn=p.sgl_rms(types.SimpleNamespace(inputs=args,params=case.params))
 files=[ROOT/'provider.py',ROOT/'native_adapters_v2.py',*(ROOT/'sgl-source').rglob('*.py')]
 return fn,{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
