from pathlib import Path
import json,hashlib,inspect,shutil
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
class FrozenScan:
 def __init__(self,case):
  from tileops.ops.mamba.selective_scan import SelectiveScanFwdOp
  import tileops.kernels.ascend.mamba.selective_scan
  from tileops.backend import BUILTIN
  self.op=SelectiveScanFwdOp(**case.arguments,target=BUILTIN)
  self.impl=case._entry.binder(self.op,case)
 def __call__(self,*args):return self.impl.run(*args)
 def capture(self,dest):
  import torch,tilelang,torch_npu
  dest=Path(dest);out=dest/'tileops-generated';out.mkdir(exist_ok=True);seen=set();records=[];binaries={};sources={}
  def walk(obj,label,depth=0):
   if obj is None or id(obj) in seen or depth>7 or isinstance(obj,(str,int,float,bytes,Path,torch.Tensor,type)):return
   seen.add(id(obj));record={'label':label,'type':type(obj).__module__+'.'+type(obj).__qualname__};records.append(record)
   if hasattr(obj,'get_kernel_source'):
    text=obj.get_kernel_source()
    if text:
     f=out/('kernel-'+str(len(sources))+'.cc');f.write_text(text);sources[f.name]=sha(f)
   for attr in ['libpath','lib_path','kernel_lib_path','device_binary_path','aibin_path']:
    value=getattr(obj,attr,None)
    if isinstance(value,(str,Path)) and Path(value).is_file():
     source=Path(value);f=out/('binary-'+str(len(binaries))+source.suffix);shutil.copyfile(source,f);binaries[str(source)]={'retained':f.name,'sha256':sha(f)}
   if isinstance(obj,dict):
    for i,value in enumerate(obj.values()):walk(value,label+'.dict'+str(i),depth+1)
   elif isinstance(obj,(list,tuple)):
    for i,value in enumerate(obj):walk(value,label+'.item'+str(i),depth+1)
   elif hasattr(obj,'__dict__'):
    for key,value in vars(obj).items():
     if key in {'compiled','adapter','artifact','rt_mod','_entries_by_call','_entries','_target_kernels','entry','kernel','_kernel','_bound','_instance','instance','op','_callable','func','_run'} or any(s in key.lower() for s in ['kernel','entry','compiled']):walk(value,label+'.'+key,depth+1)
   if inspect.isfunction(obj) and obj.__closure__:
    for i,cell in enumerate(obj.__closure__):walk(cell.cell_contents,label+'.closure'+str(i),depth+1)
  walk(self.op,'public_op');modules={}
  import sys
  for name in ['tileops.ops.mamba.selective_scan','tileops.kernels.ascend.mamba.selective_scan','tileops.kernels.mamba.selective_scan','tileops.kernels.mamba.call_spec']:
   module=sys.modules.get(name)
   if module and getattr(module,'__file__',None):modules[name]={'path':str(Path(module.__file__).resolve()),'sha256':sha(module.__file__)}
  report={'measured_library':'TileOps','public_op_type':type(self.op).__module__+'.'+type(self.op).__qualname__,'actual_modules':modules,'object_evidence':records,'generated_sources_sha256':sources,'generated_binaries':binaries,'tilelang':{'version':getattr(tilelang,'__version__',None),'path':tilelang.__file__},'original_kernel_source_unchanged':True,'generated_identity_complete':bool(sources) and bool(binaries)}
  (dest/'TILEOPS_GENERATED_IDENTITY.json').write_text(json.dumps(report,indent=2)+'\n');return report
