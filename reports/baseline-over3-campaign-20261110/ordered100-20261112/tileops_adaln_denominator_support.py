"""Unchanged frozen public TileOps AdaLN; new measurement role, never a library fallback."""
from pathlib import Path
import json,hashlib,inspect,shutil
P=Path(__file__).resolve().parent;T=P/'workspace/TileOPs';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
class FrozenAdaLN:
 def __init__(self,case):
  from tileops.ops.norm.ada_layer_norm import AdaLayerNormFwdOp
  self.op=AdaLayerNormFwdOp(**case.arguments,target='ascend');self.impl=case._entry.binder(self.op,case);self.case=case
  proof=json.loads((P/'ADALN_TILEOPS_DENOMINATOR_SOURCE_AUDIT.json').read_text());assert all(sha(T/f)==r['sha256'] for f,r in proof['selected_transitive_Ascend_sources'].items())
  self.sources={str(T/f):r['sha256'] for f,r in proof['selected_transitive_Ascend_sources'].items()};self.sources[str(P/'ADALN_TILEOPS_DENOMINATOR_SOURCE_AUDIT.json')]=sha(P/'ADALN_TILEOPS_DENOMINATOR_SOURCE_AUDIT.json');self.sources[str(Path(__file__))]=sha(Path(__file__))
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
     if key in {'compiled','adapter','artifact','rt_mod','_entries_by_call','_entries','_target_kernels','entry','kernel','_kernel','_bound','_instance','instance','op','_callable','func'} or any(s in key.lower() for s in ['kernel','entry','compiled']):walk(value,label+'.'+key,depth+1)
   if inspect.isfunction(obj) and obj.__closure__:
    for i,cell in enumerate(obj.__closure__):walk(cell.cell_contents,label+'.closure'+str(i),depth+1)
  walk(self.op,'public_op');modules={}
  import sys
  for name in ['tileops.ops.norm.ada_layer_norm','tileops.kernels.ascend.norm.ada_layer_norm','tileops.kernels.ascend.norm._ada_dispatch','tileops.kernels.ascend.norm._ada_resident','tileops.kernels.ascend.norm._ada_fp32_simd']:
   module=sys.modules.get(name)
   if module and getattr(module,'__file__',None):modules[name]={'path':str(Path(module.__file__).resolve()),'sha256':sha(module.__file__)}
  report={'measured_library':'TileOps','public_op_type':type(self.op).__module__+'.'+type(self.op).__qualname__,'actual_modules':modules,'object_evidence':records,'generated_sources_sha256':sources,'generated_binaries':binaries,'tilelang':{'version':getattr(tilelang,'__version__',None),'path':tilelang.__file__},'original_kernel_source_unchanged':True,'generated_identity_complete':bool(sources) and bool(binaries)}
  (dest/'TILEOPS_GENERATED_IDENTITY.json').write_text(json.dumps(report,indent=2)+'\n');return report
 def cleanup(self):pass

def candidate_factory(case,library):
 def build():
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  assert case._call.signature.name=='AdaLayerNormFwdOp';api=FrozenAdaLN(case)
  return Candidate(Implementation(api),'TileOps public AdaLayerNormFwdOp, unchanged frozen Ascend kernel','Denominator only; original manifest binder, full public Op/pack/checker/precision; no optimization/autotune/kernel edits',api.sources,cleanup=api.cleanup)
 return build
