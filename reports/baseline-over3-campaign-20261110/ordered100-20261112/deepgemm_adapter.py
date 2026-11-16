"""Full original bf16_bmm; no candidate kernel or torch matmul fallback."""
import pathlib,importlib.util,hashlib,os,sys,types
ROOT=pathlib.Path('/opt/tileops-all-pairs/workspace')
def factory(op,params):
 assert op=='BmmFwdOp'
 import torch
 import torch.nn.functional as F
 path=ROOT/'TileOPs/benchmarks/ascend_all_pairs/deepgemm/runtime.py'
 pkg=types.ModuleType('over3_deepgemm');pkg.__path__=[str(path.parent)];pkg.__package__='over3_deepgemm';sys.modules['over3_deepgemm']=pkg
 spec=importlib.util.spec_from_file_location('over3_deepgemm.runtime',path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 os.environ['ALLPAIRS_DEEPGEMM_JIT_CACHE']=str(ROOT/'.cache/all-pairs/over3-deepgemm-20261110')
 native,functions,proof=module.load_native('bf16_bmm');fn=functions['bf16_bmm']
 def run(a,b):
  assert a.dtype==b.dtype==torch.bfloat16
  lhs=a.contiguous();rhs=b.transpose(-1,-2).contiguous();extra=-lhs.shape[-1]%16
  if extra:lhs=F.pad(lhs,(0,extra));rhs=F.pad(rhs,(0,extra))
  out=torch.empty((a.shape[0],a.shape[1],b.shape[2]),device=a.device,dtype=a.dtype)
  fn(lhs,rhs,out);return out
 proof={str(ROOT/k):v for k,v in proof.items()};proof.update({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in path.parent.glob('*.py')})
 return run,proof
