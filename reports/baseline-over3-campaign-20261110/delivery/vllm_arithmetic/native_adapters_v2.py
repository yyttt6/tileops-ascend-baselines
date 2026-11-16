"""Original vLLM scale*x+y API; natural manifest feature shape vs singleton view."""
import pathlib,sys,types,importlib,hashlib
ROOT=pathlib.Path('/opt/tileops-all-pairs/workspace')
def factory(op,params,fixed,args):
 import torch
 sys.path.append(str(ROOT/'.cache/all-pairs/remote_triton_runtime/overlay/site-packages'))
 folder=ROOT/'TileOPs/benchmarks/ascend_all_pairs/vllm';pkg=types.ModuleType('over3_vllm');pkg.__path__=[str(folder)];sys.modules[pkg.__name__]=pkg
 rt=importlib.import_module('over3_vllm.runtime');rt.CACHE=pathlib.Path(__file__).resolve().parent/'jit_binding_cache'
 native,sources=rt.module(rt.TRITON_ROOT+'muls_add.py',hardware=True)
 shape=tuple(params['output'][0]);assert shape[-1]==4096 and op in ('AddFwdOp','SubFwdOp')
 alpha=params.get('alpha',1);scale=-alpha if op=='SubFwdOp' else alpha
 def run(x,y):
  xp=x.expand(shape).contiguous();yp=y.expand(shape).contiguous()
  if not fixed:xp=xp.reshape(-1,1);yp=yp.reshape(-1,1)
  return native.muls_add_triton(yp,xp,scale).reshape(shape)
 proof={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [*sources,*folder.glob('*.py')]}
 return run,proof
