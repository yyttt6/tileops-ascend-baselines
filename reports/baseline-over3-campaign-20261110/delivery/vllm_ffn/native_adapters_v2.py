"""CPU-staged vLLM arithmetic layout candidate; not device validated."""
import pathlib,sys,types,importlib,hashlib
ROOT=pathlib.Path('/opt/tileops-all-pairs/workspace')
def factory(op,params,fixed,args):
 import torch
 sys.path.append(str(ROOT/'.cache/all-pairs/remote_triton_runtime/overlay/site-packages'))
 folder=ROOT/'TileOPs/benchmarks/ascend_all_pairs/vllm';pkg=types.ModuleType('over3_vllm');pkg.__path__=[str(folder)];sys.modules[pkg.__name__]=pkg
 rt=importlib.import_module('over3_vllm.runtime');rt.CACHE=pathlib.Path(__file__).resolve().parent/'jit_binding_cache'
 native,sources=rt.module(rt.TRITON_ROOT+'muls_add.py',hardware=True)
 shape=tuple(params['output'][0]);assert op in ('AddFwdOp','SubFwdOp','NegFwdOp')
 block=max(shape[-1]//2,1024)
 if fixed:assert block>0 and block&(block-1)==0, 'Natural shape gives unsupported non-power-of-two native block'
 alpha=params.get('alpha',1);scale=-alpha if op=='SubFwdOp' else alpha
 def run(x,y=None):
  xp=x.expand(shape).contiguous()
  if not fixed:xp=xp.reshape(-1,1)
  if op=='NegFwdOp':return native.muls_add_triton(xp,torch.zeros_like(xp),-1.0).reshape(shape)
  yp=y.expand(shape).contiguous()
  if not fixed:yp=yp.reshape(-1,1)
  return native.muls_add_triton(yp,xp,scale).reshape(shape)
 proof={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [*sources,*folder.glob('*.py')]}
 return run,proof
