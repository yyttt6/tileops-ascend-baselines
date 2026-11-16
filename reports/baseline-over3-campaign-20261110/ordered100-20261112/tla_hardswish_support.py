"""Frozen Hardswish through complete original FMA, clamp and multiply APIs."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'workspace/isolated_wrappers'))

def fma_factory(module):
 tree=ast.parse(Path(module.__file__).read_text())
 node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='elementwise')
 before=ast.dump(node,include_attributes=False);changes=[]
 for n in ast.walk(node):
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='extra' for t in n.targets):
   try:value=ast.literal_eval(n.value)
   except (ValueError,TypeError):continue
   if value==[-1]:changes.append((n.lineno,copy.deepcopy(n.value)));n.value=ast.parse('[]',mode='eval').body
 assert len(changes)==1
 restored=copy.deepcopy(node);count=0
 for n in ast.walk(restored):
  if isinstance(n,ast.Assign) and n.lineno==changes[0][0]:n.value=copy.deepcopy(changes[0][1]);count+=1
 assert count==1 and ast.dump(restored,include_attributes=False)==before
 ns=dict(vars(module));ns['DIRECT']=dict(module.DIRECT);ns['DIRECT']['NegFwdOp']=('vec_mul_add_dst',3)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),module.__file__,'exec'),ns)
 return ns['elementwise']

def candidate_factory(case,library):
 def build():
  assert library=='tilelang-ascend' and case._call.signature.name=='HardswishFwdOp'
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  from benchmarks.ascend_all_pairs.tla import adapters,native
  import tla_cache_artifact_identity as identity
  identity.install();native.prepare_deployment();x,=case.inputs
  assert x.dtype in (torch.float16,torch.bfloat16)
  shape=tuple(case.params['output'][0]);assert shape==tuple(x.shape)
  params={'input':(shape,'float32'),'other':(shape,'float32'),'output':(shape,'float32')}
  desc=SimpleNamespace(params=params,inputs=(x,x,x))
  fma=fma_factory(adapters)('NegFwdOp',desc)
  clip=adapters.elementwise('ClampScalarFwdOp',SimpleNamespace(params={**params,'min':0.,'max':1.},inputs=(x,)))
  mul=adapters.elementwise('MulFwdOp',desc);children=(fma,clip,mul)
  assert all(c.implementation.args is None and c.implementation.reset is None for c in children)
  def run(value):
   # Both full constant tensors are allocated inside every measured invocation.
   scale=torch.full_like(value,1./6.,dtype=torch.float32)
   offset=torch.full_like(value,0.5,dtype=torch.float32)
   gate=fma.implementation.run(value,scale,offset)
   gate=clip.implementation.run(gate)
   return mul.implementation.run(value,gate).to(value.dtype)
  def cleanup():
   for c in children:
    if c.cleanup is not None:c.cleanup()
  hashes={}
  for c in children:hashes.update(c.source_sha256)
  for f in (Path(identity.__file__),Path(__file__)):
   hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  return Candidate(Implementation(run),'TLA original complete vec_mul_add_dst + original complete clamp + original complete vec_mul','Frozen F16/BF16 Hardswish selection x*clamp(x/6+0.5,0,1): complete original FMA reads three full inputs; per-call FP32 constants, all native pack/pad/crop, both FP32 intermediate materializations and final output dtype cast timed. Select original ternary API in cloned factory metadata only; no kernel/block/passflags/checker changes. Floating behavior requires exact original case verification; not generic or extreme-domain acceptance.',hashes,cleanup=cleanup)
 return build
