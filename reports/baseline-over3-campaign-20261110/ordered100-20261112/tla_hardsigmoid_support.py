"""Frozen Hardsigmoid cases through three complete original scalar/clamp APIs."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'workspace/isolated_wrappers'))

def scalar_factory(module,name,scalar):
 assert name in ('adds','vec_muls')
 tree=ast.parse(Path(module.__file__).read_text())
 node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='elementwise')
 before=ast.dump(node,include_attributes=False);count=0
 for n in ast.walk(node):
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='extra' for t in n.targets):
   try:value=ast.literal_eval(n.value)
   except (ValueError,TypeError):continue
   if value==[-1]:n.value=ast.parse(repr([scalar]),mode='eval').body;count+=1
 assert count==1
 restored=copy.deepcopy(node);count=0
 for n in ast.walk(restored):
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='extra' for t in n.targets):
   try:value=ast.literal_eval(n.value)
   except (ValueError,TypeError):continue
   if value==[scalar]:n.value=ast.parse('[-1]',mode='eval').body;count+=1
 assert count==1 and ast.dump(restored,include_attributes=False)==before
 ns=dict(vars(module));ns['DIRECT']=dict(module.DIRECT);ns['DIRECT']['NegFwdOp']=(name,1)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),module.__file__,'exec'),ns)
 return ns['elementwise']

def candidate_factory(case,library):
 def build():
  assert library=='tilelang-ascend' and case._call.signature.name=='HardsigmoidFwdOp'
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  from benchmarks.ascend_all_pairs.tla import adapters,native
  import tla_cache_artifact_identity as identity
  identity.install();native.prepare_deployment();x,=case.inputs
  assert x.dtype in (torch.float16,torch.bfloat16)
  shape=tuple(case.params['output'][0]);assert shape==tuple(x.shape)
  desc=SimpleNamespace(params={'input':(shape,'float32'),'output':(shape,'float32')},inputs=(x,))
  clipdesc=SimpleNamespace(params={**desc.params,'min':0.,'max':6.},inputs=(x,))
  plus=scalar_factory(adapters,'adds',3.)('NegFwdOp',desc)
  clip=adapters.elementwise('ClampScalarFwdOp',clipdesc)
  scale=scalar_factory(adapters,'vec_muls',1./6.)('NegFwdOp',desc)
  children=(plus,clip,scale)
  assert all(c.implementation.args is None and c.implementation.reset is None for c in children)
  def run(value):
   out=plus.implementation.run(value)
   out=clip.implementation.run(out)
   return scale.implementation.run(out).to(value.dtype)
  def cleanup():
   for c in children:
    if c.cleanup is not None:c.cleanup()
  hashes={}
  for c in children:hashes.update(c.source_sha256)
  for f in (Path(identity.__file__),Path(__file__)):
   hashes[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  return Candidate(Implementation(run),'TLA original complete adds(3) + original complete clamp(0,6) + original complete vec_muls(1/6)','Frozen F16/BF16 Hardsigmoid: all three original FP32 pack/pad/native/crop APIs, both intermediate materializations and final original dtype cast timed. Only original API selectors/scalar parameters change in cloned factory metadata; no kernel/block/passflags/checker edits.',hashes,cleanup=cleanup)
 return build
