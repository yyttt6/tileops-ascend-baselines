"""Same locked vendor complete RMS API; all original packing/weight/epsilon costs retained."""
import ast,copy,pathlib,hashlib,dataclasses,math
OLD='batch_invariant/rmsnorm.py';NEW='fla/layernorm_guard.py'
def clone(module):
 node=next(n for n in ast.parse(pathlib.Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='normalization');before=ast.dump(node,include_attributes=False)
 old=ast.parse('native.rms_norm(packed, weight, eps)',mode='eval').body
 new=ast.parse('native._layer_norm_fwd(packed, weight, None, eps, is_rms_norm=True)[0]',mode='eval').body;count=[0,0]
 class Change(ast.NodeTransformer):
  def visit_Constant(self,n):
   if n.value==OLD:count[0]+=1;return ast.copy_location(ast.Constant(NEW),n)
   return n
  def visit_Call(self,n):
   if ast.dump(n,include_attributes=False)==ast.dump(old,include_attributes=False):count[1]+=1;return copy.deepcopy(new)
   return self.generic_visit(n)
 class Restore(ast.NodeTransformer):
  def visit_Constant(self,n):
   if n.value==NEW:return ast.copy_location(ast.Constant(OLD),n)
   return n
  def visit_Subscript(self,n):
   if ast.dump(n,include_attributes=False)==ast.dump(new,include_attributes=False):return copy.deepcopy(old)
   return self.generic_visit(n)
 node=Change().visit(node);assert count==[1,1];assert ast.dump(Restore().visit(copy.deepcopy(node)),include_attributes=False)==before
 env=dict(vars(module));exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),module.__file__,'exec'),env);return env['normalization']
def build(case,library):
 if library!='vllm-ascend' or case._call.signature.name!='RMSNormFwdOp':raise RuntimeError('Only RMSNorm complete API candidate')
 p=case.params;hidden=math.prod(p.get('normalized_shape',[p['x'][0][-1]]));size={'float16':2,'bfloat16':2,'float32':4}[p['x'][1]]
 from benchmarks.ascend_baselines import UnsupportedCase
 if hidden*size>65536:raise UnsupportedCase('Original alternate API64KiBfeature limit; do not split or bypass')
 from benchmarks.ascend_all_pairs.vllm import basic
 candidate=clone(basic)(case);f=pathlib.Path(__file__);candidate.source_sha256[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
 return dataclasses.replace(candidate,api='vllm-ascend::fla/layernorm_guard.py:_layer_norm_fwd(is_rms_norm=True)[0]',adaptation='Same complete vendor RMS alternative API; unchanged epsilon/weight/dtype,percall packing and all extra outputs computed; no kernel or compiler tuning.')
