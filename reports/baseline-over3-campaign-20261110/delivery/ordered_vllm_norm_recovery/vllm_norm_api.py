"""One exact power-of-two GroupNorm case using another complete locked vendor API."""
import ast,copy,pathlib
OLD='layernorm_gated.py';NEW='fla/layernorm_guard.py'
def clone(module):
 node=next(n for n in ast.parse(pathlib.Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='group_norm');before=ast.dump(node,include_attributes=False)
 class Change(ast.NodeTransformer):
  def visit_Constant(self,n):
   if n.value==OLD:n.value=NEW
   return n
  def visit_Attribute(self,n):
   self.generic_visit(n)
   if n.attr=='layer_norm_fwd_npu':n.attr='_layer_norm_fwd'
   return n
 class Restore(ast.NodeTransformer):
  def visit_Constant(self,n):
   if n.value==NEW:n.value=OLD
   return n
  def visit_Attribute(self,n):
   self.generic_visit(n)
   if n.attr=='_layer_norm_fwd':n.attr='layer_norm_fwd_npu'
   return n
 node=Change().visit(node);assert ast.dump(Restore().visit(copy.deepcopy(node)),include_attributes=False)==before
 env=dict(vars(module));exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),module.__file__,'exec'),env);return env['group_norm']
def build(case,library,original_factory):
 from benchmarks.ascend_all_pairs.vllm import basic
 from benchmarks.ascend_baselines import source_hashes,UnsupportedCase
 if library!='vllm-ascend' or case._call.signature.name!='GroupNormFwdOp' or case.id!='image-affine-bfloat16':raise UnsupportedCase('Only exact original compile-failed power-of-two GroupNorm representative')
 original=original_factory(case,library)();expected=basic.group_norm(case)
 if original.api!=expected.api or original.source_sha256!=expected.source_sha256:raise UnsupportedCase('Registry differs from reviewed original GroupNorm route')
 candidate=clone(basic)(case);candidate.source_sha256.update(source_hashes(pathlib.Path(__file__)));return candidate
