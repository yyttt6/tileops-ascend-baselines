"""Use the other complete original mean-producing layer-norm API, keeping all wrapping."""
import ast,copy,pathlib
OLD='layer_norm_fwd_npu_simd';NEW='layer_norm_fwd_npu'
def clone(module):
 node=next(n for n in ast.parse(pathlib.Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='lerp_scalar');before=ast.dump(node,include_attributes=False)
 class Change(ast.NodeTransformer):
  def visit_Constant(self,n):
   if isinstance(n.value,str) and OLD in n.value:n.value=n.value.replace(OLD,NEW)
   return n
 class Restore(ast.NodeTransformer):
  def visit_Constant(self,n):
   if isinstance(n.value,str) and NEW in n.value:n.value=n.value.replace(NEW,OLD)
   return n
 node=Change().visit(node);assert ast.dump(Restore().visit(copy.deepcopy(node)),include_attributes=False)==before
 env=dict(vars(module));exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),module.__file__,'exec'),env);return env['lerp_scalar']
def build(case,library,original_factory):
 from benchmarks.ascend_all_pairs.sgl import scalar
 from benchmarks.ascend_baselines import source_hashes,UnsupportedCase
 if library!='sgl-kernel-npu' or case._call.signature.name!='LerpScalarFwdOp' or case.params.get('weight')!=0.5:raise UnsupportedCase('Only frozen midpoint full-API alternative')
 original=original_factory(case,library)();expected=scalar.lerp_scalar(case)
 if original.api!=expected.api or original.source_sha256!=expected.source_sha256:raise UnsupportedCase('Originalregistry exact override differs from reviewed midpoint')
 candidate=clone(scalar)(case);candidate.source_sha256.update(source_hashes(pathlib.Path(__file__)));return candidate
