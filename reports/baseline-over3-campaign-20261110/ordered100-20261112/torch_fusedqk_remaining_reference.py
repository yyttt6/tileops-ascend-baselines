"""Measure complete Torch math on device; original CPU oracle is never altered."""
import ast,copy,types,pathlib,importlib
SCOPE={'interleaved-ragged-bfloat16-float32', 'glm-4.5-partial-rope-bfloat16-float32', 'interleaved-ragged-float16-float16', 'qwen3-8b-t8192-bfloat16-bfloat16', 'qwen3-235b-t4096-bfloat16-bfloat16'}
def bind(workload):
 module=importlib.import_module(type(workload).__module__);source=pathlib.Path(module.__file__).read_text();cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='FusedQKNormRoPECall');node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='ref_program');before=ast.dump(node,include_attributes=False);calls=[]
 class Device(ast.NodeTransformer):
  def visit_Call(self,n):
   self.generic_visit(n)
   if isinstance(n.func,ast.Attribute) and n.func.attr=='cpu' and not n.args and not n.keywords:
    calls.append(ast.unparse(n));out=copy.deepcopy(n.func.value);out._restore_original_cpu_call=copy.deepcopy(n);return out
   return n
 class Restore(ast.NodeTransformer):
  def visit(self,n):
   if hasattr(n,'_restore_original_cpu_call'):return copy.deepcopy(n._restore_original_cpu_call)
   return super().visit(n)
 node=Device().visit(node);assert calls==['qkv.cpu()','cos_sin_cache.cpu()','positions.cpu()','weight.cpu()'];assert ast.dump(Restore().visit(copy.deepcopy(node)),include_attributes=False)==before
 env=dict(vars(module));exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),module.__file__,'exec'),env);return types.MethodType(env['ref_program'],workload),{'full_reverse_AST_identical':True,'only_removed_CPU_oracle_placement':calls,'complete_FP32_RMSNorm_suppliedtable_RoPE_clone_V_and_finalcast_retained':True,'original_CPU_oracle_unchanged':True}
def build(case):
 assert case._call.signature.name=='FusedQKNormRoPEFwdOp' and case.id in SCOPE
 return bind(case.workload)
