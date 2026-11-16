"""Same vendor complete linear API, honoring original segmented GEMM metadata/transpose contract."""
import ast,copy,pathlib,hashlib,dataclasses

def clone_grouped(module):
 node=next(n for n in ast.parse(pathlib.Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='grouped_gemm')
 old=ast.dump(node,include_attributes=False)
 a=ast.parse('native.matmul_persistent(aa.contiguous(), bb.contiguous())',mode='eval').body
 b=ast.parse('native.linear_persistent(aa.contiguous(), bb.transpose(-1,-2).contiguous())',mode='eval').body
 class Swap(ast.NodeTransformer):
  count=0
  def __init__(self,reverse=False):self.reverse=reverse
  def visit_Call(self,n):
   src,dst=(b,a) if self.reverse else (a,b)
   if ast.dump(n,include_attributes=False)==ast.dump(src,include_attributes=False):self.count+=1;return copy.deepcopy(dst)
   return self.generic_visit(n)
 swap=Swap();node=swap.visit(node);assert swap.count==1
 assert ast.dump(Swap(True).visit(copy.deepcopy(node)),include_attributes=False)==old
 env=dict(vars(module));exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),module.__file__,'exec'),env)
 return env['grouped_gemm'],{'reverse_grouped_AST_identical':True,'only_change':'complete linear_persistent API and necessary transposed contiguous weight','transpose_flags_and_required_copies_preserved':True,'kernel_flags_schedule_precision_not_edited':True}

def build(case,library):
 assert library=='vllm-ascend' and case._call.signature.name=='GroupedGemmFwdOp'
 from benchmarks.ascend_all_pairs.vllm import basic
 fn,proof=clone_grouped(basic);candidate=fn(case);p=pathlib.Path(__file__)
 candidate.source_sha256[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 return dataclasses.replace(candidate,api='linear_persistent segmented (original metadata read, empty groups and stack/cat retained)',adaptation='Only per-group complete vendor API and necessary weight transpose changed; per-call sizes/offsets.tolist, empty groups, all transpose flags, full K, stack/cat and copies retained. No cached input plan, kernel/compiler/block/precision changes.')
