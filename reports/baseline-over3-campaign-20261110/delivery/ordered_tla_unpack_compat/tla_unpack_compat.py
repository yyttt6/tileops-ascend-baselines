"""Keep native compare bits; construct the original eight uint8 shift values via supported arange."""
import ast,copy,pathlib,hashlib

def clone(module):
 node=next(n for n in ast.parse(pathlib.Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='elementwise');before=ast.dump(node,include_attributes=False)
 old=ast.parse('torch.arange(8, dtype=torch.uint8, device=result.device)',mode='eval').body
 new=ast.parse('torch.arange(8, dtype=torch.int32, device=result.device).to(torch.uint8)',mode='eval').body
 old_shift=ast.parse("result.reshape(-1, 1) >> shifts",mode="eval").body
 new_shift=ast.parse("result.reshape(-1, 1).expand(-1, 8).contiguous() >> shifts.expand(result.numel(), 8).contiguous()",mode="eval").body
 count=[0]
 class Change(ast.NodeTransformer):
  def visit_BinOp(self,n):
   if ast.dump(n,include_attributes=False)==ast.dump(old_shift,include_attributes=False):count[0]+=1;return copy.deepcopy(new_shift)
   return self.generic_visit(n)
  def visit_Call(self,n):
   if ast.dump(n,include_attributes=False)==ast.dump(old,include_attributes=False):count[0]+=1;return copy.deepcopy(new)
   return self.generic_visit(n)
 class Restore(ast.NodeTransformer):
  def visit_BinOp(self,n):
   if ast.dump(n,include_attributes=False)==ast.dump(new_shift,include_attributes=False):return copy.deepcopy(old_shift)
   return self.generic_visit(n)
  def visit_Call(self,n):
   if ast.dump(n,include_attributes=False)==ast.dump(new,include_attributes=False):return copy.deepcopy(old)
   return self.generic_visit(n)
 node=Change().visit(node);assert count==[2];assert ast.dump(Restore().visit(copy.deepcopy(node)),include_attributes=False)==before
 env=dict(vars(module));exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),module.__file__,'exec'),env);return env['elementwise']

def build(case,library):
 if library!='tilelang-ascend':raise RuntimeError('TLA unpack compatibility only')
 from benchmarks.ascend_all_pairs.tla import adapters,native
 op=case._call.signature.name
 if op not in adapters.COMPARISON and op!='IsnanFwdOp':raise RuntimeError('Only native comparison packed-byte unpacking')
 native.prepare_deployment();candidate=clone(adapters)(op,case)
 candidate.source_sha256[str(pathlib.Path(__file__))]=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
 return candidate
