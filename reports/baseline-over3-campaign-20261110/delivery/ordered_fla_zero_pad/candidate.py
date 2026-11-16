"""Only remove zero F.pad copies from the original FLA pointwise wrapper."""
import ast,copy,hashlib,pathlib
P=pathlib.Path(__file__).resolve().parent
def pad_if_needed(value,padding,*args,**kwargs):
 import torch.nn.functional as F
 return value if not any(padding) else F.pad(value,padding,*args,**kwargs)
def clone(module):
 source=pathlib.Path(module.__file__).read_text();tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pointwise');original=copy.deepcopy(node)
 class Rewrite(ast.NodeTransformer):
  def visit_Call(self,n):
   n=self.generic_visit(n)
   if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='F' and n.func.attr=='pad':n.func=ast.Name(id='_pad_if_needed',ctx=ast.Load())
   return n
 class Restore(ast.NodeTransformer):
  def visit_Call(self,n):
   n=self.generic_visit(n)
   if isinstance(n.func,ast.Name) and n.func.id=='_pad_if_needed':n.func=ast.Attribute(value=ast.Name(id='F',ctx=ast.Load()),attr='pad',ctx=ast.Load())
   return n
 changed=Rewrite().visit(node);assert ast.dump(Restore().visit(copy.deepcopy(changed)),include_attributes=False)==ast.dump(original,include_attributes=False)
 namespace=dict(vars(module),_pad_if_needed=pad_if_needed);exec(compile(ast.fix_missing_locations(ast.Module(body=[changed],type_ignores=[])),module.__file__,'exec'),namespace)
 return namespace['pointwise']
def build(case,library,original_factory):
 from benchmarks.ascend_all_pairs.fla import providers
 from benchmarks.ascend_baselines import source_hashes
 if library!='flash-linear-attention-npu' or case._call.signature.name!='AddFwdOp':raise ValueError('only reviewed FLA Add wrapper')
 candidate=clone(providers)(case);candidate.source_sha256.update(source_hashes(pathlib.Path(__file__)))
 return candidate
