"""One-change wrapper candidates: skip only zero padding, retain complete original SGL API."""
import pathlib,ast,copy

def pad_if_needed(value,padding,*args,**kwargs):
 import torch.nn.functional as F
 return value if not any(padding) else F.pad(value,padding,*args,**kwargs)

def cloned_namespace(module):
 tree=ast.parse(pathlib.Path(module.__file__).read_text());nodes=[copy.deepcopy(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('sub','neg','silu_and_mul')];old=ast.dump(ast.Module(body=copy.deepcopy(nodes),type_ignores=[]),include_attributes=False)
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
 changed=Rewrite().visit(ast.Module(body=nodes,type_ignores=[]));assert ast.dump(Restore().visit(copy.deepcopy(changed)),include_attributes=False)==old
 scope=dict(vars(module),_pad_if_needed=pad_if_needed);exec(compile(ast.fix_missing_locations(changed),module.__file__,'exec'),scope);return scope

def build(case,library,original_factory):
 from benchmarks.ascend_all_pairs.sgl import elementwise
 from benchmarks.ascend_baselines import source_hashes,UnsupportedCase
 names={'NegFwdOp':'neg','SiluAndMulFwdOp':'silu_and_mul'};op=case._call.signature.name
 if library!='sgl-kernel-npu' or op not in names:raise UnsupportedCase('Unreviewed wrapper route')
 x=case.inputs[0];n=x.numel() if op=='NegFwdOp' else x.numel()//2;block=4096 if op=='NegFwdOp' else 1024
 if n%block:raise UnsupportedCase('No zero-pad-only difference for this tail')
 original=original_factory(case,library)();candidate=cloned_namespace(elementwise)[names[op]](case)
 if original.api!=candidate.api or original.source_sha256!=candidate.source_sha256:raise UnsupportedCase('Original registry selected a different exact route; do not compare generic wrapper as one-change candidate')
 candidate.source_sha256.update(source_hashes(pathlib.Path(__file__)));return candidate
