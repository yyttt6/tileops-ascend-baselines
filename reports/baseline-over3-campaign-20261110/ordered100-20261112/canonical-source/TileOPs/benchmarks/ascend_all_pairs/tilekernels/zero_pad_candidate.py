"""Only adapter-level zero-tail copies are skipped; original complete native API remains."""
import ast,pathlib,hashlib,types

def clone_factory(module):
 source=pathlib.Path(module.__file__).read_text();tree=ast.parse(source);node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='int8_dequant');original=ast.dump(node,include_attributes=False);count=0
 class Change(ast.NodeTransformer):
  def visit_Call(self,n):
   nonlocal count
   self.generic_visit(n)
   if ast.unparse(n.func)=='torch.nn.functional.pad':n.func=ast.Name(id='_pad_if_needed',ctx=ast.Load());count+=1
   return n
 changed=Change().visit(node);assert count==2
 class Reverse(ast.NodeTransformer):
  def visit_Name(self,n):
   if n.id=='_pad_if_needed':return ast.parse('torch.nn.functional.pad',mode='eval').body
   return n
 assert ast.dump(Reverse().visit(ast.fix_missing_locations(ast.parse(ast.unparse(changed)).body[0])),include_attributes=False)==original
 def pad(value,amount):
  if all(x==0 for x in amount):return value
  import torch
  return torch.nn.functional.pad(value,amount)
 env=dict(module.__dict__,_pad_if_needed=pad);program=ast.fix_missing_locations(ast.Module(body=[changed],type_ignores=[]));exec(compile(program,__file__,'exec'),env)
 return env['int8_dequant'],{'module_sha256':hashlib.sha256(pathlib.Path(module.__file__).read_bytes()).hexdigest(),'original_function_ast_sha256':hashlib.sha256(original.encode()).hexdigest(),'changed_calls':count,'reverse_ast_identical':True,'kernel_or_schedule_changes':False}

def factory(proxy):
 from benchmarks.ascend_all_pairs.tilekernels import dequant
 make,proof=clone_factory(dequant);candidate=make(proxy)
 return candidate.implementation.run,proof
