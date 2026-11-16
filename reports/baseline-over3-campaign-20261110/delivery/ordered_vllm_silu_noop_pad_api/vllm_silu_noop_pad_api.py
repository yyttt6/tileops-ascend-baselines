"""Keep the complete SiLU/SwiGLU API; skip only zero-length padding copies."""
import ast,copy,pathlib,hashlib,dataclasses

def clone_activation(module):
 node=next(n for n in ast.parse(pathlib.Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='activation');old=ast.dump(node,include_attributes=False)
 patterns=[]
 for name in ['gate','up']:
  original=ast.parse(f'F.pad({name}.contiguous().reshape(-1), (0, pad))',mode='eval').body
  changed=ast.parse(f'({name}.contiguous().reshape(-1) if pad == 0 else F.pad({name}.contiguous().reshape(-1), (0, pad)))',mode='eval').body
  patterns.append((original,changed))
 counts=[0,0]
 class Swap(ast.NodeTransformer):
  def __init__(self,reverse=False):self.reverse=reverse
  def visit(self,n):
   for i,(a,b) in enumerate(patterns):
    src,dst=(b,a) if self.reverse else (a,b)
    if ast.dump(n,include_attributes=False)==ast.dump(src,include_attributes=False):
     if not self.reverse:counts[i]+=1
     return copy.deepcopy(dst)
   return super().visit(n)
 node=Swap().visit(node);assert counts==[1,1]
 assert ast.dump(Swap(True).visit(copy.deepcopy(node)),include_attributes=False)==old
 env=dict(vars(module));exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),module.__file__,'exec'),env)
 return env['activation'],{'reverse_full_factory_AST_identical':True,'only_changes':['gate zero-tail pad skipped','up zero-tail pad skipped'],'original_16column_layout_full_swiglustep_limit_inf_ones_cat_copies_retained':True,'no_kernel_block_flags_compiler_precision_edits':True}

def build(case,library):
 assert library=='vllm-ascend' and case._call.signature.name in ('SiluFwdOp','SiluAndMulFwdOp')
 import importlib
 activation=importlib.import_module('benchmarks.ascend_all_pairs.vllm.activation')
 fn,proof=clone_activation(activation);c=fn(case);p=pathlib.Path(__file__);c.source_sha256[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 return dataclasses.replace(c,api=c.api+' (skip zero-tail pad)',adaptation='Only zero-tail F.pad copies omitted; complete original16column swiglustep API, limitinf, unitup forSiLU, contiguous/concat/outputcrop remain timed. No kernel/compiler/precision changes.')
