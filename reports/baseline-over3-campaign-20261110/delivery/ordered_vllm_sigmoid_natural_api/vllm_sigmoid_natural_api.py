"""Natural head dimension for the unchanged complete gating API; sigmoid only."""
import ast,copy,pathlib,hashlib,dataclasses

def clone_activation(module):
 node=next(n for n in ast.parse(pathlib.Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='activation');old=ast.dump(node,include_attributes=False)
 patterns=[(ast.parse(a,mode='eval').body,ast.parse(b,mode='eval').body) for a,b in [('x.contiguous().reshape(-1, 1).float()','x.contiguous().reshape(-1, x.shape[-1]).float()'),('torch.zeros(1, device=x.device, dtype=torch.float32)','torch.zeros(x.shape[-1], device=x.device, dtype=torch.float32)')]];counts=[0,0]
 class Swap(ast.NodeTransformer):
  def __init__(self,reverse=False):self.reverse=reverse
  def visit_Call(self,n):
   for i,(a,b) in enumerate(patterns):
    src,dst=(b,a) if self.reverse else (a,b)
    if ast.dump(n,include_attributes=False)==ast.dump(src,include_attributes=False):
     if not self.reverse:counts[i]+=1
     return copy.deepcopy(dst)
   return self.generic_visit(n)
 node=Swap().visit(node);assert counts==[1,1];assert ast.dump(Swap(True).visit(copy.deepcopy(node)),include_attributes=False)==old
 env=dict(vars(module));exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),module.__file__,'exec'),env)
 return env['activation'],{'reverse_full_factory_AST_identical':True,'only_changes':['natural lastdim view','matching perhead zero vector'],'FP32_conversion_full_g_beta_API_extra_output_cast_and_all_copies_retained':True,'no_kernel_block_flags_compiler_precision_edits':True}

def build(case,library):
 assert library=='vllm-ascend' and case._call.signature.name=='SigmoidFwdOp'
 from benchmarks.ascend_baselines import UnsupportedCase
 if case.inputs[0].ndim<2 or case.inputs[0].shape[-1]<2:raise UnsupportedCase('No distinct natural head dimension for this layout; do not force candidate')
 import importlib
 activation=importlib.import_module('benchmarks.ascend_all_pairs.vllm.activation')
 fn,proof=clone_activation(activation);c=fn(case);p=pathlib.Path(__file__);c.source_sha256[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 return dataclasses.replace(c,api=c.api+' (natural head layout)',adaptation='Only natural lastdim and matching zero vector changed. Complete original gating API computes both g and beta; all FP32 conversion, zero allocation, contiguous copies and finalcast included. No native block/kernel/compiler/precision edits.')
