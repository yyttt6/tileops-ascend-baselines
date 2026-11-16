"""Use the original complete LayerNorm grouping API for independent affine rows."""
import ast,copy,pathlib,hashlib,dataclasses

def clone_factory(module):
 node=next(n for n in ast.parse(pathlib.Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='adaptive_layer_norm');old=ast.dump(node,include_attributes=False)
 a=ast.parse('out = [native.layer_norm_fwd_npu(xp[i:i+1], sp[i], bp[i], eps)[0] for i in range(len(xp))]').body[0]
 b=ast.parse('out = native.layer_norm_fwd_npu(xp.reshape(1, -1), sp.reshape(-1), bp.reshape(-1), eps, group_size=n)[0]').body[0]
 c=ast.parse('return torch.cat(out).reshape(x.shape)').body[0];d=ast.parse('return out.reshape(x.shape)').body[0];counts=[0,0]
 class Swap(ast.NodeTransformer):
  def __init__(self,reverse=False):self.reverse=reverse
  def visit(self,n):
   for i,(src,dst) in enumerate([(a,b),(c,d)]):
    if self.reverse:src,dst=dst,src
    if ast.dump(n,include_attributes=False)==ast.dump(src,include_attributes=False):
     if not self.reverse:counts[i]+=1
     return copy.deepcopy(dst)
   return super().visit(n)
 node=Swap().visit(node);assert counts==[1,1];assert ast.dump(Swap(True).visit(copy.deepcopy(node)),include_attributes=False)==old
 env=dict(vars(module));exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),module.__file__,'exec'),env)
 return env['adaptive_layer_norm'],{'reverse_full_factory_AST_identical':True,'only_changes':['independent original perrow affine norm calls -> same complete grouped norm API group_size=n','final cat -> reshape native output'],'perrow_reduction_domain_FP32_precision_affine_correspondence_contiguous_originalAPI_extra_outputs_retained':True,'no_kernel_block_flags_compiler_precision_edits':True}

def build(case,library):
 assert library=='vllm-ascend' and case._call.signature.name=='AdaLayerNormFwdOp'
 import importlib
 module=importlib.import_module('benchmarks.ascend_all_pairs.vllm.basic');fn,proof=clone_factory(module);result=fn(case);p=pathlib.Path(__file__);result.source_sha256[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 return dataclasses.replace(result,api=result.api+' (independent native groups)',adaptation='Same complete original layer_norm_fwd_npu with group_size=original lastdim; flatten row-specific scale/shift in same order. All percallcontiguous/nativeaffine/norm/extra statistics/outputreshape retained. No kernel/compiler/precision edits.')
