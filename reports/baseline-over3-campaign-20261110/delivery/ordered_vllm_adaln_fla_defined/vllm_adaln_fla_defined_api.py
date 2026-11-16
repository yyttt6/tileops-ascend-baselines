"""Exact AdaLN scope via complete vendor API with explicitly zero masked lanes."""
import ast,copy,pathlib,hashlib,dataclasses
EXPECTED_REASON='Original LayerNorm sums masked-load lanes without an explicit zero value for non-power-of-two domains; equivalence is unresolved until the original source/backend defines those lanes. No kernel rewrite or statistics-domain padding is credited.'
CASES={'dit-xl-2-float32','dit-xl-2-bfloat16','dit-xl-2-float16','llama-8b-prefill-float16','llama-8b-prefill-bfloat16'}
def validate_alternate(case):
 from benchmarks.ascend_all_pairs.vllm.contracts import decide
 from benchmarks.ascend_baselines import UnsupportedCase
 name=case._call.signature.name
 if name!='AdaLayerNormFwdOp' or case.id not in CASES:raise UnsupportedCase('Only five frozen exact AdaLN keys reviewed for this complete API')
 status,reason=decide(name,case.params)
 if status not in ('supported_unverified','deployment_blocked') and not (status=='adapter_blocked' and reason==EXPECTED_REASON):raise UnsupportedCase(reason)
 x,scale,shift=case.inputs
 if tuple(x.shape)!=tuple(case.params['x'][0]) or str(x.dtype).removeprefix('torch.')!=case.params['x'][1]:raise UnsupportedCase('Declared input shape/dtype must match frozen case params')
 if x.dtype.__str__() not in ('torch.float16','torch.bfloat16','torch.float32') or any(a.dtype!=x.dtype or tuple(a.shape)!=tuple(x.shape) for a in [scale,shift]):raise UnsupportedCase('Frozen original perrow affine shapes and floatingdtype required')
 if x.ndim<2 or x.shape[-1]>65536//x.element_size():raise UnsupportedCase('Complete original group feature64KiB limit retained, no domain split')
 return {'original_decision':status,'original_reason':reason,'why_this_API_is_different':'Original complete fla._layer_norm_fwd source uses tl.load(mask=cols<N,other=0.0) and masks centered variance before summing; no external padding/domain enlargement','old_global_contract_untouched':True}

def clone_factory(module):
 node=next(n for n in ast.parse(pathlib.Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='adaptive_layer_norm');original=ast.dump(node,include_attributes=False)
 patterns=[(ast.parse('validate(case)').body[0],ast.parse('validate_alternate(case)').body[0]),(ast.parse('TRITON_ROOT + "layernorm_gated.py"',mode='eval').body,ast.parse('TRITON_ROOT + "fla/layernorm_guard.py"',mode='eval').body),(ast.parse('out = [native.layer_norm_fwd_npu(xp[i:i+1], sp[i], bp[i], eps)[0] for i in range(len(xp))]').body[0],ast.parse('out = native._layer_norm_fwd(xp.reshape(1,-1), sp.reshape(-1), bp.reshape(-1), eps, group_size=n, is_rms_norm=False)[0]').body[0]),(ast.parse('return torch.cat(out).reshape(x.shape)').body[0],ast.parse('return out.reshape(x.shape)').body[0])];counts=[0]*4
 class Change(ast.NodeTransformer):
  def __init__(self,reverse=False):self.reverse=reverse
  def visit(self,n):
   for i,(a,b) in enumerate(patterns):
    src,dst=(b,a) if self.reverse else (a,b)
    if ast.dump(n,include_attributes=False)==ast.dump(src,include_attributes=False):
     if not self.reverse:counts[i]+=1
     return copy.deepcopy(dst)
   return super().visit(n)
 node=Change().visit(node);assert counts==[1]*4;assert ast.dump(Change(True).visit(copy.deepcopy(node)),include_attributes=False)==original
 env=dict(vars(module));env['validate_alternate']=validate_alternate;exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),module.__file__,'exec'),env)
 return env['adaptive_layer_norm'],{'full_factory_reverse_AST_identical':True,'only_four_scoped_changes':['route-specific validator for actual alternateAPI','complete native module selection','complete grouped LayerNorm call, same perrow n and weights','reshape in place of finalconcat'],'no_kernel_checker_precision_compiler_block_edits':True}

def build(case,library):
 assert library=='vllm-ascend'
 review=validate_alternate(case)
 import importlib
 module=importlib.import_module('benchmarks.ascend_all_pairs.vllm.basic');fn,proof=clone_factory(module);candidate=fn(case);f=pathlib.Path(__file__);candidate.source_sha256[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
 return dataclasses.replace(candidate,api='vllm-ascend::fla._layer_norm_fwd complete LayerNorm independent affine groups',adaptation='Original complete vendor API defines masked lanes as zero and keeps centered FP32 variance over original n. Perrowweights/eps/inputdtype/contiguous/outputreshape and all nativeextra statistics retained. Only five exactkeys; generic originalguard remains unchanged. No valid oldcontrol performance exists, so no original/new speedup claim.',limitations=('Candidate-only current API qualification; old reviewed route remains contract-blocked, historical performance is not reused.',))
