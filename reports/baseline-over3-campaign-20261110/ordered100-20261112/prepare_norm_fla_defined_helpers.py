from pathlib import Path
import json
P=Path('/root/dyq/workspace/reports/baseline-over3-campaign-20261110/ordered100-20261112')
helper='''"""Finite complete-API alternative; original global contracts and kernel untouched."""
import ast,copy,pathlib,hashlib,dataclasses
from math import prod
EXPECTED_REASON="Original LayerNorm sums masked-load lanes without an explicit zero value for non-power-of-two domains; equivalence is unresolved until the original source/backend defines those lanes. No kernel rewrite or statistics-domain padding is credited."
KEYS={('GroupNormFwdOp','wider-channel-affine-float16'),('GroupNormFwdOp','wider-channel-float16'),('InstanceNormFwdOp','wider-channel-affine-float16'),('InstanceNormFwdOp','wider-channel-float16'),('InstanceNormFwdOp','tail-spatial-affine-float16'),('InstanceNormFwdOp','wider-channel-affine-float32'),('InstanceNormFwdOp','tail-spatial-float16'),('LayerNormFwdOp','dit-xl-2-bfloat16')}
def validate_alternate(case):
 from benchmarks.ascend_all_pairs.vllm.contracts import decide
 from benchmarks.ascend_baselines import UnsupportedCase
 op=case._call.signature.name;p=case.params
 if (op,case.id) not in KEYS:raise UnsupportedCase('Only eight frozen previouslyblocked exact Norm keys')
 status,reason=decide(op,p)
 if status!='adapter_blocked' or reason!=EXPECTED_REASON:raise UnsupportedCase('Only exact old route maskedlane restriction is reviewed: '+reason)
 x,*args=case.inputs
 if tuple(x.shape)!=tuple(p['x'][0]) or str(x.dtype).removeprefix('torch.')!=p['x'][1]:raise UnsupportedCase('Frozen declared shape/dtype required')
 if p['x'][1] not in ('float16','bfloat16','float32'):raise UnsupportedCase('Only original floating dtype')
 if op=='InstanceNormFwdOp':
  rm,rv,w,b=args
  if rm is not None or rv is not None or not p.get('use_input_stats',True):raise UnsupportedCase('Runningstatistics require complete separate native state contract')
 elif op=='GroupNormFwdOp':w,b=args
 else:
  w=args[0] if args else None;b=args[1] if len(args)>1 else None
 width=prod(x.shape[1:])//p['num_groups'] if op=='GroupNormFwdOp' else prod(x.shape[2:]) if op=='InstanceNormFwdOp' else prod(p.get('normalized_shape',[x.shape[-1]]))
 if width>65536//x.element_size():raise UnsupportedCase('Original completeAPI64KiB feature limit, no domain splitting')
 expected=(x.shape[1],) if op!='LayerNormFwdOp' else tuple(p.get('normalized_shape',[x.shape[-1]]))
 for a in (w,b):
  if a is not None and (a.dtype!=x.dtype or tuple(a.shape)!=expected):raise UnsupportedCase('Original affine dtype/shape must match')
 return {'original_decision':status,'original_reason':reason,'group_size':width,'old_global_contract_untouched':True,'why_this_API_is_different':'Original complete fla._layer_norm_fwd explicitly loads other0 and masks centered variance over original groupN; unchanged native source, no statistics-domain padding'}
def clone_factory(module,name):
 node=next(n for n in ast.parse(pathlib.Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name);before=ast.dump(node,include_attributes=False)
 class Change(ast.NodeTransformer):
  def __init__(self,reverse=False):self.reverse=reverse
  def visit_Call(self,n):
   self.generic_visit(n)
   if isinstance(n.func,ast.Name) and n.func.id==('validate_alternate' if self.reverse else 'validate'):n.func.id='validate' if self.reverse else 'validate_alternate'
   return n
  def visit_Constant(self,n):
   a,b=('fla/layernorm_guard.py','layernorm_gated.py') if self.reverse else ('layernorm_gated.py','fla/layernorm_guard.py')
   if n.value==a:n.value=b
   return n
  def visit_Attribute(self,n):
   self.generic_visit(n);a,b=('_layer_norm_fwd','layer_norm_fwd_npu') if self.reverse else ('layer_norm_fwd_npu','_layer_norm_fwd')
   if n.attr==a:n.attr=b
   return n
 node=Change().visit(node);assert ast.dump(Change(True).visit(copy.deepcopy(node)),include_attributes=False)==before
 env=dict(vars(module));env['validate_alternate']=validate_alternate;exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),module.__file__,'exec'),env)
 return env[name],{'full_factory_reverse_AST_identical':True,'only_changes':['finite route-specific validation','original fullnative module path','complete LayerNorm method selection'],'original_groupdomain_affine_pack_dtype_eps_output_contract_unchanged':True}
def build(case,library):
 assert library=='vllm-ascend';validate_alternate(case)
 from benchmarks.ascend_all_pairs.vllm import basic
 fn,proof=clone_factory(basic,'normalization' if case._call.signature.name=='LayerNormFwdOp' else 'group_norm');c=fn(case);f=pathlib.Path(__file__);c.source_sha256[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
 return dataclasses.replace(c,api='vllm-ascend::complete fla._layer_norm_fwd original groups',adaptation='Alternate original completeAPI defines masked lanes; every original affine expand/ones/contiguous/reshape/eps/dtype and extra native statistics retained. No oldcontrol timing or ratio claimed.',limitations=('Finite eight exact originallyblocked keys; no runningstatistics contract substitution.',))
'''
(P/'workspace/isolated_wrappers/vllm_norm_fla_defined_api.py').write_text(helper)
s=(P/'adaln_fla_defined_cached_fidelity.py').read_text().replace('APPROVED=set(range(1117,1122))','APPROVED={1177,1180,1190,1191,1192,1194,1195,1199}').replace("('AdaLayerNormFwdOp',)","('GroupNormFwdOp','InstanceNormFwdOp','LayerNormFwdOp')");(P/'norm_fla_defined_cached_fidelity.py').write_text(s)
s=(P/'verify_adaln_fla_defined_cached.py').read_text().replace('adaln_fla_defined_cached_fidelity','norm_fla_defined_cached_fidelity');(P/'verify_norm_fla_defined_cached.py').write_text(s)
s=(P/'worker_vllm_adaln_fla_defined.py').read_text().replace('adaln_fla_defined_cached_fidelity','norm_fla_defined_cached_fidelity').replace('vllm_adaln_fla_defined_support','vllm_norm_fla_defined_support');(P/'worker_vllm_norm_fla_defined.py').write_text(s)
s=(P/'vllm_adaln_fla_defined_support.py').read_text().replace('vllm_adaln_fla_defined_api','vllm_norm_fla_defined_api').replace('group_size=case.inputs[0].shape[-1],','group_size=review[\'group_size\'],');(P/'vllm_norm_fla_defined_support.py').write_text(s)
print('wrote finite helpers, no active worker/source modified')
