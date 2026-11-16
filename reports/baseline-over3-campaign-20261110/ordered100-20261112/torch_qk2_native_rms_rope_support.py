"""Complete original QK program with only its FP32 RMS expression bound to a full native API."""
from pathlib import Path
import ast,copy,hashlib,inspect,types
SCOPE={'interleaved-ragged-float16-float16','interleaved-ragged-bfloat16-float32'}
def build(case,native_rms=None,native_rotary=None):
 assert case.id in SCOPE and case._call.signature.name=='FusedQKNormRoPEFwdOp'
 original=case.reference;module=inspect.getmodule(original);file=Path(module.__file__);tree=ast.parse(file.read_text());node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fused_qk_norm_rope_heads'));before=ast.dump(node,include_attributes=False)
 matches=[n for n in ast.walk(node) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='y'];assert len(matches)==1;target=matches[0];expected=ast.parse('y=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)*weight.float()').body[0];assert ast.dump(target,include_attributes=False)==ast.dump(expected,include_attributes=False)
 saved=copy.deepcopy(target);replacement=ast.parse('y,_native_rstd=native_rms(x,weight.float(),epsilon=eps)').body[0]
 class Replace(ast.NodeTransformer):
  def visit_Assign(self,n):return copy.deepcopy(replacement) if n is target else self.generic_visit(n)
 node=Replace().visit(node)
 rope=next(n for n in ast.walk(node) if isinstance(n,ast.If) and ast.unparse(n.test)=="rope_layout == 'neox'")
 original_interleave=copy.deepcopy(rope.orelse)
 assert len(original_interleave)==2 and ast.unparse(original_interleave[0])=='a, b = (rot[..., 0::2], rot[..., 1::2])'
 rope.orelse=ast.parse("rotated=native_rotary(rot.unsqueeze(1),torch.repeat_interleave(cos,2,dim=-1).unsqueeze(1),torch.repeat_interleave(sin,2,dim=-1).unsqueeze(1),rotary_mode='interleave').squeeze(1)").body
 class Restore(ast.NodeTransformer):
  def visit_Assign(self,n):return copy.deepcopy(saved) if isinstance(n.targets[0],ast.Tuple) and any(isinstance(t,ast.Name) and t.id=='_native_rstd' for t in n.targets[0].elts) else self.generic_visit(n)
  def visit_If(self,n):
   self.generic_visit(n)
   if ast.unparse(n.test)=="rope_layout == 'neox'":n.orelse=copy.deepcopy(original_interleave)
   return n
 assert ast.dump(Restore().visit(copy.deepcopy(node)),include_attributes=False)==before
 if native_rms is None or native_rotary is None:
  import torch_npu
  native_rms=torch_npu.npu_rms_norm;native_rotary=torch_npu.npu_rotary_mul
 env=dict(original.__func__.__globals__);env['native_rms']=native_rms;env['native_rotary']=native_rotary;exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(file),'exec'),env);rebound=types.FunctionType(original.__func__.__code__,env,original.__func__.__name__,original.__func__.__defaults__,original.__func__.__closure__)
 return types.MethodType(rebound,case.workload),{'original_source':str(file),'original_source_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'full_reverse_AST_identical':True,'original_outer_reference_code_object_unchanged':True,'replacement':'FP32 complete RMS and full rotary_mul(interleave); native 4D SBND view and original table cos/sin repeated inside timed program; full reverse AST retained','original_FP32_inputs_and_weights_preserved':True,'original_clone_V_tablegather_pack_finalcast_preserved':True,'native_interleave_RoPE_FP32_SBND_with_timed_coefficient_expansion':True,'not_native_qualification':True}
