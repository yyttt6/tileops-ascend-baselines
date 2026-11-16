"""Complete alternate RMS API only; original supplied-table RoPE/FP32 chain retained."""
import ast,copy,pathlib,hashlib,dataclasses
OLD='batch_invariant/rmsnorm.py';NEW='fla/layernorm_guard.py'
def clone_qk(module):
 node=next(n for n in ast.parse(pathlib.Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='fused_qk_norm_rope');old=ast.dump(node,include_attributes=False);counts=[0,0]
 class Change(ast.NodeTransformer):
  def visit_Constant(self,n):
   if n.value==OLD:counts[0]+=1;return ast.copy_location(ast.Constant(NEW),n)
   return n
  def visit_Call(self,n):
   if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='norm' and n.func.attr=='rms_norm':
    assert len(n.args)==3 and not n.keywords;counts[1]+=1
    call=ast.Call(func=ast.Attribute(value=ast.Name(id='norm',ctx=ast.Load()),attr='_layer_norm_fwd',ctx=ast.Load()),args=[copy.deepcopy(n.args[0]),copy.deepcopy(n.args[1]),ast.Constant(None),copy.deepcopy(n.args[2])],keywords=[ast.keyword(arg='is_rms_norm',value=ast.Constant(True))]);return ast.Subscript(value=call,slice=ast.Constant(0),ctx=ast.Load())
   return self.generic_visit(n)
 class Reverse(ast.NodeTransformer):
  def visit_Constant(self,n):
   if n.value==NEW:return ast.copy_location(ast.Constant(OLD),n)
   return n
  def visit_Subscript(self,n):
   if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='norm' and n.value.func.attr=='_layer_norm_fwd':
    c=n.value;assert isinstance(n.slice,ast.Constant) and n.slice.value==0 and len(c.args)==4 and c.args[2].value is None
    return ast.Call(func=ast.Attribute(value=ast.Name(id='norm',ctx=ast.Load()),attr='rms_norm',ctx=ast.Load()),args=[copy.deepcopy(c.args[0]),copy.deepcopy(c.args[1]),copy.deepcopy(c.args[3])],keywords=[])
   return self.generic_visit(n)
 node=Change().visit(node);assert counts==[1,1];assert ast.dump(Reverse().visit(copy.deepcopy(node)),include_attributes=False)==old
 env=dict(vars(module));exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),module.__file__,'exec'),env)
 return env['fused_qk_norm_rope'],{'reverse_full_factory_AST_identical':True,'only_changes':['samevendor module binding','complete RMSAPI call [0]'],'FP32_normalized_QK_supplied_tables_positions_privateclone_Vcopy_cast_cat_retained':True,'kernel_compiler_block_precision_not_edited':True}
def build(case,library):
 assert library=='vllm-ascend' and case._call.signature.name=='FusedQKNormRoPEFwdOp'
 from benchmarks.ascend_baselines import UnsupportedCase
 if case.inputs[1].numel()*4>65536:raise UnsupportedCase('Original alternateAPI64KiB featurelimit withFP32intermediate; no split/bypass')
 from benchmarks.ascend_all_pairs.vllm import basic
 fn,proof=clone_qk(basic);c=fn(case);p=pathlib.Path(__file__);c.source_sha256[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 return dataclasses.replace(c,api='complete alternate RMS + original suppliedtable RoPE',adaptation='Only full vendor RMS API changed. Original FP32intermediate, epsilon/weights, suppliedtable/positions, privateclone, Vcopy, finalcast/cat remain percall; no kernel/compiler/block/precision changes.')
