"""Per-tensor only: same complete fused_scale_shift, natural hidden axis."""
import ast,copy,pathlib,hashlib,dataclasses

def clone_functions(extra,additional):
 def node(module,name):return next(n for n in ast.parse(pathlib.Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
 mul=node(additional,'mul');old=ast.dump(mul,include_attributes=False)
 pairs=[('x.expand(shape).contiguous().reshape(1,1,-1)','x.expand(shape).contiguous().reshape(-1,1,shape[-1])'),('y.expand(shape).contiguous().reshape(-1)','y.reshape(-1).expand(shape[-1]).contiguous()')]
 patterns=[(ast.parse(a,mode='eval').body,ast.parse(b,mode='eval').body) for a,b in pairs];counts=[0,0]
 class Rewrite(ast.NodeTransformer):
  def __init__(self,reverse=False):self.reverse=reverse
  def visit_Call(self,n):
   for i,(a,b) in enumerate(patterns):
    src,dst=(b,a) if self.reverse else (a,b)
    if ast.dump(n,include_attributes=False)==ast.dump(src,include_attributes=False):
     if not self.reverse:counts[i]+=1
     return copy.deepcopy(dst)
   return self.generic_visit(n)
 mul=Rewrite().visit(mul);assert counts==[1,1];assert ast.dump(Rewrite(True).visit(copy.deepcopy(mul)),include_attributes=False)==old
 env=dict(vars(additional));exec(compile(ast.fix_missing_locations(ast.Module(body=[mul],type_ignores=[])),additional.__file__,'exec'),env);natural_mul=env['mul']
 dequant=node(extra,'dequant');old_dequant=ast.dump(dequant,include_attributes=False)
 bindings=[(i,n) for i,n in enumerate(dequant.body) if isinstance(n,ast.ImportFrom) and n.module=='additional' and n.level==1 and [x.name for x in n.names]==['mul']];assert len(bindings)==1
 index,binding=bindings[0];del dequant.body[index];restored=copy.deepcopy(dequant);restored.body.insert(index,copy.deepcopy(binding));assert ast.dump(restored,include_attributes=False)==old_dequant
 env=dict(vars(extra));env['mul']=natural_mul;exec(compile(ast.fix_missing_locations(ast.Module(body=[dequant],type_ignores=[])),extra.__file__,'exec'),env)
 return env['dequant'],{'mul_reverse_AST_identical':True,'dequant_reverse_AST_identical':True,'layout_substitutions':pairs,'native_API_kernel_precision_unchanged':True}

def build(case,library):
 assert library=='sgl-kernel-npu' and case._call.signature.name=='INT8DequantPerTensorFwdOp','exact per-tensor scope'
 q,scale=case.inputs;assert q.ndim>=1 and q.shape[-1]>0 and scale.numel()==1,'natural hidden/scalar scale contract'
 from benchmarks.ascend_all_pairs.sgl import extra_arithmetic,additional
 fn,proof=clone_functions(extra_arithmetic,additional);candidate=fn(case);f=pathlib.Path(__file__);candidate.source_sha256[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
 return dataclasses.replace(candidate,api=candidate.api+' (per-tensor natural hidden layout)',adaptation='Only native hidden axis and scalar scale expansion changed; original dequant timed body and complete native scale_shift/FP32 conversion/fullzeros/outputcast retained per call. No kernel/block/compiler/precision changes.')
