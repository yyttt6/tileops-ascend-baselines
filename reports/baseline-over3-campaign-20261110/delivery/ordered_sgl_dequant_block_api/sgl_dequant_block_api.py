"""Original full scale_shift; per128 scale becomes hidden axis, all packing timed."""
import ast,copy,pathlib,hashlib,dataclasses

def clone_functions(extra,additional):
 def find(m,n):return next(x for x in ast.parse(pathlib.Path(m.__file__).read_text()).body if isinstance(x,ast.FunctionDef) and x.name==n)
 mul=find(additional,'mul');before=ast.dump(mul,include_attributes=False);idx=next(i for i,n in enumerate(mul.body) if isinstance(n,ast.FunctionDef) and n.name=='run');original_run=copy.deepcopy(mul.body[idx])
 newrun=ast.parse('''def run(x,y):
 import torch.nn.functional as F
 m,k=x.shape;tail=-k%128
 xx=F.pad(x,(0,tail)) if tail else x
 xx=xx.reshape(-1,128).transpose(0,1).contiguous().reshape(128,1,-1)
 yy=y.contiguous().reshape(-1)
 zeros=torch.zeros_like(xx)
 return fn(xx,yy,zeros,scale_constant=0.0).reshape(128,-1).transpose(0,1).contiguous().reshape(m,k+tail)[:,:k].contiguous()
''').body[0];mul.body[idx]=newrun;restored=copy.deepcopy(mul);restored.body[idx]=original_run;assert ast.dump(restored,include_attributes=False)==before
 env=dict(vars(additional));exec(compile(ast.fix_missing_locations(ast.Module(body=[mul],type_ignores=[])),additional.__file__,'exec'),env);packed_mul=env['mul']
 deq=find(extra,'dequant');before_deq=ast.dump(deq,include_attributes=False);binding=[(i,n) for i,n in enumerate(deq.body) if isinstance(n,ast.ImportFrom) and n.module=='additional' and n.level==1];assert len(binding)==1;bi,bn=binding[0];del deq.body[bi]
 run=next(n for n in deq.body if isinstance(n,ast.FunctionDef) and n.name=='run');branch=next(n for n in run.body if isinstance(n,ast.If)).orelse[0];assert isinstance(branch,ast.If) and 'INT8DequantPerBlockFwdOp' in ast.unparse(branch.test);assignment=branch.body[0];old=copy.deepcopy(assignment.value);assert ast.dump(old,include_attributes=False)==ast.dump(ast.parse('scale.repeat_interleave(128,dim=1)[:,:q.shape[1]]',mode='eval').body,include_attributes=False);assignment.value=ast.Name(id='scale',ctx=ast.Load())
 rev=copy.deepcopy(deq);rev.body.insert(bi,copy.deepcopy(bn));rr=next(n for n in rev.body if isinstance(n,ast.FunctionDef) and n.name=='run');next(n for n in rr.body if isinstance(n,ast.If)).orelse[0].body[0].value=old;assert ast.dump(rev,include_attributes=False)==before_deq
 env=dict(vars(extra));env['mul']=packed_mul;exec(compile(ast.fix_missing_locations(ast.Module(body=[deq],type_ignores=[])),extra.__file__,'exec'),env)
 return env['dequant'],{'original_dequant_AST_restorable':True,'only_dequant_timed_change':'per128 repeat_interleave replaced by native-hidden block packing','mul_factory_original_except_explicit_packing_run_AST_restorable':True,'original_q_FP32_conversion_and_output_cast_retained':True,'native_full_zeros_and_API_retained':True}

def build(case,library):
 import torch
 assert library=='sgl-kernel-npu' and case._call.signature.name=='INT8DequantPerBlockFwdOp'
 q,scale=case.inputs;assert q.ndim==2 and q.numel()>1 and scale.dtype==torch.float32 and tuple(scale.shape)==(q.shape[0],(q.shape[1]+127)//128)
 from benchmarks.ascend_all_pairs.sgl import extra_arithmetic,additional
 fn,proof=clone_functions(extra_arithmetic,additional);candidate=fn(case);f=pathlib.Path(__file__);candidate.source_sha256[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
 return dataclasses.replace(candidate,api=candidate.api+' (per128 quant-block hidden layout)',adaptation='Original FP32 dequant and complete native scale_shift; replace scale repeat with percall Ktailpad/128group transpose/nativefullzeros/restorecrop/contiguous. All costs timed; no kernel/block/compiler/precision changes.')
