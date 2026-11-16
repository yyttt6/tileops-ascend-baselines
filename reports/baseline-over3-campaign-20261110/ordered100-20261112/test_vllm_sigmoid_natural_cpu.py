import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import pathlib,sys,types,ast,json,datetime,hashlib,itertools
import torch
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P/'workspace/isolated_wrappers'));import vllm_sigmoid_natural_api as helper
path=P/'workspace/TileOPs/benchmarks/ascend_all_pairs/vllm/activation.py';mod=types.ModuleType('frozen_activation');mod.__file__=str(path);mod.Path=pathlib.Path;mod.TRITON_ROOT='';mod.validate=lambda c:None;emitted=[]
class Native:
 def fused_gdn_gating_patch(self,alog,a,b,bias,beta,threshold):
  assert alog.numel()==a.shape[-1]==bias.numel() and a.dtype==b.dtype==torch.float32
  g=-torch.exp(alog)*torch.nn.functional.softplus(a+bias,beta=beta,threshold=threshold);out=torch.sigmoid(b);emitted.append((g.clone(),out.clone()));return g,out
mod.module=lambda name,**kw:(Native(),[]);mod.candidate=lambda fn,*a,**kw:fn
node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='activation');exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),vars(mod));alt,proof=helper.clone_activation(mod);rows=[]
def bits(x):return x.contiguous().reshape(-1).view(torch.int32 if x.dtype==torch.float32 else torch.int16)
for dtype,shape,strided,domain,beta in itertools.product([torch.float16,torch.bfloat16,torch.float32],[(3,7),(2,3,11),(17,33)],[False,True],['finite','special'],[1.,2.]):
 x=torch.randn(*shape[:-1],shape[-1]*(2 if strided else 1),dtype=dtype)
 if strided:x=x[...,::2]
 if domain=='special':x[(0,)*(x.ndim-1)+(slice(0,6),)]=torch.tensor([float('nan'),float('inf'),-float('inf'),0.,-0.,1.],dtype=dtype)
 before=x.clone();case=types.SimpleNamespace(_call=types.SimpleNamespace(signature=types.SimpleNamespace(name='SigmoidFwdOp')),params={'beta':beta,'threshold':5.});emitted.clear();a=mod.activation(case)(x);g1,b1=emitted[-1];b=alt(case)(x);g2,b2=emitted[-1]
 assert len(emitted)==2 and torch.equal(bits(a),bits(b)) and torch.equal(bits(g1),bits(g2)) and torch.equal(bits(b1),bits(b2)) and torch.equal(bits(x),bits(before));rows.append({'dtype':str(dtype),'shape':shape,'strided':strided,'domain':domain,'beta':beta,'bitwise_selected_and_extra_output_mapping_equal':True,'input_preserved':True})
# Model the real package's activation function/module name collision.
import dataclasses
for name in ['benchmarks','benchmarks.ascend_all_pairs','benchmarks.ascend_all_pairs.vllm']:
 package=types.ModuleType(name);package.__path__=[];sys.modules[name]=package
sys.modules['benchmarks.ascend_all_pairs.vllm'].activation=lambda:None
sys.modules['benchmarks.ascend_all_pairs.vllm.activation']=mod
api=types.ModuleType('benchmarks.ascend_baselines');api.UnsupportedCase=RuntimeError;sys.modules['benchmarks.ascend_baselines']=api
@dataclasses.dataclass
class Candidate:
 api:str
 adaptation:str
 source_sha256:dict
mod.candidate=lambda fn,*args,**kwargs:Candidate(str(args[0]),'CPU factory binding only',{})
case=types.SimpleNamespace(_call=types.SimpleNamespace(signature=types.SimpleNamespace(name='SigmoidFwdOp')),inputs=[torch.ones(3,7)],params={})
bound=helper.build(case,'vllm-ascend');assert 'natural head layout' in bound.api
out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':len(rows),'proof':proof,'factory_module_function_name_collision_test_passed':True,'rows':rows,'CPU_surrogate_only_not_native_acceptance':True,'no_NPU_imported':not any(x.startswith('torch_npu') for x in sys.modules),'helper_sha256':hashlib.sha256(pathlib.Path(helper.__file__).read_bytes()).hexdigest()};assert out['no_NPU_imported'];(P/'VLLM_SIGMOID_NATURAL_CPU_PROOF.json').write_text(json.dumps(out,indent=2));print('CPU_SIGMOID_AND_EXTRA_G_MAPPING_CASES',len(rows))
