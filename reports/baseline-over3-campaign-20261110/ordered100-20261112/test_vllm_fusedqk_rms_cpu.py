import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import pathlib,sys,types,ast,json,datetime,hashlib,itertools
import torch
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P/'workspace/isolated_wrappers'));import vllm_fusedqk_rms_api as helper
path=P/'workspace/TileOPs/benchmarks/ascend_all_pairs/vllm/basic.py';mod=types.ModuleType('frozen_basic');mod.__file__=str(path);mod.Path=pathlib.Path;mod.TRITON_ROOT='';mod.validate=lambda c:None;checks=[]
class Norm:
 def rms_norm(self,x,w,eps):assert x.dtype==torch.float32;checks.append('old');return x*torch.rsqrt((x*x).mean(-1,keepdim=True)+eps)*w
 def _layer_norm_fwd(self,x,w,b,eps,is_rms_norm=False):assert b is None and is_rms_norm and x.dtype==torch.float32;checks.append('new');return (x*torch.rsqrt((x*x).mean(-1,keepdim=True)+eps)*w,None,None,None)
class Rope:
 def rope_forward_triton_siso(self,x,cos_sin_cache,positions,rope_dim,is_neox_style):
  assert x.dtype==torch.float32;d=rope_dim//2;c=cos_sin_cache[positions.long(),:d,None].transpose(1,2);s=cos_sin_cache[positions.long(),d:,None].transpose(1,2)
  if is_neox_style:a,b=x[...,:d],x[...,d:rope_dim];r=torch.cat((a*c-b*s,b*c+a*s),-1)
  else:a,b=x[...,:rope_dim:2],x[...,1:rope_dim:2];r=torch.stack((a*c-b*s,b*c+a*s),-1).reshape(*x.shape[:-1],rope_dim)
  return torch.cat((r,x[...,rope_dim:]),-1)
mod.module=lambda name,**kw:(Rope() if name=='rope.py' else Norm(),[]);mod.candidate=lambda fn,*a,**kw:fn
node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='fused_qk_norm_rope');exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),vars(mod));alt,proof=helper.clone_qk(mod);rows=[]
for dtype,wdtype,style,rdim in itertools.product([torch.float16,torch.bfloat16,torch.float32],[torch.bfloat16,torch.float32],['neox','interleaved'],[4,8]):
 qkv=torch.randn(3,32,dtype=dtype);qw=torch.randn(8,dtype=wdtype);kw=torch.randn(8,dtype=wdtype);table=torch.randn(5,rdim,dtype=torch.float32);case=types.SimpleNamespace(params={'num_heads':2,'num_kv_heads':1,'eps':1e-5,'rope_layout':style});old=mod.fused_qk_norm_rope(case);new=alt(case)
 for pos in [torch.tensor([1,0,2]),torch.tensor([3,2,4])]:
  inputs=[qkv,qw,kw,table,pos];before=[x.clone() for x in inputs];checks.clear();x=old(*inputs);assert checks==['old','old'];checks.clear();y=new(*inputs);assert checks==['new','new'];assert torch.equal(x,y) and torch.equal(y.reshape(3,4,8)[:,3],qkv.reshape(3,4,8)[:,3]) and all(torch.equal(a,b) for a,b in zip(inputs,before));rows.append({'dtype':str(dtype),'weightdtype':str(wdtype),'layout':style,'rope_dim':rdim,'positions':pos.tolist(),'bitwise_equal':True,'FP32intermediate_Vcopy_inputs_preserved':True})
out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':len(rows),'rows':rows,'proof':proof,'CPU_surrogate_only_not_native_acceptance':True,'no_NPU_imported':not any(x.startswith('torch_npu') for x in sys.modules),'helper_sha256':hashlib.sha256(pathlib.Path(helper.__file__).read_bytes()).hexdigest()};assert out['no_NPU_imported'];(P/'VLLM_FUSEDQK_RMS_CPU_PROOF.json').write_text(json.dumps(out,indent=2));print('CPU_SUPPLIED_TABLE_QKV_CASES',len(rows))
