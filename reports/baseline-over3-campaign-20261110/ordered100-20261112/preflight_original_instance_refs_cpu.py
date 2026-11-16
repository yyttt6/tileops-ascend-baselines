"""CPU reference diagnosis across unvisited InstanceNorm cases; no policy change."""
from pathlib import Path
import os,sys,importlib.abc,json,hashlib,datetime,pickle
P=Path(__file__).resolve().parent;os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU diagnostic forbids '+name,name=name)
sys.meta_path.insert(0,NoNPU());sys.path[:0]=[str(P),str(P/'workspace/TileOPs'),str(P/'workspace/TileOPs/src')]
import torch
import torch.nn.functional as F
from workloads.device import set_run_device
from workloads.numerics import reference_tolerance
set_run_device('cpu');torch.set_num_threads(8);rows=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(a,b,tol):return {'mismatched':int((~torch.isclose(a,b,**tol,equal_nan=True)).sum()),'numel':a.numel(),'max_abs':float((a.double()-b.double()).abs().max())}
for i in range(309,316):
 prep=P/'prepared-rollout'/str(i)
 if not (prep/'prepare.json').exists():rows.append({'ordinal':i,'status':'missing_original_cached_input_no_devicecall'});continue
 meta=json.loads((prep/'prepare.json').read_text());assert sha(prep/'payload.pt')==meta['payload_sha256'];data=torch.load(prep/'payload.pt',weights_only=False,map_location='cpu');case=pickle.load((prep/'case.pkl').open('rb'));args=data['inputs'];x,rm,rv,w,b=args;assert rm is None and rv is None and case.params.get('use_input_stats',True);eps=case.params.get('eps');eps=torch.finfo(torch.float32).eps if eps is None else eps
 original=case.workload.ref_program(*args);z=x.double();axes=tuple(range(2,z.ndim));m=z.mean(axes,keepdim=True);v=(z-m).square().mean(axes,keepdim=True);shape=(1,x.shape[1])+(1,)*(x.ndim-2);g=(z-m)*torch.rsqrt(v+eps)
 if w is not None:g=g*w.double().reshape(shape)
 if b is not None:g=g+b.double().reshape(shape)
 f64=F.instance_norm(z,None,None,None if w is None else w.double(),None if b is None else b.double(),True,case.params.get('momentum',0.1),eps)
 fp32=F.instance_norm(x.float(),None,None,None if w is None else w.float(),None if b is None else b.float(),True,case.params.get('momentum',0.1),eps).to(x.dtype);tol=reference_tolerance(x.dtype)
 row={'ordinal':i,'key':meta['key'],'shape':list(x.shape),'dtype':str(x.dtype),'payload_sha256':meta['payload_sha256'],'tol':tol,'formula_FP64_vs_F_instance_norm_FP64':stats(g,f64,{'atol':1e-12,'rtol':1e-12}),'original_vs_FP64_cast_original_dtype':stats(original,g.to(x.dtype),tol),'FP32_upcast_vs_FP64_cast_original_dtype':stats(fp32,g.to(x.dtype),tol),'status':'CPU_diagnostic_only_reference_unchanged'};assert row['formula_FP64_vs_F_instance_norm_FP64']['mismatched']==0;rows.append(row);print(i,row['original_vs_FP64_cast_original_dtype'],flush=True)
report={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only_no_nativecall':True,'no_NPU_imported':not any(k.split('.')[0] in ('torch_npu','acl') for k in sys.modules),'reference_policy_unchanged':True,'not_acceptance':True,'rows':rows,'script_sha256':sha(Path(__file__))};(P/'server2-alignment-20261113/ORIGINAL_INSTANCE_REFERENCE_CPU_PREFLIGHT.json').write_text(json.dumps(report,indent=2)+'\n')
