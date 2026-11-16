"""Original CPU reference diagnostic only; no reference repair or NPU imports."""
import os,sys,pathlib,importlib.abc,json,pickle,hashlib,datetime,time
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU diagnosis forbids '+name,name=name)
sys.meta_path.insert(0,NoNPU());P=pathlib.Path(__file__).resolve().parent;T=P/'workspace/TileOPs';sys.path[:0]=[str(P),str(T/'src'),str(T)];sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
import torch
from workloads.device import set_run_device
set_run_device('cpu');torch.set_num_threads(8);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();start=time.time();prep=P/'prepared-rollout/271';m=json.loads((prep/'prepare.json').read_text());assert sha(prep/'payload.pt')==m['payload_sha256'] and sha(prep/'case.pkl')==m['descriptor_sha256'];saved=torch.load(prep/'payload.pt',weights_only=False);args=saved['inputs'];case=pickle.load((prep/'case.pkl').open('rb'));case.__dict__['inputs']=args
for name,value in zip(case._call.signature.inputs,args,strict=True):case.workload.tensors[name]=value
case.__dict__.pop('verification',None);gold=case.reference(*args);a,b=args;p=m['params'];a=a.T if p['trans_a'] else a;b=b.T if p['trans_b'] else b;assert gold.shape==(a.shape[0],b.shape[1]);rhs=b.double();badcount=0;maximum=0.;first=None;tol=0.001
for left in range(0,a.shape[0],128):
 stop=min(left+128,a.shape[0]);truth=(a[left:stop].double()@rhs).to(gold.dtype);value=gold[left:stop];diff=(value.float()-truth.float()).abs();bad=diff>tol+tol*truth.float().abs();badcount+=int(bad.sum());maximum=max(maximum,float(diff.max()))
 if bad.any() and first is None:
  xy=torch.nonzero(bad)[0].tolist();first={'row':left+xy[0],'column':xy[1],'original':float(value[tuple(xy)]),'FP64_rounded':float(truth[tuple(xy)])}
out={'checked_at_remote':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ordinal':271,'params':m['params'],'payload_sha256':m['payload_sha256'],'descriptor_sha256':m['descriptor_sha256'],'original_policy':m['reference_policy'],'original_CPU_reference_vs_FP64':{'mismatched':badcount,'elements':gold.numel(),'max_absolute_difference':maximum,'first_bad':first,'atol':tol,'rtol':tol},'CPU_only_no_NPU_import':True,'seconds':time.time()-start,'diagnostic_only_not_native_acceptance_or_reference_repair':True,'source_sha256':sha(__file__)};f=P/'MATH_GEMM271_REFERENCE_CPU_DIAGNOSIS.json';assert not f.exists();f.write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
