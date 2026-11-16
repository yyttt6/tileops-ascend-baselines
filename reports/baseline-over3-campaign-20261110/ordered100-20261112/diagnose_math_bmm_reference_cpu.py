"""CPU-only diagnosis; never changes accepted oracle/checker or performs NPU work."""
import os,sys,pathlib,importlib.abc,json,pickle,datetime,hashlib,time
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU diagnosis forbids '+name,name=name)
sys.meta_path.insert(0,NoNPU());P=pathlib.Path(__file__).resolve().parent;T=P/'workspace/TileOPs';sys.path[:0]=[str(P),str(T/'src'),str(T)];sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
import torch
from workloads.device import set_run_device
set_run_device('cpu');torch.set_num_threads(8);sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();rows=[];start=time.time()
failed='rollout-math-bmm-transformer-alternative-240-h1-v1-004532025004'
for i in [240,241,243,244,246,247]:
 prep=P/'prepared-rollout'/str(i);m=json.loads((prep/'prepare.json').read_text());assert m['reference_policy']=='original_cpu';assert sha(prep/'payload.pt')==m['payload_sha256'] and sha(prep/'case.pkl')==m['descriptor_sha256'];payload=torch.load(prep/'payload.pt',weights_only=False);a,b=payload['inputs'];case=pickle.load((prep/'case.pkl').open('rb'));case.__dict__['inputs']=(a,b)
 if hasattr(case.workload,'tensors'):
  for name,value in zip(case._call.signature.inputs,(a,b),strict=True):case.workload.tensors[name]=value
 case.__dict__.pop('verification',None)
 if i==240:
  p=P/'runs'/failed/str(i)/'CHECKER_DIAGNOSTIC.pt';d=torch.load(p,weights_only=False);gold=d['expected'];native=d['produced'];assert native is not None and gold is not None;dsha=sha(p)
 else:gold=case.reference(a,b);native=None;dsha=None
 assert tuple(gold.shape)==(a.shape[0],a.shape[1],b.shape[2]);tol=0.016 if gold.dtype==torch.bfloat16 else 0.001;tot=gold.numel();stats={'original_CPU_reference_vs_FP64':{'mismatched':0,'elements':tot,'max_absolute_difference':0.0,'first_bad':None}}
 if native is not None:stats['retained_native_vs_FP64']={'mismatched':0,'elements':tot,'max_absolute_difference':0.0,'first_bad':None}
 for batch in range(a.shape[0]):
  rhs=b[batch].double()
  for left in range(0,a.shape[1],128):
   stop=min(left+128,a.shape[1]);truth=(a[batch,left:stop].double()@rhs).to(gold.dtype)
   for label,value in [('original_CPU_reference_vs_FP64',gold),('retained_native_vs_FP64',native)]:
    if value is None:continue
    v=value[batch,left:stop];diff=(v.float()-truth.float()).abs();bad=diff>tol+tol*truth.float().abs();s=stats[label];s['mismatched']+=int(bad.sum());s['max_absolute_difference']=max(s['max_absolute_difference'],float(diff.max()))
    if bad.any() and s['first_bad'] is None:
     xy=torch.nonzero(bad)[0].tolist();s['first_bad']={'batch':batch,'row':left+xy[0],'column':xy[1],'original':float(v[tuple(xy)]),'FP64_rounded':float(truth[tuple(xy)])}
  del rhs
 r={'ordinal':i,'params':m['params'],'payload_sha256':m['payload_sha256'],'descriptor_sha256':m['descriptor_sha256'],'diagnostic_output_sha256':dsha,'original_tolerance':{'atol':tol,'rtol':tol},'comparisons':stats,'original_reference_cpu_source':str(case.reference.__code__.co_filename),'diagnostic_only_not_authorized_reference_repair':True};rows.append(r);print(json.dumps({'ordinal':i,'comparisons':stats}),flush=True);del a,b,gold,native,case,payload
out={'checked_at_remote':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seconds':time.time()-start,'CPU_only_no_NPU_import':True,'source_sha256':sha(__file__),'failed240_checkpoint':failed,'rows':rows,'no_reference_kernel_checker_precision_change':True};path=P/'MATH_BMM_REFERENCE_CPU_DIAGNOSIS_20261116.json';assert not path.exists();path.write_text(json.dumps(out,indent=2))
