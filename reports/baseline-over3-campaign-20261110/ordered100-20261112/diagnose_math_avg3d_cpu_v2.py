"""CPU-only exact frozen reference availability and higherprecision diagnosis; no oracle activation."""
from pathlib import Path
import os,sys,importlib.abc,json,pickle,hashlib,datetime
P=Path(__file__).resolve().parent;sys.path[:0]=[str(P),str(P/'workspace/TileOPs/src'),str(P/'workspace/TileOPs')];os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPUdiagnosis forbids '+name,name=name)
sys.meta_path.insert(0,NoNPU())
import torch
from workloads.device import set_run_device
from workloads.numerics import reference_tolerance
set_run_device('cpu');torch.set_num_threads(4);rows=[]
for i in [224,225]:
 p=P/'prepared-rollout'/str(i);m=json.loads((p/'prepare.json').read_text());case=pickle.load((p/'case.pkl').open('rb'));saved=torch.load(p/'payload.pt',weights_only=False);args=saved['inputs'];assert all(x.device.type=='cpu' for x in args if isinstance(x,torch.Tensor));case.__dict__['inputs']=args
 if hasattr(case.workload,'tensors'):
  for name,x in zip(case._call.signature.inputs,args,strict=True):case.workload.tensors[name]=x
 case.__dict__.pop('verification',None);ref=case.reference;r={'ordinal':i,'key':m['key'],'payload_sha256':hashlib.sha256((p/'payload.pt').read_bytes()).hexdigest(),'reference_policy':m['reference_policy'],'params':m['params']};assert r['payload_sha256']==m['payload_sha256']
 try:
  ref(*args);r['original_CPU_reference']='passed'
 except Exception as e:r.update(original_CPU_reference='failed',exception_type=type(e).__name__,error=str(e))
 r.update(actual_input_dtypes=[str(x.dtype) for x in args if isinstance(x,torch.Tensor)],reference_repr=str(ref));rows.append(r);print(json.dumps(r),flush=True)

result={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_NPU_imported':True,'rows':rows,'oracle_unchanged':True,'proposal_not_activated':True};(P/'MATH_AVG3D_CPU_REFERENCE_DIAGNOSIS.json').write_text(json.dumps(result,indent=2)+'\n');print('AVG3D_FROZEN_REFERENCE_AVAILABILITY_CAPTURED',len(rows))
