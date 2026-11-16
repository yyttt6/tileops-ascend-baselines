"""CPU-only original-reference thread stability; no oracle repair/native measurement."""
import os,sys,pathlib,importlib.abc,json,pickle,hashlib,datetime,time,gc
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU diagnosis forbids '+name,name=name)
sys.meta_path.insert(0,NoNPU());P=pathlib.Path(__file__).resolve().parent;T=P/'workspace/TileOPs';sys.path[:0]=[str(P),str(T/'src'),str(T)];sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
import torch
from workloads.device import set_run_device
set_run_device('cpu');sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();tensor_sha=lambda x:hashlib.sha256(memoryview(x.contiguous().view(torch.uint8).numpy())).hexdigest();scope=[240,241,243,244,246,247,270,271,272,273,274,276,277,278,279,280,281,282,284,294,295,296,297,298];start=time.time();rows=[];out=P/'MATH_REFERENCE_CPU_THREAD_STABILITY.json';assert not out.exists()
for i in scope:
 d=P/'prepared-rollout'/str(i);m=json.loads((d/'prepare.json').read_text());assert sha(d/'payload.pt')==m['payload_sha256'] and sha(d/'case.pkl')==m['descriptor_sha256'];saved=torch.load(d/'payload.pt',weights_only=False);args=saved['inputs'];case=pickle.load((d/'case.pkl').open('rb'));case.__dict__['inputs']=args
 if hasattr(case.workload,'tensors'):
  for name,value in zip(case._call.signature.inputs,args,strict=True):case.workload.tensors[name]=value
 case.__dict__.pop('verification',None);digests={}
 for threads in [8,4]:
  torch.set_num_threads(threads);gold=case.reference(*args);digests[str(threads)]=tensor_sha(gold);del gold;gc.collect()
 r={'ordinal':i,'payload_sha256':m['payload_sha256'],'descriptor_sha256':m['descriptor_sha256'],'reference_tensor_sha256_by_threads':digests,'reference8_and_formal4_bitwise_equal':digests['8']==digests['4']}
 if i==240:
  diag=P/'runs/rollout-math-bmm-transformer-alternative-240-h1-v1-004532025004/240/CHECKER_DIAGNOSTIC.pt';x=torch.load(diag,weights_only=False);r['retained_actual_formal_expected_sha256']=tensor_sha(x['expected']);r['formal4_replays_actual240_expected_bitwise']=r['retained_actual_formal_expected_sha256']==digests['4'];del x
 rows.append(r);out.write_text(json.dumps({'checked_at_remote':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':scope,'rows':rows,'seconds':time.time()-start,'CPU_only_no_NPU_import':True,'source_sha256':sha(__file__),'no_reference_checker_kernel_precision_changes':True},indent=2));print(json.dumps(r),flush=True);del saved,args,case;gc.collect()
