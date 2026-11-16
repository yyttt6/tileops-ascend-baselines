import pathlib,sys,os,json,pickle,hashlib,datetime,importlib.abc,gc
P=pathlib.Path('/opt/baseline-over3-20261110/ordered100-20261112');T=P/'workspace/TileOPs';N=pathlib.Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-gemm10-v1');sys.path[:0]=[str(P),str(T/'src'),str(T)]
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPUreference preflight forbids '+name,name=name)
sys.meta_path.insert(0,NoNPU());import torch
from workloads.device import set_run_device
set_run_device('cpu');torch.set_num_threads(4);ids=[488,487,489,491,492,493,495,498];out={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'running','rows':[],'not_device_acceptance_or_reference_change':True,'CPU_threads':4,'placement_matches_canonical_cpu_reference_min4':True}
def save():
 p=N/'GEMM10_REMAINING_REFERENCE_PREFLIGHT_THREADS4.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(out,indent=2,default=str)+'\n');t.replace(p)
save()
try:
 for i in ids:
  prep=P/'prepared-rollout'/str(i);case=pickle.load((prep/'case.pkl').open('rb'));data=torch.load(prep/'payload.pt',weights_only=False);xs=data['inputs'];case.__dict__['inputs']=xs
  if hasattr(case.workload,'tensors'):
   for name,x in zip(case._call.signature.inputs,xs,strict=True):case.workload.tensors[name]=x
  p=case.params;a,w=xs;aa=a.T if p['trans_a'] else a;bb=w.T if p['trans_b'] else w;expected=case.reference(*xs);truth=(aa.double()@bb.double()).to(expected.dtype);ok=torch.isclose(expected,truth,atol=.016,rtol=.016,equal_nan=True);bad=int((~ok).sum());out['rows'].append({'ordinal':i,'key':['ops-nn','GemmFwdOp',case.id],'params':p,'elements':expected.numel(),'full_FP64_original_tolerance_failures':bad,'maxabs':float((expected.float()-truth.float()).abs().max()),'payload_sha256':hashlib.sha256((prep/'payload.pt').read_bytes()).hexdigest(),'classification':'reference_pending_no_device_attempt' if bad else 'CPU_reference_preflight_consistent_native_still_unverified','checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat()});save();print('CPU_REFERENCE_PREFLIGHT',i,bad,expected.numel(),flush=True);del case,data,xs,a,w,aa,bb,expected,truth,ok;gc.collect()
 assert 'torch_npu' not in sys.modules;out['status']='all8_full_CPU_reference_preflight_complete'
except BaseException as e:out.update(status='CPU_preflight_failed_requires_diagnosis',error=repr(e));raise
finally:out['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
