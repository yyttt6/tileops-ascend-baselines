from pathlib import Path
import os,sys,json,pickle,hashlib,datetime,importlib.abc,gc
P=Path('/opt/baseline-over3-20261110/ordered100-20261112');T=P/'workspace/TileOPs';N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-gemm-pad4-v1');sys.path[:0]=[str(P),str(T/'src'),str(T)]
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPUdiagnosis forbids '+name,name=name)
sys.meta_path.insert(0,NoNPU());import torch
from workloads.device import set_run_device
set_run_device('cpu');torch.set_num_threads(4);out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'original_tolerance':{'atol':.016,'rtol':.016},'no_reference_or_checker_change':True,'not_formal_acceptance':True,'rows':[]};sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for i in [496,490,494]:
 prep=P/'prepared-rollout'/str(i);meta=json.loads((prep/'prepare.json').read_text());assert sha(prep/'payload.pt')==meta['payload_sha256'];case=pickle.load((prep/'case.pkl').open('rb'));data=torch.load(prep/'payload.pt',weights_only=False,map_location='cpu');xs=data['inputs'];case.__dict__['inputs']=xs
 if hasattr(case.workload,'tensors'):
  for name,x in zip(case._call.signature.inputs,xs,strict=True):case.workload.tensors[name]=x
 p=case.params;a,w=xs;aa=a.T if p['trans_a'] else a;bb=w.T if p['trans_b'] else w;truth=(aa.double()@bb.double());expected=case.reference(*xs);truth=truth.to(expected.dtype)
 def assess(x):
  t=truth.to(x.dtype);ok=torch.isclose(x.float(),t.float(),atol=.016,rtol=.016,equal_nan=True);return {'failures':int((~ok).sum()),'elements':x.numel(),'maxabs':float((x.float()-t.float()).abs().max()),'dtype':str(x.dtype)}
 row={'ordinal':i,'key':meta['key'],'payload_sha256':meta['payload_sha256'],'params':p,'reference_vs_full_FP64':assess(expected)}
 if i==496:
  file=P/'runs'/'rollout-ops-nn-gemm-pad4-496-l1-v1-200213557964'/str(i)/'CHECKER_DIAGNOSTIC.pt';d=torch.load(file,weights_only=False,map_location='cpu');assert d['produced'] is not None;row.update(saved_native_vs_full_FP64=assess(d['produced']),saved_checker_reference_vs_full_FP64=assess(d['expected']),saved_diagnostic_sha256=sha(file));torch.save({'native':d['produced'],'saved_expected':d['expected'],'full_FP64_cast':truth},N/'PAD4_496_CPU_FULL_DIAGNOSIS.pt')
 out['rows'].append(row);print('CPU_FP64_DIAGNOSIS',i,row,flush=True);del case,data,xs,a,w,aa,bb,expected,truth;gc.collect()
assert 'torch_npu' not in sys.modules and 'acl' not in sys.modules;out['status']='complete_CPU_diagnosis_only';(N/'PAD4_BF16_CPU_DIAGNOSIS.json').write_text(json.dumps(out,indent=2,default=str)+'\n')
