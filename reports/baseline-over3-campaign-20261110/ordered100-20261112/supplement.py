"""Original canonical checker on SAME stored inputs; correctness only, no profiling."""
import pathlib,sys,os,json,hashlib,pickle,copy,importlib.util,types,time,gc,dataclasses
P=pathlib.Path(__file__).resolve().parent;W=P/'workspace';T=W/'TileOPs';B=pathlib.Path('/opt/torch-all2208-20261105');tag=sys.argv[1];indices=list(map(int,sys.argv[2:]));assert 0<len(indices)<=8
sys.path[:0]=[str(P),str(T/'src'),str(T),str(W)];sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
import torch,torch_npu
from benchmarks import api
from benchmarks.ascend_all_pairs import registry
from benchmarks.ascend_baselines import Candidate
from workloads.device import set_run_device
from reference_overrides import corrected_reference
# Template entry installs the exact captured ATVC adapter; original DeepGEMM
# fresh-process adapter is selected below, not a different extension/kernel.
rows=json.loads((P/'TASKS.json').read_text());lib=rows[indices[0]]['library'];assert all(rows[i]['library']==lib for i in indices)
torch.set_num_threads(8);torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
O=P/'runs'/tag;O.mkdir(parents=True,exist_ok=False)
spec=importlib.util.spec_from_file_location('canonical_original_checker',T/'scripts/cache_frozen_pair_check.py');checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker);checker.require_guard()
original_cases=api.cases;original_factory=registry.candidate_factory;reuse={r['ordinal']:r for r in json.loads((P/'ATVC_REUSE_REVIEW.json').read_text())['rows']};oldatvc={tuple(r['key']):r for r in json.loads((P/'ATVC_EXISTING_INDEX.json').read_text())};selected={x['ordinal']:x for x in json.loads((P/'SELECTION.json').read_text())};templates={};result=[];started=time.perf_counter()
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def cv(x,d):
 if isinstance(x,torch.Tensor):return x.to(d)
 if isinstance(x,tuple):return tuple(cv(v,d) for v in x)
 if isinstance(x,list):return [cv(v,d) for v in x]
 if isinstance(x,dict):return {k:cv(v,d) for k,v in x.items()}
 return x
for i in indices:
 r=rows[i];key=(lib,r['op'],r['case']);set_run_device('cpu')
 if i in reuse:
  old=oldatvc[key];prep=P.parent/'atvc-server1/runs'/str(old['index'])
 else:prep=P/'prepared'/str(i)
 meta=json.loads((prep/'prepare.json').read_text());assert sha(prep/'payload.pt')==meta['payload_sha256'];saved=torch.load(prep/'payload.pt',weights_only=False)
 if r['op'] not in templates:templates[r['op']]=list(original_cases(r['op']))
 case=copy.copy(next(c for c in templates[r['op']] if c.id==r['case']));case.__dict__['inputs']=saved['inputs'];workload=case.workload
 args=cv(saved['inputs'],'npu');case.__dict__['inputs']=args
 if hasattr(workload,'tensors'):
  for name,value in zip(case._call.signature.inputs,args,strict=True):workload.tensors[name]=value
 case.__dict__.pop('verification',None);set_run_device('npu')
 correction=corrected_reference(r['op'],r['case'])
 if correction is not None:case.__dict__['reference']=checker.cpu_reference(torch,correction,args)
 api.cases=lambda op:[case] if op==r['op'] else []
 def factory(current,family):
  def build():
   if family=='DeepGEMM-Ascend':
    import deepgemm_adapter
    fn,proof=deepgemm_adapter.factory(r['op'],case.params)
    return Candidate(api.Implementation(fn),'deep_gemm._C.bf16_bmm','same original guarded fresh-process API as measured',proof)
   candidate=original_factory(current,family)()
   if selected.get(i,{}).get('selection')=='zero_pad_adapter':
    from benchmarks.ascend_all_pairs.tilekernels import dequant
    from benchmarks.ascend_all_pairs.tilekernels.zero_pad_candidate import clone_factory
    make,proof=clone_factory(dequant);candidate=make(current)
   return candidate
  return build
 registry.candidate_factory=factory
 dest=O/('check-'+str(i)+'.json');sys.argv=[str(T/'scripts/cache_frozen_pair_check.py'),'--op',r['op'],'--case-id',r['case'],'--library',lib,'--output',str(dest)]
 rc=checker.main();receipt=json.loads(dest.read_text())
 result.append({'ordinal':i,'key':key,'status':receipt.get('status'),'exit_code':rc,'receipt_sha256':sha(dest),'payload_sha256':meta['payload_sha256'],'original_manifest_and_input_values_retained':True,'reference_correction':bool(correction),'performance_windows':0,'source':'original canonical cache_frozen_pair_check.py; no profile'})
 print('CHECK',i,receipt.get('status'),receipt.get('error','')[:160],flush=True)
 del case,workload,args,saved;api.cases=original_cases;registry.candidate_factory=original_factory;torch.npu.synchronize();gc.collect();torch.npu.empty_cache()
(O/'BATCH.json').write_text(json.dumps({'tag':tag,'requested':indices,'results':result,'runtime':{'torch':torch.__version__,'torch_npu':torch_npu.__version__},'process_seconds':time.perf_counter()-started,'performance_measurement':False},indent=2))

# Actual loaded runtime identities for dependency-aware reuse of prior timings.
maps=pathlib.Path("/proc/self/maps").read_text();(O/"PROCESS_MAPS.txt").write_text(maps)
libs={line.split()[-1] for line in maps.splitlines() if len(line.split())>=6 and line.split()[-1].startswith("/") and ".so" in line.split()[-1]}
(O/"LOADED_LIBRARIES.json").write_text(json.dumps({x:sha(x) for x in sorted(libs) if pathlib.Path(x).is_file()},indent=2))
