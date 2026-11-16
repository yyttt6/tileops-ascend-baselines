"""Bounded unchanged-provider batch. CPU gold outside guard; all device imports inside."""
import sys,os,pathlib,json,hashlib,time,dataclasses,gc,ctypes,pickle,copy,traceback,importlib.abc,fcntl
START=time.perf_counter();P=pathlib.Path(__file__).resolve().parent;W=P/'workspace';T=W/'TileOPs';B=pathlib.Path('/opt/torch-all2208-20261105')
mode,tag=sys.argv[1:3];indices=list(map(int,sys.argv[3:]));assert 0<len(indices)<=8 and len(set(indices))==len(indices)
rows=json.loads((P/'TASKS.json').read_text());lib=rows[indices[0]]['library'];assert all(rows[i]['library']==lib for i in indices)
sys.path[:0]=[str(P),str(T/'src'),str(T),str(B)]
sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
if lib=='DeepGEMM-Ascend':
 os.environ['ASCEND_HOME_PATH']='/opt/tileops-all-pairs/workspace/.cache/all-pairs/remote_native_deepgemm_deepep/sdk-920-release-headers'
 os.environ['ASCEND_TOOLKIT_HOME']=os.environ['ASCEND_HOME_PATH'];os.environ['ALLPAIRS_DEEPGEMM_BUILD_ROOT']='/opt/tileops-all-pairs/workspace/.cache/all-pairs/deepgemm/torch210-pristine'
if mode=='prepare':
 class NoNPU(importlib.abc.MetaPathFinder):
  def find_spec(self,fullname,path=None,target=None):
   if fullname.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU gold forbids '+fullname,name=fullname)
 sys.meta_path.insert(0,NoNPU())
else:
 with open('/run/lock/ascend-npu/device-0.lock') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:pass
  else:raise RuntimeError('host guard required')
import torch
from benchmarks.api import cases
from workloads.device import set_run_device
from workloads.numerics import compare_outputs,describe
from reference_correction_registry import apply as correct_reference

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,default=str));os.replace(tmp,p)
def cv(x,device):
 if isinstance(x,torch.Tensor):return x.to(device)
 if isinstance(x,tuple):return tuple(cv(v,device) for v in x)
 if isinstance(x,list):return [cv(v,device) for v in x]
 if isinstance(x,dict):return {k:cv(v,device) for k,v in x.items()}
 return x
source={str(p):sha(p) for folder in [P] for p in folder.rglob('*.py') if not any(s in p.parts for s in ['runs','prepared','__pycache__'])}
# B tools are the frozen full32 parser/timer/flush implementation.
source.update({str(p):sha(p) for p in (B/'tools').glob('*.py')})
torch.set_num_threads(8);manifest_cache={}
if mode=='prepare':
 set_run_device('cpu')
 for i in indices:
  r=rows[i];out=P/'prepared'/str(i)
  if (out/'prepare.json').exists():continue
  torch.manual_seed(0);t=time.perf_counter()
  if r['op'] not in manifest_cache:manifest_cache[r['op']]=list(cases(r['op']))
  case=next(c for c in manifest_cache[r['op']] if c.id==r['case']);out.mkdir(parents=True,exist_ok=True)
  with (out/'case.pkl').open('wb') as f:pickle.dump(case,f,protocol=5)
  correction=correct_reference(case);xs=case.inputs;expected=case.reference(*copy.deepcopy(xs));controls=[]
  for ctl in getattr(case.verification,'controls',()):
   try:compare_outputs(ctl.run(case.reference,copy.deepcopy(xs)),expected,case.verification)
   except AssertionError:controls.append({'name':ctl.name,'rejected':True})
   else:raise AssertionError('negative control accepted '+ctl.name)
  torch.save({'inputs':xs,'expected':expected},out/'payload.pt')
  put(out/'prepare.json',{'key':[r['library'],r['op'],r['case']],'params':case.params,'verification':describe(case.verification),'controls':controls,'correction':correction,'seed':0,'torch':torch.__version__,'payload_sha256':sha(out/'payload.pt'),'descriptor_sha256':sha(out/'case.pkl'),'source':source,'seconds':time.perf_counter()-t})
  print('PREPARED',i,r['case'],time.perf_counter()-t,flush=True);del xs,expected,case;gc.collect()
 sys.exit(0)
import torch_npu
from tools import kernel_task_profile_full32 as parser
sys.modules['tools.kernel_task_profile']=parser
from tools import measure_baselines as core
core.ROOT=B;core.TILEOPS_ROOT=T
from restricted_profiler.capture import make_timer
from restricted_profiler.partition import partition
set_run_device('npu');torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
prop=torch.npu.get_device_properties(0);assert (prop.cube_core_num,prop.vector_core_num)==(32,64)
runtime={'torch':torch.__version__,'torch_npu':torch_npu.__version__,'cube':prop.cube_core_num,'vector':prop.vector_core_num,'ASCEND_HOME_PATH':os.environ.get('ASCEND_HOME_PATH'),'tilelang':None}
O=P/'runs'/tag;O.mkdir(parents=True,exist_ok=False);results=[]
fd=B/'flush';fm=json.loads((fd/'SOURCE.json').read_text());assert sha(fd/'host_kernel.asc')==fm['generated_source_sha256'] and sha(fd/'flush.aibin')==fm['device_binary_sha256']
dll=ctypes.CDLL(str(fd/'flush.so'));dll.init.argtypes=[ctypes.c_char_p];dll.init.restype=ctypes.c_int;dll.call.argtypes=[ctypes.c_void_p,ctypes.c_void_p];dll.call.restype=ctypes.c_int;assert dll.init(str(fd/'flush.aibin').encode())==0
def flush(buffer):
 if dll.call(buffer.data_ptr(),torch.npu.current_stream().npu_stream):raise RuntimeError('flush failure')
buffer=torch.ones(268435456,dtype=torch.int32,device='npu');flush(buffer);torch.npu.synchronize();assert torch.count_nonzero(buffer).item()==0;core._flush_kernel_cache[0]=(flush,buffer)
initialization=time.perf_counter()-START
for i in indices:
 r=rows[i];D=O/str(i);D.mkdir();stage='load';started=time.perf_counter()
 try:
  if (P/'STOP').exists():raise RuntimeError('STOP')
  prep=P/'prepared'/str(i);meta=json.loads((prep/'prepare.json').read_text());assert meta['torch']==torch.__version__
  assert all(sha(k)==v for k,v in meta['source'].items())
  assert sha(prep/'payload.pt')==meta['payload_sha256'] and sha(prep/'case.pkl')==meta['descriptor_sha256']
  with (prep/'case.pkl').open('rb') as f:case=pickle.load(f)
  correct_reference(case);saved=torch.load(prep/'payload.pt',weights_only=False);case.__dict__['inputs']=saved['inputs'];assert case.params==meta['params'] or json.loads(json.dumps(case.params))==meta['params']
  args=cv(saved['inputs'],'npu');stage='factory'
  if lib=='DeepGEMM-Ascend':
   import deepgemm_adapter
   fn,ns=deepgemm_adapter.factory(r['op'],case.params);api='deep_gemm._C.bf16_bmm';metric='device_async';reset=None
  else:
   from benchmarks.ascend_all_pairs.registry import candidate_factory
   import types
   proxy=types.SimpleNamespace(inputs=args,params=case.params,_call=case._call)
   candidate=candidate_factory(proxy,lib)();fn=candidate.implementation.run;api=candidate.api;metric=candidate.timing_mode;reset=getattr(candidate.implementation,'reset',None)
   ns={str(W/k) if not pathlib.Path(k).is_absolute() else k:v for k,v in candidate.source_sha256.items()}
  if lib=='TileKernels':
   from benchmarks.ascend_all_pairs.tilekernels.zero_pad_candidate import factory as pad_factory
   fn,pad_proof=pad_factory(proxy);api+=' / skip zero-tail adapter copy'
   put(D/'CANDIDATE_PROOF.json',pad_proof)
  assert all(sha(k)==v for k,v in ns.items());source.update(ns)
  before=core._tensor_digest(args);counts={'calls':0};last=[];wall=[]
  def measured(*values):
   t=time.perf_counter();value=fn(*values);torch.npu.synchronize();wall.append((time.perf_counter()-t)*1000);counts['calls']+=1
   if counts['calls']==19:last.append(value)
   return value
  stage='measurement';os.environ['TILEOPS_PROFILE_DIR']=str(D/'profiles');t=time.perf_counter()
  # Host/reset paths retain original timer before executing any session work.
  use_session=mode!='isolated' and metric=='device_async' and reset is None
  timing=(make_timer(D/'session') if use_session else core._time_reference)(torch,torch_npu,measured,args,3,15,reset)
  measurement_seconds=time.perf_counter()-t
  assert counts['calls']==19 and len(last)==1 and len(wall)==19
  stage='correctness';frozen=cv(last.pop(),'cpu');check=compare_outputs(frozen,saved['expected'],case.verification)
  assert check.checked_outputs==check.total_outputs and check.checked_outputs and not check.unchecked_reason
  if getattr(case.verification,'probe',None):case.verification.probe(fn,args)
  assert before==core._tensor_digest(args)
  stage='raw_audit'
  if use_session:
   info=timing['single_session'];assert parser.trace_digest(info['path'])==info['sha256'];pieces,proof=partition(info['path'],15);proof['samples']=pieces;assert proof==json.loads((D/'session/partition.json').read_text())
  else:
   pieces=[]
   for x in timing['retained_profiles']:
    assert parser.trace_digest(x['profile_path'])==x['profile_trace_sha256'];a=parser.parse_profile(x['profile_path'],1);assert a['profile_named_task_coverage_complete'];pieces.append(a)
  assert len(pieces)==15 and all(abs(x['latency_ms']-v)<1e-9 for x,v in zip(pieces,timing['latency_samples_ms']))
  assert abs(sum(x['latency_ms'] for x in pieces)/15-timing['latency_ms'])<1e-9
  result={'status':'raw_checked_pending_guard','ordinal':i,'key':[lib,r['op'],r['case']],'api':api,'metric':metric,'check':dataclasses.asdict(check),'payload_sha256':meta['payload_sha256'],'prepare_sha256':sha(prep/'prepare.json'),'timing':timing,'device_us':timing['latency_ms']*1000,'host_us_diagnostic':sum(wall[-15:])/15*1000,'call_count':counts['calls'],'inputs_preserved':True,'controls':meta['controls'],'measurement_seconds':measurement_seconds,'case_seconds':time.perf_counter()-started,'aicpu_counts':[p['profile_aicpu_task_count'] for p in pieces],'topology':[(v['name'],v['core_roles']) for v in pieces[0]['profile_kernel_tasks']],'sources':ns,'selected':'zero_pad_adapter_candidate','selection_reason':'only zero-tail F.pad copies skipped; identical original native full API, all necessary copies/casts retained; reverse AST proof'}
  put(D/'result.json',result);results.append(result);print('MEASURED',i,result['device_us'],result['case_seconds'],flush=True)
  if lib!="DeepGEMM-Ascend":
   if candidate.cleanup is not None:candidate.cleanup()
   del candidate,proxy
  del args,saved,case,fn,frozen,timing,pieces,last;torch.npu.synchronize();gc.collect();torch.npu.empty_cache()
 except Exception as e:
  if stage=='correctness' and 'frozen' in locals() and 'saved' in locals():
   torch.save({'produced':frozen,'expected':saved['expected']},D/'CHECKER_FAILURE.pt')
  failure={'status':'unsupported' if type(e).__name__=='UnsupportedCase' else 'failed','ordinal':i,'key':[lib,r['op'],r['case']],'phase':stage,'error':repr(e),'traceback':traceback.format_exc(),'seconds':time.perf_counter()-started}
  put(D/'result.json',failure);results.append(failure);print('FAILED',i,stage,repr(e),flush=True)
  # Do not continue a process after an unqualified runtime error. Previous rows
  # only become valid if guard postflight is valid; failed row is never accepted.
  try:torch.npu.synchronize()
  except Exception:raise
  break
assert all(sha(k)==v for k,v in source.items())
if 'tilelang' in sys.modules:runtime['tilelang']={'path':sys.modules['tilelang'].__file__,'version':getattr(sys.modules['tilelang'],'__version__',None)}
put(O/'SOURCE.json',source)
maps=pathlib.Path('/proc/self/maps').read_text();(O/'PROCESS_MAPS.txt').write_text(maps);libs={x.split()[-1] for x in maps.splitlines() if len(x.split())>=6 and x.split()[-1].startswith('/') and '.so' in x.split()[-1]};put(O/'LOADED_LIBRARIES.json',{q:sha(q) for q in sorted(libs) if pathlib.Path(q).is_file()})
put(O/'BATCH.json',{'tag':tag,'requested':indices,'results':results,'runtime':runtime,'process_seconds':time.perf_counter()-START,'initialization_seconds':initialization,'source_sha256':sha(O/'SOURCE.json'),'status':'pending_guard'})
print('BATCH_COMPLETE',tag,len(results),flush=True)
