"""Bounded batch using CI measured-output verification and canonical oracle placement."""
import pathlib,sys,os,json,time,pickle,copy,gc,ctypes,hashlib,dataclasses,importlib.util,types,traceback,fcntl
START=time.perf_counter();P=pathlib.Path(__file__).resolve().parent;W=P/'workspace';T=W/'TileOPs';B=pathlib.Path('/opt/torch-all2208-20261105');tag=sys.argv[1];ids=list(map(int,sys.argv[2:]));assert 0<len(ids)<=8
sys.path[:0]=[str(P),str(T/'src'),str(T)];sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
with open('/run/lock/ascend-npu/device-0.lock') as f:
 try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:pass
 else:raise RuntimeError('whole process host guard required')
# Frozen full32 decoder already counts named AICPU once.
tools=types.ModuleType('tools');tools.__path__=[str(B/'tools'),str(W/'tools')];sys.modules['tools']=tools
# Set originalnative component discovery before anyTorch/NPU import or ELF load.
N=pathlib.Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-fp8quant-v1')
manifest_path=N/'FP8QUANT_NATIVE_PACKAGE.json';native_manifest=json.loads(manifest_path.read_text());qpath=N/'PRIVATE_FP8QUANT_QUALIFICATION.json';qualification=json.loads(qpath.read_text());assert qualification['private_original_device_kernels_verified'] is True
if os.environ.get('BASELINE_NATIVE92_ENV_BOUND')!='1':
 root=native_manifest['staging_root'];dirs=[root+'/op_api/lib',root+'/op_proto/lib/linux/aarch64',root+'/op_impl/ai_core/tbe/op_tiling']
 os.environ['ASCEND_CUSTOM_OPP_PATH']=root;os.environ['LD_LIBRARY_PATH']=':'.join(dirs)+':'+os.environ.get('LD_LIBRARY_PATH','');os.environ['BASELINE_NATIVE92_ENV_BOUND']='1'
 os.execv(sys.executable,[sys.executable,str(P/'rollout_bootstrap.py'),'ops-nn-fp8quant',*sys.argv[1:]])
assert os.environ['ASCEND_CUSTOM_OPP_PATH']==native_manifest['staging_root']
import torch,torch_npu
from benchmarks.api import Implementation
from ops_nn_fp8quant_cached_fidelity import verify_runtime_source_identity
from workloads.device import set_run_device
from workloads.numerics import _copy_outputs,_snapshot,_unique,_require_unchanged
from tools import kernel_task_profile_full32 as parser
sys.modules['tools.kernel_task_profile']=parser
from tools import measure_baselines as core
from benchmarks.ascend_all_pairs.registry import candidate_factory
from reference_overrides import corrected_reference
from reference_policy import policy
from measured_correctness import measure_and_verify
from restricted_profiler.capture import make_timer
from restricted_profiler.partition import partition
core.ROOT=B;core.TILEOPS_ROOT=T;rows=json.loads((P/'TASKS.json').read_text());lib=rows[ids[0]]['library'];assert all(rows[i]['library']==lib for i in ids)
O=P/'runs'/tag;O.mkdir(parents=True,exist_ok=False)
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def put(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,default=str))
def cv(x,d):
 if isinstance(x,torch.Tensor):
  out=x.to(d)
  if out.dtype!=x.dtype or tuple(out.shape)!=tuple(x.shape):raise TypeError('Device transfer changed declared input/output dtype or shape')
  return out
 if isinstance(x,tuple):return tuple(cv(v,d) for v in x)
 if isinstance(x,list):return [cv(v,d) for v in x]
 if isinstance(x,dict):return {k:cv(v,d) for k,v in x.items()}
 return x
spec=importlib.util.spec_from_file_location('canonical_checker_helpers',T/'scripts/cache_frozen_pair_check.py');canonical=importlib.util.module_from_spec(spec);spec.loader.exec_module(canonical);canonical.require_guard()
torch.set_num_threads(8);torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
prop=torch.npu.get_device_properties(0);assert (prop.cube_core_num,prop.vector_core_num)==(32,64)
runtime={'torch':torch.__version__,'torch_npu':torch_npu.__version__,'cube':32,'vector':64,'sdk':os.environ.get('ASCEND_HOME_PATH'),'hostname':os.uname().nodename}
source={str(p):sha(p) for folder in [T/'benchmarks',T/'workloads',P/'restricted_profiler',B/'tools'] for p in folder.rglob('*.py') if '__pycache__' not in p.parts}
for p in [pathlib.Path(__file__),P/'torch_ready_cached_fidelity.py',P/'measured_correctness.py',P/'compact_timing.py',P/'reference_policy.py',T/'scripts/cache_frozen_pair_check.py',P/'reference_overrides.py',P/'TASKS.json']:source[str(p)]=sha(p)
fd=B/'flush';m=json.loads((fd/'SOURCE.json').read_text());assert sha(fd/'host_kernel.asc')==m['generated_source_sha256'] and sha(fd/'flush.aibin')==m['device_binary_sha256']
dll=ctypes.CDLL(str(fd/'flush.so'));dll.init.argtypes=[ctypes.c_char_p];dll.init.restype=ctypes.c_int;dll.call.argtypes=[ctypes.c_void_p,ctypes.c_void_p];dll.call.restype=ctypes.c_int;assert dll.init(str(fd/'flush.aibin').encode())==0
def flush(buf):
 if dll.call(buf.data_ptr(),torch.npu.current_stream().npu_stream):raise RuntimeError('original flush failed')
buf=torch.ones(268435456,dtype=torch.int32,device='npu');flush(buf);torch.npu.synchronize();assert torch.count_nonzero(buf).item()==0;core._flush_kernel_cache[0]=(flush,buf)
active=[False];results=[];initialization=time.perf_counter()-START
class UnsupportedCapturedTopology(RuntimeError):pass
# Reuse the qualified full32 capture byte-for-byte except for an exception
# boundary around partitioning, so only topology failures can trigger fallback.
import restricted_profiler.capture as capture
original_partition=capture.partition
def checked_partition(path,repeats):
 try:
  pieces,proof=original_partition(path,repeats)
  if proof.get('whole_aicpu_count',0):raise ValueError('named AICPU retains original isolated protocol in this rollout')
  return pieces,proof
 except (ValueError,AssertionError) as e:raise UnsupportedCapturedTopology(str(e)) from e
capture.partition=checked_partition
for i in ids:
 r=rows[i];D=O/str(i);D.mkdir();stage='load';beg=time.perf_counter();candidate=None;last=[];diagnostic_reference=[]
 try:
  if (P/'STOP').exists():raise RuntimeError('STOP')
  prep=P/'prepared-rollout'/str(i);meta=json.loads((prep/'prepare.json').read_text());assert meta['key']==[lib,r['op'],r['case']] and meta['torch']==torch.__version__
  assert verify_runtime_source_identity(P,i,meta);assert sha(prep/'payload.pt')==meta['payload_sha256'] and sha(prep/'case.pkl')==meta['descriptor_sha256']
  set_run_device('cpu');case=pickle.load((prep/'case.pkl').open('rb'));saved=torch.load(prep/'payload.pt',weights_only=False);case.__dict__['inputs']=saved['inputs'];workload=case.workload
  args=cv(saved['inputs'],'npu');case.__dict__['inputs']=args
  if hasattr(workload,'tensors'):
   for name,value in zip(case._call.signature.inputs,args,strict=True):workload.tensors[name]=value
  case.__dict__.pop('verification',None);set_run_device('npu');raw_reference=case.reference;stage='factory'
  assert lib=='ops-nn' and r['op'] in ('FP8QuantFwdOp',) and i in {470, 471, 472}, 'Finite original FP8Quant3 scope'
  from ops_nn_fp8quant_api import FP8QuantAPI
  root=pathlib.Path(native_manifest['staging_root']);native=FP8QuantAPI(torch,W,manifest_path,qpath)
  impl=Implementation(native.prepare(case));api='original opsnn_fp8_cases run AST; fullDynamicBlockQuant with FP32widen, samegroup1e-4floorpad, originalblock128/256/rint/dst36, scales/codes/crop alltimed, originalquantizationchecker unchanged';metric='device_async'
  ns={str(P/name):sha(P/name) for name in ['ops_nn_fp8quant_api.py','ops_nn_fp8quant_cached_fidelity.py']};ns[str(manifest_path)]=sha(manifest_path);ns[str(qpath)]=sha(qpath);ns.update({str(root/name):digest['sha256'] for name,digest in native_manifest['all_files'].items()});assert all(sha(path)==digest for path,digest in ns.items());source.update(ns)
  rule=policy(lib,r['op'],r['case']);assert rule==meta['reference_policy']
  if rule=='audited_cpu_fp64':reference=canonical.cpu_reference(torch,corrected_reference(r['op'],r['case']),args)
  elif rule=='original_cpu':reference=canonical.cpu_reference(torch,raw_reference,args)
  else:reference=raw_reference
  def watched_reference(*values):
   out=reference(*values)
   if not diagnostic_reference:diagnostic_reference.append(out)
   return out
  case.__dict__['reference']=watched_reference
  actual_args=args if impl.args is None else impl.args;reset=impl.reset;fn=impl.run;wall=[];children=[];active=[False];calls=[0]
  def audit(event,event_args):
   if active[0] and event in ('subprocess.Popen','os.fork','os.posix_spawn'):children.append(event)
  sys.addaudithook(audit)
  def observed(*values):
   start=time.perf_counter();n=len(children);active[0]=True
   try:out=fn(*values);torch.npu.synchronize()
   finally:active[0]=False
   wall.append({'ms':(time.perf_counter()-start)*1000,'child':len(children)!=n});calls[0]+=1
   if calls[0]==19:last.append(out)
   return out
  checked_impl=dataclasses.replace(impl,run=observed)
  def measure(run,timer,profile_name):
   os.environ['TILEOPS_PROFILE_DIR']=str(D/profile_name);result=timer(torch,torch_npu,run,actual_args,3,15,reset)
   assert calls[0]==19 and len(last)==1 and len(wall)==19 and not any(v['child'] for v in wall[-15:])
   # Freeze outside all timed windows, before any reference/probe.
   frozen=_copy_outputs(last.pop());last.append(frozen);return result
  stage='measurement_and_original_verification';t=time.perf_counter();session=False  # independent original-isolated comparison for both arms
  if rule in ('canonical_cpu_fp64_validator','canonical_cpu_fft'):
   # Qualified original CPU validator route; no CPU implementation is substituted
   # for the measured device baseline. Keep its established isolated protocol.
   timing=measure(observed,core._time_reference,'profiles');frozen=last[0];first=[True]
   def first_frozen_then_real(*xs):
    if first[0]:first[0]=False;return frozen
    return fn(*xs)
   if rule=='canonical_cpu_fp64_validator':checks=canonical.verify_fp64_on_cpu(torch,case,first_frozen_then_real,actual_args,reset,lib,[]);check=dataclasses.asdict(checks[lib])
   else:raise NotImplementedError('canonical FFT CPU-validator route retained pending dedicated adapter; no fake device measurement')
   session=False
  else:
   try:check,timing=measure_and_verify(case,checked_impl,lambda run:measure(run,make_timer(D/'session') if session else core._time_reference,'profiles'),expected_calls=19)
   except UnsupportedCapturedTopology as e:
    put(D/'SESSION_FALLBACK.json',{'reason':str(e),'not_a_numerical_retry':True});calls[0]=0;wall.clear();last.clear();diagnostic_reference.clear();session=False
    check,timing=measure_and_verify(case,checked_impl,lambda run:measure(run,core._time_reference,'fallback-profiles'),expected_calls=19)
  elapsed=time.perf_counter()-t;assert check['checked_outputs'] and check['checked_outputs']==check['total_outputs'] and not check['unchecked_reason']
  stage='original_edge_probes';edge={}
  if r['op']=='RoPEFwdOp':
   edge=canonical.rope_table_diagnostics(torch,case,impl,lib);assert edge['cpu_rotation_with_device_tables']['passed'],'original same-coefficient CPU rotation failed'
  if r['op']=='FP8QuantFwdOp':edge['fp8']=canonical.fp8_edge_check(torch,case,impl)
  stage='raw_audit'
  if session:
   info=timing['single_session'];assert parser.trace_digest(info['path'])==info['sha256'];pieces,proof=partition(info['path'],15);proof['samples']=pieces;assert proof==json.loads((D/'session/partition.json').read_text())
  else:
   pieces=[]
   for profile in timing['retained_profiles']:
    assert parser.trace_digest(profile['profile_path'])==profile['profile_trace_sha256'];piece=parser.parse_profile(profile['profile_path'],1);assert piece['profile_named_task_coverage_complete'];pieces.append(piece)
  assert len(pieces)==15 and all(abs(x['latency_ms']-v)<1e-9 for x,v in zip(pieces,timing['latency_samples_ms']))
  assert abs(sum(x['latency_ms'] for x in pieces)/15-timing['latency_ms'])<1e-9
  from compact_timing import compact
  timing=compact(timing)
  result={'ordinal':i,'key':[lib,r['op'],r['case']],'status':'raw_checked_pending_guard','api':api,'metric':metric,'reference_policy':rule,'check':check,'inputs_preserved_by_original_verifier':True,'payload_sha256':meta['payload_sha256'],'prepare_sha256':sha(prep/'prepare.json'),'timing':timing,'device_us':timing['latency_ms']*1000,'host_us_diagnostic':sum(v['ms'] for v in wall[4:19])/15*1000,'wall_calls':wall,'host_wall_formal_slice':[4,19],'extra_probe_wall_calls':len(wall)-19,'edge_probes':edge,'measurement_and_verification_seconds':elapsed,'case_seconds':time.perf_counter()-beg,'sources':ns,'selected':'original_complete_API_pending_candidate_review','timing_summary_schema':'compact-raw-backed-v1; taskarrays reconstructed from unchanged retained raw'}
  put(D/'result.json',result);results.append(result);print('VERIFIED',i,r['op'],result['device_us'],round(result['case_seconds'],2),flush=True)
 except Exception as e:
  if last or diagnostic_reference:
   torch.save({'produced':cv(last[0],'cpu') if last else None,'expected':cv(diagnostic_reference[0],'cpu') if diagnostic_reference else None},D/'CHECKER_DIAGNOSTIC.pt')
  failure={'ordinal':i,'key':[lib,r['op'],r['case']],'status':'unsupported' if type(e).__name__=='UnsupportedCase' else 'failed','phase':stage,'error':repr(e),'traceback':traceback.format_exc(),'seconds':time.perf_counter()-beg}
  put(D/'result.json',failure);results.append(failure);print('FAILED',i,stage,repr(e)[:240],flush=True)
  try:torch.npu.synchronize()
  except Exception:raise
  break
 finally:
  active[0]=False if 'active' in locals() else False
  if candidate is not None and candidate.cleanup is not None:candidate.cleanup()
 if 'case' in locals():del case
 for name in ['args','actual_args','saved','workload','impl','checked_impl','fn','frozen','candidate']:
  if name in globals():globals().pop(name,None)
 last.clear();diagnostic_reference.clear();torch.npu.synchronize();gc.collect();torch.npu.empty_cache()
assert all(sha(p)==v for p,v in source.items());put(O/'SOURCE.json',source)
maps=pathlib.Path('/proc/self/maps').read_text();(O/'PROCESS_MAPS.txt').write_text(maps);libs={s.split()[-1] for s in maps.splitlines() if len(s.split())>=6 and s.split()[-1].startswith('/') and '.so' in s.split()[-1]};put(O/'LOADED_LIBRARIES.json',{s:sha(s) for s in sorted(libs) if pathlib.Path(s).is_file()})
put(O/'BATCH.json',{'tag':tag,'requested':ids,'results':results,'runtime':runtime,'initialization_seconds':initialization,'process_seconds':time.perf_counter()-START,'source_sha256':sha(O/'SOURCE.json'),'status':'pending_guard'})
