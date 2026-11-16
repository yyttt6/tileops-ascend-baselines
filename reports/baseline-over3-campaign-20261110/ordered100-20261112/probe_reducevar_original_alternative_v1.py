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
if os.environ.get('BASELINE_GRAPH92_ENV_BOUND')!='1':
 import shlex
 command='source /opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-reducevar-original-v1/runtime-env-formal-quiet-v3.sh; export BASELINE_GRAPH92_ENV_BOUND=1; exec /opt/baseline-over3-20261110/server2-aligned-stack/job-python-v1/bin/python '+shlex.quote(str(P/'worker_ops_transformer_gmmv5_api.py'))+' '+shlex.join(sys.argv[1:])
 os.execv('/bin/bash',['bash','-c',command])
JIT_WORK=pathlib.Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-reducevar-original-v1')/'jit-work'/tag;JIT_WORK.mkdir(parents=True,exist_ok=False);os.environ['ASCEND_WORK_PATH']=str(JIT_WORK)
import torch,torch_npu
from benchmarks.api import Implementation
from reducevar_exact_input_fidelity import verify_runtime_source_identity
from workloads.device import set_run_device
from workloads.numerics import _copy_outputs,_snapshot,_unique,_require_unchanged
from tools import kernel_task_profile_full32 as parser
sys.modules['tools.kernel_task_profile']=parser
from tools import measure_baselines as core
from ops_math_reducevar_alternative_api import candidate_factory
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


for i in ids:
 assert i in [683,684];r=rows[i];prep=P/'prepared-rollout'/str(i);stage='load';candidate=None;Q=O/str(i);Q.mkdir()
 try:
  assert not (P/'STOP').exists() and not (P.parent/'STOP').exists();meta=json.loads((prep/'prepare.json').read_text());assert verify_runtime_source_identity(P,i,meta);assert sha(prep/'payload.pt')==meta['payload_sha256'] and sha(prep/'case.pkl')==meta['descriptor_sha256'];set_run_device('cpu');case=pickle.load((prep/'case.pkl').open('rb'));saved=torch.load(prep/'payload.pt',weights_only=False);case.__dict__['inputs']=saved['inputs'];workload=case.workload;args=cv(saved['inputs'],'npu');case.__dict__['inputs']=args
  if hasattr(workload,'tensors'):
   for name,value in zip(case._call.signature.inputs,args,strict=True):workload.tensors[name]=value
  case.__dict__.pop('verification',None);set_run_device('npu');raw_reference=case.reference;rule=policy(lib,r['op'],r['case']);assert rule==meta['reference_policy'] and rule=='original_cpu';case.__dict__['reference']=canonical.cpu_reference(torch,raw_reference,args);candidate=candidate_factory(case,lib)();stage='one_original_contract_call_original_checker'
  def once(run):run(*args);torch.npu.synchronize();return {'calls':1,'not_formal15':True}
  check,execution=measure_and_verify(case,candidate.implementation,once,expected_calls=1);maps=pathlib.Path('/proc/self/maps').read_text();host=pathlib.Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-reducevar-original-v1')/'private-v1/vendors/originalvar_math/op_api/lib/libcust_opapi.so';assert str(host) in maps;put(Q/'PROBE.json',{'status':'passed_once_original_checker_not_coverage','alternative_API':candidate.api,'original_library':lib,'check':check,'execution':execution,'loaded_host_path':str(host),'loaded_host_sha':sha(host),'payload_sha':meta['payload_sha256'],'actual_device_binary_source':'requires_formalprofile_binding_before_acceptance'});(Q/'loaded-maps.txt').write_text(maps)
 except BaseException as e:put(Q/'PROBE_FAILED.json',{'stage':stage,'error':repr(e),'traceback':traceback.format_exc(),'no_same_condition_retry':True});raise
 finally:
  if candidate is not None and candidate.cleanup is not None:candidate.cleanup()
