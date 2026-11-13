"""Guarded first-batch measurement; acceptance requires external guard and provenance audit."""
import argparse,ctypes,dataclasses,inspect,fcntl,hashlib,json,os,pathlib,sys,time,traceback

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--index',type=int,default=0);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--windows',type=int,choices=[1,15],required=True);p.add_argument('--claim',type=pathlib.Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 # Require a held host reservation; the caller runs this whole foreground process under guard.
 with open('/run/lock/ascend-npu/device-0.lock') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:pass
  else:raise RuntimeError('No held host device reservation')
 root=a.root;s=root/'frozen/source';sys.path[:0]=[str(s/'src'),str(s)]
 def event(name,**extra):
  with (a.output/'timeline.jsonl').open('a') as f:f.write(json.dumps({'event':name,'wall_ns':time.time_ns(),'mono_ns':time.monotonic_ns(),**extra})+'\n')
 event('process_start')
 report={'status':'running','kind':'first_batch_measurement_pending_guard_and_environment_acceptance','observation_protocol':'reservation-driver-v1','fd_scan_performed':False,'limitation':'No full process FD visibility; alternate observation pending publisher acceptance'}
 try:
  import torch,torch_npu
  from benchmarks.api import cases
  from workloads.device import set_run_device
  from workloads.numerics import compare_outputs,describe
  torch.set_num_threads(2);torch.manual_seed(0);set_run_device('cpu')
  row=json.loads(a.claim.read_text())['rows'][a.index]
  c=next(c for c in cases(row['op']) if c.id==row['case_id'])
  assert json.loads(json.dumps(c.params))==row['case']['params']
  # Capture loaded source identities before any reference invocation can fail.
  sys.path.insert(0,str(root/'frozen'))
  from tools import kernel_task_profile_full32 as early_parser
  sys.modules['tools.kernel_task_profile']=early_parser
  from tools import measure_baselines as early_core
  from benchmarks.baselines import private_inputs as early_private_inputs
  report['execution_source_receipt']={}
  for name,func in [('reference',c.reference),('checker',compare_outputs),('core',early_core._time_reference),('parser',early_parser.parse_profile),('private_inputs',early_private_inputs)]:
   source_path=pathlib.Path(inspect.getsourcefile(func))
   report['execution_source_receipt'][name]={'path':str(source_path),'file_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),'body_sha256':hashlib.sha256(inspect.getsource(func).encode()).hexdigest()}
  (a.output/'execution-sources.json').write_text(json.dumps(report['execution_source_receipt'],indent=2))
  report['runner_revision']='driver-v2-early-provenance'
  report['runner_sha256']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
  xs=c.inputs
  expected=c.reference(*tuple(x.clone() if isinstance(x,torch.Tensor) else x for x in xs))
  torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
  prop=torch.npu.get_device_properties(0);assert (prop.cube_core_num,prop.vector_core_num)==(32,64)
  def cv(x,device):
   if isinstance(x,torch.Tensor):return x.to(device)
   if isinstance(x,tuple):return tuple(cv(v,device) for v in x)
   if isinstance(x,list):return [cv(v,device) for v in x]
   if isinstance(x,dict):return {k:cv(v,device) for k,v in x.items()}
   return x
  def zeroed(x):
   if isinstance(x,torch.Tensor):return torch.zeros_like(x)
   if isinstance(x,tuple):return tuple(zeroed(v) for v in x)
   if isinstance(x,list):return [zeroed(v) for v in x]
   if isinstance(x,dict):return {k:zeroed(v) for k,v in x.items()}
   return x
  args=cv(xs,'npu');set_run_device('npu')
  report.update(op=row['op'],case=row['case_id'],contract_sha256=row['contract_sha256'],torch=torch.__version__,torch_npu=torch_npu.__version__,compiler_backend=os.environ.get('TORCHINDUCTOR_NPU_BACKEND'),cube=32,vector=64,verification=describe(c.verification))
  report['phase']='eager_correctness';out=c.reference(*args);torch.npu.synchronize();compare_outputs(cv(out,'cpu'),expected,c.verification);report['eager_correctness']='passed'
  report['phase']='compile_first_call';torch._dynamo.config.suppress_errors=False
  fn=torch.compile(c.reference,backend='inductor',fullgraph=True,dynamic=False)
  event('compile_first_call_begin');t=time.monotonic();out=fn(*args);torch.npu.synchronize();report['cold_first_call_seconds']=time.monotonic()-t
  event('compile_first_call_end');report['phase']='compile_correctness';compare_outputs(cv(out,'cpu'),expected,c.verification);report['compile_correctness']='passed'
  bad=zeroed(cv(out,'cpu'))
  try:compare_outputs(bad,expected,c.verification)
  except AssertionError:report['negative_control']={'candidate':'compile','control':'replace complete output with zeros','rejected':True}
  else:raise RuntimeError('Negative control accepted')
  report['graph_counters']={str(k):dict(v) for k,v in torch._dynamo.utils.counters.items()}
  sys.path.insert(0,str(root/'frozen'))
  from tools import kernel_task_profile_full32 as full32
  sys.modules['tools.kernel_task_profile']=full32
  from tools import measure_baselines as core
  report['actual_sources']={}
  for label,func in [('reference',c.reference),('checker',compare_outputs),('measure_core',core._time_reference),('parser',full32.parse_profile)]:
   path=pathlib.Path(inspect.getsourcefile(func))
   report['actual_sources'][label]={'path':str(path),'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'body_sha256':hashlib.sha256(inspect.getsource(func).encode()).hexdigest()}
  report['phase']='flush_validation'
  fd=root/'flush-rebuild/build01'
  binary_hash=hashlib.sha256((fd/'flush.aibin').read_bytes()).hexdigest()
  assert binary_hash=='693be2fb78b6c749ec2afee81e1ed3d8c0de58ea3397daaeaeef06588cc2e865'
  dll=ctypes.CDLL(str(fd/'flush.so'))
  dll.init.argtypes=[ctypes.c_char_p];dll.init.restype=ctypes.c_int
  dll.call.argtypes=[ctypes.c_void_p,ctypes.c_void_p];dll.call.restype=ctypes.c_int
  assert dll.init(str(fd/'flush.aibin').encode())==0
  def exported_flush(buffer):
   assert dll.call(buffer.data_ptr(),torch.npu.current_stream().npu_stream)==0
  core._flush_kernel_cache[0]=(exported_flush,torch.empty(268435456,dtype=torch.int32,device='npu'))
  report['flush_build']={'sha256':binary_hash,'compiler':'CANN9.2 20261015 ARM64','historical_byte_identical':False}
  report['windows']=a.windows
  flush,buffer=core._get_flush_kernel_and_buffer(torch,torch_npu)
  assert buffer.numel()*buffer.element_size()==1024**3
  buffer.fill_(1);flush(buffer);torch.npu.synchronize()
  assert torch.count_nonzero(buffer.cpu()).item()==0, 'flush did not zero entire 1GiB buffer'
  report['flush_bytes']=1024**3
  report['timings']={}
  original_window=core._time_one_isolated_repeat
  def observed_window(*values,**kwargs):
   event('window_begin',variant=report['phase']);value=original_window(*values,**kwargs);event('window_end',variant=report['phase']);return value
  core._time_one_isolated_repeat=observed_window
  for variant,call in [('eager',c.reference),('compile',fn)]:
   report['phase']='timing_'+variant
   os.environ['TILEOPS_PROFILE_DIR']=str(a.output/variant/'profiles')
   last={}
   def capture(*values):
    value=call(*values);last['output']=value;return value
   event('measurement_begin',variant=variant)
   before={str(k):dict(v) for k,v in torch._dynamo.utils.counters.items()}
   timing=core._time_reference(torch,torch_npu,capture,args,warmup=3,repeats=a.windows)
   event('measurement_end',variant=variant)
   after={str(k):dict(v) for k,v in torch._dynamo.utils.counters.items()}
   report.setdefault('measurement_graph_counters',{})[variant]={'before':before,'after':after}
   check=compare_outputs(cv(last['output'],'cpu'),expected,c.verification)
   assert check.checked_outputs==check.total_outputs and check.checked_outputs>0
   report.setdefault('timed_checks',{})[variant]=dataclasses.asdict(check)
   try:compare_outputs(zeroed(cv(last['output'],'cpu')),expected,c.verification)
   except AssertionError:report.setdefault('negative_controls',{})[variant]={'complete_output_zeroed':True,'rejected':True}
   else:raise RuntimeError('Timed output negative control accepted')
   samples=[]
   for prof in timing['retained_profiles']:
    assert full32.trace_digest(prof['profile_path'])==prof['profile_trace_sha256']
    parsed=full32.parse_profile(pathlib.Path(prof['profile_path']),1)
    assert parsed['profile_named_task_coverage_complete']
    assert parsed['profile_kernel_tasks']==prof['profile_kernel_tasks']
    samples.append(parsed['profile_kernel_us']/1000)
   assert len(samples)==a.windows
   assert all(abs(x-y)<1e-9 for x,y in zip(samples,timing['latency_samples_ms']))
   report['timings'][variant]={'measurement':timing,'offline_replay':str(a.windows)+'/'+str(a.windows)+' passed','timed_output_correctness':'passed'}
   (a.output/'progress.json').write_text(json.dumps(report,indent=2,default=str))
  report['status']='measured_pending_guard_and_environment_acceptance';report['phase']='finished' 
 except Exception as e:
  report['status']='failed';report['error']=repr(e);report['traceback']=traceback.format_exc();raise
 finally:
  event('process_report',status=report['status'])
  (a.output/'admission.json').write_text(json.dumps(report,indent=2,default=str));print(json.dumps(report,default=str),flush=True)
if __name__=='__main__':main()
