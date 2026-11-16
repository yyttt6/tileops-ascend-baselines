"""Correctness-only supplement under the exact frozen legacy checker; no performance sampling."""
from pathlib import Path
import os,sys,json,ast,types,pickle,hashlib,fcntl,importlib.util,datetime,dataclasses
P=Path(__file__).resolve().parent;T=P/'workspace/TileOps-08f-original';OLD=P/'workspace/TileOPs';sys.path[:0]=[str(T/'src'),str(T),str(P)];tag=sys.argv[1];O=P/'runs'/tag;O.mkdir(parents=True,exist_ok=False)
with open('/run/lock/ascend-npu/device-0.lock') as lock:
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:pass
 else:raise RuntimeError('Host guard must cover entire import and command')
guard_source=OLD/'scripts/cache_frozen_pair_check.py';guard_node=next(n for n in ast.parse(guard_source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='require_guard');guard_env={'json':json,'Path':Path,'os':os};exec(compile(ast.fix_missing_locations(ast.Module(body=[guard_node],type_ignores=[])),str(guard_source),'exec'),guard_env);guard_env['require_guard']()
CHECKPOINT_ROOT=P/'qk2-frozen-checker-checkpoints-v1'
checkpoint_manifest=json.loads((CHECKPOINT_ROOT/'MANIFEST.json').read_text());assert len(checkpoint_manifest)==8
for name,digest in checkpoint_manifest.items():assert hashlib.sha256((CHECKPOINT_ROOT/name).read_bytes()).hexdigest()==digest
import torch,torch_npu
from workloads.device import set_run_device
from torch_qk2_native_rms_support import build as rms_build
from torch_qk2_native_rms_rope_support import build as rope_build
from tileops_qk2_denominator_support import FrozenQK
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();put=lambda p,v:p.write_text(json.dumps(v,indent=2,default=str))
# Load the unchanged frozen numeric interpreter in its own namespace.
spec=importlib.util.spec_from_file_location('frozen_qk_numeric_policy',OLD/'workloads/numerics.py');num=importlib.util.module_from_spec(spec);sys.modules[spec.name]=num;spec.loader.exec_module(num)
source=OLD/'workloads/ascend_extended.py';tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FusedQKNormRoPECall');nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in {'ref_program','verification'}];assert len(nodes)==2;env={'torch':torch,'Custom':num.Custom,'reference_tolerance':num.reference_tolerance,'assert_normalized_error':num.assert_normalized_error};exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source),'exec'),env)
files=[Path(__file__),source,OLD/'workloads/numerics.py',T/'workloads/norm.py',P/'torch_qk2_native_rms_support.py',P/'torch_qk2_native_rms_rope_support.py',P/'tileops_qk2_denominator_support.py'];sources={str(f):sha(f) for f in files};sources[str(guard_source)]=sha(guard_source);torch.set_num_threads(8);torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False;rows=[];qualified={}
for i in [1356,1358]:
 d=P/'prepared-torch-qk2-08f-v2'/str(i);meta=json.loads((d/'prepare.json').read_text());assert sha(d/'payload.pt')==meta['payload_sha256'] and sha(d/'case.pkl')==meta['descriptor_sha256'];set_run_device('cpu');case=pickle.load((d/'case.pkl').open('rb'));xs=torch.load(d/'payload.pt',weights_only=False)['inputs'];case.__dict__['inputs']=xs;legacy_ref=types.MethodType(env['ref_program'],case.workload);expected=legacy_ref(*xs);gpu=[x.to('npu') for x in xs];case.__dict__['inputs']=tuple(gpu);case.__dict__.pop('verification',None);set_run_device('npu');evidence=types.MethodType(env['verification'],case.workload)(*gpu);inputs_before=[x.contiguous().view(torch.uint8).clone() for x in xs];denom=FrozenQK(case);rms,_=rms_build(case);rope,_=rope_build(case)
 roles=[('original',case.reference,None,None,f'rollout-torch-qk2-original-08f-{i}-v3.CHECKPOINT.json'),('RMS',rms,None,None,f'rollout-torch-qk2-native-rms-{i}-v1.CHECKPOINT.json'),('RMS_RoPE',rope,None,None,f'rollout-torch-qk2-native-rms-rope-{i}-v1.CHECKPOINT.json'),('TileOps',denom.impl.run,denom.impl.args,denom.impl.reset,f'rollout-tileops-qk2-denominator-{i}-v1.CHECKPOINT.json')]
 for role,run,actual,reset,cp_name in roles:
  cp=CHECKPOINT_ROOT/cp_name;event=json.loads(cp.read_text());record=next(r for r in event['batch']['results'] if r['ordinal']==i);assert record['payload_sha256']==meta['payload_sha256'] and record['status']=='raw_checked_pending_guard';assert event['guard']['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip();qk=case.arguments;end=(qk['num_heads']+qk['num_kv_heads'])*gpu[1].numel()
  if reset is not None:reset()
  got=run(*(gpu if actual is None else actual));torch.npu.synchronize();cpu=got.cpu();check=num.compare_outputs(cpu,expected,evidence);assert check.checked_outputs==check.total_outputs and not check.unchecked_reason;torch.testing.assert_close(cpu[...,end:],xs[0][...,end:],rtol=0,atol=0);assert torch.equal(cpu[...,end:].contiguous().view(torch.uint8),xs[0][...,end:].contiguous().view(torch.uint8));assert all(torch.equal(x.cpu().contiguous().view(torch.uint8),b) for x,b in zip(gpu,inputs_before,strict=True))
  bad=cpu.clone();bits=bad.view(torch.int16);bits[0,end]=torch.bitwise_xor(bits[0,end],1)
  try:num.compare_outputs(bad,expected,evidence)
  except (AssertionError,RuntimeError):exactV_negative=True
  else:raise AssertionError('Frozen exact V checker did not reject 1-bit perturbation')
  qbad=cpu.clone();qbad[...,:end]=0
  try:num.compare_outputs(qbad,expected,evidence)
  except (AssertionError,RuntimeError):qk_negative=True
  else:raise AssertionError('Frozen QK checker did not reject zeroed QK')
  qualified[cp_name]=sha(cp);rows.append({'ordinal':i,'role':role,'checkpoint':cp_name,'checkpoint_sha256':sha(cp),'payload_sha256':meta['payload_sha256'],'old_checker_AST_unmodified':True,'legacy_custom_reference_tolerance_and_normalized_error_and_exactV_passed':True,'both_negative_controls_rejected':True,'original_shared_inputs_bit_preserved':True,'check':dataclasses.asdict(check),'no_performance_remeasurement':True});print('FROZEN_QK_CHECKER_PASS',i,role,flush=True)
 del gpu,case,expected,xs,denom,roles,rms,rope;torch.npu.synchronize();torch.npu.empty_cache()
assert all(sha(f)==v for f,v in sources.items());proof={'checked_at':now(),'status':'passed','rows':rows,'qualified_checkpoint_sha256':qualified,'frozen_checker_source':str(source),'frozen_checker_source_sha256':sources[str(source)],'frozen_numerics_source_sha256':sources[str(OLD/'workloads/numerics.py')],'frozen_ref_and_verification_FunctionDef_AST_unmodified':True,'supplements_stricter_frozen_protocol_than_original_published_workload_default':True,'original15raw_and_all_performance_checkpoints_unchanged':True,'correctness_only_no_timing_or_profiler_runs':True,'SOURCE':sources};put(O/'QK2_FROZEN_CHECKER_SUPPLEMENT.json',proof);print('ALL8_FROZEN_CHECKER_PASSED',flush=True)
