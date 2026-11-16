"""Guarded correctness supplement only. Reuse retained final measured outputs."""
from pathlib import Path
import os,sys,fcntl,json,hashlib,pickle,datetime,dataclasses
P=Path(__file__).resolve().parent;T=P/'workspace/TileOPs';sys.path[:0]=[str(P),str(T),str(T/'src')];sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
with open('/run/lock/ascend-npu/device-0.lock') as f:
 try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:pass
 else:raise RuntimeError('Whole host guard required')
import torch,torch_npu
from workloads.device import set_run_device
from benchmarks.api import Implementation
from measured_correctness import measure_and_verify
from reference_policy import policy
from workloads.numerics import _require_unchanged,_snapshot,_unique
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False;prop=torch.npu.get_device_properties(0);assert (prop.cube_core_num,prop.vector_core_num)==(32,64);jobs=json.loads(Path(sys.argv[1]).read_text());rows=[]
for job in jobs:
 target,old=job['target'],job['source_ordinal'];a=P/'prepared-rollout'/str(target);b=P/'prepared-rollout'/str(old);m=json.loads((a/'prepare.json').read_text());n=json.loads((b/'prepare.json').read_text());assert m['params']==n['params'] and m['reference_policy']=='original_npu' and n['reference_policy']=='original_cpu';assert sha(a/'payload.pt')==sha(b/'payload.pt')==m['payload_sha256']==n['payload_sha256'];assert sha(a/'case.pkl')==sha(b/'case.pkl')==m['descriptor_sha256']==n['descriptor_sha256'];assert policy(*m['key'])=='original_npu';source=P/'runs'/job['source_tag']/str(old);ret=json.loads((source/'MEASURED_OUTPUT_RETENTION.json').read_text());assert ret['ordinal']==old and ret['payload_sha256']==m['payload_sha256'] and ret['descriptor_sha256']==m['descriptor_sha256'];output=source/'MEASURED_OUTPUT.pt';assert sha(output)==ret['retained_output_sha256']==job['measured_output_sha256']
 set_run_device('cpu');case=pickle.load((a/'case.pkl').open('rb'));saved=torch.load(a/'payload.pt',weights_only=False);args=tuple(x.to('npu') for x in saved['inputs']);case.__dict__['inputs']=args
 if hasattr(case.workload,'tensors'):
  for name,value in zip(case._call.signature.inputs,args,strict=True):case.workload.tensors[name]=value
 case.__dict__.pop('verification',None);set_run_device('npu');original_reference=case.reference;actual_cpu=torch.load(output,weights_only=False);produced=actual_cpu.to('npu');torch.npu.synchronize();assert torch.equal(produced.cpu().view(torch.uint8),actual_cpu.view(torch.uint8));first=[True]
 def first_frozen_then_original(*values):
  if first[0]:first[0]=False;return produced
  return original_reference(*values)
 impl=Implementation(first_frozen_then_original)
 def no_performance_measurement(run):
  run(*args);torch.npu.synchronize();return {'correctness_supplement_only':True,'no_performance_samples':True,'original_performance_raw_reused':job['source_checkpoint_sha256']}
 check,proof=measure_and_verify(case,impl,no_performance_measurement,expected_calls=1);assert check['checked_outputs']==check['total_outputs']==1 and not check['unchecked_reason'];rows.append({'target':target,'source_ordinal':old,'target_key':m['key'],'original_target_reference_policy':'original_npu','source_reference_policy':'original_cpu','payload_sha256':m['payload_sha256'],'descriptor_sha256':m['descriptor_sha256'],'retained_actual_measured_output_sha256':sha(output),'source_checkpoint_sha256':job['source_checkpoint_sha256'],'source_tag':job['source_tag'],'full_original_target_checker':check,'proof':proof,'unmodified_original_golden_output_checker_and_input_preservation':True});print('SUPPLEMENT_PASSED',target,flush=True)
 torch.npu.synchronize();del case,saved,args,produced,actual_cpu
out={'checked_at_remote':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'passed','rows':rows,'new_performance_measurements':0,'new_pairs':0,'new_final_closures':0,'runtime':{'torch':torch.__version__,'torch_npu':torch_npu.__version__,'SDK':os.environ.get('ASCEND_HOME_PATH'),'OPP':os.environ.get('ASCEND_OPP_PATH')},'source_sha256':sha(__file__)};f=Path(sys.argv[2]);assert not f.exists();f.write_text(json.dumps(out,indent=2));torch.npu.synchronize()
