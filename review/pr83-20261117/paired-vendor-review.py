"""Bounded PR83 same-process qualification on the canonical runner, not a bulk baseline campaign."""
import sys,pathlib,json,hashlib,time,traceback,dataclasses,types
S=pathlib.Path(sys.argv[1]).resolve();O=pathlib.Path(sys.argv[2]).resolve()
sys.path[:0]=[str(S/'src'),str(S),str(S/'scripts/ci'),str(S/'scripts/ci/nightly_support')]
from benchmarks.ascend.guard import require_reservation
require_reservation()
import torch,torch_npu
from benchmarks.api import cases,Implementation
from benchmarks._cases import entry
from benchmarks.ascend.runner import source_fingerprint
from measured_correctness import measure_profile_and_verify
from golden_cache import input_digest
from workloads.device import set_run_device
from tileops.gemm import GEMMW4A16FwdOp
from tileops.backend import _registry
set_run_device('npu');torch.npu.set_device(0);_registry.ensure_loaded();assert not _registry.LOAD_FAILURES
props=torch.npu.get_device_properties(0);assert (props.cube_core_num,props.vector_core_num)==(32,64)
O.mkdir(parents=True,exist_ok=False)
def write(p,x):p.write_text(json.dumps(x,indent=2,default=str)+'\n')
source=source_fingerprint();write(O/'sources.json',source);write(O/'hardware.json',{'device':props.name,'cube':props.cube_core_num,'vector':props.vector_core_num,'clock':'platform-default','torch':torch.__version__,'torch_npu':torch_npu.__version__,'adapter_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()})
selected=cases('GEMMW4A16FwdOp');assert len(selected)==15
rows=[];failed=0
with (O/'results.jsonl').open('x') as stream:
 for index,c in enumerate(selected):
  item={'case':c.id,'params':c.params};start=time.monotonic()
  try:
   seed=int.from_bytes(hashlib.sha256(f'GEMMW4A16FwdOp::{c.id}'.encode()).digest()[:4],'little');torch.manual_seed(seed)
   args=c.inputs;act,packed,scale,zero=args;assert act.dtype==torch.float16 and c.workload.group_size==128
   row_major=c.workload.row_major_weight
   # Each provider receives its own checkpoint packing, prepared once outside timing.
   # XOR each nibble's sign bit maps unsigned q to signed q-8 without changing logical weights.
   signed=torch.empty((row_major.shape[0], row_major.shape[1]*2),dtype=torch.int32,device=act.device)
   signed[:,0::2]=(row_major & 15).to(torch.int32)-8
   signed[:,1::2]=(row_major >> 4).to(torch.int32)-8
   weight=torch_npu.npu_convert_weight_to_int4pack(signed).t()
   del signed
   native_scale=scale.t();offset=(8-zero.to(act.dtype)).t()
   native_args=(act,weight,native_scale,offset)
   assert row_major.dtype==torch.uint8 and weight.dtype==torch.int32
   descriptors=lambda xs:[{'shape':list(x.shape),'stride':list(x.stride()),'dtype':str(x.dtype)} for x in xs]
   item.update(seed=seed,input_sha256=input_digest(args),native_input_sha256=input_digest(native_args),input_descriptors=descriptors(args),native_descriptors=descriptors(native_args),native_contract='(signed(q-8)+(8-zero))*scale == (q-zero)*scale; provider-specific checkpoint packing outside timing; complete per-call matmul, candidate activation permutation included')
   def native(x,w,s,z):return torch_npu.npu_weight_quant_batchmatmul(x,w,s,antiquant_offset=z,antiquant_group_size=0 if s.shape[0]==1 else 128)
   candidate=entry('GEMMW4A16FwdOp').binder(GEMMW4A16FwdOp(**c.arguments),c)
   # Treat prepared native tensors as shared inputs so preservation is checked before restore.
   native_case=types.SimpleNamespace(inputs=native_args,reference=lambda *unused:c.reference(*args),verification=c.verification)
   implementations={'tileops':(c,candidate),'torch-npu-native':(native_case,Implementation(run=native))}
   # Fixed alternating order; retain every window and final timed output for both providers.
   order=['tileops','torch-npu-native'] if index%2==0 else ['torch-npu-native','tileops']
   item['native_group_attribute']=0 if native_scale.shape[0]==1 else 128;item['order']=order;item['implementations']={}
   for tag in order:
    case,impl=implementations[tag]
    check,measurement=measure_profile_and_verify(case,impl,torch,torch_npu,impl.args if impl.args is not None else case.inputs,O/f'{index:02d}'/tag,mode='auto')
    item['implementations'][tag]={'correctness':check,'measurement':measurement}
   assert input_digest(args)==item['input_sha256'] and input_digest(native_args)==item['native_input_sha256']
   item['ratio']=item['implementations']['torch-npu-native']['measurement']['latency_ms']/item['implementations']['tileops']['measurement']['latency_ms'];item['status']='passed';item['performance_passed']=item['ratio']>=.8
  except Exception as exc:
   failed+=1;item.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
  finally:torch.npu.synchronize()
  item['seconds']=time.monotonic()-start;stream.write(json.dumps(item,default=str)+'\n');stream.flush();rows.append(item);print(json.dumps({'case':c.id,'status':item['status'],'ratio':item.get('ratio'),'error':item.get('error')}),flush=True)
  c._release()
  if failed:break
  del args,act,packed,scale,zero,row_major,weight,native_scale,offset,native_args,candidate,native_case,implementations
  torch.npu.empty_cache()
changed=source_fingerprint()!=source
passed=[x for x in rows if x['status']=='passed'];result={'status':'passed' if not failed and len(rows)==15 and not changed else 'failed','cases':len(rows),'correctness_passed':len(passed),'failed':failed,'source_changed':changed,'performance_passed':sum(x.get('performance_passed',False) for x in rows),'minimum_ratio':min((x['ratio'] for x in passed),default=None),'official_baseline_modified':False};write(O/'result.json',result);print(json.dumps(result),flush=True)
raise SystemExit(0 if result['status']=='passed' else 1)
