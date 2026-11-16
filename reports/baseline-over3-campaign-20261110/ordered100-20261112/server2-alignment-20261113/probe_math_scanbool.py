from pathlib import Path
import os,sys,json,hashlib,datetime,traceback
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-scanbool-v1');owner=sys.argv[1];dest=N/sys.argv[2];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert g['phase']=='running' and g['owner']==owner and g['child']['pid']==os.getpgrp();out={'status':'initializing','zero_baseline_coverage':True,'checks':[]}
def save():dest.write_text(json.dumps(out,indent=2)+'\n')
save()
try:
 import torch,torch_npu
 from ops_math_scanbool_api import ScanAPI
 torch.set_num_threads(8);torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});p=torch.npu.get_device_properties(0);assert (p.cube_core_num,p.vector_core_num)==(32,64);out['runtime']={'torch':torch.__version__,'torch_npu':torch_npu.__version__,'cube':32,'vector':64}
 for op in ['LogicalOr','NeTensor']:
  api=ScanAPI(torch,op,dim=-1)
  dtypes=[torch.float16,torch.bfloat16,torch.float32] if op in ('Cumsum','Cumprod') else [torch.bool]
  for dt in dtypes:
   x=(torch.arange(51).reshape(3,17)%3).float().div(2).to(dt);cpu=[x]
   if op in ('LogicalOr','NeTensor'):cpu.append((torch.arange(17).reshape(1,17)%2).bool())
   gpu=[x.to('npu') for x in cpu];out.update(status='before_API',op=op,dtype=str(dt));save();actual=api(*gpu);torch.npu.synchronize()
   gold=torch.cumsum(cpu[0],dim=-1) if op=='Cumsum' else torch.cumprod(cpu[0],dim=-1) if op=='Cumprod' else torch.logical_or(*cpu) if op=='LogicalOr' else torch.ne(*cpu)
   torch.testing.assert_close(actual.cpu(),gold)
   for a,b in zip(gpu,cpu):assert torch.equal(a.cpu().contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
   out['checks'].append({'op':op,'dtype':str(dt),'all_outputs_and_input_bits_verified':True})
  api.cleanup()
 marker=torch.zeros(1024,dtype=torch.int32,device='npu');marker.add_(1);torch.npu.synchronize();assert torch.equal(marker.cpu(),torch.ones(1024,dtype=torch.int32));out.update(status='full_original_APIs_functional_passed',full_output_and_inputs_verified=True,original_INT32_marker_passed=True)
except BaseException as e:out.update(status='failed_requires_diagnosis',error=repr(e),traceback=traceback.format_exc());raise
finally:
 out['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();maps=Path('/proc/self/maps').read_text();(N/(sys.argv[2]+'.maps.txt')).write_text(maps);out['loaded_library_sha256']={l.split()[-1]:sha(l.split()[-1]) for l in maps.splitlines() if len(l.split())>=6 and l.split()[-1].startswith('/') and '.so' in l.split()[-1] and Path(l.split()[-1]).is_file()};save()
