from pathlib import Path
import os,sys,json,hashlib,datetime,traceback
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-dequant-v1');owner=sys.argv[1];dest=N/sys.argv[2];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert g['phase']=='running' and g['owner']==owner and g['child']['pid']==os.getpgrp();out={'status':'initializing','zero_baseline_coverage':True,'checks':[]}
def save():dest.write_text(json.dumps(out,indent=2)+'\n')
save()
try:
 import torch,torch_npu
 from ops_math_dequant_api import MathDequantAPI
 torch.set_num_threads(8);torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});p=torch.npu.get_device_properties(0);assert (p.cube_core_num,p.vector_core_num)==(32,64);out['runtime']={'torch':torch.__version__,'torch_npu':torch_npu.__version__,'cube':32,'vector':64}
 for mode in ['tensor','channel','block']:
  for dt in [torch.float16,torch.bfloat16]:
   shape=(17,4099);cpuq=(torch.arange(shape[0]*shape[1]).reshape(shape)%255-127).to(torch.int8)
   scaleshape={'tensor':(),'channel':(shape[0],),'block':(shape[0],(shape[1]+127)//128)}[mode]
   scale=((torch.arange(__import__('math').prod(scaleshape)).reshape(scaleshape)%13)+1).float()/16
   q=cpuq.to('npu');sc=scale.to('npu');api=MathDequantAPI(torch,'Mul')
   out.update(status='before_API',op='INT8Dequant',mode=mode,dtype=str(dt));save()
   native_scale=sc[:,None] if mode=='channel' else sc.repeat_interleave(128,dim=1)[:,:q.shape[1]] if mode=='block' else sc
   actual=api(q.float(),native_scale).to(dt);torch.npu.synchronize()
   cpu_scale=scale[:,None] if mode=='channel' else scale.repeat_interleave(128,dim=1)[:,:cpuq.shape[1]] if mode=='block' else scale
   gold=(cpuq.float()*cpu_scale).to(dt);torch.testing.assert_close(actual.cpu(),gold,rtol=0,atol=0)
   assert torch.equal(q.cpu().contiguous().view(torch.uint8),cpuq.contiguous().view(torch.uint8))
   assert torch.equal(sc.cpu().reshape(-1).view(torch.uint8),scale.reshape(-1).view(torch.uint8))
   out['checks'].append({'op':'INT8Dequant','mode':mode,'dtype':str(dt),'shape':shape,'all_outputs_and_input_bits_verified':True});api.cleanup()
 marker=torch.zeros(1024,dtype=torch.int32,device='npu');marker.add_(1);torch.npu.synchronize();assert torch.equal(marker.cpu(),torch.ones(1024,dtype=torch.int32));assert torch.count_nonzero(marker).item()==1024;import ctypes as C
 m=json.loads((N/'original-flush/SOURCE.json').read_text());assert sha(N/'original-flush/host_kernel.asc')==m['generated_source_sha256'] and sha(N/'original-flush/flush.aibin')==m['device_binary_sha256'];dll=C.CDLL(str(N/'original-flush/flush.so'));dll.init.argtypes=[C.c_char_p];dll.init.restype=C.c_int;dll.call.argtypes=[C.c_void_p,C.c_void_p];dll.call.restype=C.c_int;assert dll.init(str(N/'original-flush/flush.aibin').encode())==0;buf=torch.ones(268435456,dtype=torch.int32,device='npu');assert dll.call(buf.data_ptr(),torch.npu.current_stream().npu_stream)==0;torch.npu.synchronize();assert torch.count_nonzero(buf).item()==0
 out.update(status='full_original_APIs_functional_passed',full_output_and_inputs_verified=True,original_INT32_marker_passed=True,original_1GiB_flush_selfcheck_passed=True)
except BaseException as e:out.update(status='failed_requires_diagnosis',error=repr(e),traceback=traceback.format_exc());raise
finally:
 out['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();maps=Path('/proc/self/maps').read_text();(N/(sys.argv[2]+'.maps.txt')).write_text(maps);out['loaded_library_sha256']={l.split()[-1]:sha(l.split()[-1]) for l in maps.splitlines() if len(l.split())>=6 and l.split()[-1].startswith('/') and '.so' in l.split()[-1] and Path(l.split()[-1]).is_file()};save()
