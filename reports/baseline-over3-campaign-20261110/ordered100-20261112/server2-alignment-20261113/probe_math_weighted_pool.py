from pathlib import Path
import os,sys,json,hashlib,datetime,traceback
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-weighted-pool-v1');owner=sys.argv[1];dest=N/sys.argv[2];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert g['phase']=='running' and g['owner']==owner and g['child']['pid']==os.getpgrp();out={'status':'initializing','zero_baseline_coverage':True,'checks':[]}
def save():dest.write_text(json.dumps(out,indent=2)+'\n')
save()
try:
 import torch,torch_npu
 P=Path('/opt/baseline-over3-20261110/ordered100-20261112');sys.path[:0]=[str(P),str(P/"workspace/TileOPs/src"),str(P/"workspace/TileOPs")]
 from ops_math_weighted_pool_support import weighted_pool_candidate
 from ops_math_weighted_pool_api import WeightedMeanAPI
 from types import SimpleNamespace as T
 from workloads.pool import AvgPoolWorkload
 from workloads.numerics import reference_tolerance
 torch.set_num_threads(8);torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});p=torch.npu.get_device_properties(0);assert (p.cube_core_num,p.vector_core_num)==(32,64);out['runtime']={'torch':torch.__version__,'torch_npu':torch_npu.__version__,'cube':32,'vector':64}
 configs=[(2,(1,2,7,9),torch.bfloat16,True,False),(2,(1,2,7,9),torch.float32,True,False),(3,(1,2,5,7,9),torch.bfloat16,False,True),(3,(1,2,5,7,9),torch.float32,False,True)]
 for nd,shape,dt,ceil,include in configs:
  opts=dict(kernel_size=3,stride=2,padding=1,ceil_mode=ceil,count_include_pad=include,divisor_override=7);cpu=(torch.arange(__import__('math').prod(shape)).reshape(shape)%97-48).to(dt)/16;w=AvgPoolWorkload(nd,3,2,1,ceil,include,7,dt);gold=w.ref_program(cpu);x=cpu.to('npu');params={**opts,'input':(shape,str(dt).split('.')[-1]),'output':(tuple(gold.shape),str(dt).split('.')[-1])};apis=[]
  def child(c,name):
   assert name=='MeanFwdOp' and c.params['_weighted'];api=WeightedMeanAPI(torch);apis.append(api);return T(implementation=T(run=api),cleanup=api.cleanup)
  candidate=weighted_pool_candidate(T(params=params),'AvgPool'+str(nd)+'dFwdOp',child,lambda:{});mark={'ndim':nd,'shape':shape,'dtype':str(dt)};print('BASELINE_WEIGHTED_POOL_CALL_BEGIN '+json.dumps(mark),flush=True);out.update(status='before_complete_weighted_API',**mark);save();actual=candidate.implementation.run(x);torch.npu.synchronize();torch.testing.assert_close(actual.cpu(),gold,**reference_tolerance(dt));assert torch.equal(x.cpu().view(torch.uint8),cpu.view(torch.uint8));print('BASELINE_WEIGHTED_POOL_CALL_END '+json.dumps(mark),flush=True);out['checks'].append({**mark,'full_output_original_reference_tolerance_passed':True,'inputbits_preserved':True,'tolerance':reference_tolerance(dt)});candidate.cleanup()
 marker=torch.zeros(1024,dtype=torch.int32,device='npu');marker.add_(1);torch.npu.synchronize();assert torch.equal(marker.cpu(),torch.ones(1024,dtype=torch.int32));assert torch.count_nonzero(marker).item()==1024;import ctypes as C
 m=json.loads((N/'original-flush/SOURCE.json').read_text());assert sha(N/'original-flush/host_kernel.asc')==m['generated_source_sha256'] and sha(N/'original-flush/flush.aibin')==m['device_binary_sha256'];dll=C.CDLL(str(N/'original-flush/flush.so'));dll.init.argtypes=[C.c_char_p];dll.init.restype=C.c_int;dll.call.argtypes=[C.c_void_p,C.c_void_p];dll.call.restype=C.c_int;assert dll.init(str(N/'original-flush/flush.aibin').encode())==0;buf=torch.ones(268435456,dtype=torch.int32,device='npu');assert dll.call(buf.data_ptr(),torch.npu.current_stream().npu_stream)==0;torch.npu.synchronize();assert torch.count_nonzero(buf).item()==0
 out.update(status='full_WeightedPool_completeAPI_functional_passed',full_output_and_inputs_verified=True,original_INT32_marker_passed=True,original_1GiB_flush_selfcheck_passed=True)
except BaseException as e:out.update(status='failed_requires_diagnosis',error=repr(e),traceback=traceback.format_exc());raise
finally:
 out['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();maps=Path('/proc/self/maps').read_text();(N/(sys.argv[2]+'.maps.txt')).write_text(maps);out['loaded_library_sha256']={l.split()[-1]:sha(l.split()[-1]) for l in maps.splitlines() if len(l.split())>=6 and l.split()[-1].startswith('/') and '.so' in l.split()[-1] and Path(l.split()[-1]).is_file()};save()
