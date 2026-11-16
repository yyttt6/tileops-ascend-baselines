from pathlib import Path
import os,sys,json,hashlib,traceback,datetime
P=Path('/opt/baseline-over3-20261110/ordered100-20261112');N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-adaln3-v1');sys.path[:0]=[str(P),str(P/'workspace/TileOPs'),str(P/'workspace/TileOPs/src')];owner=sys.argv[1];dest=N/sys.argv[2];g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert g['owner']==owner and g['phase']=='running' and g['child']['pid']==os.getpgrp();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();r={'status':'initializing','checks':[],'no_formal_or_performance_acceptance':True};saved=[];sources={}
def save():dest.write_text(json.dumps(r,indent=2)+'\n')
save()
try:
 import torch,torch_npu
 from types import SimpleNamespace as T
 from workloads.norm import AdaLayerNormWorkload
 from workloads.device import set_run_device
 from workloads.numerics import reference_tolerance
 from ops_math_adaln3_support import make_candidate
 torch.set_num_threads(4);torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});prop=torch.npu.get_device_properties(0);assert (prop.cube_core_num,prop.vector_core_num)==(32,64);set_run_device('cpu');r['runtime']={'hostname':os.uname().nodename,'torch':torch.__version__,'torch_npu':torch_npu.__version__,'cube':32,'vector':64}
 for dt,n,broadcast in [(torch.float16,17,False),(torch.bfloat16,17,True),(torch.float32,129,False),(torch.float16,16385,True)]:
  w=AdaLayerNormWorkload(3,n,dt);torch.manual_seed(718);cpu=w.gen_inputs()
  if broadcast:cpu=(cpu[0],cpu[1][:1],cpu[2][:1])
  gold=w.ref_program(*cpu);args=tuple(x.to('npu') for x in cpu);candidate=make_candidate(T(inputs=args,params={'eps':w.eps}),False);sources.update(candidate.source_sha256);mark={'dtype':str(dt),'shape':[3,n],'broadcast':broadcast};r.update(status='calling_three_complete_APIs',current=mark);save();print('ADALN3_BEGIN '+json.dumps(mark),flush=True)
  try:out=candidate.implementation.run(*args);torch.npu.synchronize();produced=out.cpu();saved.append({'inputs':cpu,'expected':gold,'produced':produced,**mark});torch.save(saved,N/(sys.argv[2]+'.pt'));torch.testing.assert_close(produced,gold,**reference_tolerance(dt));assert all(torch.equal(x.cpu().contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)) for x,b in zip(args,cpu))
  finally:candidate.cleanup()
  r['checks'].append({**mark,'full_original_reference_and_inputbits_passed':True});save();print('ADALN3_END '+json.dumps(mark),flush=True)
 import ctypes as C
 f=Path('/opt/torch-all2208-20261105/flush');m=json.loads((f/'SOURCE.json').read_text());assert sha(f/'host_kernel.asc')==m['generated_source_sha256'] and sha(f/'flush.aibin')==m['device_binary_sha256'];dll=C.CDLL(str(f/'flush.so'));dll.init.argtypes=[C.c_char_p];dll.init.restype=C.c_int;dll.call.argtypes=[C.c_void_p,C.c_void_p];dll.call.restype=C.c_int;assert dll.init(str(f/'flush.aibin').encode())==0;buf=torch.ones(268435456,dtype=torch.int32,device='npu');assert dll.call(buf.data_ptr(),torch.npu.current_stream().npu_stream)==0;torch.npu.synchronize();assert torch.count_nonzero(buf).item()==0;r.update(status='three_complete_API_4groups_passed',full_output_and_inputs_verified=True,original_1GiB_flush_verified=True)
except BaseException as e:r.update(status='failed_requires_diagnosis',error=repr(e),traceback=traceback.format_exc());raise
finally:
 maps=Path('/proc/self/maps').read_text();(N/(sys.argv[2]+'.maps.txt')).write_text(maps);r['component_sources_sha256']=sources;r['component_source_sha256_unchanged']=all(sha(f)==h for f,h in sources.items());r['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
