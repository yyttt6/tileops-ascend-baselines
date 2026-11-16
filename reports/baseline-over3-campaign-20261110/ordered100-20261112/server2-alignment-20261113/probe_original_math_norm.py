from pathlib import Path
import json,os,sys,hashlib,traceback,datetime
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-layernorm-original-v1');P=Path('/opt/baseline-over3-20261110/ordered100-20261112');sys.path[:0]=[str(P),str(P/'workspace/TileOPs/src'),str(P/'workspace/TileOPs')];owner=sys.argv[1];dest=N/sys.argv[2];g=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text());assert g['owner']==owner and g['phase']=='running' and g['child']['pid']==os.getpgrp();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();r={'status':'initializing','no_formal_or_performance_acceptance':True,'checks':[]};saved=[]
def save():dest.write_text(json.dumps(r,indent=2)+'\n')
save()
try:
 import torch,torch_npu
 from types import SimpleNamespace as T
 from workloads.device import set_run_device
 from workloads.norm import LayerNormWorkload,AdaLayerNormWorkload,GroupNormWorkload,InstanceNormWorkload
 from workloads.numerics import reference_tolerance
 from ops_math_original_norm_layout import norm_layout_run
 torch.set_num_threads(4);set_run_device('cpu');torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});prop=torch.npu.get_device_properties(0);assert (prop.cube_core_num,prop.vector_core_num)==(32,64);build=json.loads((N/'BUILD_RECEIPT.json').read_text());assert build['status']=='CPU_build_complete_runtime_qualification_required' and sha(N/'libops_math_layernorm.so')==build['binary_sha256'] and sha(N/'layer_norm.asc')==build['source_sha256'];torch.ops.load_library(str(N/'libops_math_layernorm.so'));native=torch.ops.allpairs_math_example.layer_norm;r.update(runtime={'hostname':os.uname().nodename,'torch':torch.__version__,'torch_npu':torch_npu.__version__,'cube':32,'vector':64},build_receipt_sha256=sha(N/'BUILD_RECEIPT.json'),original_source_sha256=sha(N/'layer_norm.asc'),loaded_original_binary_sha256=sha(N/'libops_math_layernorm.so'))
 configs=[('LayerNormFwdOp',torch.float16),('AdaLayerNormFwdOp',torch.bfloat16),('GroupNormFwdOp',torch.float32),('InstanceNormFwdOp',torch.float16)]
 for op,dt in configs:
  if op=='LayerNormFwdOp':w=LayerNormWorkload(2,16385,dt);params={'eps':w.eps,'normalized_shape':(16385,)}
  elif op=='AdaLayerNormFwdOp':w=AdaLayerNormWorkload(3,17,dt);params={'eps':w.eps}
  elif op=='GroupNormFwdOp':w=GroupNormWorkload(2,8,(5,7),4,dt);params={'eps':w.eps,'num_groups':4}
  else:w=InstanceNormWorkload(2,3,(5,7),dt);params={'eps':w.eps,'use_input_stats':True}
  torch.manual_seed(718);cpu=w.gen_inputs()
  if op=='AdaLayerNormFwdOp':cpu=(cpu[0],cpu[1][:1],cpu[2][:1])
  gold=w.ref_program(*cpu);args=tuple(x.to('npu') if isinstance(x,torch.Tensor) else x for x in cpu);run=norm_layout_run(T(params=params,inputs=args),op,native);mark={'op':op,'dtype':str(dt),'input_shape':list(cpu[0].shape)};r.update(status='before_original_norm_fullAPI',current=mark);save();print('BASELINE_ORIGINAL_NORM_CALL_BEGIN '+json.dumps(mark),flush=True);actual=run(*args);torch.npu.synchronize();produced=actual.cpu();saved.append({'input':cpu,'expected':gold,'produced':produced,**mark});torch.save(saved,N/(sys.argv[2]+'.pt'));torch.testing.assert_close(produced,gold,**reference_tolerance(dt));assert all(torch.equal(x.cpu().contiguous().view(torch.uint8),before.contiguous().view(torch.uint8)) if isinstance(x,torch.Tensor) else x is before for x,before in zip(args,cpu));r['checks'].append({**mark,'full_original_reference_and_inputbits_passed':True,'tolerance':reference_tolerance(dt)});save();print('BASELINE_ORIGINAL_NORM_CALL_END '+json.dumps(mark),flush=True)
 marker=torch.zeros(1024,dtype=torch.int32,device='npu');marker.add_(1);torch.npu.synchronize();assert torch.equal(marker.cpu(),torch.ones(1024,dtype=torch.int32));import ctypes as C
 f=Path('/opt/torch-all2208-20261105/flush');m=json.loads((f/'SOURCE.json').read_text());assert sha(f/'host_kernel.asc')==m['generated_source_sha256'] and sha(f/'flush.aibin')==m['device_binary_sha256'];dll=C.CDLL(str(f/'flush.so'));dll.init.argtypes=[C.c_char_p];dll.init.restype=C.c_int;dll.call.argtypes=[C.c_void_p,C.c_void_p];dll.call.restype=C.c_int;assert dll.init(str(f/'flush.aibin').encode())==0;buf=torch.ones(268435456,dtype=torch.int32,device='npu');assert dll.call(buf.data_ptr(),torch.npu.current_stream().npu_stream)==0;torch.npu.synchronize();assert torch.count_nonzero(buf).item()==0;r.update(status='original_norm_4full_API_groups_passed',full_output_and_inputs_verified=True,original_1GiB_flush_verified=True,INT32_marker_verified=True)
except BaseException as e:r.update(status='failed_requires_diagnosis',error=repr(e),traceback=traceback.format_exc());raise
finally:
 maps=Path('/proc/self/maps').read_text();(N/(sys.argv[2]+'.maps.txt')).write_text(maps);r['loaded_original_private_binary_present']=str(N/'libops_math_layernorm.so') in maps;r['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
