"""Guarded existing Trunc API host-shape qualification, not frozen benchmark timing."""
from pathlib import Path
import sys,os,json,hashlib,ctypes as C,fcntl,datetime,traceback
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-tensor-trunc-host-tile-v1');owner=sys.argv[1];dest=N/sys.argv[2];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
with open('/run/lock/ascend-npu/device-0.lock') as f:
 g=json.load(f);assert g['phase']=='running' and g['owner']==owner and g['child']['pid']==os.getpgrp()
 try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:pass
 else:raise RuntimeError('Hostguard must cover imports and complete native run')
proof=json.loads((N/'CPU_BOUNDARY.json').read_text());assert any(x['guard_boot_id']==g['boot_id'] and x['device_node']==g['device_node'] and x['verified_UB_capacity_lower_bound_bytes']>65536 for x in proof['host_hardware_UB_evidence'].values());build=json.loads((N/'BUILD_RECEIPT.json').read_text());assert build['exit_code']==0 and sha(N/'libops_tensor_trunc_tile.so')==build['binary_sha256'];out={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'initializing','rows':[],'not_formal15raw_or_coverage_acceptance':True,'guard_owner':owner,'build_receipt_sha256':sha(N/'BUILD_RECEIPT.json'),'CPU_boundary_sha256':sha(N/'CPU_BOUNDARY.json'),'source_sha256':sha(__file__)}
def put():dest.write_text(json.dumps(out,indent=2)+'\n')
put()
try:
 import torch,torch_npu
 torch.set_num_threads(8);torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});props=torch.npu.get_device_properties(0);assert (props.cube_core_num,props.vector_core_num)==(32,64) and '950' in props.name;lib=C.CDLL(str(N/'libops_tensor_trunc_tile.so'),mode=os.RTLD_NOW|os.RTLD_LOCAL);fn=lib.allpairs_opstensor_trunc_tile;fn.argtypes=[C.c_void_p,C.c_void_p,C.c_int64,C.c_void_p];fn.restype=C.c_int
 values=torch.tensor([0.,-0.,.1,-.1,.9,-.9,1.1,-1.1,float('inf'),float('-inf'),float('nan')],dtype=torch.float16);x=values.repeat((8192+len(values)-1)//len(values))[:8192];device=x.to('npu');packed=device.float();result=torch.full_like(packed,-23456.5);torch.npu.synchronize();assert fn(packed.data_ptr(),result.data_ptr(),8192,torch.npu.current_stream().npu_stream)==0;torch.npu.synchronize()
 stages={'CPU_input_F16':x[:11],'device_input_F16':device[:11].cpu(),'widened_input_F32':packed[:11].cpu(),'native_output_F32':result[:11].cpu(),'CPU_cast_native_to_F16':result[:11].cpu().to(torch.float16),'NPU_cast_native_to_F16':result[:11].to(torch.float16).cpu(),'CPU_expected_F16':torch.trunc(x[:11]),'NPU_cast_control_F16':torch.tensor([-0.,0.],device='npu',dtype=torch.float32).to(torch.float16).cpu()}
 out['stages']={k:{'dtype':str(t.dtype),'values_repr':[repr(v) for v in t.tolist()],'signbit':torch.signbit(t).tolist(),'bytes':t.view(torch.uint8).tolist()} for k,t in stages.items()};out['input_bits_preserved']=torch.equal(device.cpu().view(torch.uint8),x.view(torch.uint8));out['diagnostic_only_not_qualification']=True
 out['status']='stage_capture_completed_not_acceptance'
except BaseException as e:out.update(status='failed_requires_diagnosis',error=repr(e),traceback=traceback.format_exc());raise
finally:
 out['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();maps=Path('/proc/self/maps').read_text();(N/(sys.argv[2]+'.maps.txt')).write_text(maps);libs={l.split()[-1] for l in maps.splitlines() if len(l.split())>=6 and l.split()[-1].startswith('/') and '.so' in l.split()[-1]};out['loaded_library_sha256']={s:sha(s) for s in libs if Path(s).is_file()};put()
