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
 for dtype in [torch.float16,torch.bfloat16,torch.float32]:
  for count in [8192+37,16384]:
   values=torch.tensor([float('-inf'),float('inf'),float('nan'),0.,-0.,.1,-.1,.9,-.9,1.1,-1.1,65504.,-65504.]);x=values.repeat((count+len(values)-1)//len(values))[:count].to(dtype);gold=torch.trunc(x);device=x.to('npu');packed=torch.nn.functional.pad(device.contiguous().reshape(-1).float(),(0,-count%8192));length=packed.numel();src=torch.full((length+128,),12345.5,dtype=torch.float32,device='npu');dst=torch.full_like(src,-23456.5);inside=src[64:-64];result=dst[64:-64];inside.copy_(packed);torch.npu.synchronize();assert fn(inside.data_ptr(),result.data_ptr(),length,torch.npu.current_stream().npu_stream)==0;torch.npu.synchronize();got=result[:count].to(dtype).cpu();torch.testing.assert_close(got,gold,rtol=0,atol=0,equal_nan=True);zeros=gold==0;assert torch.equal(torch.signbit(got[zeros]),torch.signbit(gold[zeros]));assert torch.equal(device.cpu().view(torch.uint8),x.view(torch.uint8));assert bool(torch.all(src[:64]==12345.5)) and bool(torch.all(src[-64:]==12345.5)) and bool(torch.all(dst[:64]==-23456.5)) and bool(torch.all(dst[-64:]==-23456.5));assert torch.equal(inside.cpu().view(torch.uint8),packed.cpu().view(torch.uint8));out['rows'].append({'dtype':str(dtype),'count':count,'all_values_exact_toward_zero_and_NaNs_checked':True,'signed_zeros_preserved':True,'input_bits_preserved':True,'both_input_output_64element_canaries_intact':True,'full_padded_output_computed_before_crop':True});put();print('TRUNC_EXISTING_KERNEL_HOST_TILE_FULL_OUTPUT_CANARIES_PASS',str(dtype),count,flush=True)
 out['status']='functional_full6_host_tile_API_passed_not_formal_large_case_acceptance'
except BaseException as e:out.update(status='failed_requires_diagnosis',error=repr(e),traceback=traceback.format_exc());raise
finally:
 out['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();maps=Path('/proc/self/maps').read_text();(N/(sys.argv[2]+'.maps.txt')).write_text(maps);libs={l.split()[-1] for l in maps.splitlines() if len(l.split())>=6 and l.split()[-1].startswith('/') and '.so' in l.split()[-1]};out['loaded_library_sha256']={s:sha(s) for s in libs if Path(s).is_file()};put()
