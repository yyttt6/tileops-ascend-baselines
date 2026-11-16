"""Guard-only functional probe of rebuilt complete ForeachMulList API; not baseline timing."""
import pathlib,sys,os,json,hashlib,ctypes as C,fcntl,datetime,traceback
N=pathlib.Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-unary-trig-v1');owner=sys.argv[1];dest=N/sys.argv[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ticks(pid):return int(pathlib.Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19])
with pathlib.Path('/run/lock/ascend-npu/device-0.lock').open() as f:
 g=json.load(f);ch=g['child'];assert g['phase']=='running' and g['device']==0 and g['owner']==owner and ch['pid']==os.getpgrp() and ticks(ch['pid'])==ch['start_ticks'] and ticks(g['pid'])==g['start_ticks'] and g['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:pass
 else:raise RuntimeError('Hostguard kernel lock not held')
out={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'initializing','rows':[],'not_baseline_15raw_or_kernel_attribution_acceptance':True,'device_kernel_provenance_pending':True,'guard_owner':owner,'probe_source_sha256':sha(pathlib.Path(__file__))}
def save():dest.write_text(json.dumps(out,indent=2)+'\n')
save()
try:
 import torch,torch_npu
 torch.set_num_threads(8);torch.npu.set_device(0);torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype'});prop=torch.npu.get_device_properties(0);assert (prop.cube_core_num,prop.vector_core_num)==(32,64)
 out['runtime']={'torch':torch.__version__,'torch_npu':torch_npu.__version__,'cube':32,'vector':64};m=json.loads((N/'TRIG_UNARY_PACKAGE.v2.json').read_text());root=pathlib.Path(m['staging_root']);binary=root/'op_api/lib/libcust_opapi.so';assert sha(binary)==m['libraries']['op_api/lib/libcust_opapi.so']['sha256'];out['binary_sha256']=sha(binary)
 lib=C.CDLL(str(binary),mode=os.RTLD_NOW|os.RTLD_LOCAL);base=C.CDLL('libnnopbase.so');base.aclCreateTensor.argtypes=[C.POINTER(C.c_int64),C.c_uint64,C.c_int,C.POINTER(C.c_int64),C.c_int64,C.c_int,C.POINTER(C.c_int64),C.c_uint64,C.c_void_p];base.aclCreateTensor.restype=C.c_void_p;base.aclCreateTensorList.argtypes=[C.POINTER(C.c_void_p),C.c_uint64];base.aclCreateTensorList.restype=C.c_void_p;base.aclDestroyTensorList.argtypes=[C.c_void_p];base.aclDestroyTensorList.restype=C.c_int;base.aclDestroyTensor.argtypes=[C.c_void_p];base.aclDestroyTensor.restype=C.c_int
 for op in ['ForeachCos','ForeachSign','ForeachSin','ForeachTanh']:
  get=getattr(lib,'aclnn'+op+'GetWorkspaceSize');get.argtypes=[C.c_void_p,C.c_void_p,C.POINTER(C.c_uint64),C.POINTER(C.c_void_p)];get.restype=C.c_int
  run=getattr(lib,'aclnn'+op);run.argtypes=[C.c_void_p,C.c_uint64,C.c_void_p,C.c_void_p];run.restype=C.c_int
 class DlInfo(C.Structure):_fields_=[('fname',C.c_char_p),('base',C.c_void_p),('sname',C.c_char_p),('saddr',C.c_void_p)]
 dl=C.CDLL(None);dl.dladdr.argtypes=[C.c_void_p,C.POINTER(DlInfo)];dl.dladdr.restype=C.c_int
 for name in [x+y for x in ['aclnnForeachCos','aclnnForeachSign','aclnnForeachSin','aclnnForeachTanh'] for y in ['', 'GetWorkspaceSize']]:
  info=DlInfo();assert dl.dladdr(C.cast(getattr(lib,name),C.c_void_p),C.byref(info)) and pathlib.Path(info.fname.decode()).resolve()==binary.resolve()
 def check(rc,name):
  if rc:raise RuntimeError(name+' returned '+str(rc))
 for op,dtype,code in [(op,dt,code) for op in ['ForeachCos','ForeachSign','ForeachSin','ForeachTanh'] for dt,code in [(torch.float16,1),(torch.bfloat16,27),(torch.float32,0)]]:
  keep=[];lists=[];unowned=[];row={'op':op,'dtype':str(dtype),'status':'running'};out['rows'].append(row);save()
  try:
   xs=[torch.tensor([[1,-2,.5],[3.25,-4.5,5.75]],dtype=dtype),torch.tensor([-3,.5,2,4,1,2,.25,8,4,-2,1,1.25,3,.75,4,-8,2],dtype=dtype)];ys=[torch.tensor([[2,3,-1],[-2,.5,4]],dtype=dtype),torch.tensor([2]*17,dtype=dtype)];gold=[({"ForeachCos":torch.cos,"ForeachSign":torch.sign,"ForeachSin":torch.sin,"ForeachTanh":torch.tanh}[op](x)) for x in xs];a=[x.to('npu') for x in xs];b=[y.to('npu') for y in ys];outputs=[torch.empty_like(x) for x in a];keep.extend(a+b+outputs)
   def tensorlist(values):
    handles=[]
    for t in values:
     dims=(C.c_int64*t.ndim)(*t.shape);strides=(C.c_int64*t.ndim)(*t.stride());keep.extend([dims,strides]);ptr=base.aclCreateTensor(dims,t.ndim,code,strides,0,2,dims,t.ndim,t.data_ptr());assert ptr;handles.append(ptr);unowned.append(ptr)
    array=(C.c_void_p*len(handles))(*handles);keep.append(array);ptr=base.aclCreateTensorList(array,len(handles));assert ptr;lists.append(ptr)
    # Original SDKsample: destroyingTensorList also owns/destroys its tensor descriptors.
    for h in handles:unowned.remove(h)
    return ptr
   left,result=tensorlist(a),tensorlist(outputs);size=C.c_uint64();executor=C.c_void_p();get=getattr(lib,'aclnn'+op+'GetWorkspaceSize');run=getattr(lib,'aclnn'+op);check(get(left,result,C.byref(size),C.byref(executor)),'GetWorkspaceSize');workspace=torch.empty(size.value,dtype=torch.uint8,device='npu') if size.value else None;keep.append(workspace);check(run(workspace.data_ptr() if workspace is not None else None,size.value,executor,torch.npu.current_stream().npu_stream),op);torch.npu.synchronize()
   for x,y,orig_a,orig_b in zip(outputs,gold,a,b):torch.testing.assert_close(x.cpu(),y,rtol={torch.float16:1e-3,torch.bfloat16:1.6e-2,torch.float32:1e-5}[dtype],atol={torch.float16:1e-3,torch.bfloat16:1.6e-2,torch.float32:1e-5}[dtype])
   for actual,original in zip(a+b,xs+ys):torch.testing.assert_close(actual.cpu(),original,rtol=0,atol=0)
   row.update(status='full_outputs_and_inputs_verified',outputs=2,workspace_bytes=size.value,output_zero_sign_diagnostic=[bool(torch.equal(torch.signbit(x.cpu()),torch.signbit(y))) for x,y in zip(outputs,gold)])
  finally:
   torch.npu.synchronize()
   for ptr in lists:check(base.aclDestroyTensorList(ptr),'DestroyTensorList')
   for ptr in unowned:check(base.aclDestroyTensor(ptr),'DestroyTensor')
   save()
 out['status']='functional_probe_passed_not_formal_baseline'
except BaseException as e:out.update(status='failed_requires_diagnosis',error=repr(e),traceback=traceback.format_exc());raise
finally:
 out['checked_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();maps=pathlib.Path('/proc/self/maps').read_text();(N/(sys.argv[2]+'.maps.txt')).write_text(maps);names={s.split()[-1] for s in maps.splitlines() if len(s.split())>=6 and s.split()[-1].startswith('/') and '.so' in s.split()[-1]};out['loaded_library_sha256']={s:sha(pathlib.Path(s)) for s in names if pathlib.Path(s).is_file()};save()
print(json.dumps({'status':out['status'],'rows':out['rows']}))
