import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import pathlib,sys,types,ast,json,datetime,hashlib
import torch
P=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(P/'workspace/isolated_wrappers'))
import sgl_lerp_tensor_pad_api as helper
source=P/'workspace/TileOPs/benchmarks/ascend_all_pairs/sgl/extra_arithmetic.py'
mod=types.ModuleType('frozen_extra');mod.__file__=str(source)
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_affine')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),vars(mod))
factory=types.ModuleType('benchmarks.ascend_baselines');factory.LIBRARIES=P;factory.source_hashes=lambda *paths:{}
factory.local_function=lambda *a:lambda x,y,factor:x*factor+y
sys.modules['benchmarks']=types.ModuleType('benchmarks');sys.modules['benchmarks.ascend_baselines']=factory
old,_=mod._affine(None);newfactory,proof=helper.clone_affine(mod);new,_=newfactory(None)
cases=[]
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 for shape in [(17,), (1,4096),(3,4097),(2,4096)]:
  for kind in ['ordinary','nonfinite_signed_zero']:
   x=torch.randn(shape,dtype=dtype);y=torch.randn(shape,dtype=dtype)
   if kind!='ordinary':x.flatten()[:6]=torch.tensor([float('nan'),float('inf'),-float('inf'),0.,-0.,1.],dtype=dtype)
   for factor in [-1.,1.]:
    a,b=old(x,y,factor),new(x,y,factor)
    assert torch.equal(a.view(torch.int16 if dtype!=torch.float32 else torch.int32),b.view(torch.int16 if dtype!=torch.float32 else torch.int32))
    cases.append({'dtype':str(dtype),'shape':shape,'kind':kind,'factor':factor,'bitwise_equal':True,'tail':-x.numel()%4096})
out={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':len(cases),'rows':cases,'proof':proof,'CPU_surrogate_only_not_native_acceptance':True,'helper_sha256':hashlib.sha256(pathlib.Path(helper.__file__).read_bytes()).hexdigest(),'no_NPU_imported':not any(k.startswith('torch_npu') for k in sys.modules)}
assert out['no_NPU_imported'];(P/'SGL_LERP_TENSOR_CPU_PROOF.json').write_text(json.dumps(out,indent=2));print('CPU_ONLY_BITWISE_CASES',len(cases))
