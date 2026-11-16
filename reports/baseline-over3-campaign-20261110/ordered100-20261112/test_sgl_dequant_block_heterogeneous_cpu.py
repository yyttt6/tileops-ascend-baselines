"""CPU surrogate only: same packing/casts/zeros/source AST, never native acceptance."""
import pathlib,sys,types,dataclasses,importlib.util,json,datetime,hashlib
P=pathlib.Path(__file__).resolve().parent
import torch
assert not any(n in sys.modules for n in ['torch_npu','acl'])
@dataclasses.dataclass
class Implementation:run:object
@dataclasses.dataclass
class Candidate:
 implementation:object;api:str;adaptation:str;source_sha256:dict
api=types.ModuleType('benchmarks.api');api.Implementation=Implementation;sys.modules['benchmarks.api']=api
ab=types.ModuleType('benchmarks.ascend_baselines');ab.Candidate=Candidate;ab.LIBRARIES=P/'workspace/ascend_baselines';ab.source_hashes=lambda *paths:{str(p):hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in paths if pathlib.Path(p).is_file()}
calls=[]
def native(x,scale,shift,scale_constant=0):
 assert x.ndim==3 and scale_constant==0 and scale.numel()==x.shape[-1];calls.append((list(x.shape),scale.numel(),shift.numel()));return x*scale.reshape(1,1,-1)+shift
ab.local_function=lambda *a,**k:native;sys.modules['benchmarks.ascend_baselines']=ab
sgl=types.ModuleType('benchmarks.ascend_all_pairs.sgl');sys.modules[sgl.__name__]=sgl
for name in ['additional','extra_arithmetic']:
 f=P/'workspace/TileOPs/benchmarks/ascend_all_pairs/sgl'/(name+'.py');spec=importlib.util.spec_from_file_location(sgl.__name__+'.'+name,f);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);setattr(sgl,name,m)
sys.path.insert(0,str(P/'workspace/isolated_wrappers'));import sgl_dequant_block_api as helper
fn,proof=helper.clone_functions(sgl.extra_arithmetic,sgl.additional)
rows=[];torch.manual_seed(0)
for dtype in ['float16','bfloat16','float32']:
 for shape in [(17,4099),(3,256),(2,129),(13,7)]:
  for scalar in ['ramp','alternating','random']:
   q=torch.randint(-128,128,shape,dtype=torch.int8)
   if shape==(13,7):q=q.T.contiguous().T
   scale=(torch.linspace(-1.25,2.0,shape[0]*((shape[1]+127)//128),dtype=torch.float32).reshape((shape[0],(shape[1]+127)//128)) if scalar=='ramp' else (torch.arange(shape[0]*((shape[1]+127)//128),dtype=torch.float32).remainder(2).mul(2).sub(1).reshape((shape[0],(shape[1]+127)//128)) if scalar=='alternating' else torch.randn((shape[0],(shape[1]+127)//128),dtype=torch.float32)));case=types.SimpleNamespace(inputs=(q,scale),params={'out_dtype':dtype},_call=types.SimpleNamespace(signature=types.SimpleNamespace(name='INT8DequantPerBlockFwdOp')))
   old=sgl.extra_arithmetic.dequant(case);new=fn(case);calls.clear();a=old.implementation.run(q,scale);oldcall=calls[-1];b=new.implementation.run(q,scale);newcall=calls[-1]
   assert a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
   assert oldcall[1]==q.numel() and newcall[1]==scale.numel() and oldcall[2]==q.numel() and newcall[2]==shape[0]*((shape[1]+127)//128)*128
   rows.append({'dtype':dtype,'shape':shape,'scalar':repr(scalar),'bitwise_equal':True,'native_fullzeros_preserved':True,'old_scale_elements':oldcall[1],'new_scale_elements':newcall[1]})
(P/'SGL_DEQUANT_BLOCK_HETEROGENEOUS_CPU_PROOF.json').write_text(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cases':len(rows),'heterogeneous_per_row_or_block_scales':True,'proof':proof,'rows':rows,'helper_sha256':hashlib.sha256(pathlib.Path(helper.__file__).read_bytes()).hexdigest(),'CPU_surrogate_only_not_native_acceptance':True,'no_NPU_imported':not any(n in sys.modules for n in ['torch_npu','acl'])},indent=2));print('CPU_MAPPING_AND_AST',len(rows),'passed; no native performance claim')
