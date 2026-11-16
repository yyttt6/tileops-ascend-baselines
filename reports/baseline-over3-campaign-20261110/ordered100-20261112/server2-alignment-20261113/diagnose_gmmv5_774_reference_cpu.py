from pathlib import Path
import os,sys,ast,json,hashlib,time,datetime,types,importlib.abc
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0'
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='torch_npu' or fullname.startswith('torch_npu.'):raise RuntimeError('CPU-only diagnosis prohibits torch_npu imports')
sys.meta_path.insert(0,NoNPU())
import torch
assert 'torch_npu' not in sys.modules;torch.set_num_threads(8);start=time.perf_counter();P=Path("/opt/baseline-over3-20261110/ordered100-20261112");prep=P/'prepared-rollout/774';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();meta=json.loads((prep/'prepare.json').read_text());assert sha(prep/'payload.pt')==meta['payload_sha256'];source=P/'workspace/TileOPs/workloads/gemm.py';assert sha(source)=="18b739700d065e64f67bcf27ab8263e8252fd93748f411868efa95a07ca9283c"
tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GroupedGemmWorkload');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='ref_program');mod=ast.Module(body=[fn],type_ignores=[]);ast.fix_missing_locations(mod);ns={'torch':torch};exec(compile(mod,str(source),'exec'),ns);payload=torch.load(prep/'payload.pt',map_location='cpu',weights_only=False);args=payload['inputs'];assert all(x.device.type=='cpu' for x in args);a,b,sizes,offsets=args;assert meta['params']['transpose_a']==False and meta['params']['transpose_b']==True
original=ns['ref_program'](types.SimpleNamespace(transpose_a=False,transpose_b=True),*args);exact=torch.empty_like(original);startrow=0;groups=[]
for i,size in enumerate(sizes.tolist()):
 assert int(offsets[i])==startrow;end=startrow+size
 if size:exact[startrow:end]=(a[startrow:end].double()@b[i].double().T).to(a.dtype)
 groups.append({'group':i,'start':startrow,'rows':size});startrow=end
assert startrow==a.shape[0]
def comparison(x,y):
 flatx=x.reshape(-1);flaty=y.reshape(-1);close=torch.isclose(flatx,flaty,rtol=0.001,atol=0.001,equal_nan=True);idx=(~close).nonzero().flatten();return {'elements':x.numel(),'mismatches':int(idx.numel()),'first1048576_mismatches':int((~close[:1048576]).sum()),'max_abs':float((x.double()-y.double()).abs().max()),'examples':[{'flat_index':int(i),'original_cpu':float(flatx[i]),'fp64_rounded_original_dtype':float(flaty[i])} for i in idx[:8]]}
r={'checked_at_remote':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'torch':torch.__version__,'torch_npu_imported':'torch_npu' in sys.modules,'source_sha':sha(source),'ref_function_AST_sha':hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest(),'original_payload_sha':meta['payload_sha256'],'params':meta['params'],'groups':groups,'original_cpu_vs_fullK_FP64_rounded_to_original_storage':comparison(original,exact),'native_result_available':False,'limitation':'NPU produced output was not retained on original checker failure. This diagnoses CPU reference only; does not prove native output equals FP64 or authorize reference replacement.','not_coverage_or_oracle_change':True,'seconds':time.perf_counter()-start};out=P/'server2-alignment-20261113/GMMV5_774_CPU_REFERENCE_FP64_DIAGNOSIS.json';assert not out.exists();out.write_text(json.dumps(r,indent=2));print(json.dumps(r))
