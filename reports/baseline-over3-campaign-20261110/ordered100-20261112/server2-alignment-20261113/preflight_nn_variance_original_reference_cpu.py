from pathlib import Path
import os,sys,ast,json,hashlib,datetime,time,importlib.abc
assert os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD')=='0'
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ['torch_npu','acl']:raise RuntimeError('CPU-onlypreflight prohibits '+name)
sys.meta_path.insert(0,NoNPU())
import torch
assert 'torch_npu' not in sys.modules;torch.set_num_threads(8);P=Path("/opt/baseline-over3-20261110/ordered100-20261112");sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();source=P/'workspace/TileOPs/workloads/reduction.py';assert sha(source)=="b0b4dbc73b94ed137bfba3d523d9172b79817ddfeccea6021ca1ef99ce3908cb";tree=ast.parse(source.read_text());fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_reduction_reference');mod=ast.Module(body=[fn],type_ignores=[]);ast.fix_missing_locations(mod);ns={'torch':torch};exec(compile(mod,str(source),'exec'),ns);rows=[];beg=time.perf_counter()
for i in [683,684]:
 d=P/'prepared-rollout'/str(i);m=json.loads((d/'prepare.json').read_text());assert sha(d/'payload.pt')==m['payload_sha256'];v=torch.load(d/'payload.pt',map_location='cpu',weights_only=False);x=v['inputs'][0];assert x.device.type=='cpu' and x.dtype==torch.float32 and m['params']['correction']==1;ref=ns['_reduction_reference'](m['key'][1],x,m['params']);gold=torch.var_mean(x.double(),dim=-1,correction=1)
 gold=(gold[0].float(),gold[1].float()) if i==684 else (gold[0].float(),);ref=ref if isinstance(ref,(tuple,list)) else (ref,);assert len(ref)==len(gold);checks=[]
 for a,b in zip(ref,gold):
  close=torch.isclose(a,b,rtol=1e-5,atol=1e-5,equal_nan=True);checks.append({'elements':a.numel(),'mismatches':int((~close).sum()),'max_abs':float((a.double()-b.double()).abs().max())})
 rows.append({'ordinal':i,'payload_sha256':m['payload_sha256'],'params':m['params'],'outputs':checks,'all_original_reference_outputs_consistent':all(c['mismatches']==0 for c in checks)})
r={'checked_at_remote':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'torch_npu_imported':False,'source_sha':sha(source),'original_reference_AST_sha':hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest(),'rows':rows,'seconds':time.perf_counter()-beg,'no_reference_checker_or_precision_change':True,'not_native_acceptance':True};out=P/'server2-alignment-20261113/NN_VARIANCE683684_CPU_REFERENCE_PREFLIGHT.json';assert not out.exists();out.write_text(json.dumps(r,indent=2));print(json.dumps(r))
