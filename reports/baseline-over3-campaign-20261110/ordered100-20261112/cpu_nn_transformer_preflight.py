import sys,pathlib,importlib.abc,importlib,json,datetime,os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
P=pathlib.Path(__file__).resolve().parent;T=P/'workspace/TileOPs'
sys.path[:0]=[str(T/'src'),str(T),'/opt/baseline-over3-20261110/unified-stack-v1/references/fla-v1/site-packages','/opt/tileops-ci/runtime/lib/python3.12/site-packages']
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU registry forbids '+name,name=name)
sys.meta_path.insert(0,NoNPU());out={}
for name in ['ops_nn','ops_transformer','ops_math']:
 try:
  m=importlib.import_module('benchmarks.ascend_all_pairs.'+name);out[name]={'imported':True,'providers':len(m.PROVIDERS),'provider_names':sorted(m.PROVIDERS)}
 except Exception as e:
  import traceback
  out[name]={'imported':False,'error':repr(e),'traceback':traceback.format_exc()}
assert 'torch_npu' not in sys.modules and 'acl' not in sys.modules
r={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'NoNPU':True,'libraries':out};(P/'NN_TRANSFORMER_CPU_PREFLIGHT.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
