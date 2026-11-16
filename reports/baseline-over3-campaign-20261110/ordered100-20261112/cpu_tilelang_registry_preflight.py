"""CPU-only registry import: device Python modules are forbidden."""
import sys,pathlib,importlib.abc,importlib,json,datetime
P=pathlib.Path(__file__).resolve().parent;T=P/'workspace/TileOPs';sys.path[:0]=[str(T/'src'),str(T)];sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU registry preflight forbids '+name,name=name)
sys.meta_path.insert(0,NoNPU());out={}
for name in ['tilelang','tla']:
 try:
  m=importlib.import_module('benchmarks.ascend_all_pairs.'+name);out[name]={'imported':True,'providers':len(m.PROVIDERS)}
 except Exception as e:
  import traceback
  out[name]={'imported':False,'error':repr(e),'traceback':traceback.format_exc()}
assert 'torch_npu' not in sys.modules and 'acl' not in sys.modules
report={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'NoNPU':True,'libraries':out};(P/'TILELANG_CPU_REGISTRY_PREFLIGHT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
