import pathlib,sys,os,importlib.abc,json
P=pathlib.Path(__file__).resolve().parent;T=P/'workspace/TileOPs';sys.path[:0]=[str(T/'src'),str(T)];sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages');os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU registration proof forbids '+name,name=name)
sys.meta_path.insert(0,NoNPU())
from benchmarks.ascend_all_pairs.registry import LIBRARY_MODULES
module=__import__('importlib').import_module('benchmarks.ascend_all_pairs.'+LIBRARY_MODULES[sys.argv[1]])
assert isinstance(module.PROVIDERS,dict);assert 'torch_npu' not in sys.modules and 'acl' not in sys.modules
print(json.dumps({'library':sys.argv[1],'status':'CPU_registry_import_verified','registered_ops':len(module.PROVIDERS),'no_factory_or_device_calls':True}))
