"""Prove op-level registry availability without any device module or factory execution."""
import sys,pathlib,importlib.abc,importlib,json,datetime,os,hashlib,ast
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0';P=pathlib.Path(__file__).resolve().parent;T=P/'workspace/TileOPs';sys.path[:0]=[str(T/'src'),str(T),'/opt/baseline-over3-20261110/unified-stack-v1/references/fla-v1/site-packages','/opt/tileops-ci/runtime/lib/python3.12/site-packages']
class NoNPU(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch_npu','acl'):raise ModuleNotFoundError('CPU route catalog forbids '+name,name=name)
sys.meta_path.insert(0,NoNPU())
from benchmarks.ascend_all_pairs import tla,registry
from benchmarks.ops import bench_ascend_classics as classic
node=next(n for n in ast.parse(pathlib.Path(classic.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='candidate_factories')
expected=ast.parse('''def candidate_factories(case):
 """Resolve each provider separately so a failed import cannot hide its peers."""
 return {library: partial(factory, case) for library, factory in PROVIDERS.get(case._call.signature.name, {}).items()}
''').body[0]
assert ast.dump(node,include_attributes=False)==ast.dump(expected,include_attributes=False)
ops=set(tla.PROVIDERS)|{op for op,providers in classic.PROVIDERS.items() if 'tilelang-ascend' in providers};files=[*pathlib.Path(tla.__file__).parent.rglob('*.py'),pathlib.Path(registry.__file__),pathlib.Path(classic.__file__)];sources={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files};assert 'torch_npu' not in sys.modules and 'acl' not in sys.modules
rows=json.loads((P/'TASKS.json').read_text());missing=[i for i,r in enumerate(rows) if r['library']=='tilelang-ascend' and r['op'] not in ops]
r={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'NoNPU':True,'no_factory_execution':True,'classic_lookup_AST_proved_op_only':True,'libraries':{'tilelang-ascend':{'available_ops':sorted(ops),'missing_route_indices':missing}},'sources':sources,'registry_absence_not_native_API_absence_or_performance_acceptance':True};(P/'CPU_TLA_ROUTE_CATALOG.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
