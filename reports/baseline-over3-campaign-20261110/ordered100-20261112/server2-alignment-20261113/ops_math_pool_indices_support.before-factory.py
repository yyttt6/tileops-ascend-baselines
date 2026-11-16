"""Keep original complete pooling indices run/cleanup AST and coordinate mapping."""
from pathlib import Path
from types import SimpleNamespace
from itertools import product
from math import prod
import ast,copy,hashlib
P=Path(__file__).resolve().parent;N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-pool-indices-v1')
def pool_indices_candidate(case,op,child_factory,deployment_hashes):
 f=P/'workspace/TileOPs/benchmarks/ascend_all_pairs/ops_math/pool_indices.py';raw=f.read_bytes();node=copy.deepcopy(next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='make'));before={n.name:ast.dump(n,include_attributes=False) for n in node.body if isinstance(n,ast.FunctionDef)};changed=0
 for i,n in enumerate(node.body):
  if isinstance(n,ast.ImportFrom) and n.level==1 and n.module=='graph':node.body[i]=ast.Pass();changed+=1
  elif isinstance(n,ast.ImportFrom) and n.module=='benchmarks.ascend_baselines':n.names=[x for x in n.names if x.name!='source_hashes']
 assert changed==1 and before=={n.name:ast.dump(n,include_attributes=False) for n in node.body if isinstance(n,ast.FunctionDef)}
 scope={'__file__':str(f),'Path':Path,'SimpleNamespace':SimpleNamespace,'prod':prod,'product':product,'SOURCE':N/'source','graph_make':child_factory,'deployment_hashes':deployment_hashes,'source_hashes':lambda *paths:{str(Path(p)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(f),'exec'),scope);return scope['make'](case,op)
