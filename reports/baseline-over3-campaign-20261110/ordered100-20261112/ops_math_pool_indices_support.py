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

def candidate_factory(case,library):
 def build():
  import torch,os,json
  from dataclasses import replace
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  from ops_math_pool_indices_api import MaxDimAPI
  assert library=='ops-math';op=case._call.signature.name;assert op in {'MaxPool1dIndicesFwdOp','MaxPool2dIndicesFwdOp','MaxPool3dIndicesFwdOp','AdaptiveMaxPool2dIndicesFwdOp'}
  f=N/'POOL_INDICES_HOST_QUALIFICATION.json';q=json.loads(f.read_text());assert q['full_API_and_original_flush_and_private_kernel_verified'] and q['execution_hostname']==os.uname().nodename;assert hashlib.sha256((N/'MATH_POOL_INDICES_PACKAGE.json').read_bytes()).hexdigest()==q['native_package_manifest_sha256']
  for name,h in q['qualified_API_source_sha256'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h
  api=MaxDimAPI(torch);sources={**api.sources,str(f):hashlib.sha256(f.read_bytes()).hexdigest()}
  for name in ['ops_math_pool_indices_api.py','ops_math_pool_indices_support.py','ops_math_l1_api.py']:
   f=P/name;sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  def child(c,childop):
   assert childop=='AmaxFwdOp' and c.params['_argvalue'];return Candidate(Implementation(api),'ops-math complete aclnnMaxDim bothoutputs','Full original two-stage API returns int64indices+values; all native casts/descriptors/workspaces timed',sources,cleanup=api.cleanup)
  out=pool_indices_candidate(case,op,child,lambda:sources);return replace(out,api='ops-math complete aclnnMaxDim, original full pool window+indices AST',source_sha256={**out.source_sha256,**sources})
 return build
