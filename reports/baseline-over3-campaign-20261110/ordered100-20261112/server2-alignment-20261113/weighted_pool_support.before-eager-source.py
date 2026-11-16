"""Original pool run/reduce AST preserved; only the complete native Mean child bound privately."""
from pathlib import Path
from types import SimpleNamespace
from math import prod
from itertools import product
import ast,copy,hashlib,json,os
P=Path(__file__).resolve().parent
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-pool-mean-v1')
V=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-pool-mean-v1')
OPS={'AdaptiveAvgPool2dFwdOp','AvgPool1dFwdOp','AvgPool2dFwdOp','AvgPool3dFwdOp'}
def weighted_pool_candidate(case,op,child_factory,deployment_hashes):
 assert op in OPS and case.params.get('divisor_override') is not None
 f=P/'workspace/TileOPs/benchmarks/ascend_all_pairs/ops_math/pooling.py';raw=f.read_bytes();assert hashlib.sha256(raw).hexdigest()=='0feeabb79d21cc953803b7c01b5192e69448bf7f3f062181025e4be62b892296';tree=ast.parse(raw);node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make'));before={n.name:ast.dump(n,include_attributes=False) for n in node.body if isinstance(n,ast.FunctionDef)};changed=0
 for i,n in enumerate(node.body):
  if isinstance(n,ast.ImportFrom) and n.level==1 and n.module=='graph':node.body[i]=ast.Pass();changed+=1
  elif isinstance(n,ast.ImportFrom) and n.module=='benchmarks.ascend_baselines':n.names=[alias for alias in n.names if alias.name!='source_hashes']
 assert changed==1
 assert before=={n.name:ast.dump(n,include_attributes=False) for n in node.body if isinstance(n,ast.FunctionDef)}
 source=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-add-v1/source/math/reduce_mean')
 def files(name):
  assert name=='MeanFwdOp';return [p for folder in ['op_host','op_kernel','op_api','op_graph'] for p in (source/folder).rglob('*') if p.is_file() and p.suffix in ('.cpp','.h','.py')]
 scope={'__file__':str(f),'Path':Path,'SimpleNamespace':SimpleNamespace,'prod':prod,'product':product,'files':files,'source_hashes':lambda *paths:{str(Path(p)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},'graph_make':child_factory,'deployment_hashes':deployment_hashes};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(f),'exec'),scope);return scope['make'](case,op)
def candidate_factory(case,library):
 def build():
  import torch
  from dataclasses import replace
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  from ops_math_weighted_pool_api import WeightedMeanAPI
  assert library=='ops-math' and case._call.signature.name in {'AvgPool2dFwdOp','AvgPool3dFwdOp'} and case.params['divisor_override']==7
  qfile=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-weighted-pool-v1/WEIGHTED_POOL_HOST_QUALIFICATION.json');q=json.loads(qfile.read_text());assert q['full_API_and_original_flush_and_private_kernel_verified'] and q['execution_hostname']==os.uname().nodename
  for name,h in q['qualified_API_source_sha256'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h
  apis=[];sources={str(qfile):hashlib.sha256(qfile.read_bytes()).hexdigest()}
  def child(c,op):
   assert op=='MeanFwdOp' and c.params['_weighted'] and c.params['dim']==-1 and not c.params['keepdim'];api=WeightedMeanAPI(torch);apis.append(api);sources.update(api.sources);return Candidate(Implementation(api),'complete Mul+Div+ReduceSum','Original DAG order, all inputs/auxiliary allocations/casts/pack/nativework retained',api.sources,cleanup=api.cleanup)
  out=weighted_pool_candidate(case,case._call.signature.name,child,lambda:sources);sources[str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();return replace(out,api='ops-math complete Mul+Div+ReduceSum, original weighted pooling AST',source_sha256={**out.source_sha256,**sources})
 return build
