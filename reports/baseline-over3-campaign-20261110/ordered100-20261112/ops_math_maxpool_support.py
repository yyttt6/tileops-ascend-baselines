"""Original pool run/reduce AST preserved; only the complete native Amax child bound privately."""
from pathlib import Path
from types import SimpleNamespace
from math import prod
from itertools import product
import ast,copy,hashlib,json,os
P=Path(__file__).resolve().parent
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-maxpool-v1')
V=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-vectorinf-v1')
OPS={'AdaptiveMaxPool2dFwdOp','MaxPool1dFwdOp','MaxPool2dFwdOp','MaxPool3dFwdOp'}
def pool_candidate(case,op,child_factory,deployment_hashes):
 assert op in OPS and case.params.get('divisor_override') is None
 f=P/'workspace/TileOPs/benchmarks/ascend_all_pairs/ops_math/pooling.py';raw=f.read_bytes();assert hashlib.sha256(raw).hexdigest()=='0feeabb79d21cc953803b7c01b5192e69448bf7f3f062181025e4be62b892296';tree=ast.parse(raw);node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make'));before={n.name:ast.dump(n,include_attributes=False) for n in node.body if isinstance(n,ast.FunctionDef)};changed=0
 for i,n in enumerate(node.body):
  if isinstance(n,ast.ImportFrom) and n.level==1 and n.module=='graph':node.body[i]=ast.Pass();changed+=1
  elif isinstance(n,ast.ImportFrom) and n.module=='benchmarks.ascend_baselines':n.names=[alias for alias in n.names if alias.name!='source_hashes']
 assert changed==1
 assert before=={n.name:ast.dump(n,include_attributes=False) for n in node.body if isinstance(n,ast.FunctionDef)}
 source=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-add-v1/source/math/reduce_max')
 def files(name):
  assert name=='AmaxFwdOp';return [p for folder in ['op_host','op_kernel','op_api','op_graph'] for p in (source/folder).rglob('*') if p.is_file() and p.suffix in ('.cpp','.h','.py')]
 scope={'__file__':str(f),'Path':Path,'SimpleNamespace':SimpleNamespace,'prod':prod,'product':product,'files':files,'source_hashes':lambda *paths:{str(Path(p)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},'graph_make':child_factory,'deployment_hashes':deployment_hashes};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(f),'exec'),scope);return scope['make'](case,op)
def candidate_factory(case,library):
 def build():
  import torch
  from dataclasses import replace
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  from ops_math_maxpool_api import MaxOnlyAPI
  assert library=='ops-math';op=case._call.signature.name;assert op in OPS
  proof=N/'MAXPOOL_HOST_QUALIFICATION.json';q=json.loads(proof.read_text());assert q['full_API_and_original_flush_and_private_kernel_verified'] is True and q['execution_hostname']==os.uname().nodename
  manifest=V/'VECTORINF_PACKAGE.json';assert q['native_package_manifest_sha256']==hashlib.sha256(manifest.read_bytes()).hexdigest()
  for name,h in q['qualified_API_source_sha256'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h
  m=json.loads(manifest.read_text());sources={str(manifest):hashlib.sha256(manifest.read_bytes()).hexdigest(),str(proof):hashlib.sha256(proof.read_bytes()).hexdigest()}
  for k,r in m['files'].items():sources[str(V/'private-v1'/k)]=r['sha256']
  for name in ['ops_math_maxpool_api.py','ops_math_maxpool_support.py','ops_math_vectorinf_api.py']:sources[str(P/name)]=hashlib.sha256((P/name).read_bytes()).hexdigest()
  def child_factory(child,child_op):
   assert child_op=='AmaxFwdOp' and child.params['dim']==-1 and child.params['keepdim'] is False and not child.params['_weighted'];api=MaxOnlyAPI(torch,dim=-1,shape=child.params['output'][0]);return Candidate(Implementation(api),'ops-math complete original aclnnAmax','Full native contiguous/axis/descriptors/workspaces/output; signed maximum, no abs preprocessing.',sources,cleanup=api.cleanup)
  out=pool_candidate(case,op,child_factory,lambda:sources);return replace(out,api='ops-math complete aclnnAmax, original full window-packing run AST',source_sha256={**out.source_sha256,**sources})
 return build
