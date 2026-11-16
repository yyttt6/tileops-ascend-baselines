"""Frozen chunk/tail/metadata run+reduce AST, complete original Mean with FP32 accumulator."""
from pathlib import Path
from types import SimpleNamespace
import ast,copy,hashlib,json,os
P=Path(__file__).resolve().parent;N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-chunk-mean-v1');V=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-pool-mean-v1')
def chunk_candidate(case,child_factory,deployment_hashes):
 assert case.params.get('accum_dtype','float32')=='float32';f=P/'workspace/TileOPs/benchmarks/ascend_all_pairs/ops_math/mean_pooling.py';raw=f.read_bytes();assert hashlib.sha256(raw).hexdigest()=='946298a8b6fe83b89a894e2da7999758983e2506bb7ad33e6c4174bf2bae1bbb';node=copy.deepcopy(next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='make'));before={n.name:ast.dump(n,include_attributes=False) for n in node.body if isinstance(n,ast.FunctionDef)};changed=0
 for i,n in enumerate(node.body):
  if isinstance(n,ast.ImportFrom) and n.level==1 and n.module=='graph':node.body[i]=ast.Pass();changed+=1
  elif isinstance(n,ast.ImportFrom) and n.module=='benchmarks.ascend_baselines':n.names=[x for x in n.names if x.name!='source_hashes']
 assert changed==1 and before=={n.name:ast.dump(n,include_attributes=False) for n in node.body if isinstance(n,ast.FunctionDef)}
 scope={'__file__':str(f),'Path':Path,'SimpleNamespace':SimpleNamespace,'SOURCE':V/'source','graph_make':child_factory,'deployment_hashes':deployment_hashes,'source_hashes':lambda *paths:{str(Path(p)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(f),'exec'),scope);return scope['make'](case)
def candidate_factory(case,library):
 def build():
  import torch
  from dataclasses import replace
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  from ops_math_chunk_mean_api import ChunkMeanAPI
  assert library=='ops-math' and case._call.signature.name=='MeanPoolingFwdOp' and case.params['accum_dtype']=='float32';qfile=N/'CHUNK_MEAN_HOST_QUALIFICATION.json';q=json.loads(qfile.read_text());assert q['full_API_and_original_flush_and_private_kernel_verified'] and q['execution_hostname']==os.uname().nodename;mf=V/'MATH_POOL_MEAN_PACKAGE.json';assert hashlib.sha256(mf.read_bytes()).hexdigest()==q['native_package_manifest_sha256']
  for name,h in q['qualified_API_source_sha256'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h
  sources={str(mf):hashlib.sha256(mf.read_bytes()).hexdigest(),str(qfile):hashlib.sha256(qfile.read_bytes()).hexdigest()};m=json.loads(mf.read_text())
  for k,r in m['files'].items():sources[str(Path(m['package'])/k)]=r['sha256']
  for name in ['ops_math_chunk_mean_api.py','ops_math_chunk_mean_support.py','ops_math_pool_mean_api.py','ops_math_l1_api.py']:sources[str(P/name)]=hashlib.sha256((P/name).read_bytes()).hexdigest()
  def child(c,op):
   assert op=='MeanFwdOp' and c.params['dim']==2 and not c.params['keepdim'];api=ChunkMeanAPI(torch,shape=c.params['output'][0]);return Candidate(Implementation(api),'ops-math complete aclnnMean FP32 accumulator, original outputdtype','All contiguous/cast/axis/descriptors/workspace/nativeMean/outputcast work timed; no Torch mean fallback.',sources,cleanup=api.cleanup)
  out=chunk_candidate(case,child,lambda:sources);return replace(out,api='ops-math complete aclnnMean axis2 FP32 accumulator, original chunk/tail/order AST',source_sha256={**out.source_sha256,**sources})
 return build
