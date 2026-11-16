"""Frozen complete distance pack/zeros/outputcast; three original same-library APIs."""
from pathlib import Path
from types import SimpleNamespace
from math import prod
import ast,hashlib,json,os
P=Path(__file__).resolve().parent
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-l2-sqrt-v1')
def candidate_factory(case,library):
 def build():
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  from ops_math_l2_api import MathL2API
  assert library=='ops-math' and case._call.signature.name=='VectorNormFwdOp';p=case.params;assert p['ord']==2 and not p['keepdim'] and p['x'][1] in ('float16','bfloat16')
  proof=N/'MATH_L2_HOST_QUALIFICATION_V2.json';q=json.loads(proof.read_text());assert q['full_API_and_original_flush_and_private_kernel_verified'] is True and q['execution_hostname']==os.uname().nodename;assert q['package_manifest_sha256']==hashlib.sha256((N/'MATH_L2_SQRT_PACKAGE.json').read_bytes()).hexdigest()
  for name,h in q['qualified_API_source_sha256'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h
  f=P/'workspace/TileOPs/benchmarks/ascend_all_pairs/ops_math/distance.py';raw=f.read_bytes();assert hashlib.sha256(raw).hexdigest()=='34fbc366c1281052f1cbe5667ea8102c67cb55735243a5b99e352945c9c829c3';tree=ast.parse(raw);make=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make');run=next(n for n in make.body if isinstance(n,ast.FunctionDef) and n.name=='run');original=ast.dump(run,include_attributes=False)
  shape,dtype=p['x'];out_shape,out_dtype=p['output'];ndim=len(shape);dim=p.get('dim');axes=list(range(ndim)) if dim is None else [d%ndim for d in dim] if isinstance(dim,(tuple,list)) else [dim%ndim];kept=[i for i in range(ndim) if i not in axes];k=prod(shape[i] for i in axes);rows=prod(shape[i] for i in kept);api=MathL2API(torch);child=SimpleNamespace(implementation=Implementation(api));scope=dict(torch=torch,child=child,kept=kept,axes=axes,rows=rows,k=k,out_shape=out_shape,out_dtype=out_dtype);exec(compile(ast.fix_missing_locations(ast.Module(body=[run],type_ignores=[])),str(f),'exec'),scope);assert ast.dump(run,include_attributes=False)==original
  sources={**api.sources,str(f):hashlib.sha256(raw).hexdigest(),str(proof):hashlib.sha256(proof.read_bytes()).hexdigest(),str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
  return Candidate(Implementation(scope['run']),'ops-math complete Mul+ReduceSum+Sqrt L2 (three existing APIs)','Finite real L2 only; original permute/contiguous/reshape/float32 widen, known-zero allocation, full native threeAPIs/descriptors/workspaces/cleanup/sync/outputcast timed. Not general Cdist; no torch arithmetic/kernel/precision/checker edits.',sources,cleanup=api.cleanup)
 return build
