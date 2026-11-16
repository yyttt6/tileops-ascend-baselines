"""Original frozen dequant timed run AST unchanged; full original Mul CAPI child."""
from pathlib import Path
from types import SimpleNamespace
import ast,hashlib,json
P=Path(__file__).resolve().parent
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-math-dequant-v1')
def candidate_factory(case,library):
 def build():
  import torch
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  import ops_math_dequant_api as capi
  assert library=='ops-math';op=case._call.signature.name;assert op in ('INT8DequantPerTensorFwdOp','INT8DequantPerChannelFwdOp','INT8DequantPerBlockFwdOp')
  proof=N/'DEQUANT_HOST_QUALIFICATION.json';q=json.loads(proof.read_text());assert q['full_API_and_original_flush_and_private_kernel_verified'] is True
  import os
  assert q['execution_hostname']==os.uname().nodename
  assert q['package_manifest_sha256']==hashlib.sha256((N/'DEQUANT_PACKAGE.json').read_bytes()).hexdigest()
  f=P/'workspace/TileOPs/benchmarks/ascend_all_pairs/ops_math/dequant.py';source=f.read_bytes();assert hashlib.sha256(source).hexdigest()=='c9e63af0f00230d0070c38907771dd81d6c4170ef3414600481bbad936c17beb'
  tree=ast.parse(source);make=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='make');run=next(n for n in make.body if isinstance(n,ast.FunctionDef) and n.name=='run');body_ast=ast.dump(run,include_attributes=False)
  api=capi.MathDequantAPI(torch,'Mul');child=SimpleNamespace(implementation=Implementation(api));p=case.params;scope={'torch':torch,'op':op,'p':p,'child':child};exec(compile(ast.fix_missing_locations(ast.Module(body=[run],type_ignores=[])),str(f),'exec'),scope)
  assert ast.dump(run,include_attributes=False)==body_ast
  files={**api.sources,str(f):hashlib.sha256(source).hexdigest(),str(proof):hashlib.sha256(proof.read_bytes()).hexdigest()}
  for path in [Path(__file__),Path(capi.__file__)]:files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
  return Candidate(Implementation(scope['run']),'ops-math full original aclnnMul, original dequant timedbody AST','Original q.float, per-channel view or per-block repeat128/crop, full native Mul and outputcast all timed; no newkernel/precision/checker change.',files,cleanup=api.cleanup)
 return build
