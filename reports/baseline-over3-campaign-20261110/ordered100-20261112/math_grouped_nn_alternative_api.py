from pathlib import Path
import ast,copy,hashlib,json,os
from ops_nn_gemm_pad4_api import GemmPad4API
P=Path(__file__).resolve().parent;N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-gemm-pad4-v1')
def candidate_factory(case,library):
 def build():
  import torch
  import torch.nn.functional as F
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  assert library in ('ops-math','ops-transformer') and case._call.signature.name=='GroupedGemmFwdOp';assert case.id in ['moe-wgrad-float16','qwen3-30b-ep8-empty-bfloat16','moe-fwd-float16','moe-fwd-bfloat16','qwen3-30b-ep8-routed1k-bfloat16','qwen3-30b-ep8-ragged-bfloat16']
  qf=N/'GEMM_PAD4_HOST_QUALIFICATION.json';q=json.loads(qf.read_text());assert q['private_original_device_kernels_verified'] and q['execution_hostname']==os.uname().nodename
  mf=N/'GEMM_PAD4_NATIVE_PACKAGE.json';assert hashlib.sha256(mf.read_bytes()).hexdigest()==q['package_manifest_sha256'];m=json.loads(mf.read_text());root=Path(m['staging_root']);sources={str(root/name):r['sha256'] for name,r in m['all_files'].items()}
  for name,h in sources.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h
  binding=GemmPad4API(torch,mf,qf);src=P/'workspace/TileOPs/benchmarks/ascend_all_pairs/ops_nn/matrix.py';tree=ast.parse(src.read_text());mm=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_mm'));group=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='grouped');run=copy.deepcopy(next(n for n in group.body if isinstance(n,ast.FunctionDef) and n.name=='run'));before=[ast.dump(x,include_attributes=False) for x in [mm,run]];ns={'torch':torch,'F':F,'binding':binding,'p':case.params};exec(compile(ast.fix_missing_locations(ast.Module(body=[mm,run],type_ignores=[])),str(src),'exec'),ns);assert before==[ast.dump(x,include_attributes=False) for x in [mm,run]]
  for f in [Path(__file__),P/'ops_nn_gemm_pad4_api.py',src,mf,qf]:sources[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  return Candidate(Implementation(ns['run']),'ops-nn::original matrix.grouped complete MatMulV3 API per logicalgroup','OriginalNN _mm/run AST exact, including group-sizes read, TN/NT transpose, emptygroups, original1024/256/64padding and stack/cat; originalCAPI tensorbinding and dtype, no newtensor arithmetic/kernel. AlternativeforMathcontract.',sources,timing_mode='host_roundtrip')
 return build
