from pathlib import Path
import json,hashlib,os
P=Path(__file__).resolve().parent
N=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/ops-nn-gemm-pad4-v1')
def candidate_factory(case,library):
 def build():
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  from ops_nn_gemm_pad4_api import GemmPad4API
  import torch
  assert library=='ops-nn' and case._call.signature.name=='GemmFwdOp'
  proof=N/'GEMM_PAD4_HOST_QUALIFICATION.json';q=json.loads(proof.read_text());assert q['private_original_device_kernels_verified'] and q['execution_hostname']==os.uname().nodename
  manifest=N/'GEMM_PAD4_NATIVE_PACKAGE.json';assert hashlib.sha256(manifest.read_bytes()).hexdigest()==q['package_manifest_sha256'];m=json.loads(manifest.read_text());root=Path(m['staging_root'])
  files={str(root/k):v['sha256'] for k,v in m['all_files'].items()}
  for f,h in files.items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
  api=GemmPad4API(torch,manifest,proof);fn=api.prepare(case)
  for f in [manifest,proof,Path(__file__),P/'ops_nn_gemm_pad4_api.py',N/'original-ascend_gemm_cases.py',N/'original-ascend_nn_aligned_gemm.py']:files[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  return Candidate(Implementation(fn),'ops-nn complete original aclnnMatmul, original alignedpacking and timedrun AST','Original transpose/contiguous/full1024x256x64zero pad, nativeF16BF16 MatMul, outputcrop/contiguous; allwork timed, no source/kernel/precision/checker change.',files)
 return build
