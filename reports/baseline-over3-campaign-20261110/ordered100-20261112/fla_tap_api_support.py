from pathlib import Path
import hashlib,json

def candidate_factory(case,library):
 def build():
  from benchmarks.ascend_all_pairs.fla import PROVIDERS
  import benchmarks.ascend_baselines as provenance
  import fla_tap_triton_identity as identity
  assert library=='flash-linear-attention-npu' and case._call.signature.name in ('INT8DequantPerTensorFwdOp','BmmFwdOp','GroupedGemmFwdOp','SSDStatePassingFwdOp','SSDChunkCouplingFwdOp','SSDChunkCumsumFwdOp')
  if case._call.signature.name=='BmmFwdOp':
   import torch
   assert all(x.dtype==torch.float16 for x in case.inputs) and case.inputs[0].shape[-1]>128
  if case._call.signature.name=='GroupedGemmFwdOp':
   import torch
   assert all(x.dtype==torch.float16 for x in case.inputs[:2])
  if case._call.signature.name=='SSDChunkCouplingFwdOp':
   import torch
   assert all(x.dtype==torch.bfloat16 for x in case.inputs)
  identity.install();original=provenance.source_hashes
  try:
   provenance.source_hashes=lambda *paths:{str(Path(p)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
   candidate=PROVIDERS[case._call.signature.name](case)
  finally:provenance.source_hashes=original
  assert candidate.api in ('FLA causal_conv1d_fwd_impl_old(W=1, FP32 tap=scale)','FLA causal_conv1d_fwd_impl_old(window dot product)(matrix rows as independent 1D windows)','FLA causal_conv1d_fwd_impl_old(grouped matrix window dots)','FLA ChunkFwdH exponentials + original FP32 causal tap/residual recurrence','FLA native matrix product + original constant causal tap mask','FLA KdaGateCumsum(T=1 raw gate), native tap product, original chunk_local_cumsum') and candidate.implementation.run.__code__.co_filename.endswith(('/fla/dequant.py','/fla/spatial.py','/fla/grouped_gemm.py','/fla/sequence.py'))
  if case._call.signature.name in ('SSDStatePassingFwdOp','SSDChunkCouplingFwdOp','SSDChunkCumsumFwdOp'):
   package_manifest=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/fla-original-gate-v1/PACKAGE_AOT_V1_MANIFEST.json') if case._call.signature.name=='SSDChunkCumsumFwdOp' else Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/fla-original-chunk-v1/PACKAGE_AOT_V4_MANIFEST.json');m=json.loads(package_manifest.read_text())
   for name,h in m['artifact_sha256'].items():
    file=Path(m['package'])/name;assert hashlib.sha256(file.read_bytes()).hexdigest()==h;candidate.source_sha256[str(file)]=h
   candidate.source_sha256[str(package_manifest)]=hashlib.sha256(package_manifest.read_bytes()).hexdigest()
  if case._call.signature.name=='SSDChunkCumsumFwdOp':
   marker=Path('/opt/baseline-over3-20261110/ordered100-20261112/workspace/.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/fla/partial_native_ready.json');h=hashlib.sha256(marker.read_bytes()).hexdigest();assert h=='4c11280c0798b3186fccc7846ed6382d0305a04d67653c534b4e874dd04620ba';candidate.source_sha256[str(marker)]=h
  for p in [Path(__file__),Path(identity.__file__),Path(__file__).with_name('FLA_TAP_API_FORMAL_PREREQUISITES.json'),Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/fla-original-tap-v1/SDK_CACHE_IDENTITY.json'),Path('/opt/baseline-over3-20261110/server2-aligned-stack/native/fla-original-tap-v1/sdk-cache-identity-v1/aarch64-linux/ascend_toolkit_install.info')]:candidate.source_sha256[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
  return candidate
 return build
