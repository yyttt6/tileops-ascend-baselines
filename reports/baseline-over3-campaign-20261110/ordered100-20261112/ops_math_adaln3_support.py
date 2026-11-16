"""Complete same-pin LayerNorm/Mul/Add, private prequalified components; no new kernel."""
from pathlib import Path
import json,hashlib,os
from types import SimpleNamespace as T
from ops_math_adaln3_layout import composed_run
P=Path(__file__).resolve().parent;S=Path('/opt/baseline-over3-20261110/server2-aligned-stack/native');N=S/'ops-math-adaln3-v1';LN=S/'ops-math-layernorm-original-v1';ADD=S/'ops-math-add-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def make_candidate(case,require_composition_qualification=True):
 import torch
 from benchmarks.api import Implementation
 from benchmarks.ascend_baselines import Candidate
 from ops_math_dequant_api import MathDequantAPI
 from ops_math_add_api import OpsMathAddAPI
 q=json.loads((LN/'ORIGINAL_NORM_HOST_QUALIFICATION.json').read_text());assert q['full_API_embedded_kernel_source_build_guard_and_dual_archive_verified'] and q['execution_hostname']==os.uname().nodename
 assert sha(LN/'libops_math_layernorm.so')==q['build_record']['binary_sha256'];assert sha(LN/'layer_norm.asc')==q['original_source_sha256']
 if require_composition_qualification:
  cq=json.loads((N/'ADALN3_HOST_QUALIFICATION.json').read_text());assert cq['full_API_original_reference_inputbits_flush_private_kernels_guard_and_dual_archive_verified'] and cq['execution_hostname']==os.uname().nodename
  for name,h in cq['formal_source_sha256'].items():assert sha(P/name)==h
 am=ADD/'OPSMATH_ADD_NATIVE_PACKAGE_V2.json';aq=ADD/'PRIVATE_OPSMATH_ADD_QUALIFICATION_V2.json';ar=json.loads(aq.read_text());assert ar['private_original_device_kernels_verified'] and ar['package_manifest_sha256']==sha(am)
 torch.ops.load_library(str(LN/'libops_math_layernorm.so'));native=torch.ops.allpairs_math_example.layer_norm;mul=MathDequantAPI(torch,'Mul');binding=OpsMathAddAPI(torch,P/'workspace',am);binding.ns['WORKSPACE']=N/'add-binding-workspace'
 # Metadata-only tensors establish Add's shape/dtype; actual upcasts stay in run.
 shape=tuple(case.inputs[0].shape);metadata=torch.empty(shape,dtype=torch.float32,device='meta');add=binding.prepare(T(inputs=(metadata,metadata),params={'alpha':1}))
 sources={**mul.sources,**add.source_sha256}
 for f in [LN/'libops_math_layernorm.so',LN/'layer_norm.asc',LN/'BUILD_RECEIPT.json',LN/'ORIGINAL_NORM_HOST_QUALIFICATION.json',am,aq,P/'ops_math_adaln3_layout.py',P/'ops_math_adaln3_support.py',P/'ops_math_dequant_api.py',P/'ops_math_add_api.py']:
  sources[str(f)]=sha(f)
 if require_composition_qualification:sources[str(N/'ADALN3_HOST_QUALIFICATION.json')]=sha(N/'ADALN3_HOST_QUALIFICATION.json')
 def cleanup():mul.cleanup();add.cleanup()
 return Candidate(Implementation(composed_run(case,native,mul,add.implementation.run)),'ops-math complete original LayerNorm + Mul + Add (3 APIs)','Complete normalization and affine; original FP32 widening/scale/shift/output cast; no new kernels or reduced outputs',sources,cleanup=cleanup)
def candidate_factory(case,library):
 def build():
  assert library=='ops-math' and case._call.signature.name=='AdaLayerNormFwdOp';return make_candidate(case)
 return build
