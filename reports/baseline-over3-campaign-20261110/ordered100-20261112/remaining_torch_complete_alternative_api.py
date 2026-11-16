"""Use the frozen complete Torch reference program as an actual device API."""
from pathlib import Path
import inspect,hashlib
def candidate_factory(case,library):
 def build():
  from benchmarks.api import Implementation
  from benchmarks.ascend_baselines import Candidate
  assert library in ('ops-math','ops-tensor','tilelang-ascend') and case._call.signature.name in ('SSDChunkCouplingFwdOp','TruncFwdOp','CumprodFwdOp','VarFwdOp','VarMeanFwdOp')
  fn=case.reference
  assert fn is not None
  files=[Path(__file__),Path(inspect.getsourcefile(type(case.workload)))]
  return Candidate(Implementation(fn),'original frozen complete Torch program, executed on NPU','Original callable without AST/body changes; input/output/storage dtype, mathematics and complete return structure preserved. Original library-specific reference placement and checker remain independent. Not a CPU baseline.',{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
 return build
