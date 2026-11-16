import sys,pathlib
P=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(P/'workspace/isolated_wrappers'))
from benchmarks.ascend_all_pairs.registry import candidate_factory as original_factory
import sgl_noop_pad,sgl_midpoint_api
def candidate_factory(case,library):
 module=sgl_midpoint_api if case._call.signature.name=='LerpScalarFwdOp' else sgl_noop_pad
 return lambda:module.build(case,library,original_factory)
