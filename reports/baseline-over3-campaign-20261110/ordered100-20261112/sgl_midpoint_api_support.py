import sys,pathlib
P=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(P/'workspace/isolated_wrappers'))
import sgl_midpoint_api
def candidate_factory(case,library):
 from benchmarks.ascend_all_pairs.registry import candidate_factory as original_factory
 return lambda:sgl_midpoint_api.build(case,library,original_factory)
