import sys,pathlib
P=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(P/'workspace/isolated_wrappers'))
from benchmarks.ascend_all_pairs.registry import candidate_factory as original_factory
import vllm_norm_api_v2
def candidate_factory(case,library):return lambda:vllm_norm_api_v2.build(case,library,original_factory)
