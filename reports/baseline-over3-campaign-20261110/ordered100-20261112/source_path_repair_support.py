import pathlib,sys
P=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(P/'workspace/isolated_wrappers'))
from benchmarks.ascend_all_pairs.registry import candidate_factory as original_factory
import vllm_external_source_hashes
def candidate_factory(case,library):return lambda:vllm_external_source_hashes.build(case,library,original_factory)
