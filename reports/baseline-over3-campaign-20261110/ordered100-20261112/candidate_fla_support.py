from benchmarks.ascend_all_pairs.registry import candidate_factory as original_factory
from benchmarks.ascend_all_pairs.fla.zero_pad_candidate import build
def candidate_factory(case,library):
 return lambda:build(case,library,original_factory)
