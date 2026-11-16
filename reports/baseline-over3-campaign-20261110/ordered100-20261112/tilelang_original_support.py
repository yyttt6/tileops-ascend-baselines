import sys,pathlib
P=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(P/'workspace/isolated_wrappers'))
import tilelang_source_identity
tilelang_source_identity.install()
from benchmarks.ascend_all_pairs.registry import candidate_factory as original_factory
def candidate_factory(case,library):return lambda:tilelang_source_identity.build(case,library,original_factory)
