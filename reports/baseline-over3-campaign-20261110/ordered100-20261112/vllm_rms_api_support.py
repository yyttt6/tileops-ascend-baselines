import sys,pathlib
P=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(P/'workspace/isolated_wrappers'))
import vllm_rms_complete_api
def candidate_factory(case,library):return lambda:vllm_rms_complete_api.build(case,library)
