import pathlib,sys
P=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(P/'workspace/isolated_wrappers'))
import vllm_fusedqk_rms_api
def candidate_factory(case,library):return lambda:vllm_fusedqk_rms_api.build(case,library)
