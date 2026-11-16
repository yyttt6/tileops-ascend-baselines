import pathlib,sys
P=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(P/'workspace/isolated_wrappers'))
import vllm_sigmoid_natural_api
def candidate_factory(case,library):return lambda:vllm_sigmoid_natural_api.build(case,library)
