import sys,pathlib
P=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(P/'workspace/isolated_wrappers'))
import sgl_dequant_natural_api
def candidate_factory(case,library):return lambda:sgl_dequant_natural_api.build(case,library)
