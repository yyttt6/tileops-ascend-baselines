import pathlib,sys,json,hashlib
P=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(P/'workspace/isolated_wrappers'))
import vllm_adaln_fla_defined_api as helper
def candidate_factory(case,library):
 def create():
  review=helper.validate_alternate(case);candidate=helper.build(case,library)
  review.update(case_id=case.id,op=case._call.signature.name,input_shape=list(case.inputs[0].shape),input_dtype=str(case.inputs[0].dtype),group_size=case.inputs[0].shape[-1],no_generic_guard_or_checker_change=True,api=candidate.api)
  dest=P/'runs'/sys.argv[1]/'ALTERNATE_API_CONTRACT_REVIEW.json';dest.write_text(json.dumps(review,indent=2)+'\n')
  return candidate
 return create
