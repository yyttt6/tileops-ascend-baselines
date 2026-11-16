from ops_transformer_bmm_api_support import candidate_factory as original_factory
def candidate_factory(case,library):
 assert library=='ops-math' and case._call.signature.name=='BmmFwdOp'
 return original_factory(case,'ops-transformer')
