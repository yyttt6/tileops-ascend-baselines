"""Case-level CPU gold registry: the same corrected reference for every implementation."""
import pathlib,hashlib
from reference_overrides import corrected_reference

def apply(case):
 op=case._call.signature.name
 corrected=corrected_reference(op,case.id)
 if corrected is None:return {'applied':False,'key':[op,case.id]}
 record={'applied':True,'key':[op,case.id],'module_sha256':hashlib.sha256(pathlib.Path(__file__).with_name('reference_overrides.py').read_bytes()).hexdigest(),'kind':'CPU FP64 accumulation then declared BF16 output','tolerance_changed':False,'implementation_independent':True}
 case.__dict__['reference']=corrected
 case.__dict__['_audited_reference_correction']=record
 return record
