"""Reference placement follows cache_frozen_pair_check.py; never choose by pass/fail."""
def policy(library,op,case_id):
 if (op,case_id) in {('BmmFwdOp','mha-decode-pv-bfloat16'),('BmmFwdOp','square-512-bfloat16')}:return 'audited_cpu_fp64'
 if op in ('GDNFwdOp','KDAFwdOp'):return 'canonical_cpu_fp64_validator'
 if op=='RoPEFwdOp':return 'npu_with_same_coefficient_cpu_rotation'
 if op=='FFTC2CFwdOp':return 'canonical_cpu_fft'
 if (library in ('ops-math','ops-nn','ops-transformer') and op!='DeltaNetFwdOp') or op=='AdaptiveMaxPool2dFwdOp':return 'original_cpu'
 return 'original_npu'
