"""Explicit process-local exact6 input-preservation backend compatibility."""
from pathlib import Path
import hashlib,json,datetime
SCOPE={1292,1293,1294,1295,1298,1299}
def install(P,i,meta,args,out,source):
 import torch
 import workloads.numerics as numerics
 from complex_input_preservation_compat_proposal import exact_original_comparison_on_cpu
 assert i in SCOPE and meta['key'][0]=='torch-eager' and meta['key'][1] in ('AllFwdOp','AnyFwdOp','CountNonzeroFwdOp')
 assert meta['key'][2] in ('3d-multidim-complex64','3d-multidim-complex128') and len(args)==1 and args[0].dtype in (torch.complex64,torch.complex128) and args[0].device.type=='npu'
 checker=Path(numerics.__file__);digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 assert digest(checker)=='ef600c156d927cb9a7401debe056ceaaad57dc0450e949d5f7ba27f14f6ac173'
 old=numerics._unchanged;stats={'complex_comparisons':0,'noncomplex_comparisons':0,'original_checker_sha256':digest(checker),'CPU_backend_only_for_complex_inputs':True,'original_rtol':0,'original_atol':0,'original_equal_nan':True,'outside_timed_windows':True}
 def unchanged(tensors,snapshots):
  if any(t.dtype.is_complex for t in tensors):
   stats['complex_comparisons']+=1
   return exact_original_comparison_on_cpu(old,tensors,snapshots)
  stats['noncomplex_comparisons']+=1
  return old(tensors,snapshots)
 # Real guarded NPU/CPU-transfer negative controls, before any formal timer.
 controls={}
 for dtype in (torch.complex64,torch.complex128):
  cpu=torch.tensor([1+2j,0+0j,complex(float('nan'),1)],dtype=dtype);a=cpu.to(args[0].device);same=cpu.clone().to(args[0].device)
  assert unchanged([a],[same]);bad=cpu.clone();bad[0]=1+3j;assert not unchanged([a],[bad.to(args[0].device)])
  bad=cpu.clone();bad[0]=2+2j;assert not unchanged([a],[bad.to(args[0].device)])
  assert not unchanged([a],[same.reshape(1,3)]);assert not unchanged([a],[cpu.to(torch.complex128 if dtype==torch.complex64 else torch.complex64).to(args[0].device)])
  controls[str(dtype)]={'same_and_equal_nan_pass':True,'imaginary_corruption_rejected':True,'real_corruption_rejected':True,'shape_and_dtype_changes_rejected':True}
 torch.npu.synchronize();stats['guarded_NPU_transfer_negative_controls']=controls;stats['negative_control_complex_comparisons']=stats['complex_comparisons']
 for p in [checker,Path(__file__),P/'complex_input_preservation_compat_proposal.py',P/'COMPLEX6_USER_AUTHORIZATION_20261116.json']:source[str(p)]=digest(p)
 numerics._unchanged=unchanged
 def finish():
  assert numerics._unchanged is unchanged;numerics._unchanged=old
  stats['original_restored']=True;stats['formal_complex_comparisons']=stats['complex_comparisons']-stats['negative_control_complex_comparisons'];stats['checked_at_remote']=datetime.datetime.now(datetime.timezone.utc).isoformat()
  (out/'COMPLEX_INPUT_PRESERVATION.json').write_text(json.dumps(stats,indent=2))
 return finish
