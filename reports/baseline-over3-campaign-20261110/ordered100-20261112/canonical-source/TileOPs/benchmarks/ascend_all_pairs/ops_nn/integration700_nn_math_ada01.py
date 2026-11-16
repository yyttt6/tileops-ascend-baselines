"""Private complete API bindings; no internal kernel/schedule changes."""
import ctypes as C
from pathlib import Path

def make(case):
 import torch
 from benchmarks.api import Implementation
 from benchmarks.ascend_baselines import Candidate,LIBRARIES,source_hashes
 from benchmarks.ascend_math_more import _MathCall
 op=case._call.signature.name;p=case.params
 if op=='FFTC2CFwdOp':
  b=_MathCall('aclStft',[C.c_void_p]*3+[C.c_int64]*3+[C.c_bool]*3);b.codes[torch.complex128]=17
  def run(x):
   n=x.shape[-1];out=torch.empty((*x.shape,1),device=x.device,dtype=x.dtype);r=b.record()
   b.call([b.tensor(x.contiguous(),r),None,b.tensor(out,r),n,n,n,False,False,True],r,x.device)
   return out.reshape(x.shape)
  directory='ops-math/math/stft'
 elif op=='AdaLayerNormFwdOp':
  b=_MathCall('aclnnLayerNorm',[C.c_void_p]*4+[C.c_double]+[C.c_void_p]*3,library='ops-nn')
  def run(x,scale,shift):
   n=x.shape[-1];xx=x.float().reshape(-1,n).contiguous();ss=scale.float().expand_as(x).reshape(-1,n).contiguous();zz=shift.float().expand_as(x).reshape(-1,n).contiguous();out=torch.empty_like(xx)
   for i in range(xx.shape[0]):
    r=b.record();b.call([b.tensor(xx[i:i+1],r),b.array([n],r),b.tensor(ss[i],r),b.tensor(zz[i],r),float(p['eps']),b.tensor(out[i:i+1],r),None,None],r,x.device)
   return out.reshape(x.shape).to(x.dtype)
  directory='ops-nn/norm/layer_norm_v4'
 elif op in ('FusedAddRMSNormFwdOp','FusedAddLayerNormFwdOp'):
  rms=op=='FusedAddRMSNormFwdOp'
  api='aclnnAddRmsNorm' if rms else 'aclnnAddLayerNorm'
  sig=([C.c_void_p]*3+[C.c_double]+[C.c_void_p]*3) if rms else ([C.c_void_p]*5+[C.c_double,C.c_bool]+[C.c_void_p]*4)
  b=_MathCall(api,sig,library='ops-nn')
  def run(x,residual,w,*bias):
   out=torch.empty_like(x);summed=torch.empty_like(x);stat=torch.empty((*x.shape[:-1],1),device=x.device,dtype=torch.float32);r=b.record()
   def t(v):return b.tensor(v.contiguous(),r)
   args=[t(x),t(residual),t(w),float(p['eps']),t(out),t(stat),t(summed)] if rms else [t(x),t(residual),t(w),t(bias[0]),None,float(p['eps']),True,t(out),t(stat),t(torch.empty_like(stat)),t(summed)]
   b.call(args,r,x.device);return out,summed
  directory='ops-nn/norm/'+('add_rms_norm' if rms else 'add_layer_norm')
 elif op=='BatchNormFwdOp':
  b=_MathCall('aclnnBatchNorm',[C.c_void_p]*5+[C.c_bool,C.c_double,C.c_double]+[C.c_void_p]*3,library='ops-nn')
  def run(x,rm,rv,w,bias):
   r=b.record();mean=None if rm is None else rm.float().clone();var=None if rv is None else rv.float().clone();out=torch.empty_like(x);stat=torch.empty((x.shape[1],),device=x.device,dtype=torch.float32)
   def t(v):return None if v is None else b.tensor(v.contiguous(),r)
   b.call([t(x),t(None if w is None else w.float()),t(None if bias is None else bias.float()),t(mean),t(var),bool(p['training']),float(p['momentum']),float(p['eps']),t(out),t(stat),t(torch.empty_like(stat))],r,x.device)
   return out,mean,var
  directory='ops-nn/norm/batch_norm_v3'
 else:raise ValueError(op)
 return Candidate(Implementation(run),b.api,'Original complete API with shape/dtype/affine/state binding; preserve all canonical outputs.',b.hashes(LIBRARIES/directory)|source_hashes(Path(__file__)),cleanup=b.cleanup)
