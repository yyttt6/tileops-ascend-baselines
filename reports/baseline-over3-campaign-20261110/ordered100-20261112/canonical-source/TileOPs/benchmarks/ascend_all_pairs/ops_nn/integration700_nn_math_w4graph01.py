"""Original graph APIs, descriptor-only metadata adaptation."""
import ctypes as C
from pathlib import Path

def make(case):
 import torch
 from benchmarks.ascend_all_pairs.ops_nn.graph import GraphCall
 from benchmarks.ascend_all_pairs.ops_nn.common import candidate
 op=case._call.signature.name;p=case.params
 if op=='INT8DequantPerChannelFwdOp':
  from benchmarks.ascend_all_pairs.ops_math.dequant import make as factory
  return factory(case,op)
 if op=='InstanceNormFwdOp' and __import__('os').environ.get('NN_MATH_BASELINE')=='ops-math':
  from benchmarks.ascend_all_pairs.ops_math.layernorm import make as factory
  return factory(case,op)
 b=GraphCall('ops-nn')
 if op in ('AdaLayerNormFwdOp','InstanceNormFwdOp'):
  def ln(x,w,bias):
   n=x.shape[-1];xx=x.float().contiguous();shape=torch.tensor([n],device=x.device,dtype=torch.int32);y=torch.empty_like(xx);stat=torch.empty((*x.shape[:-1],1),device=x.device,dtype=torch.float32)
   b.call('LayerNormV4',[xx,shape,w.float().contiguous(),bias.float().contiguous()],[y,stat,torch.empty_like(stat)],{'epsilon':float(p['eps'])},constants={1:(C.c_int32*1)(n)})
   return y
  if op=='AdaLayerNormFwdOp':
   def run(x,scale,shift):
    n=x.shape[-1];xx=x.reshape(-1,n);s=scale.expand_as(x).reshape(-1,n);z=shift.expand_as(x).reshape(-1,n)
    return torch.cat([ln(xx[i:i+1],s[i],z[i]) for i in range(xx.shape[0])]).reshape(x.shape).to(x.dtype)
  else:
   assert case.inputs[1] is None,'Tracked evaluation handled separately'
   def run(x,rm,rv,w,bias):
    n,c=x.shape[:2];xx=x.reshape(n,c,-1);k=xx.shape[-1]
    return torch.stack([ln(xx[:,j],w[j:j+1].expand(k),bias[j:j+1].expand(k)) for j in range(c)],dim=1).reshape(x.shape).to(x.dtype)
  api='LayerNormV4';directory='norm/layer_norm_v4'
 elif op in ('FusedAddRMSNormFwdOp','FusedAddLayerNormFwdOp'):
  rms=op=='FusedAddRMSNormFwdOp';api='AddRmsNorm' if rms else 'AddLayerNorm';directory='norm/'+('add_rms_norm' if rms else 'add_layer_norm')
  def run(x,residual,w,*bias):
   out=torch.empty_like(x);summed=torch.empty_like(x);stat=torch.empty((*x.shape[:-1],1),device=x.device,dtype=torch.float32)
   ins=[x.contiguous(),residual.contiguous(),w.contiguous()];outs=[out,stat,summed] if rms else [out,stat,torch.empty_like(stat),summed];attrs={'epsilon':float(p['eps'])}
   if not rms:ins.extend([bias[0].contiguous(),None]);attrs['additional_output']=True
   b.call(api,ins,outs,attrs);return out,summed
 elif op=='GemmW4A16FwdOp':
  from workloads.gemm import unrepack_w4a16_weight
  b.codes['int4']=29
  class Packed4:
   dtype='int4';ndim=2
   def __init__(self,data,shape):self.data=data;self.shape=shape
   def is_contiguous(self):return True
   def data_ptr(self):return self.data.data_ptr()
   def numel(self):return self.data.numel()
   def element_size(self):return 1
  def run(a,packed,scale,zero):
   raw=unrepack_w4a16_weight(packed);n,kp=raw.shape;k=kp*2
   q=torch.empty((n,k),device=raw.device,dtype=torch.uint8);q[:,0::2]=raw&15;q[:,1::2]=raw>>4;q=q.T.contiguous()^8
   packed_native=(q[:,0::2]|(q[:,1::2]<<4)).contiguous();weight=Packed4(packed_native,(k,n));out=torch.empty((a.shape[0],n),device=a.device,dtype=a.dtype)
   b.call('WeightQuantBatchMatmulV2',[a.contiguous(),weight,scale.T.contiguous(),(8-zero.to(torch.int16)).to(a.dtype).T.contiguous(),None,None,None],[out],{'transpose_x':False,'transpose_weight':False,'antiquant_group_size':k//scale.shape[1]})
   return out
  api='WeightQuantBatchMatmulV2';directory='matmul/weight_quant_batch_matmul_v2'
 elif op=='BatchNormFwdOp':
  def run(x,rm,rv,w,bias):
   c=x.shape[1];w=torch.ones(c,device=x.device) if w is None else w.float();bias=torch.zeros(c,device=x.device) if bias is None else bias.float();mean=torch.zeros(c,device=x.device) if rm is None else rm.float().clone();var=torch.ones(c,device=x.device) if rv is None else rv.float().clone()
   xx=x.contiguous().reshape(x.shape[0],c,1,-1);out=torch.empty_like(xx);stat=torch.empty_like(mean)
   b.call('BatchNormV3',[xx,w,bias,mean,var],[out,mean,var,stat,torch.empty_like(stat)],{'epsilon':float(p['eps']),'momentum':float(p['momentum']),'is_training':bool(p['training'])},formats=[0,2,2,2,2,0,2,2,2,2])
   return out.reshape(x.shape),None if rm is None else mean,None if rv is None else var
  api='BatchNormV3';directory='norm/batch_norm_v3'
 else:raise ValueError(op)
 return candidate(b,run,api,'One original complete API per logical independent row/group; preserve complete outputs, exact layout/dtype/metadata binding.',directory)
