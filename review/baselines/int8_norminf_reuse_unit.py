"""Full existing APIs; preserve the frozen FP32 scale formula and INT8 range."""
import torch
import torch.nn.functional as F
import torch_npu
from workloads.quantization.quantize import _int8_scale

def row_quant(xf,scale,unit):
    # Native qint8 saturates to -128; the original mathematical contract uses -127.
    normalized=(xf/scale[:,None]).clamp(-127,127)
    return torch_npu.npu_quantize(normalized,unit,None,torch.qint8,-2,False)

def factory(op,params):
    # Immutable API parameter, independent of every input; arithmetic/normalization stays timed.
    unit=torch.ones(1,device='npu',dtype=torch.float32)
    def exact_scale(am):
        # Reuse the existing input-independent FP32 API parameter 1.0.
        # Keep the original integer127 RealDiv and zero guard exactly.
        return torch.where(am>0,am/127,unit[0])
    if op=='INT8QuantPerBlockFwdOp':
        def run(x):
            m,k=x.shape;xf=x.float();pad=(-k)%128
            blocks=(F.pad(xf,(0,pad)) if pad else xf).reshape(-1,128)
            scale=exact_scale(torch.linalg.vector_norm(blocks,ord=float('inf'),dim=-1))
            q=row_quant(blocks,scale,unit).reshape(m,-1)[:,:k]
            return q,scale.reshape(m,(k+127)//128)
        return run
    if op=='INT8QuantPerChannelFwdOp':
        def run(w):
            wf=w.float();scale=exact_scale(torch.linalg.vector_norm(wf,ord=float('inf'),dim=1))
            return row_quant(wf,scale,unit),scale
        return run
    if op=='INT8QuantPerTensorFwdOp':
        def run(x):
            xf=x.float();scale=exact_scale(torch.linalg.vector_norm(xf,ord=float('inf'))).reshape(1)
            normalized=(xf/scale).clamp(-127,127)
            q=torch_npu.npu_quantize(normalized,unit,None,torch.qint8,-2,False)
            return q,scale
        return run
    if op=='SmoothQuantFwdOp':
        def run(x,smooth):
            xs=x.float()/smooth.view(x.shape[1]);scale=exact_scale(torch.linalg.vector_norm(xs,ord=float('inf'),dim=1))
            return row_quant(xs,scale,unit),scale
        return run
    raise ValueError('Unreviewed explicit-scale native quantization contract: '+op)
