"""Complete existing API composition, private FP32 work buffer only."""
import torch
import torch_npu
from workloads.quantization.quantize import _int8_scale

def factory(op, params):
    assert op=='INT8QuantPerChannelFwdOp'
    unit=torch.ones(1,device='npu',dtype=torch.float32)
    def run(w):
        assert w.dtype in (torch.float16,torch.bfloat16)
        # The upcast owns its buffer: in-place operations never alias original input.
        wf=w.float()
        # Infinity norm is exactly max(abs(wf)); same FP32 output, existing full API.
        scale=_int8_scale(torch.linalg.vector_norm(wf,ord=float('inf'),dim=1))
        wf.div_(scale[:,None]).clamp_(-127,127)
        q=torch_npu.npu_quantize(wf,unit,None,torch.qint8,-2,False)
        return q,scale
    return run
