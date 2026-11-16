"""Exact representation adapters for the original p=0 predicate/reduction DAG."""
def truth01(binding,x):
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    if x.dtype==torch.bool:return x.float()
    if x.dtype in (torch.float16,torch.bfloat16,torch.float32):
        lanes=x.float().contiguous().reshape(-1,1);second=False
    elif x.dtype in (torch.int8,torch.uint8,torch.int16,torch.int32,torch.int64):
        # Integer zero has all-zero bytes; retain every bit, including the sign.
        lanes=x.contiguous().view(torch.uint8).reshape(x.numel(),x.element_size()).float();second=True
    elif x.dtype==torch.complex64:
        # Either real or imaginary nonzero makes complex truth true, including NaNs.
        lanes=torch.view_as_real(x.contiguous()).reshape(-1,2);second=True
    else:raise UnsupportedCase('Original p=0 has no FP64 lane; this dtype has no inspected exact representation specialization.')
    out=torch.empty((x.numel(),),device=x.device,dtype=torch.float32)
    binding.call('LpNormV2',[lanes],[out],{'p':0.0,'axes':[1],'keepdim':False,'epsilon':0.0})
    if second:
        result=torch.empty_like(out)
        binding.call('LpNormV2',[out.reshape(-1,1)],[result],{'p':0.0,'axes':[1],'keepdim':False,'epsilon':0.0})
        out=result
    return out.reshape(x.shape)
