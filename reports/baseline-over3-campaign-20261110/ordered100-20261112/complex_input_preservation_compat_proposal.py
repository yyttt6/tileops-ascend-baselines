"""Proposal only. No installation or global monkeypatch occurs on import."""
def exact_original_comparison_on_cpu(original,tensors,snapshots):
 import torch
 left=[];right=[]
 for tensor,source in zip(tensors,snapshots,strict=True):
  if tensor.dtype!=source.dtype or tensor.device!=source.device or tensor.shape!=source.shape or tensor.layout!=source.layout:return False
  if tensor.dtype.is_complex:
   # Byte-preserving transfer; original complex dtype restored only on CPU.
   # Original tensor/storage/math remains complex on NPU. Outside timed windows.
   def cpu_bits(x):return x.detach().contiguous().reshape(-1).view(torch.uint8).cpu().view(x.dtype).reshape(x.shape)
   left.append(cpu_bits(tensor));right.append(cpu_bits(source))
  else:left.append(tensor);right.append(source)
 return original(left,right)
