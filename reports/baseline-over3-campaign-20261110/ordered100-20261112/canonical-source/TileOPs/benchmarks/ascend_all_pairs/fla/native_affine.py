"""Bounded original FLA tap/residual composition, with copy-only broadcasting."""
from pathlib import Path


def make_affine():
    import torch
    import torch.nn.functional as F
    from benchmarks.ascend_baselines import source_hashes
    from .providers import _triton_module
    conv,source=_triton_module('causal_conv1d')

    def affine(x,weight=1.0,residual=None):
        shape=x.shape
        value=x.contiguous().reshape(-1).float()
        tensor_weight=isinstance(weight,torch.Tensor)
        weights=weight.expand(shape).contiguous().reshape(-1).float() if tensor_weight else None
        add=None if residual is None else residual.expand(shape).contiguous().reshape(-1).float()
        parts=[]
        for start in range(0,value.numel(),1048576):
            data=value[start:start+1048576];count=data.numel();extent=count+-count%256
            data=F.pad(data,(0,extent-count)).reshape(1,1,extent)
            tap=F.pad(weights[start:start+count],(0,extent-count)).reshape(1,extent) if tensor_weight else torch.full((1,extent),weight,device=x.device,dtype=torch.float32)
            tail=None if add is None else F.pad(add[start:start+count],(0,extent-count)).reshape(1,1,extent)
            out,_=conv.causal_conv1d_fwd_impl_old(data,tap,bias=None,residual=tail)
            parts.append(out.reshape(-1)[:count])
        return torch.cat(parts).reshape(shape)

    return affine,source_hashes(Path(__file__),source)
