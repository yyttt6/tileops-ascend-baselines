"""Exact unsigned16 address storage; integer indices, original complete decode table."""
import torch,sys

def decode(packed_weight,group_scale,group_size,pair_table):
    assert sys.byteorder=='little' and packed_weight.dtype==torch.uint8 and group_scale.dtype==torch.float8_e4m3fn
    n=packed_weight.shape[0];k=packed_weight.shape[1]*2
    packed=packed_weight.reshape(-1,group_size//2)
    scales=group_scale.view(torch.uint8).reshape(-1,1).expand_as(packed)
    index=torch.stack((packed,scales),-1).contiguous().view(torch.uint16).reshape(-1).to(torch.int32)
    pairs=torch.index_select(pair_table,0,index)
    return pairs.view(torch.float8_e4m3fn).reshape(n,k)
