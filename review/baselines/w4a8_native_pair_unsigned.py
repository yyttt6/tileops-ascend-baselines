"""Full existing NPU APIs with an exhaustive, input-independent local decode table."""
import hashlib
import torch
import torch_npu
from w4a8_universal_pair_lut import cpu_pair_table
from w4a8_pair_unsigned_index import decode
TABLE_SHA256='ae57f77c961fd946be2312170fa71456d3747dff7b1f6b160cb4c357accd3f25'

def factory(params):
    m,k=params['activation'][0];n=params['packed_weight'][0][0];group_size=params['group_size']
    table_cpu=cpu_pair_table()
    # Pin the exact original CPU FP32->FP8 local-operation semantics.
    assert hashlib.sha256(table_cpu.view(torch.uint8).numpy().tobytes()).hexdigest()==TABLE_SHA256
    pair_table=table_cpu.to('npu')
    unit=torch.ones(1,device='npu',dtype=torch.float32)
    def run(activation,packed_weight,weight_group_scale,weight_channel_scale,activation_scale):
        assert activation.dtype==torch.float8_e4m3fn
        # Every input-dependent address, gather, matmul and scale is executed here.
        rounded_weight=decode(packed_weight,weight_group_scale,group_size,pair_table)
        out=torch_npu.npu_quant_matmul(activation,rounded_weight.T,unit,output_dtype=torch.float32)
        assert out.dtype==torch.float32
        return (out*activation_scale.float()[:,None]*weight_channel_scale.float()[None,:]).to(torch.bfloat16)
    return run
