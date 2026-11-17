"""Complete native API composition: original INT4 decode and FP8 rounding, FP32 product/output scaling."""
import torch
import torch_npu

def factory(params):
    m,k=params['activation'][0];n=params['packed_weight'][0][0];group_size=params['group_size']
    # Immutable API scales, input independent. All real weight/input work stays timed.
    unit_channel=torch.ones(n,device='npu',dtype=torch.float32)
    unit_token=torch.ones(m,device='npu',dtype=torch.float32)
    def run(activation,packed_weight,weight_group_scale,weight_channel_scale,activation_scale):
        assert activation.dtype==torch.float8_e4m3fn
        packed=packed_weight.to(torch.int32)
        codes=torch.stack((packed & 15,packed >> 4),-1).flatten(-2)
        signed=((codes ^ 8)-8).float()
        weight=(signed.reshape(n,k//group_size,group_size)*weight_group_scale.float().unsqueeze(-1)).reshape(n,k)
        rounded_weight=weight.to(torch.float8_e4m3fn)
        # FP8 operands are the original exact representable values. Request FP32 output;
        # retain original token-then-channel FP32 scaling and final BF16 rounding.
        out=torch_npu.npu_quant_matmul(activation,rounded_weight.T,unit_channel,pertoken_scale=unit_token,output_dtype=torch.float32)
        assert out.dtype==torch.float32
        return (out*activation_scale.float()[:,None]*weight_channel_scale.float()[None,:]).to(torch.bfloat16)
    return run
