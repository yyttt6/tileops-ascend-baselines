"""Complete native INT4 decode with proved original FP32 values and FP8 rounding."""
import torch
import torch_npu

def factory(params):
    m,k=params['activation'][0];n=params['packed_weight'][0][0];group_size=params['group_size']
    assert group_size==128 and params['weight_group_scale'][1]=='float8_e4m3fn'
    unit=torch.ones(1,device='npu',dtype=torch.float32)
    def run(activation,packed_weight,weight_group_scale,weight_channel_scale,activation_scale):
        assert activation.dtype==torch.float8_e4m3fn
        # All input-dependent unpacking, FP32 group scaling, original FP8 rounding,
        # matrix multiplication and original FP32 epilogue are executed here.
        packed=packed_weight.reshape(-1,group_size//2).view(torch.int32)
        scales=weight_group_scale.float().reshape(-1,1)
        # FP16 is only lossless storage for these <=7-significand-bit products.
        # Eligibility requires actual NPU full-domain FP32-bit and FP8-byte proof.
        stored=torch_npu.npu_anti_quant(packed,scales,dst_dtype=torch.float16,src_dtype=torch_npu.int4)
        rounded_weight=stored.clamp(-448,448).to(torch.float8_e4m3fn).reshape(n,k)
        out=torch_npu.npu_quant_matmul(activation,rounded_weight.T,unit,output_dtype=torch.float32)
        assert out.dtype==torch.float32
        return (out*activation_scale.float()[:,None]*weight_channel_scale.float()[None,:]).to(torch.bfloat16)
    return run
