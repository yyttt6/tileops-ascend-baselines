"""INT8 dequantization is the original native FP32 weight product."""
from .extended import native, result


def int8_dequant(case):
    import torch
    fn = native('mhc.norm_fn_kernel', 'mhc_fn_normw_merge_fwd')
    name = case._call.signature.name
    dtype = getattr(torch, case.params['out_dtype'])
    def run(q, scale):
        rows, cols = q.shape
        if name == 'INT8DequantPerTensorFwdOp':
            scales = scale.expand(rows, cols)
        elif name == 'INT8DequantPerChannelFwdOp':
            scales = scale[:, None].expand(rows, cols)
        else:
            scales = scale[:, :, None].expand(rows, scale.shape[1], 128).reshape(rows, -1)[:, :cols]
        count = q.numel()
        left = torch.nn.functional.pad(q.float().contiguous().reshape(-1), (0, -count % 256))
        right = torch.nn.functional.pad(scales.contiguous().reshape(-1), (0, -count % 256))
        out = torch.empty_like(left).reshape(1, -1)
        fn(left.reshape(1, -1), right, out)
        return out.reshape(-1)[:count].reshape_as(q).to(dtype)
    return result(run, 'mhc.norm_fn_kernel.mhc_fn_normw_merge_fwd',
                  'INT8-to-FP32 is lossless; expand supplied scale tensor per tensor/channel/block, call original native pointwise product, restore manifest output dtype. Scale expansion is inside each call.',
                  'mhc/norm_fn_kernel.py', 'mhc/norm_fn_asc.py')
