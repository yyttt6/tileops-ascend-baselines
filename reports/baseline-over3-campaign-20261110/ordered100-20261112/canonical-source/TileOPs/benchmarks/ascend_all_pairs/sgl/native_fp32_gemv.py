"""Original SGL full-K SGEMMV, retaining the native FP32 result buffer."""
from pathlib import Path


def bind(device):
    import importlib
    import sys
    import torch
    import torch.nn.functional as F
    from benchmarks.ascend_baselines import LIBRARIES, _owned_native, source_hashes
    root = LIBRARIES / 'sgl-kernel-npu'
    package = root / 'python/sgl_kernel_npu/sgl_kernel_npu'
    sys.path.insert(0, str(package.parent))
    loaded = importlib.import_module('sgl_kernel_npu')
    if Path(loaded.__file__).resolve() != (package / '__init__.py').resolve():
        raise RuntimeError('Unexpected SGL native package source')
    binary = package / 'lib/libsgl_kernel_npu.so'
    _owned_native(binary, ['_ZN6sglang10npu_kernel13sgemmv_shrinkERN2at6TensorES3_S3_S3_S3_S3_S3_l'])
    indices = torch.zeros(1, device=device, dtype=torch.int32)
    scales = torch.ones(1, device=device, dtype=torch.float16)
    buffers = {}

    def matmul(left, weight):
        if left.dtype in (torch.int8, torch.float8_e4m3fn):
            left, weight = left.to(torch.float16), weight.to(torch.float16)
        if left.dtype not in (torch.float16, torch.bfloat16) or weight.dtype != left.dtype:
            raise TypeError('SGEMMV requires matching FP16/BF16 or losslessly widened FP8/INT8 inputs')
        rows, k = left.shape
        columns = weight.shape[0]
        output = torch.empty((rows, columns), device=left.device, dtype=torch.float32)
        # Keep the original K without large dummy tails when K is short.
        # This only partitions independent output columns at the public API.
        column_block = min(1024, max(16, ((k-1)//16)*16))
        for c in range(0, columns, column_block):
            n = min(column_block, columns-c)
            padded_n = (n+15)//16*16
            padded_k = max((k+15)//16*16, padded_n+16)
            ww = F.pad(weight[c:c+n], (0, padded_k-k, 0, padded_n-n)).contiguous().unsqueeze(0)
            for r in range(0, rows, 4096):
                m = min(4096, rows-r)
                key = (m, padded_n)
                if key not in buffers:
                    buffers[key] = (
                        torch.tensor([m], device=device, dtype=torch.int32),
                        torch.tensor([padded_n], device=device, dtype=torch.int32),
                        torch.empty((m, padded_n), device=device, dtype=torch.float32),
                    )
                lengths, ranks, result = buffers[key]
                xx = F.pad(left[r:r+m], (0, padded_k-k)).contiguous()
                torch.ops.npu.sgemmv_shrink(xx, ww, indices, lengths, ranks, scales, result, 1)
                output[r:r+m, c:c+n].copy_(result[:, :n])
        return output

    sources = [p for p in (root/'csrc/lora').rglob('*') if p.suffix in ('.cpp', '.h', '.hpp')]
    return matmul, source_hashes(Path(__file__), binary, package/'__init__.py', root/'csrc/pytorch_extensions.cpp', *sources)
