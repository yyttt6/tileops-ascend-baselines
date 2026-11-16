"""BF16 instantiation of original CATLASS example56; unchanged native schedule."""
import ctypes as C
from pathlib import Path

from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, source_hashes
from benchmarks.ascend_catlass_baselines import _native


def catlass_compact_conv(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    x, weight, *extra = case.inputs
    p = case.params
    if any(v is not None for v in extra) or p.get("groups", 1) != 1:
        raise UnsupportedCase("The inspected original example has no bias or grouped convolution.")
    if x.dtype not in (torch.float16, torch.bfloat16) or weight.dtype != x.dtype:
        raise UnsupportedCase("Original compact example56 requires matching16bit operands.")
    batch, channels, hi, wi = x.shape
    out_channels, _, kh, kw = weight.shape
    stride = p.get("stride", (1, 1))
    stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
    pad = p.get("padding", (0, 0))
    pad = (pad, pad) if isinstance(pad, int) else tuple(pad)
    dilation = p.get("dilation", (1, 1))
    dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
    ho = (hi + 2 * pad[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    wo = (wi + 2 * pad[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
    ci1, co1 = (channels + 15) // 16, (out_channels + 15) // 16
    kind = "f16" if x.dtype == torch.float16 else "bf16"
    symbol = "classic_catlass_conv_compact_" + kind
    lib, hashes = _native("conv_compact_" + kind, "56_ascend950_basic_conv2d_tla/basic_conv2d_tla.cpp", symbol)
    fn = getattr(lib, symbol)
    fn.argtypes = [C.c_void_p] * 3 + [C.POINTER(C.c_uint32), C.c_void_p]
    fn.restype = C.c_int
    sizes = (C.c_uint32 * 15)(
        batch, hi, wi, ci1 * 16, out_channels, kh, kw, *stride,
        pad[1], pad[1], pad[0], pad[0], *dilation)
    out = torch.empty((batch, co1, ho, wo, 16), device=x.device, dtype=x.dtype)

    def run(x, weight, *extra):
        x = F.pad(x, (0, 0, 0, 0, 0, ci1 * 16 - channels))
        weight = F.pad(weight, (0, 0, 0, 0, 0, ci1 * 16 - channels))
        x = x.reshape(batch, ci1, 16, hi, wi).permute(0, 1, 3, 4, 2).contiguous()
        weight = weight.reshape(out_channels, ci1, 16, kh, kw).permute(1, 3, 4, 0, 2).contiguous()
        rc = fn(x.data_ptr(), weight.data_ptr(), out.data_ptr(), sizes,
                torch.npu.current_stream().npu_stream)
        if rc:
            raise RuntimeError(f"CATLASS convolution host status {rc}")
        return out.permute(0, 1, 4, 2, 3).contiguous().view(batch, co1 * 16, ho, wo)[:, :out_channels]

    return Candidate(
        Implementation(run), "CATLASS example56 compact L1 tile, original16bit native templates",
        "NCHW/OIHW -> NC1HWC0/CI1KHKWCOCI0 and back; all copies inside run; only L1 tile extents changed to2x8x1/96x1; all stagecounts, synchronization and computation unchanged.",
        hashes | source_hashes(Path(__file__), LIBRARIES / "catlass/tests/optest/tests/conv_common.py"),
    )


def catlass_compact_cases(case):
    from benchmarks.ascend_bias_coordinate_conv import _bias_coordinate, catlass_bias_coordinate_cases
    if tuple(case.inputs[1].shape[-2:]) == (1, 1) or case.params['groups'] != 1:
        return catlass_bias_coordinate_cases(case)
    return _bias_coordinate(case, catlass_compact_conv)
