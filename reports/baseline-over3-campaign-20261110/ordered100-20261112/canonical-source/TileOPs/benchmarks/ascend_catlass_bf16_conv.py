"""BF16 instantiation of original CATLASS example56; unchanged native schedule."""
import ctypes as C
from pathlib import Path

from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, source_hashes
from benchmarks.ascend_catlass_baselines import _native


def catlass_conv_all_types(case):
    import torch
    from benchmarks.ascend_catlass_conv_cases import catlass_conv_cases

    if case.inputs[0].dtype == torch.bfloat16 and tuple(case.inputs[1].shape[-2:]) != (1, 1):
        return _bf16_conv(case)
    return catlass_conv_cases(case)


def _bf16_conv(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    x, weight, *extra = case.inputs
    p = case.params
    if any(v is not None for v in extra) or p.get("groups", 1) != 1:
        raise UnsupportedCase("The inspected original example has no bias or grouped convolution.")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise UnsupportedCase("The original example-56 compute is instantiated for BF16.")
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
    lib, hashes = _native("conv_bf16", "56_ascend950_basic_conv2d_tla/basic_conv2d_tla.cpp",
                          "classic_catlass_conv_bf16")
    fn = lib.classic_catlass_conv_bf16
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
        Implementation(run), "CATLASS example56 Ascend950BasicConv2dTLA BF16 template",
        "NCHW/OIHW -> NC1HWC0/CI1KHKWCOCI0 and back; all copies inside run; original compute and schedule.",
        hashes | source_hashes(Path(__file__), LIBRARIES / "catlass/tests/optest/tests/conv_common.py"),
    )
