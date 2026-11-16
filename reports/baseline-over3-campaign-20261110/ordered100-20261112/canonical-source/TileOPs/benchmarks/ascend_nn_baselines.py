"""Thin calls to ops-nn's original native APIs; no device kernel changes."""
import ctypes as C
from pathlib import Path

from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, source_hashes
from benchmarks.ascend_math_more import _MathCall


def _candidate(binding, run, api, adaptation, *directories):
    from benchmarks.api import Implementation

    hashes = source_hashes(Path(__file__))
    for directory in directories:
        hashes.update(binding.hashes(LIBRARIES / "ops-nn" / directory))
    return Candidate(
        Implementation(run), "ops-nn::" + api, adaptation, hashes,
        limitations=("Requires isolated vendor deployment and own device-kernel trace.",),
        cleanup=binding.cleanup,
    )


def _add_rms(case, *, select_sum):
    import torch

    if select_sum and case.params.get("alpha", 1) != 1:
        raise UnsupportedCase("Original fused residual output requires alpha=1.")
    binding = _MathCall(
        "aclnnAddRmsNorm", [C.c_void_p] * 3 + [C.c_double] + [C.c_void_p] * 3,
        library="ops-nn",
    )
    epsilon = case.params.get("eps", 1e-6)

    def run(x, second):
        x = x.contiguous()
        if select_sum:
            residual = second.contiguous()
            gamma = torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)
        else:
            residual = torch.zeros_like(x)
            gamma = second.contiguous()
        normalized = torch.empty_like(x)
        added = torch.empty_like(x)
        rstd = torch.empty((*x.shape[:-1], 1), dtype=torch.float32, device=x.device)
        record = binding.record()
        binding.call([
            binding.tensor(x, record), binding.tensor(residual, record),
            binding.tensor(gamma, record), epsilon, binding.tensor(normalized, record),
            binding.tensor(rstd, record), binding.tensor(added, record),
        ], record, x.device)
        return added if select_sum else normalized

    return _candidate(
        binding, run, "aclnnAddRmsNorm",
        "Select native residual sum output." if select_sum else
        "Zero residual; select original weighted RMS output with manifest epsilon.",
        "norm/add_rms_norm",
    )


def opsnn_add(case):
    return _add_rms(case, select_sum=True)


def opsnn_rms(case):
    return _add_rms(case, select_sum=False)


def opsnn_amax(case):
    import torch

    x, = case.inputs
    if x.ndim != 2 or case.params["dim"] != -1 or case.params["keepdim"]:
        raise UnsupportedCase("This pilot maps complete rows to adaptive 2D pooling.")
    binding = _MathCall("aclnnAdaptiveMaxPool2d", [C.c_void_p] * 4, library="ops-nn")

    def run(x):
        packed = x.contiguous().reshape(x.shape[0], 1, 1, x.shape[1])
        out = torch.empty((x.shape[0], 1, 1, 1), dtype=x.dtype, device=x.device)
        indices = torch.empty(out.shape, dtype=torch.int64, device=x.device)
        record = binding.record()
        binding.call([
            binding.tensor(packed, record, format_code=0), binding.array((1, 1), record),
            binding.tensor(out, record, format_code=0),
            binding.tensor(indices, record, format_code=0),
        ], record, x.device)
        return out.reshape(x.shape[0])

    return _candidate(binding, run, "aclnnAdaptiveMaxPool2d",
                      "Each row becomes NCHW 1x1xW; pool to1x1 and select values.",
                      "pooling/adaptive_max_pool2d")


def opsnn_gemm(case):
    import torch

    binding = _MathCall("aclnnMm", [C.c_void_p] * 3 + [C.c_int8], library="ops-nn")
    ta, tb = case.params["trans_a"], case.params["trans_b"]

    def run(a, b):
        lhs, rhs = (a.T if ta else a).contiguous(), (b.T if tb else b).contiguous()
        out = torch.empty((lhs.shape[0], rhs.shape[1]), dtype=a.dtype, device=a.device)
        record = binding.record()
        binding.call([binding.tensor(lhs, record), binding.tensor(rhs, record),
                      binding.tensor(out, record), 0], record, a.device)
        return out

    return _candidate(binding, run, "aclnnMm", "Manifest transpose flags; original BF16 matrix multiply.",
                      "matmul/mat_mul_v3")


def opsnn_conv(case):
    import torch

    binding = _MathCall(
        "aclnnConvolution",
        [C.c_void_p] * 6 + [C.c_bool, C.c_void_p, C.c_int64, C.c_void_p, C.c_int8],
        library="ops-nn",
    )
    params = case.params

    def pair(value):
        return (value, value) if isinstance(value, int) else tuple(value)

    stride, padding, dilation = [pair(params[k]) for k in ("stride", "padding", "dilation")]
    groups = params["groups"]

    def run(x, weight, bias=None):
        x, weight = x.contiguous(), weight.contiguous()
        shape = (x.shape[0], weight.shape[0], *[
            (x.shape[i + 2] + 2 * padding[i] - dilation[i] * (weight.shape[i + 2] - 1) - 1)
            // stride[i] + 1 for i in range(2)
        ])
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        record = binding.record()
        binding.call([
            binding.tensor(x, record, format_code=0),
            binding.tensor(weight, record, format_code=0),
            binding.tensor(bias.contiguous(), record) if bias is not None else None,
            binding.array(stride, record), binding.array(padding, record),
            binding.array(dilation, record), False, binding.array((0, 0), record),
            groups, binding.tensor(out, record, format_code=0), 0,
        ], record, x.device)
        return out

    return _candidate(binding, run, "aclnnConvolution",
                      "Original NCHW convolution API with all manifest spatial parameters.",
                      "conv/convolution_forward", "conv/conv2d_v2")


def opsnn_fp8(case):
    import os
    import torch
    import torch.nn.functional as F

    x, = case.inputs
    if x.shape[-1] != 64:
        raise UnsupportedCase("This pilot pads each64-channel row to128.")
    # This repository publishes the device kernel and uses CANN's public host
    # API in its original example. State this delegation explicitly; acceptance
    # still requires a launch trace identifying our compiled device binary.
    host = Path(os.environ["ASCEND_HOME_PATH"]) / "lib64/libopapi.so"
    binding = _MathCall(
        "aclnnDynamicBlockQuant",
        [C.c_void_p, C.c_double, C.c_char_p] + [C.c_int64] * 3 + [C.c_void_p] * 2,
        library="ops-nn", host_library=host,
    )

    def run(x):
        # The native API caps scale from above, whereas the manifest clamps
        # absmax from below. Padding with1e-4 makes the original reduction
        # produce max(abs(original64),1e-4) without changing its kernel.
        packed = F.pad(x.float().reshape(-1, 64), (0, 64), value=1e-4)
        codes = torch.empty(packed.shape, dtype=torch.float8_e4m3fn, device=x.device)
        scale = torch.empty((packed.shape[0], 1), dtype=torch.float32, device=x.device)
        record = binding.record()
        binding.call([
            binding.tensor(packed, record), 0.0, b"rint", 36, 1, 128,
            binding.tensor(codes, record), binding.tensor(scale, record),
        ], record, x.device)
        return scale.reshape(x.shape[:-1]), codes[:, :64].contiguous().reshape(x.shape)

    return _candidate(
        binding, run, "DynamicBlockQuant (CANN public host API; custom device kernel)",
        "Exact FP16->FP32 widening; pad64 dummy values1e-4; native128 block; crop codes. "
        "Stock host API is explicit, own device binary attribution is mandatory.",
        "quant/dynamic_block_quant",
    )
