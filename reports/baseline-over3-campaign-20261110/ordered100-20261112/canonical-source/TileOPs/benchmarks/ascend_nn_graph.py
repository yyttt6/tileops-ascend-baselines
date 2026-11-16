"""Original ops-nn graph operators with manifest shape and attribute bindings."""
from pathlib import Path

from benchmarks.ascend_baselines import Candidate, LIBRARIES, WORKSPACE, UnsupportedCase, source_hashes
from benchmarks.ascend_graph_call import GraphCall


def _candidate(binding, run, op, adaptation, *directories, extra_sources=()):
    from benchmarks.api import Implementation

    files = [Path(__file__), Path(__file__).with_name("ascend_graph_call.py"), *extra_sources]
    for root in [binding.vendor, *(LIBRARIES / "ops-nn" / d for d in directories)]:
        files.extend(p for p in root.rglob("*") if p.is_file() and
                     p.suffix in (".cpp", ".h", ".py", ".json", ".so", ".o"))
    return Candidate(
        Implementation(run), "ops-nn::" + op + " via CANN graph JIT", adaptation,
        source_hashes(*files),
        limitations=("Own source compilation and device launch must be proven from CANN logs.",),
        cleanup=binding.cleanup,
    )


def _add_rms(case, select_sum):
    import torch

    if select_sum and case.params.get("alpha", 1) != 1:
        raise UnsupportedCase("The original residual output requires alpha=1.")
    binding = GraphCall("ops-nn")

    def run(x, second):
        # The native graph path may reuse its inputs for the residual output.
        # Per-call copies preserve the manifest inputs and are part of its cost.
        x = x.clone(memory_format=torch.contiguous_format)
        second = second.clone(memory_format=torch.contiguous_format)
        residual = second if select_sum else torch.zeros_like(x)
        gamma = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype) if select_sum else second
        y, added = torch.empty_like(x), torch.empty_like(x)
        rstd = torch.empty((*x.shape[:-1], 1), device=x.device, dtype=torch.float32)
        binding.call("AddRmsNorm", [x, residual, gamma], [y, rstd, added],
                     {"epsilon": float(case.params.get("eps", 1e-6))})
        return added if select_sum else y

    return _candidate(binding, run, "AddRmsNorm",
                      "Select original residual sum." if select_sum else
                      "Zero residual; original weighted RMS output and manifest epsilon.",
                      "norm/add_rms_norm")


def opsnn_graph_add(case):
    return _add_rms(case, True)


def opsnn_graph_rms(case):
    return _add_rms(case, False)


def opsnn_graph_amax(case):
    import torch

    if case.inputs[0].ndim != 2 or case.params["dim"] != -1 or case.params["keepdim"]:
        raise UnsupportedCase("This pilot pools each complete row to one value.")
    binding = GraphCall("ops-nn")

    def run(x):
        packed = x.contiguous().reshape(x.shape[0], 1, 1, x.shape[1])
        out = torch.empty((x.shape[0], 1, 1, 1), device=x.device, dtype=x.dtype)
        indices = torch.empty_like(out, dtype=torch.int64)
        binding.call("AdaptiveMaxPool2d", [packed], [out, indices],
                     {"output_size": [1, 1], "argmax_dtype": 9})
        return out.reshape(x.shape[0])

    return _candidate(binding, run, "AdaptiveMaxPool2d",
                      "Rows become NCHW 1x1xW; native adaptive pooling to1x1.",
                      "pooling/adaptive_max_pool2d")


def opsnn_graph_gemm(case):
    import torch

    binding = GraphCall("ops-nn")

    def run(a, b):
        lhs = (a.T if case.params["trans_a"] else a).contiguous()
        rhs = (b.T if case.params["trans_b"] else b).contiguous()
        out = torch.empty((lhs.shape[0], rhs.shape[1]), device=a.device, dtype=a.dtype)
        binding.call("MatMulV3", [lhs, rhs, None, None], [out],
                     {"transpose_x1": False, "transpose_x2": False})
        return out

    return _candidate(binding, run, "MatMulV3",
                      "Manifest transpose flags; isolated host dtype registration binds output to input dtype.",
                      "matmul/mat_mul_v3", extra_sources=(
                          WORKSPACE / ".cache/ascend-classic/native-pilot-20261020/ops-nn/source/"
                          "matmul/mat_mul_v3/op_host/mat_mul_v3_infershape.cpp",))


def opsnn_graph_conv(case):
    import torch

    binding = GraphCall("ops-nn")
    params = case.params

    def pair(value):
        return (value, value) if isinstance(value, int) else tuple(value)

    stride, padding, dilation = [pair(params[k]) for k in ("stride", "padding", "dilation")]

    def run(x, weight, bias=None):
        x, weight = x.contiguous(), weight.contiguous()
        shape = (x.shape[0], weight.shape[0], *[
            (x.shape[i + 2] + 2 * padding[i] - dilation[i] * (weight.shape[i + 2] - 1) - 1)
            // stride[i] + 1 for i in range(2)])
        out = torch.empty(shape, device=x.device, dtype=x.dtype)
        binding.call("Conv2DV2", [x, weight, bias, None], [out], {
            "strides": [1, 1, *stride],
            "pads": [padding[0], padding[0], padding[1], padding[1]],
            "dilations": [1, 1, *dilation], "groups": int(params["groups"]),
            "data_format": "NCHW", "pad_mode": "SPECIFIC", "enable_hf32": False,
        }, formats=[0, 0, 2, 2, 0])
        return out

    return _candidate(binding, run, "Conv2DV2",
                      "NCHW tensors and all original manifest convolution attributes.",
                      "conv/conv2d_v2")


def opsnn_graph_fp8(case):
    import torch
    import torch.nn.functional as F

    if case.inputs[0].shape[-1] != 64:
        raise UnsupportedCase("This pilot pads each64-channel row to128.")
    binding = GraphCall("ops-nn")

    def run(x):
        packed = F.pad(x.float().reshape(-1, 64), (0, 64), value=1e-4)
        codes = torch.empty(packed.shape, device=x.device, dtype=torch.float8_e4m3fn)
        scale = torch.empty((packed.shape[0], 1), device=x.device, dtype=torch.float32)
        binding.call("DynamicBlockQuant", [packed], [codes, scale], {
            "min_scale": 0.0, "round_mode": "rint", "dst_type": 36,
            "row_block_size": 1, "col_block_size": 128,
        })
        return scale.reshape(x.shape[:-1]), codes[:, :64].contiguous().reshape(x.shape)

    return _candidate(binding, run, "DynamicBlockQuant",
                      "Exact FP16->FP32 widening and64 dummy1e-4 values enforce the original scale floor; crop codes.",
                      "quant/dynamic_block_quant")
