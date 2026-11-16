"""Layout/API adapters to unchanged DeepGEMM kernels, with no torch math fallback.

All input-dependent packing, metadata reads, casts and copies are in run().
Torch is used only for storage/layout and the declared output cast. In particular
there are no torch matmul, convolution, activation or reduction implementations.
"""

from .layouts import convolution_plan, group_ranges, moe_ranges
from .runtime import candidate, load_native


def _unsupported(message):
    from benchmarks.ascend_baselines import UnsupportedCase

    raise UnsupportedCase(message)


def _bf16(*values):
    import torch

    if any(value.dtype != torch.bfloat16 for value in values if value is not None):
        _unsupported("Native GEMM operands are BF16; FP16/FP32 narrowing is not a lossless adaptation.")


def _device(*values):
    values = [value for value in values if value is not None]
    if any(value.device != values[0].device for value in values):
        _unsupported("All runtime inputs must be on the same device.")
    if values[0].device.type not in ("npu", "privateuseone"):
        _unsupported("This provider is an NPU implementation; there is no CPU fallback.")


def _nt_operands(a, bt):
    """Contiguous NT operands; pad short/unaligned reduction axes with zeros."""
    import torch.nn.functional as F

    if a.shape[-1] != bt.shape[-1]:
        _unsupported("The two reduction extents must match.")
    k = a.shape[-1]
    # Fractal BF16 K alignment is 16 elements. Padding also removes the
    # (stride(-2), stride(-1)) == (1, 1) ambiguity for K=1 matrices.
    extra = -k % 16
    a, bt = a.contiguous(), bt.contiguous()
    if extra:
        a, bt = F.pad(a, (0, extra)), F.pad(bt, (0, extra))
    return a, bt


def gemm(case):
    a, b = case.inputs
    _bf16(a, b)
    _device(a, b)
    ta, tb = case.params.get("trans_a", False), case.params.get("trans_b", False)
    if a.ndim != 2 or b.ndim != 2:
        _unsupported("Dense GEMM requires rank-two operands.")
    _, functions, hashes = load_native("bf16_gemm_nt")
    native = functions["bf16_gemm_nt"]

    def run(a, b):
        import torch

        lhs, rhs = _nt_operands(a.T if ta else a, b if tb else b.T)
        output = torch.empty((lhs.shape[0], rhs.shape[0]), device=a.device, dtype=a.dtype)
        native(lhs, rhs, output)
        return output

    return candidate(run, "deep_gemm.bf16_gemm_nt", "Per-call logical transpose, contiguous NT packing and zero K padding to 16; unchanged native GEMM.", hashes)


def bmm(case):
    a, b = case.inputs
    _bf16(a, b)
    _device(a, b)
    if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0] or a.shape[2] != b.shape[1]:
        _unsupported("BMM requires equal batches and [B,M,K] @ [B,K,N].")
    _, functions, hashes = load_native("bf16_bmm")
    native = functions["bf16_bmm"]

    def run(a, b):
        import torch

        lhs, rhs = _nt_operands(a, b.transpose(-1, -2))
        output = torch.empty((a.shape[0], a.shape[1], b.shape[2]), device=a.device, dtype=a.dtype)
        native(lhs, rhs, output)
        return output

    return candidate(run, "deep_gemm._C.bf16_bmm", "Per-call transpose of B into native [batch,N,K], contiguous copies and zero K padding; original batched scheduler.", hashes)


def grouped_gemm(case):
    a, b, sizes, offsets = case.inputs
    _bf16(a, b)
    _device(a, b, sizes, offsets)
    ta, tb = case.params["transpose_a"], case.params["transpose_b"]
    if a.ndim != 2 or b.ndim != (2 if ta else 3):
        _unsupported("Grouped GEMM operand ranks disagree with transpose contract.")
    _, functions, hashes = load_native("bf16_gemm_nt")
    native = functions["bf16_gemm_nt"]

    def run(a, b, sizes, offsets):
        import torch

        ranges = group_ranges(sizes.tolist(), offsets.tolist(), a.shape[0])
        if ta:
            n = b.shape[0] if tb else b.shape[1]
            output = torch.zeros((len(ranges), a.shape[1], n), device=a.device, dtype=a.dtype)
        else:
            n = b.shape[1] if tb else b.shape[2]
            output = torch.zeros((a.shape[0], n), device=a.device, dtype=a.dtype)
        for group, (start, end) in enumerate(ranges):
            if start == end:
                continue
            if ta:
                lhs = a[start:end].T
                rhs = b[:, start:end] if tb else b[start:end].T
                target = output[group]
            else:
                lhs = a[start:end]
                rhs = b[group] if tb else b[group].T
                target = output[start:end]
            native(*_nt_operands(lhs, rhs), target)
        return output

    return candidate(run, "deep_gemm.bf16_gemm_nt (one original call per group)",
                     "Read and validate current counts/offsets inside run; slice M groups or K groups, transpose/pack/pad and invoke unchanged dense API for each nonempty group. Empty groups are zero padding.",
                     hashes, host_roundtrip=True)


def _fp8_setup(case, batched):
    import torch

    a, b, sa, sb, *bias = case.inputs
    _device(*case.inputs)
    if a.dtype != torch.float8_e4m3fn or b.dtype != a.dtype:
        _unsupported("The manifest route requires E4M3 inputs, exactly representable in BF16.")
    if sa.numel() != 1 or sb.numel() != 1:
        _unsupported("Arbitrary FP32 block scales cannot be represented by UE8M0; only scalar scales map to native Alpha.")
    if sa.dtype != torch.float32 or sb.dtype != torch.float32:
        _unsupported("Expected FP32 scalar scales.")
    if batched and bias:
        _unsupported("BMM manifest has no bias input.")
    dtype_name = case.params.get("out_dtype", "bfloat16")
    if dtype_name not in ("bfloat16", "float16"):
        _unsupported("Unsupported manifest output dtype.")
    symbols = ("bf16_bmm",) if batched else ("bf16_gemm_nt",)
    module, functions, hashes = load_native(*symbols)
    return module, functions[symbols[0]], hashes, getattr(torch, dtype_name)


def fp8_gemm(case):
    _, native, hashes, out_dtype = _fp8_setup(case, False)

    def run(a, b, scale_a, scale_b, bias=None):
        import torch

        # Host scalar conversion is part of API mapping, never cached. Actual
        # output scaling is performed by the vendor ScaleByAlpha epilogue.
        alpha = float(scale_a.item()) * float(scale_b.item())
        lhs, rhs = _nt_operands(a.to(torch.bfloat16), b.to(torch.bfloat16))
        output = torch.empty((a.shape[0], b.shape[0]), device=a.device, dtype=torch.float32)
        if bias is not None:
            # The native API forbids simultaneous Alpha + separate [1,N] bias.
            # Its original C=D accumulation mode supports the same equation.
            output.copy_(bias.to(torch.float32).reshape(1, -1).expand_as(output))
        native(lhs, rhs, output, c=output if bias is not None else None, alpha=alpha)
        return output.to(out_dtype)

    return candidate(run, "deep_gemm.bf16_gemm_nt(alpha=scale_a*scale_b, c=D when bias)",
                     "Exact E4M3-to-BF16 operand widening; per-call scalar API mapping to native FP32 Alpha epilogue; optional bias broadcast into private FP32 C=D; manifest output cast. No UE8M0 scale rounding. Floating-point reassociation remains numerically unverified.",
                     hashes, host_roundtrip=True)


def fp8_bmm(case):
    module, native, hashes, out_dtype = _fp8_setup(case, True)
    trans_b = case.params.get("trans_b", False)

    def run(a, b, scale_a, scale_b):
        import torch

        alpha = float(scale_a.item()) * float(scale_b.item())
        lhs, rhs = _nt_operands(a.to(torch.bfloat16),
                               (b if trans_b else b.transpose(-1, -2)).to(torch.bfloat16))
        output = torch.empty((lhs.shape[0], lhs.shape[1], rhs.shape[1]), device=a.device, dtype=torch.float32)
        native(lhs, rhs, output, epilogue=module._C.epilogue.Alpha(alpha))
        return output.to(out_dtype)

    return candidate(run, "deep_gemm._C.bf16_bmm(epilogue=deep_gemm._C.epilogue.Alpha)",
                     "Exact E4M3-to-BF16 widening, current scalar scales mapped to native Alpha, transpose B, pad K, run unchanged batched GEMM and cast manifest output; all transformations and host synchronization timed.",
                     hashes, host_roundtrip=True)


def convolution(case):
    x, weight, bias = case.inputs
    _bf16(x, weight, bias)
    _device(x, weight, bias)
    plan = convolution_plan(case.params)
    _, functions, hashes = load_native("bf16_gemm_nt")
    native = functions["bf16_gemm_nt"]

    def run(x, weight, bias=None):
        import torch
        import torch.nn.functional as F

        rank = plan["rank"]
        # Spatial padding precedes a read-only patch view. Materialization is
        # repeated for every call and every group, including static weights.
        padding = tuple(v for pair in reversed(plan["pads"]) for v in pair)
        padded = F.pad(x, padding).contiguous() if any(padding) else x.contiguous()
        source_strides = padded.stride()
        columns_per_group = plan["output_channels_per_group"]
        channels_per_group = plan["input_channels_per_group"]
        output = torch.empty(plan["output"], device=x.device, dtype=x.dtype)
        for group in range(plan["groups"]):
            channel_start = group * channels_per_group
            group_input = padded[:, channel_start:channel_start + channels_per_group]
            # [batch, output_spatial..., channel, kernel_spatial...]
            patch_shape = (x.shape[0], *plan["spatial"], channels_per_group, *plan["kernel"])
            patch_strides = (source_strides[0],
                             *(s * st for s, st in zip(source_strides[2:], plan["stride"])),
                             source_strides[1],
                             *(s * d for s, d in zip(source_strides[2:], plan["dilation"])))
            patches = group_input.as_strided(patch_shape, patch_strides)
            lhs = patches.contiguous().view(plan["rows"], plan["reduction"])
            begin = group * columns_per_group
            end = begin + columns_per_group
            rhs = weight[begin:end].contiguous().view(columns_per_group, plan["reduction"])
            lhs, rhs = _nt_operands(lhs, rhs)
            values = torch.empty((plan["rows"], columns_per_group), device=x.device, dtype=x.dtype)
            bias_row = None if bias is None else bias[begin:end].reshape(1, -1).contiguous()
            native(lhs, rhs, values, c=bias_row)
            values = values.view(x.shape[0], *plan["spatial"], columns_per_group)
            output[:, begin:end].copy_(values.permute(0, rank + 1, *range(1, rank + 1)))
        return output

    return candidate(run, "deep_gemm.bf16_gemm_nt(c=optional [1,N] bias)",
                     "Per-call zero spatial padding, read-only strided patches for 1D/2D/3D stride/dilation, channel-group packing, zero K padding, original GEMM/bias, and NC-spatial layout restoration; no convolution arithmetic outside the native kernel.",
                     hashes)


def moe_grouped_gemm(case):
    import torch

    a, b, metadata = case.inputs
    _bf16(a, b)
    _device(a, b, metadata)
    params = case.params
    if params.get("activation") is not None:
        _unsupported("DeepGEMM BF16 epilogues do not expose the required gated activation.")
    layout = params["layout"]
    if "masked" not in layout and layout.get("contiguous", {}).get("metadata_kind") != "physical_psum":
        _unsupported("This adapter supports masked counts and physical prefix sums.")
    output_dtype = a.dtype if params.get("out_dtype") is None else getattr(torch, params["out_dtype"])
    if output_dtype not in (torch.bfloat16, torch.float32):
        _unsupported("Native BF16 GEMM emits BF16 or FP32.")
    _, functions, hashes = load_native("bf16_gemm_nt")
    native = functions["bf16_gemm_nt"]

    def run(a, b, metadata):
        import torch

        ranges = moe_ranges(layout, metadata.tolist(), tuple(a.shape), b.shape[0])
        flat = a.reshape(-1, a.shape[-1])
        output = torch.zeros((flat.shape[0], b.shape[1]), device=a.device, dtype=output_dtype)
        for expert, (start, end) in enumerate(ranges):
            if start < end:
                native(*_nt_operands(flat[start:end], b[expert]), output[start:end])
        return output.view(*a.shape[:-1], b.shape[1])

    return candidate(run, "deep_gemm.bf16_gemm_nt (per expert; masked/prefix layouts)",
                     "Read current metadata inside run; crop each expert's defined rows, call unchanged BF16 GEMM with native BF16/FP32 output, and retain zero padding for undefined rows. No gated activation or quantization substitution.",
                     hashes, host_roundtrip=True)


def ssd_chunk_coupling(case):
    c_mat, b_mat = case.inputs
    _bf16(c_mat, b_mat)
    _device(c_mat, b_mat)
    length = case.params["chunk_len"]
    if c_mat.shape != b_mat.shape or c_mat.ndim != 4 or c_mat.shape[1] % length:
        _unsupported("Chunk coupling requires equal [B,T,G,N] operands and whole chunks.")
    batch, sequence, groups, state = c_mat.shape
    chunks = sequence // length
    count = batch * chunks * groups
    _, functions, hashes = load_native("bf16_bmm")
    native = functions["bf16_bmm"]

    def run(c_mat, b_mat):
        import torch

        def pack(value):
            return value.reshape(batch, chunks, length, groups, state).permute(0, 1, 3, 2, 4).contiguous().view(count, length, state)

        lhs, rhs = _nt_operands(pack(c_mat), pack(b_mat))
        full = torch.empty((count, length, length), device=c_mat.device, dtype=c_mat.dtype)
        native(lhs, rhs, full)
        # Row-wise causal crop with zero padding, no tensor arithmetic. The
        # complete output, including the upper triangle, is returned to oracle.
        output = torch.zeros_like(full)
        for row in range(length):
            output[:, row, :row + 1].copy_(full[:, row, :row + 1])
        return output.view(batch, chunks, groups, length, length)

    return candidate(run, "deep_gemm._C.bf16_bmm (chunked C @ B.T)",
                     "Reshape/permute chunks and groups into native BMM batch; per-row causal crop and zero upper-triangle padding; all input transformations and full output copies run each invocation.",
                     hashes)
