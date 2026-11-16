"""Lazy original ops-tensor entry points; no tensor imports at module scope.

Factories and run closures require a future authorized device session. This worker
only imports the module under a device-import blocker for static registration tests.
"""
from __future__ import annotations

from math import prod
from pathlib import Path

from .contract import convolution_plan, pooling_plan, support_reason


def _check(op, case):
    from benchmarks.ascend_baselines import UnsupportedCase
    reason = support_reason(op, case.params)
    if reason:
        raise UnsupportedCase(reason)


def _native(kind):
    import ctypes as C
    from benchmarks.ascend_baselines import _owned_native
    from .build import compile_native
    binary, hashes = compile_native(kind)
    symbol = "allpairs_opstensor_" + kind
    lib = _owned_native(binary, [symbol])
    fn = getattr(lib, symbol)
    if kind == "gemm":
        fn.argtypes = [C.c_void_p] * 4 + [C.c_int64] * 3 + [C.c_int, C.c_void_p]
    elif kind == "add":
        fn.argtypes = [C.c_void_p] * 3 + [C.c_int64, C.c_void_p]
    else:
        fn.argtypes = [C.c_void_p] * 2 + [C.c_int64, C.c_void_p]
    fn.restype = C.c_int
    return fn, hashes


def _candidate(run, api, adaptation, hashes):
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    from .build import fingerprint
    files = [Path(__file__), Path(__file__).with_name("reductions.py"),
             Path(__file__).with_name("contract.py"), Path(__file__).with_name("__init__.py")]
    return Candidate(
        implementation=Implementation(run), api=api, adaptation=adaptation,
        source_sha256=hashes | fingerprint(files), timing_mode="host_roundtrip",
        limitations=("ARM64 device validation was prohibited; no current numerical receipt.",
                     "Host loops, metadata reads, allocation, packing, native launches and cropping are all timed."),
    )


def _status(code, api):
    if code:
        raise RuntimeError(f"{api} returned host status {code}")


def _float_inputs(*values):
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    actual = [v for v in values if v is not None]
    if not actual or any(v.dtype not in (torch.float16, torch.bfloat16, torch.float32) for v in actual):
        raise UnsupportedCase("Only FP16/BF16/FP32 operands; no lossy input conversion.")
    if any(v.device != actual[0].device for v in actual):
        raise UnsupportedCase("All operands must be on the same device.")
    if actual[0].device.type not in ("npu", "privateuseone"):
        raise UnsupportedCase("Native ops-tensor requires NPU inputs.")


def add(case):
    _check("AddFwdOp", case)
    import torch
    import torch.nn.functional as F
    fn, hashes = _native("add")

    def run(a, b):
        _float_inputs(a, b)
        shape = tuple(torch.broadcast_shapes(a.shape, b.shape))
        count = prod(shape)
        # Widening is exact; padding is excluded by the final crop.
        lhs = F.pad(a.expand(shape).contiguous().reshape(-1).float(), (0, -count % 64))
        rhs = F.pad(b.expand(shape).contiguous().reshape(-1).float(), (0, -count % 64))
        out = torch.empty_like(lhs)
        _status(fn(lhs.data_ptr(), rhs.data_ptr(), out.data_ptr(), lhs.numel(),
                   torch.npu.current_stream().npu_stream), "add")
        return out[:count].to(a.dtype).reshape(shape)

    return _candidate(run, "ops-tensor::add<float> (original 64-element example)",
                      "Per-call broadcast/contiguous packing, exact widening, zero-pad to 64, host repetition of original one-block launch, crop and restore dtype.", hashes)


def trunc(case):
    _check("TruncFwdOp", case)
    import torch
    import torch.nn.functional as F
    fn, hashes = _native("trunc")

    def run(x):
        _float_inputs(x)
        count = x.numel()
        packed = F.pad(x.contiguous().reshape(-1).float(), (0, -count % 256))
        out = torch.empty_like(packed)
        _status(fn(packed.data_ptr(), out.data_ptr(), packed.numel(),
                   torch.npu.current_stream().npu_stream), "trunc<float>")
        return out[:count].to(x.dtype).reshape(x.shape)

    return _candidate(run, "ops-tensor::trunc<float>(...,16,16)",
                      "Exact FP16/BF16 widening; 256-element padding/host batching of unchanged native Trunc; crop and restore dtype, including signed zeros.", hashes)


def _gemm_engine():
    import torch
    import torch.nn.functional as F
    from benchmarks.ascend_baselines import UnsupportedCase
    fn, hashes = _native("gemm")

    def mm(a, b, bias=None, out_dtype=None):
        _float_inputs(a, b, bias)
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0] or a.dtype != b.dtype:
            raise UnsupportedCase("GEMM needs matching two-dimensional operands.")
        dtype = a.dtype if out_dtype is None else out_dtype
        if dtype not in (a.dtype, torch.float32):
            raise UnsupportedCase("Only original output dtype or exact FP32 operand widening is allowed.")
        m, k = a.shape
        n = b.shape[1]
        if min(m, n, k) == 0:
            return torch.zeros((m, n), device=a.device, dtype=dtype)
        if bias is not None and (tuple(bias.shape) != (n,) or bias.dtype != a.dtype):
            raise UnsupportedCase("Native bias is a matching-dtype vector of output columns.")
        if dtype == torch.float32 and m > 128:
            # Select independent output rows to avoid the original scheduler's
            # unwritten final wave at N=128; each call retains the complete K.
            return torch.cat([mm(a[start:start + 128], b, bias, out_dtype=dtype)
                              for start in range(0, m, 128)], dim=0)
        # The original FP32 full 256-column tile would consume 64 KiB in
        # each 32 KiB L0B ping/pong slot. Partition independent OUTPUT columns
        # into <=128; native GetTilingConfig/ComputeTiling handle their tails.
        # The complete K dimension is preserved in every original invocation.
        tile_m = 128 if dtype == torch.float32 else 256
        column_width = 128 if dtype == torch.float32 else n
        column_alignment = 128 if dtype == torch.float32 else 256
        pm, pk = -m % tile_m, -k % 128
        aa = F.pad(a.to(dtype).contiguous(), (0, pk, 0, pm))
        result = torch.empty((m, n), device=a.device, dtype=dtype)
        for start in range(0, n, column_width):
            stop = min(start + column_width, n)
            count = stop - start
            pn = -count % column_alignment
            bb = F.pad(b[:, start:stop].to(dtype).contiguous(), (0, pn, 0, pk))
            bias_packed = None if bias is None else F.pad(
                bias[start:stop].to(dtype).contiguous(), (0, pn))
            out = torch.empty((m + pm, count + pn), device=a.device, dtype=dtype)
            _status(fn(aa.data_ptr(), bb.data_ptr(), out.data_ptr(),
                       None if bias_packed is None else bias_packed.data_ptr(),
                       m + pm, count + pn, k + pk,
                       {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}[dtype],
                       torch.npu.current_stream().npu_stream), "matmul_basic_kernel")
            result[:, start:stop].copy_(out[:m, :count])
        return result

    return mm, hashes


def gemm(case):
    _check("GemmFwdOp", case)
    mm, hashes = _gemm_engine()
    ta, tb = case.params.get("trans_a", False), case.params.get("trans_b", False)

    def run(a, b):
        return mm(a.T if ta else a, b.T if tb else b)

    return _candidate(run, "ops-tensor::LaunchKernel / Blaze::GemmUniversal",
                      "Original dtype/API and tiling; per-call transpose, full-tile zero padding and crop; FP32 selects native non-HF32 mode.", hashes)


def bmm(case):
    _check("BmmFwdOp", case)
    import torch
    mm, hashes = _gemm_engine()

    def run(a, b):
        if a.ndim != 3 or b.ndim != 3 or a.shape[0] != b.shape[0]:
            raise ValueError("BMM requires matching batch axes")
        return torch.stack([mm(aa, bb) for aa, bb in zip(a.unbind(0), b.unbind(0))])

    return _candidate(run, "ops-tensor::LaunchKernel / Blaze::GemmUniversal [each batch]",
                      "Unbind independent batch coordinates; original full GEMM per batch; stack/crop output. No reduction split or scheduling change.", hashes)


def grouped_gemm(case):
    _check("GroupedGemmFwdOp", case)
    import torch
    mm, hashes = _gemm_engine()
    ta, tb = case.params["transpose_a"], case.params["transpose_b"]

    def run(a, b, batch_sizes, batch_offsets):
        # Input-dependent metadata must be reread for every invocation and timed.
        sizes, offsets = batch_sizes.tolist(), batch_offsets.tolist()
        start, products = 0, []
        for group, (size, offset) in enumerate(zip(sizes, offsets)):
            if size < 0 or offset != start:
                raise ValueError("Grouped GEMM requires exclusive-prefix offsets matching current sizes")
            end = start + size
            aa = a[start:end].T if ta else a[start:end]
            bb = (b[:, start:end].T if tb else b[start:end]) if ta else (b[group].T if tb else b[group])
            products.append(mm(aa, bb))
            start = end
        if start != a.shape[0] or len(sizes) != len(offsets):
            raise ValueError("Grouped GEMM metadata does not cover the input")
        return torch.stack(products) if ta else torch.cat(products, dim=0)

    return _candidate(run, "ops-tensor::LaunchKernel / Blaze::GemmUniversal [each current group]",
                      "Read current sizes/offsets inside run; slice independent groups, select original transpose layouts, run full native GEMMs and concatenate/stack.", hashes)


def moe_grouped_gemm(case):
    _check("MoEGroupedGemmFwdOp", case)
    import torch
    mm, hashes = _gemm_engine()
    layout = case.params["layout"]
    out_dtype_name = case.params.get("out_dtype")

    def run(a, b, layout_metadata):
        dtype = a.dtype if out_dtype_name is None else getattr(torch, out_dtype_name)
        out = torch.zeros((*a.shape[:-1], b.shape[1]), dtype=dtype, device=a.device)
        metadata = layout_metadata.tolist()
        if len(metadata) != b.shape[0]:
            raise ValueError("One current count/end per expert is required")
        start = 0
        for group, end in enumerate(metadata):
            if "masked" in layout:
                if end < 0 or end > a.shape[1]:
                    raise ValueError("Masked expert count outside allocated extent")
                if end:
                    out[group, :end].copy_(mm(a[group, :end], b[group].T, out_dtype=dtype))
            else:
                alignment = layout["contiguous"].get("alignment", 1)
                start = -(-start // alignment) * alignment
                if end < start or end > a.shape[0]:
                    raise ValueError("Invalid physical expert prefix sum")
                if end > start:
                    out[start:end].copy_(mm(a[start:end], b[group].T, out_dtype=dtype))
                start = end
        return out

    return _candidate(run, "ops-tensor::LaunchKernel / Blaze::GemmUniversal [each expert]",
                      "Per-call expert metadata, layout slices and full original GEMM; zero-fill invalid padding rows. FP32 output exactly widens operands and selects original FP32 GEMM; gated activation rejected.", hashes)


def convolution(case):
    rank = len(case.params["input"][0]) - 2
    _check(f"Conv{rank}dFwdOp", case)
    import torch
    import torch.nn.functional as F
    plan = convolution_plan(case.params)
    mm, hashes = _gemm_engine()

    def run(x, weight, bias=None):
        _float_inputs(x, weight, bias)
        padding = tuple(v for p in reversed(plan["padding"]) for v in (p, p))
        patches = F.pad(x, padding)
        # unfold/as_strided and gather are layout transformations, not convolution arithmetic.
        for dim, (k, d, s) in enumerate(zip(plan["kernel"], plan["dilation"], plan["stride"])):
            patches = patches.unfold(2 + dim, d * (k - 1) + 1, s)
        patches = patches[(slice(None),) * (rank + 2) + tuple(slice(None, None, d) for d in plan["dilation"])]
        n, g, cg, co, positions, reduction = (plan[k] for k in (
            "batch", "groups", "channels_per_group", "outputs_per_group", "output_positions", "reduction"))
        # Keep Conv3d packing within the runtime's rank-8 copy ABI by
        # selecting each independent group before the contiguous copy.
        order = (0, *range(2, 2 + rank), 1, *range(2 + rank, 2 + 2 * rank))
        weights = weight.reshape(g, co, reduction)
        outputs = []
        for group in range(g):
            aa = patches[:, group * cg:(group + 1) * cg].permute(order).contiguous()
            aa = aa.reshape(n * positions, reduction)
            bb = weights[group].T
            bias_group = None if bias is None else bias[group * co:(group + 1) * co]
            value = mm(aa, bb, bias_group)
            outputs.append(value.reshape(n, *plan["output"], co).movedim(-1, 1))
        return torch.cat(outputs, dim=1).contiguous()

    return _candidate(run, "ops-tensor::LaunchKernel / Blaze::GemmUniversal (packed convolution)",
                      "Per-call zero padding, strided/dilated sliding-window layout, groups and native column bias; original full-reduction GEMM; restore channel-first output. Covers 1D/2D/3D without the demo Conv2D L0 whole-patch limit.", hashes)


def ssd_chunk_coupling(case):
    _check("SSDChunkCouplingFwdOp", case)
    import torch
    mm, hashes = _gemm_engine()
    chunk = case.params["chunk_len"]

    def run(c, b):
        batch, length, groups, state = c.shape
        if length % chunk or b.shape != c.shape:
            raise ValueError("Chunk coupling needs equal complete chunk shapes")
        chunks = length // chunk
        cc = c.reshape(batch, chunks, chunk, groups, state).permute(0, 1, 3, 2, 4)
        bb = b.reshape(batch, chunks, chunk, groups, state).permute(0, 1, 3, 2, 4)
        out = torch.zeros((batch, chunks, groups, chunk, chunk), dtype=c.dtype, device=c.device)
        for i in range(batch):
            for j in range(chunks):
                for g in range(groups):
                    value = mm(cc[i, j, g], bb[i, j, g].T)
                    # Crop each causal output row; no external multiplication/matmul.
                    for row in range(chunk):
                        out[i, j, g, row, :row + 1].copy_(value[row, :row + 1])
        return out

    return _candidate(run, "ops-tensor::LaunchKernel / Blaze::GemmUniversal (chunk C @ B.T)",
                      "Chunk/group layout and original complete GEMMs; copy causal lower-triangle slices into zero-initialized output. All per-call layout, row copies and launches included.", hashes)


def max_pool(case):
    # The exact registered name is inferred from the original case metadata.
    if "output_size" in case.params:
        op = "AdaptiveMaxPool2dFwdOp"
    else:
        kernel = case.params["kernel_size"]
        rank = len(kernel) if isinstance(kernel, (tuple, list)) else 1
        op = f"MaxPool{rank}dFwdOp"
    _check(op, case)
    import torch
    import torch.nn.functional as F
    plan = pooling_plan(op, case.params)
    fn, hashes = _native("reduction")

    def max_rows(rows):
        count, width = rows.shape
        result = torch.empty(count, device=rows.device, dtype=rows.dtype)
        # Bounded scratch does not change any original kernel launch or reduction.
        for start in range(0, count, 4096):
            stop = min(start + 4096, count)
            packed = F.pad(rows[start:stop].float().contiguous(), (0, 64 - width), value=float("-inf"))
            scratch = torch.empty((stop - start, 192), device=rows.device, dtype=torch.float32)
            _status(fn(packed.data_ptr(), scratch.data_ptr(), stop - start,
                       torch.npu.current_stream().npu_stream), "reduction_compute")
            result[start:stop].copy_(scratch[:, 64].to(rows.dtype))
        return result

    def run(x):
        _float_inputs(x)
        if plan["adaptive"]:
            result = torch.empty((*x.shape[:-2], *plan["output"]), device=x.device, dtype=x.dtype)
            for i, j, hs, he, ws, we in plan["windows"]:
                rows = x[..., hs:he, ws:we].reshape(-1, (he - hs) * (we - ws))
                result[..., i, j].copy_(max_rows(rows).reshape(x.shape[:-2]))
            return result
        rank = plan["rank"]
        lead = x.ndim - rank
        padding = tuple(v for pair in reversed(tuple(zip(plan["padding"], plan["right"]))) for v in pair)
        windows = F.pad(x, padding, value=float("-inf"))
        for dim, (k, d, s) in enumerate(zip(plan["kernel"], plan["dilation"], plan["stride"])):
            windows = windows.unfold(lead + dim, d * (k - 1) + 1, s)
        windows = windows[(slice(None),) * x.ndim + tuple(slice(None, None, d) for d in plan["dilation"])]
        rows = windows.reshape(-1, plan["max_window"])
        return max_rows(rows).reshape(*x.shape[:-rank], *plan["output"])

    return _candidate(run, "ops-tensor::reduction_compute(...)[max_output_index=64]",
                      "Pack each complete pooling window into the original 64-lane reduction; pad with -infinity; exact FP32 widening; select max output and restore layout. Ceil-mode extension and adaptive window slicing are layout only; no reduction tree or index synthesis.", hashes)
