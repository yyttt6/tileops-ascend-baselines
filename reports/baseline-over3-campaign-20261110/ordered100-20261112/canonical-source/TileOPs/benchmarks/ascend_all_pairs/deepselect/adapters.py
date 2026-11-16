"""Value selection adapters; no target math is delegated to torch.

All input-dependent layout work is inside run. Allocation, copy, broadcast,
strided windows, constant padding and exact dtype representation are the only
torch operations used here. The maximum is always DeepSelect's original TopK.
"""

from math import prod
from pathlib import Path

from .semantics import adaptive_windows, axes_plan, pool_plan, support

# Bound each original native launch; the existing loops still visit every row.
ROW_BATCH = 32768


def factory(op, case):
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase, source_hashes

    status, reason = support(op, case.params)
    if status != "supported_unverified":
        raise UnsupportedCase(reason)

    from .loader import load_owned, vendor_hashes

    # Resolve/validate the owned artifact before materializing Case.inputs.
    module, native, receipt, compatibility = load_owned()
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    params = case.params
    inputs = case.inputs
    if any(x.device.type != "npu" for x in inputs):
        raise UnsupportedCase("DeepSelect candidate requires NPU inputs under the host guard.")
    boolean = op in ("AnyFwdOp", "LogicalOrFwdOp")
    alignment = module.get_stride_requirement()[0] // 2
    if alignment <= 0:
        raise RuntimeError("Invalid original DeepSelect input alignment")

    def select(packed):
        """One value per packed row; source data is freshly packed every call."""
        rows, width = packed.shape
        padded = (width + alignment - 1) // alignment * alignment
        dtype = torch.bfloat16 if boolean else packed.dtype
        # Always copy into private aligned storage: original inputs stay untouched.
        buffer = torch.empty((rows, padded), dtype=dtype, device=packed.device)
        buffer[:, :width].copy_(packed)  # bool -> BF16 is exact 0/1 widening.
        values, _ = module.topk(
            buffer[:, :width].view(torch.bfloat16), 1, sorted=False,
            indices_type=torch.int32, return_value=True, abort_when_nan_found=True,
        )
        values = values.contiguous()
        return values.to(torch.bool).reshape(rows) if boolean else values.view(packed.dtype).reshape(rows)

    def select_batches(packed):
        out = torch.empty((packed.shape[0],), dtype=torch.bool if boolean else packed.dtype, device=packed.device)
        for start in range(0, packed.shape[0], ROW_BATCH):
            end = min(start + ROW_BATCH, packed.shape[0])
            out[start:end].copy_(select(packed[start:end]))
        return out

    if op in ("AmaxFwdOp", "AnyFwdOp"):
        shape = tuple(params["x"][0])
        order, rows, width, output_shape = axes_plan(shape, params.get("dim"), params.get("keepdim", False))

        def run(x):
            packed = x.permute(order).contiguous().reshape(rows, width)
            return select_batches(packed).reshape(output_shape)

        adaptation = "Permute/fold reduction axes; private 32-byte-aligned row packing; original k=1 values; restore keepdim/output layout."
    elif op in ("MaximumFwdOp", "LogicalOrFwdOp", "ReluFwdOp"):
        output_shape = tuple(params["output"][0])

        def run(x, *other):
            left = x.expand(output_shape).contiguous().reshape(-1)
            right = other[0].expand(output_shape).contiguous().reshape(-1) if other else None
            out = torch.empty_like(left, dtype=torch.bool if boolean else x.dtype)
            for start in range(0, left.numel(), ROW_BATCH):
                end = min(start + ROW_BATCH, left.numel())
                pair = torch.empty((end - start, 2), dtype=x.dtype, device=x.device)
                pair[:, 0].copy_(left[start:end])
                if right is None:
                    pair[:, 1].fill_(0)  # ReLU is selection from [input, constant zero].
                else:
                    pair[:, 1].copy_(right[start:end])
                out[start:end].copy_(select(pair))
            return out.reshape(output_shape)

        adaptation = "Broadcast operands and pack two columns per output; original k=1 selection; ReLU pads each single-value row with constant zero."
    elif op.startswith("MaxPool"):
        shape, kernel, stride, padding, dilation, output_spatial, right = pool_plan(params)
        planes, width, locations = prod(shape[:2]), prod(kernel), prod(output_spatial)
        output_shape = shape[:2] + output_spatial
        pad = tuple(v for pair in reversed(tuple(zip(padding, right))) for v in pair)

        def run(x):
            padded = F.pad(x.reshape(planes, *shape[2:]), pad, value=float("-inf"))
            base_stride = padded.stride()
            windows = padded.as_strided(
                (planes, *output_spatial, *kernel),
                (base_stride[0], *(s * t for s, t in zip(stride, base_stride[1:])),
                 *(d * t for d, t in zip(dilation, base_stride[1:]))),
            )
            out = torch.empty((planes, locations), dtype=x.dtype, device=x.device)
            plane_batch = max(1, ROW_BATCH // locations)
            for start in range(0, planes, plane_batch):
                end = min(start + plane_batch, planes)
                packed = windows[start:end].contiguous().reshape(-1, width)
                out[start:end].copy_(select_batches(packed).reshape(end - start, locations))
            return out.reshape(output_shape)

        adaptation = "Pad spatial borders with -inf, including ceil-mode right tail; extract strided/dilated pooling windows; original k=1 value selection; restore N,C,spatial output."
    else:
        shape, out_spatial, windows = adaptive_windows(params)
        planes = prod(shape[:-2])

        def run(x):
            view = x.reshape(planes, *shape[-2:])
            out = torch.empty((planes, len(windows)), dtype=x.dtype, device=x.device)
            for location, (hs, he, ws, we) in enumerate(windows):
                packed = view[:, hs:he, ws:we].contiguous().reshape(planes, -1)
                out[:, location].copy_(select_batches(packed))
            return out.reshape(*shape[:-2], *out_spatial)

        adaptation = "Use exact integer floor/ceil adaptive-window boundaries, slice/pack each window, call original k=1 value selection and restore batched/unbatched output."

    representation = " Bool inputs widen exactly to BF16 0/1; selected 0/1 values convert back to bool." if boolean else " FP16/BF16 use bit-preserving ordered 16-bit views, with no numeric FP16-to-BF16 conversion."
    own = Path(__file__).parent
    return Candidate(
        Implementation(run=run), "deep_select.topk(input, 1, sorted=False, indices_type=torch.int32, return_value=True)[0]",
        adaptation + representation + " All packing, copies, padding and native calls are inside run and count toward timing.",
        vendor_hashes() | compatibility | source_hashes(native, receipt, own / "adapters.py", own / "semantics.py", own / "loader.py"),
        limitations=(
            "CPU static inspection only in this task; no manifest numerical validation or runtime ownership receipt yet.",
            "Finite manifest inputs: the original selector's NaN-abort policy is not torch NaN propagation.",
            "Original radix selection orders +0 above -0; returned zero signs are not promised bitwise identical.",
            "Selection of millions of tiny rows is mathematically valid but has not been benchmarked.",
        ),
    )
