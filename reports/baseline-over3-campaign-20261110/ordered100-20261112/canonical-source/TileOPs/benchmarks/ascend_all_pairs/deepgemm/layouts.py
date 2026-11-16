"""Pure Python shape planning; no tensor or device imports."""

from math import prod


def ntuple(value, rank):
    values = (value,) * rank if isinstance(value, int) else tuple(value)
    if len(values) != rank or not all(isinstance(x, int) for x in values):
        raise ValueError(f"Expected {rank} integer spatial parameters, got {value!r}")
    return values


def convolution_plan(params):
    xshape, _ = params["input"]
    wshape, _ = params["weight"]
    rank = len(xshape) - 2
    if rank not in (1, 2, 3) or len(wshape) != rank + 2:
        raise ValueError("Expected NC[spatial] and OC,IC/group,[kernel] tensors")
    stride = ntuple(params.get("stride", 1), rank)
    dilation = ntuple(params.get("dilation", 1), rank)
    kernel = tuple(wshape[2:])
    extent = tuple(d * (k - 1) + 1 for d, k in zip(dilation, kernel))
    padding = params.get("padding", 0)
    if padding == "valid":
        pads = ((0, 0),) * rank
    elif padding == "same":
        if stride != (1,) * rank:
            raise ValueError("Manifest same convolution requires stride one")
        pads = tuple(((e - 1) // 2, e - 1 - (e - 1) // 2) for e in extent)
    else:
        pads = tuple((p, p) for p in ntuple(padding, rank))
    if min(stride + dilation + kernel) < 1 or any(min(p) < 0 for p in pads):
        raise ValueError("Invalid convolution geometry")
    groups = params.get("groups", 1)
    if groups < 1 or xshape[1] != groups * wshape[1] or wshape[0] % groups:
        raise ValueError("Invalid channel grouping")
    spatial = tuple((s + lo + hi - e) // st + 1
                    for s, (lo, hi), e, st in zip(xshape[2:], pads, extent, stride))
    if min(spatial) < 1:
        raise ValueError("Empty convolution output is outside current manifest")
    output = (xshape[0], wshape[0], *spatial)
    if "output" in params and tuple(params["output"][0]) != output:
        raise ValueError(f"Manifest output disagrees with convolution geometry: {output}")
    return dict(rank=rank, stride=stride, dilation=dilation, kernel=kernel,
                pads=pads, spatial=spatial, groups=groups, output=output,
                rows=xshape[0] * prod(spatial), reduction=wshape[1] * prod(kernel),
                input_channels_per_group=wshape[1], output_channels_per_group=wshape[0] // groups)


def group_ranges(sizes, offsets, total):
    """Validate prefix metadata afresh; never cache input-dependent partitions."""
    if len(sizes) != len(offsets):
        raise ValueError("Group counts and offsets differ in length")
    end = 0
    result = []
    for count, offset in zip(sizes, offsets):
        if count < 0 or offset != end or end + count > total:
            raise ValueError("Grouped GEMM requires nonnegative contiguous prefix offsets")
        result.append((end, end + count))
        end += count
    if end != total:
        raise ValueError("Group counts do not cover the input")
    return result


def moe_ranges(layout, metadata, shape, experts):
    if len(metadata) != experts:
        raise ValueError("Expected one metadata entry per expert")
    if "masked" in layout:
        maximum = layout["masked"]["max_m"]
        if len(shape) != 3 or shape[:2] != (experts, maximum):
            raise ValueError("Masked slab shape disagrees with layout")
        if any(count < 0 or count > maximum for count in metadata):
            raise ValueError("Masked count exceeds its slab")
        return [(g * maximum, g * maximum + count) for g, count in enumerate(metadata)]
    spec = layout.get("contiguous", {})
    if spec.get("metadata_kind") != "physical_psum":
        raise ValueError("Only physical prefix sums or masked counts are implemented")
    alignment = spec.get("alignment", 1)
    if alignment < 1 or len(shape) != 2:
        raise ValueError("Invalid contiguous layout")
    previous = 0
    result = []
    for end in metadata:
        start = (previous + alignment - 1) // alignment * alignment
        if not start <= end <= shape[0]:
            raise ValueError("Invalid physical prefix sum")
        result.append((start, end))
        previous = end
    return result
