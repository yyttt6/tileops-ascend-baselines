"""Pure Python admission rules, shared by runtime factories and the CPU audit."""

from math import prod


def plan(op, params):
    """Return static layout metadata or raise ValueError with a concrete restriction."""
    if op == "CountNonzeroFwdOp":
        shape, dtype = params["x"]
        if dtype != "bool":
            raise ValueError("Only bool values widen losslessly to expert IDs 0/1; general numeric nonzero predicates are absent.")
        dims = params.get("dim")
        rank = len(shape)
        dims = list(range(rank)) if dims is None else [dims] if isinstance(dims, int) else list(dims)
        if not dims:
            dims = list(range(rank))
        if any(not (-rank <= d < rank if rank else -1 <= d <= 0) for d in dims):
            raise ValueError("Invalid reduction axes.")
        normalized = [d % rank if rank else 0 for d in dims]
        if len(set(normalized)) != len(normalized):
            raise ValueError("Invalid reduction axes.")
        axes = normalized if rank else []
        kept = [d for d in range(len(shape)) if d not in axes]
        elements = prod(shape[d] for d in axes)
        if not 0 < elements <= 0xFFFFFF:
            raise ValueError("Native route counting has a positive 24-bit token bound per reduction row.")
        return dict(tokens=elements, hidden=128, experts=2, topk=1,
                    batches=prod(shape[d] for d in kept), permutation=tuple(kept + axes),
                    output_shape=tuple(params["output"][0]))
    if op == "PagedKVCacheGatherFwdOp":
        cache_shape, dtype = params["cache"]
        dst_shape, out_dtype = params["dst"]
        if dtype not in ("float16", "bfloat16") or out_dtype != dtype:
            raise ValueError("Engram fetch copies payload bytes; it does not dequantize FP8 or apply scale.")
        if params.get("scale") is not None:
            raise ValueError("Engram fetch has no arithmetic scale epilogue.")
        if len(cache_shape) < 3 or list(cache_shape[2:]) != list(dst_shape[1:]):
            raise ValueError("Paged cache and destination must have the same per-row payload shape.")
        pages, page_size = cache_shape[:2]
        rows, hidden = pages * page_size, prod(cache_shape[2:])
        if min(pages, page_size, hidden, dst_shape[0]) <= 0 or rows >= 2**31:
            raise ValueError("Engram fetch requires positive extents and int32-addressable cache rows.")
        return dict(kind="engram", entries=rows, tokens=dst_shape[0], hidden=hidden,
                    page_size=page_size, pages=pages, output_shape=tuple(dst_shape))
    if op == "AddFwdOp":
        shape, dtype = params["output"]
        if params.get("alpha", 1) != 1:
            raise ValueError("DeepEP combine bias addition has no alpha multiplier; alpha must be 1.")
        if any(params[key][1] != "bfloat16" for key in ("input", "other", "output")):
            raise ValueError("Native combine and its two biases require BF16; FP16/FP32 -> BF16 is lossy.")
        if not shape or min(shape) <= 0:
            raise ValueError("Native EP buffer needs positive token and hidden dimensions.")
        # A width of 128 meets the original 256-byte vector-register alignment;
        # flattening and zero-padding the tail do not alter any addition.
        elements = prod(shape)
        rows = (elements + 127) // 128
        if rows > 0xFFFFFF:
            raise ValueError("Native dispatch uses a 24-bit per-rank token counter.")
        return dict(shape=tuple(shape), elements=elements, tokens=rows, hidden=128, topk=1, experts=1)
    if op == "MoEPrePermuteFwdOp":
        layout = params.get("layout", {}).get("contiguous", {})
        if (layout.get("packing"), layout.get("metadata_kind"), layout.get("alignment")) != (
            "tight", "physical_psum", 1
        ):
            raise ValueError("This adapter returns tight expert-major rows and physical prefix sums, alignment=1.")
        (tokens, hidden), dtype = params["hidden_states"]
        (id_tokens, topk), id_dtype = params["local_expert_ids"]
        experts = params["num_local_experts"]
        if dtype not in ("float16", "bfloat16") or id_dtype != "int32":
            raise ValueError("Prepermute preserves 16-bit payloads and accepts int32 routing IDs.")
        if tokens != id_tokens or min(tokens, hidden, topk, experts) <= 0:
            raise ValueError("Invalid routing dimensions.")
        if experts > 512 or topk > 32:
            raise ValueError("Original dispatch copy epilogue permits <=512 local experts and <=32 routes/token.")
        if tokens > 0xFFFFFF:
            raise ValueError("Native dispatch uses a 24-bit per-rank token counter.")
        return dict(tokens=tokens, hidden=((hidden + 127) // 128) * 128,
                    logical_hidden=hidden, topk=topk, experts=experts, dtype=dtype)
    if op == "MoEPermuteAlignFwdOp":
        shape, dtype = params["topk_ids"]
        routes = prod(shape)
        experts, block = params["num_experts"], params["block_size"]
        if dtype != "int32" or routes <= 0 or not 0 < experts <= 512:
            raise ValueError("Original dispatch needs positive routes, int32 IDs and <=512 local experts.")
        if block <= 0 or block & (block - 1):
            raise ValueError("Original expert_alignment must be a positive power of two.")
        if routes > 0xFFFFFF or ((routes + block - 1) // block) * block > 0xFFFFFF:
            raise ValueError("Aligned native host expert-count notification exceeds 24 bits.")
        capacity = routes + (experts + 1) * (block - 1)
        return dict(tokens=routes, hidden=128, topk=1, experts=experts,
                    routes=routes, block=block, capacity=capacity,
                    blocks=(capacity + block - 1) // block)
    raise ValueError(f"No own DeepEP kernel adapter for {op}.")


def block_owners(aligned_counts, block, capacity_blocks, experts):
    """Reformat the original host notification, which ALREADY includes padding.

    No histogram, routing or prefix sum is recomputed from input IDs here.
    """
    owners = []
    for expert, count in enumerate(aligned_counts):
        if count < 0 or count % block:
            raise RuntimeError("DeepEP returned an invalid alignment-padded expert count.")
        owners.extend([expert] * (count // block))
    if len(owners) > capacity_blocks or len(aligned_counts) != experts:
        raise RuntimeError("DeepEP count metadata exceeds the declared manifest capacity.")
    padded_tokens = len(owners) * block
    return owners + [experts] * (capacity_blocks - len(owners)), padded_tokens


def paged_indices(table, offsets, starts, page_size, num_pages, tokens):
    """Metadata-only address translation; payload gathering remains native.

    All arguments are CPU lists copied from current call's metadata in run().
    """
    if len(offsets) != len(table) + 1 or offsets[0] != 0 or offsets[-1] != tokens:
        raise ValueError("Paged sequence offsets do not cover the full destination.")
    starts = [0] * len(table) if starts is None else starts
    if len(starts) != len(table):
        raise ValueError("Paged sequence starts have wrong length.")
    indices = []
    for batch, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        start = starts[batch]
        if begin > end or start < 0 or start + end - begin > len(table[batch]) * page_size:
            raise ValueError("Paged sequence extends outside its block table.")
        for position in range(start, start + end - begin):
            page, in_page = divmod(position, page_size)
            physical = table[batch][page]
            if not 0 <= physical < num_pages:
                raise ValueError("Paged block table contains an out-of-range page.")
            indices.append(physical * page_size + in_page)
    return indices
