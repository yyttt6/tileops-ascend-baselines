"""Complete cache results using the original state-copy gather/scatter kernel."""
from math import prod
from pathlib import Path
from .runtime import TRITON_ROOT, VENDOR, candidate, module, validate


def cache_transfer(case):
    validate(case)
    import torch
    import triton
    native, sources = module(TRITON_ROOT + "kda_state_copy_kernel.py")
    sources += [VENDOR / "vllm_ascend/ops/kda_state_copy_plan.py"]
    op, p = case._call.signature.name, case.params
    if "scale" in p:
        scaler, extra = module(TRITON_ROOT + "muls_add.py", hardware=True)
        sources += extra
    if p.get("fuse_rope"):
        rotation, extra = module(TRITON_ROOT + "rope.py", hardware=True)
        sources += extra


    def run(*args):
        if op == "PagedKVCacheGatherFwdOp":
            dst, cache, table, offsets, starts, scale = args
            bounds = offsets.tolist()
            starts = [0]*(len(bounds)-1) if starts is None else starts.tolist()
            pages = table.tolist()
            slots = [pages[b][i//cache.shape[1]]*cache.shape[1]+i%cache.shape[1]
                     for b, (begin, end) in enumerate(zip(bounds[:-1], bounds[1:]))
                     for i in range(starts[b], starts[b]+end-begin)]
            indices = torch.tensor(slots, device=cache.device, dtype=torch.int64)
            out = copy_payload(native, cache, None, indices, False)
            if scale is not None:
                flat = out.float().reshape(-1, 1)
                out = scaler.muls_add_triton(flat, torch.zeros_like(flat), float(scale.item()))
            return out.reshape(dst.shape).to(dst.dtype)
        if op == "PagedKVCacheWriteFwdOp":
            k, v, kc, vc, indices, _ks, _vs = args
            return copy_payload(native, kc, k, indices, True), copy_payload(native, vc, v, indices, True)
        latent, pe, cache, indices, _scale, positions, tables = args
        if p.get("fuse_rope"):
            pe = rotation.rope_forward_triton_siso(pe.clone().unsqueeze(1).contiguous(),
                cos_sin_cache=tables.contiguous(), positions=positions.contiguous(),
                rope_dim=pe.shape[-1], is_neox_style=p["rope_layout"] == "neox").squeeze(1)
        return copy_payload(native, cache, torch.cat((latent, pe), -1), indices, True)

    return candidate(run, "_kda_state_copy_kernel(gather/scatter)",
        "Flatten cache page/row layouts into original state-copy payloads; use original default 8192-element tiles, invalid-slot behavior and unique-write precondition. Clone full caches and return every updated element; bit-preserving BF16 views handle FP16 payloads and exact FP32 widening handles FP8 reads. Scalar gather dequantization and optional MLA rotation use original owned kernels; quantized writes are excluded.",
        [Path(__file__), *sources], timing_mode="host_completed")


def copy_payload(native, cache, data, indices, to_cache):
    import torch
    import triton
    # The original copy kernel performs no arithmetic on payloads. Reinterpret
    # FP16 as BF16 so even NaN payloads and untouched cache bytes survive.
    raw_half = cache.dtype == torch.float16
    dtype = torch.bfloat16 if cache.dtype in (torch.float16, torch.bfloat16) else torch.float32
    rows, width = cache.shape[0]*cache.shape[1], prod(cache.shape[2:])
    storage = (cache.view(torch.bfloat16) if raw_half else cache.to(dtype)).clone().contiguous().reshape(rows, width)
    packed = ((data.to(cache.dtype).contiguous().view(torch.bfloat16) if raw_half else data.to(dtype).contiguous()).reshape(indices.numel(), width) if to_cache else
              torch.empty((indices.numel(), width), device=cache.device, dtype=dtype))
    native._kda_state_copy_kernel[(indices.numel(), triton.cdiv(width, native.DEFAULT_KDA_BLOCK_SIZE))](
        storage, packed, indices, indices, rows, storage.stride(0), width, indices.stride(0), 0,
        TO_CACHE=to_cache, HAS_FLAGS=False, BLOCK_SIZE=native.DEFAULT_KDA_BLOCK_SIZE)
    if to_cache:
        result = storage.reshape(cache.shape)
        return result.view(torch.float16) if raw_half else result.to(cache.dtype)
    return packed.view(torch.float16) if raw_half else packed
