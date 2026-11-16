"""Owned native extension routes. No extension is imported at package load time."""
from pathlib import Path
import os
import platform
import struct

from .runtime import CACHE, TRITON_ROOT, VENDOR, candidate, module, validate


def native_ops(subdir):
    import torch
    binary = Path(os.environ.get("VLLM_ALL_PAIRS_EXTENSION", str(CACHE / "install/vllm_ascend/vllm_ascend_C.abi3.so"))).resolve()
    opp = Path(os.environ.get("VLLM_ALL_PAIRS_OPP", str(CACHE / "install/vendors/custom_transformer"))).resolve()
    if not any(binary.is_relative_to(root.resolve()) for root in (CACHE, VENDOR)):
        raise RuntimeError("Extension must come from this library's own source/build directory")
    if not any(opp.is_relative_to(root.resolve()) for root in (CACHE, VENDOR)):
        raise RuntimeError("Custom OPP must come from this library's own source/build directory")
    if not binary.is_file():
        raise RuntimeError(f"deployment_blocked: build original vllm-ascend for ARM64/CANN9.1; missing {binary}")
    header = binary.read_bytes()[:20]
    machine = 183 if platform.machine() == "aarch64" else 62
    if len(header) < 20 or header[:4] != b'\x7fELF' or struct.unpack('<H', header[18:20])[0] != machine:
        raise RuntimeError("Extension ELF architecture differs from the current host; foreign .so reuse prohibited")
    api = opp / "op_api/lib/libcust_opapi.so"
    if not api.is_file():
        raise RuntimeError(f"deployment_blocked: missing original custom OPP {api}")
    configured = {Path(p).resolve() for p in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":") if p}
    if opp not in configured:
        raise RuntimeError("ASCEND_CUSTOM_OPP_PATH must select VLLM_ALL_PAIRS_OPP before process startup")
    torch.ops.load_library(str(binary))
    sources = [Path(__file__), binary, api, VENDOR / "csrc/torch_binding.cpp"]
    sources += [p for p in (VENDOR / subdir).rglob("*") if p.suffix in (".h", ".cpp", ".json")]
    sources += [p for p in opp.rglob("*") if p.is_file() and p.suffix in (".so", ".o", ".json")]
    return torch.ops._C_ascend, sources


def fused_topk(case):
    validate(case)
    ops, sources = native_ops("csrc/moe/moe_gating_top_k")
    p = case.params

    if p["routed_scaling_factor"] != 1.0:
        scaling, scale_sources = module(TRITON_ROOT + "muls_add.py", hardware=True)
        sources += scale_sources

    def run(logits, correction_bias=None):
        import torch
        xp = logits.float().contiguous()
        weights, indices, scores = ops.moe_gating_top_k(
            xp, p["top_k"], p["topk_group"], p["num_expert_group"],
            1 if correction_bias is not None else 0, 0,
            0 if p["scoring_func"] == "softmax" else 1, True, 1.0, 0.0,
            None if correction_bias is None else correction_bias.float().contiguous())
        if p["renormalize"] and p["scoring_func"] == "softmax":
            # Arch35 host only accepts renorm=0. Softmax of the selected ORIGINAL
            # logits equals original probabilities divided by their selected sum.
            selected = xp.gather(1, indices.long()).contiguous()
            _, _, weights = ops.moe_gating_top_k(selected, 1, 1, 1, 0, 0, 0, True, 1.0, 0.0, None)
        elif not p["renormalize"]:
            weights = scores.gather(1, indices.long())
        if p["routed_scaling_factor"] != 1.0:
            wp = weights.reshape(-1, 1)
            weights = scaling.muls_add_triton(wp, torch.zeros_like(wp), p["routed_scaling_factor"]).reshape(weights.shape)
        return weights, indices

    return candidate(run, "moe_gating_top_k", "Exact FP32 widening; original group selection and optional correction bias. Arch35 renorm=0: sigmoid is natively normalized, softmax normalization uses a second original softmax on selected logits; optional route scale uses original muls_add. Both outputs and all launches are included.", sources)


def mhc_post(case):
    validate(case)
    import torch
    ops, sources = native_ops("csrc/moe/hc_post")

    def run(x_layer_out, h_post, x_res):
        batch, dim = x_layer_out.shape
        expand = h_post.shape[-1]
        identity = torch.eye(expand, device=x_res.device, dtype=h_post.dtype).reshape(1, 1, expand, expand).expand(batch, 1, expand, expand).contiguous()
        result = ops.npu_hc_post(x_layer_out.unsqueeze(1).contiguous(),
                x_res.reshape(batch, 1, expand, dim).contiguous(), h_post.unsqueeze(1).contiguous(), identity)
        return result.reshape(x_res.shape)

    return candidate(run, "npu_hc_post", "Singleton sequence axis and identity residual combination matrix specialize the original HcPost to manifest post-weighted layer output plus residual; both terms computed in original kernel.", sources)


def amax(case):
    validate(case)
    import torch
    import torch.nn.functional as F
    from math import prod
    ops, sources = native_ops("csrc/attention/sparse_flash_attention")
    p = case.params
    dims = p.get("dim")
    rank = len(p["x"][0])
    dims = tuple(range(rank)) if dims is None else ((dims,) if isinstance(dims, int) else tuple(dims))
    dims = tuple(d % rank for d in dims)
    order = tuple(i for i in range(rank) if i not in dims) + dims

    def run(x):
        width = prod(x.shape[d] for d in dims)
        data = x.permute(order).contiguous().reshape(-1, width)
        rows = data.shape[0]
        keys = F.pad(data.reshape(rows, width, 1, 1), (0, 511))
        query = torch.zeros((rows, 1, 1, 512), device=x.device, dtype=x.dtype)
        query[..., 0] = 1
        qr = torch.zeros((rows, 1, 1, 64), device=x.device, dtype=x.dtype)
        kr = torch.zeros((rows, width, 1, 64), device=x.device, dtype=x.dtype)
        indices = torch.arange(width, device=x.device, dtype=torch.int32).reshape(1, 1, 1, width).expand(rows, 1, 1, width).contiguous()
        _, maximum, _ = ops.npu_sparse_flash_attention(query, keys, keys, indices, 1.0,
            query_rope=qr, key_rope=kr, sparse_block_size=1, sparse_mode=0, attention_mode=2,
            layout_query="BSND", layout_kv="BSND", return_softmax_lse=True)
        return maximum.to(x.dtype).reshape(p["output"][0])

    return candidate(run, "npu_sparse_flash_attention.softmax_max", "Pack complete reduction axes; basis query and padded keys give native QK=x; original API exposes the row maximum. All native attention and packing costs included.", sources)


def dense_attention(case):
    validate(case)
    if case.params["q"][1] == "float8_e4m3fn":
        # Existing owned C binding is preserved exactly; its source-tree OPP
        # deployment requirement is explicit and recorded separately.
        from dataclasses import replace
        from benchmarks.ascend_vllm_fp8_attention import vllm_fp8_gqa_cases
        from .runtime import hashes
        result = vllm_fp8_gqa_cases(case)
        return replace(result, source_sha256=result.source_sha256 | hashes(Path(__file__)),
                       limitations=(*result.limitations, "Not executed on this ARM64 host."))
    run, sources = _dense_route(case.params)
    return candidate(run, "npu_sparse_flash_attention (complete visible rows)", "Original sparse shared-KV attention on [K,V,0] and [Q,0], zero RoPE coordinates, all-valid indices. Each query sees its complete right-aligned causal prefix; original native RoPE consumes supplied tables when requested. Packing and every launch stay in run.", sources)


def _dense_route(p):
    import torch
    import torch.nn.functional as F
    ops, sources = native_ops("csrc/attention/sparse_flash_attention")
    if p.get("pos_encoding_mode") == "rope":
        rotation, rope_sources = module(TRITON_ROOT + "rope.py", hardware=True)
        sources += rope_sources

    def rotate(x, cos, sin):
        batch, seq, heads, dim = x.shape
        # clone after exact widening also protects a hypothetical FP32 input.
        packed = x.float().clone().reshape(batch*seq, heads, dim)
        out = rotation.rope_forward_triton_siso(packed, cos=cos.repeat(batch, 1).float(),
            sin=sin.repeat(batch, 1).float(), rope_dim=p.get("rotary_dim") or dim, is_neox_style=p["rope_layout"] == "neox")
        return out.reshape(x.shape).to(x.dtype)

    def run(q, k, v, qs=None, ks=None, vs=None, cos=None, sin=None):
        batch, sq, heads, dim = q.shape
        sk, kh, vd = k.shape[1], k.shape[2], v.shape[-1]
        if p.get("pos_encoding_mode") == "rope":
            q, k = rotate(q, cos[sk-sq:], sin[sk-sq:]), rotate(k, cos, sin)
        group = heads // kh
        pad_group = 128 if group > 64 else group
        query = q.reshape(batch, sq, kh, group, dim).permute(0, 2, 1, 3, 4)
        query = F.pad(query, (0, 512-dim, 0, pad_group-group)).reshape(batch*kh, sq, pad_group, 512)
        kv = torch.cat((k, v), -1).permute(0, 2, 1, 3).contiguous()
        kv = F.pad(kv, (0, 512-dim-vd)).reshape(batch*kh, sk, 1, 512)
        outputs = []
        scale = dim ** -0.5 if p.get("sm_scale") is None else p["sm_scale"]
        for i in range(sq):
            stop = sk-sq+i+1 if p.get("is_causal") else sk
            start = max(0, sk-sq+i-p["window_size_left"]) if p.get("window_size_left", -1) >= 0 else 0
            if p.get("window_size_right", -1) >= 0:
                stop = min(stop, sk-sq+i+p["window_size_right"]+1)
            length = max(0, stop-start)
            if not length:
                outputs.append(torch.zeros((batch*kh, 1, group, vd), device=q.device, dtype=q.dtype))
                continue
            qi, ki = query[:, i:i+1].contiguous(), kv[:, start:stop].contiguous()
            qr = torch.zeros((*qi.shape[:-1], 64), device=q.device, dtype=q.dtype)
            kr = torch.zeros((*ki.shape[:-1], 64), device=q.device, dtype=q.dtype)
            indices = torch.arange(length, device=q.device, dtype=torch.int32).reshape(1, 1, 1, length).expand(batch*kh, 1, 1, length).contiguous()
            out, _, _ = ops.npu_sparse_flash_attention(qi, ki, ki, indices, scale,
                query_rope=qr, key_rope=kr, sparse_block_size=1, layout_query="BSND", layout_kv="BSND",
                sparse_mode=0, attention_mode=2, return_softmax_lse=False)
            outputs.append(out[..., :group, dim:dim+vd])
        result = torch.cat(outputs, dim=1).reshape(batch, kh, sq, group, vd)
        result = result.permute(0, 2, 1, 3, 4).contiguous().reshape(batch, sq, heads, vd)
        return result.to(getattr(torch, p.get("out_dtype") or str(q.dtype).split(".")[-1]))

    return run, sources


def softmax(case):
    validate(case)
    import torch
    ops, sources = native_ops("csrc/moe/moe_gating_top_k")
    p = case.params
    dim = p["dim"] % len(p["x"][0])

    def run(x):
        packed = x.movedim(dim, -1).contiguous()
        _, _, scores = ops.moe_gating_top_k(packed.float().reshape(-1, packed.shape[-1]),
                                           1, 1, 1, 0, 0, 0, True, 1.0, 0.0, None)
        return scores.reshape(packed.shape).movedim(-1, dim).contiguous().to(getattr(torch, p["output"][1]))

    return candidate(run, "moe_gating_top_k(out_flag=True).out", "Original gating kernel exposes the complete FP32 softmax scores before top-k; select that output, restore axis and manifest dtype. Original <=2048 expert-width host limit applies.", sources)


def causal_conv(case):
    validate(case)
    import torch
    ops, sources = native_ops("csrc/moe/causal_conv1d_v310")
    prefill = case._call.signature.name == "CausalConv1dFwdOp"

    def run(x, weight, conv_state, *args):
        if prefill:
            boundaries, bias, has_initial = args
            data = x.transpose(0, 1).contiguous()
        else:
            bias, = args
            boundaries, has_initial = None, None
            data = x.contiguous()
        # Decode manifests shift a W-element state before convolution; the
        # native API consumes and updates only the W-1 history elements.
        history = conv_state if prefill else conv_state[..., 1:]
        state = history.transpose(-1, -2).clone().contiguous()
        indices = torch.arange(state.shape[0], device=x.device, dtype=torch.int32)
        output = ops.npu_causal_conv1d_310(data, weight.transpose(0, 1).contiguous(), bias,
            state, boundaries, indices, has_initial, None,
            int(case.params.get("activation") is not None), -1, 0 if prefill else 1)
        result = output.transpose(0, 1).contiguous() if prefill else output
        updated = state.transpose(-1, -2).contiguous()
        if not prefill:
            updated = torch.cat((conv_state[..., 1:2], updated), dim=-1)
        return result, updated

    return candidate(run, "npu_causal_conv1d_310", "Transpose packed channel/token and state/weight layouts; native prefill/update with original metadata, bias and SiLU flag. Return the complete updated state as required by the shared workload binder; keep original inputs protected.", sources, limitations=("Original OpDef is FP16 and ascend310p-only. This route requires a legal target deployment before use on Ascend950; BF16 is not narrowed.",))
