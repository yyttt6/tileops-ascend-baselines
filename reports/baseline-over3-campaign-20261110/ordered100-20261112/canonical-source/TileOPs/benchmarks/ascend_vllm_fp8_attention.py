"""FP8 GQA through vLLM's original packed, quantized shared-KV attention."""
import ctypes as C
import os
from pathlib import Path

from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, _owned_native, source_hashes
from benchmarks.ascend_math_more import _MathCall


class _PackedAttentionCall(_MathCall):
    """Bind the deployed original C API without requiring a rebuilt Torch extension."""

    def __init__(self):
        import torch

        self.torch, self.pending = torch, []
        self.vendor = (LIBRARIES / 'vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer').resolve()
        configured = {Path(p).resolve() for p in os.environ.get('ASCEND_CUSTOM_OPP_PATH', '').split(':') if p}
        if self.vendor not in configured:
            raise RuntimeError('Select the original vLLM vendor with baseline-run.')
        self.binary = self.vendor / 'op_api/lib/libcust_opapi.so'
        self.api = 'aclnnKvQuantSparseFlashAttentionVllm'
        self.lib = _owned_native(self.binary, [self.api, self.api + 'GetWorkspaceSize'])
        self.size = getattr(self.lib, self.api + 'GetWorkspaceSize')
        self.size.argtypes = ([C.c_void_p] * 9 + [C.c_double] + [C.c_int64] * 3
                             + [C.c_char_p] * 2 + [C.c_int64] * 7 + [C.c_bool]
                             + [C.c_void_p] * 3 + [C.POINTER(C.c_uint64), C.POINTER(C.c_void_p)])
        self.size.restype = C.c_int
        self.launch = getattr(self.lib, self.api)
        self.launch.argtypes = [C.c_void_p, C.c_uint64, C.c_void_p, C.c_void_p]
        self.launch.restype = C.c_int
        self.base = C.CDLL('libnnopbase.so')
        self.base.aclCreateTensor.argtypes = [C.POINTER(C.c_int64), C.c_uint64, C.c_int,
            C.POINTER(C.c_int64), C.c_int64, C.c_int, C.POINTER(C.c_int64), C.c_uint64, C.c_void_p]
        self.base.aclCreateTensor.restype = C.c_void_p
        self.base.aclDestroyTensor.argtypes = [C.c_void_p]
        self.base.aclDestroyTensor.restype = C.c_int
        self.codes = {torch.float16: 1, torch.float32: 0, torch.int32: 3, torch.float8_e4m3fn: 36}


def vllm_fp8_gqa_cases(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    if case.inputs[0].dtype != torch.float8_e4m3fn:
        from dataclasses import replace
        from benchmarks.ascend_vllm_causal_rows import vllm_rope_causal_cases
        native = vllm_rope_causal_cases(case)
        return replace(native, source_sha256=native.source_sha256 | source_hashes(Path(__file__)))
    if len(case.inputs) != 8:
        raise UnsupportedCase('FP8 attention requires Q/K/V, scales and optional rotation tables.')
    q, k, v, qs, ks, vs, cos, sin = case.inputs
    p = case.params
    batch, queries, heads, dim = q.shape
    keys, kvheads, vdim = k.shape[1:]
    if (batch != 1 or dim != 128 or vdim != 128 or k.shape != v.shape
            or k.dtype != q.dtype or v.dtype != q.dtype or heads % kvheads
            or heads // kvheads not in (1, 2, 4, 8, 16, 32, 48, 64, 128)
            or keys % 128 or queries > keys or not p.get('is_causal')
            or p.get('softcap') is not None or p.get('pos_encoding_mode', 'none') != 'none'
            or p.get('window_size_left', -1) != -1 or p.get('window_size_right', -1) != -1
            or p.get('out_dtype') != 'float16' or cos is not None or sin is not None
            or any(s is None or s.dtype != torch.float32 or tuple(s.shape) != (batch, kvheads)
                   for s in (qs, ks, vs))):
        raise UnsupportedCase('Focused packed FP8 attention: B1/D128, integral128-token pages, per-KV-head supplied scales and right-aligned causal visibility.')
    from benchmarks.ascend_all_pairs.vllm.runtime import CACHE, TRITON_ROOT, module
    from benchmarks.ascend_all_pairs.jit_evidence import triton_run
    query_scaler, query_sources = module(TRITON_ROOT + "muls_add.py", hardware=True)
    binding = _PackedAttentionCall()
    group = heads // kvheads
    page_size = 128
    pages = keys // page_size
    table = torch.arange(pages, device=q.device, dtype=torch.int32).reshape(1, pages)
    indices = torch.arange(keys, device=q.device, dtype=torch.int32)
    indices = indices.reshape(1, 1, keys).expand(queries, 1, keys).contiguous()
    q_lengths = torch.tensor([queries], device=q.device, dtype=torch.int32)
    kv_lengths = torch.tensor([keys], device=q.device, dtype=torch.int32)
    scale = dim ** -0.5 if p.get('sm_scale') is None else p['sm_scale']

    def run(q, k, v, qs, ks, vs, cos, sin):
        # Each KV group is independent. Materialize the manifest FP16 rounding
        # after Q scaling through the original scalar-multiply kernel.
        # K and V occupy distinct128-feature tiles with their own native scales.
        outputs = []
        for head in range(kvheads):
            query = q[0, :, head * group:(head + 1) * group].to(torch.float16)
            flat_query = query.contiguous().reshape(-1, 1)
            query = query_scaler.muls_add_triton(
                flat_query, torch.zeros_like(flat_query), float(qs[0, head].item())
            ).reshape(query.shape)
            query = F.pad(query, (0, 512 - dim)).contiguous()
            packed_bytes = torch.cat((k[0, :, head].contiguous().view(torch.uint8),
                                      v[0, :, head].contiguous().view(torch.uint8),
                                      torch.zeros((keys, 256), device=q.device, dtype=torch.uint8)), dim=-1)
            scales = torch.cat((ks[:, head:head + 1], vs[:, head:head + 1],
                                torch.ones((1, 2), device=q.device, dtype=torch.float32)), dim=-1)
            scales = scales.expand(keys, 4).contiguous()
            cache = torch.cat((packed_bytes, scales.view(torch.uint8)), dim=-1)
            cache = cache.view(torch.float8_e4m3fn).reshape(pages, page_size, 1, 528)
            out = torch.empty_like(query)
            maximum = torch.empty((0,), device=q.device, dtype=torch.float32)
            total = torch.empty_like(maximum)
            record = binding.record()
            desc = lambda t: binding.tensor(t, record)
            args = [desc(query), desc(cache), desc(cache), desc(indices), None, None,
                    desc(table), desc(q_lengths), desc(kv_lengths),
                    float(scale), 2, 2, 1, b'TND', b'PA_BSND',
                    3, 2**63 - 1, 2**63 - 1, 2, 1, 128, 0, False,
                    desc(out), desc(maximum), desc(total)]
            binding.call(args, record, q.device)
            outputs.append(out[..., dim:dim + vdim])
        return torch.cat(outputs, dim=1).unsqueeze(0).contiguous()

    root = LIBRARIES / 'vllm-ascend/csrc/attention/kv_quant_sparse_flash_attention_vllm'
    header = binding.vendor / 'op_api/include/aclnnop/aclnn_kv_quant_sparse_flash_attention_vllm.h'
    owned_hashes = binding.hashes(root) | source_hashes(
        Path(__file__), header, *query_sources, Path(triton_run.__code__.co_filename))
    return Candidate(Implementation(triton_run(run, owned_hashes, CACHE)),
        'vllm-ascend::muls_add_triton + aclnnKvQuantSparseFlashAttentionVllm',
        'Original C API binding; independent KV groups, exact FP8-to-FP16 query cast, original native Q scaling with the required FP16 intermediate rounding, shared cache [K,V,0] with supplied K/V scales in separate native128-feature tiles; native causal mode3; crop V coordinates. All packing/scalar reads count in timing.',
        owned_hashes,
        limitations=('Native FP16 dequantization rounding must pass unchanged manifest tolerance.',
                     'Own vendor kernel launch attribution is mandatory; no correctness credit for host-symbol availability.'),
        timing_mode="host_completed", cleanup=binding.cleanup)
