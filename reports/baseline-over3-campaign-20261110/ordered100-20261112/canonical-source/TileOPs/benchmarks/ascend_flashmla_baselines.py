"""Host layout adapters for FlashMLA's unchanged BF16 sparse-prefill kernel."""
import importlib
import math
from pathlib import Path

from benchmarks.ascend_baselines import (
    Candidate, LIBRARIES, WORKSPACE, UnsupportedCase, source_hashes,
)


def _native():
    root = LIBRARIES / "FlashMLA"
    installed = WORKSPACE / ".venv/lib/python3.12/site-packages/flash_mla"
    python_files = list((root / "flash_mla").glob("*.py"))
    for original in python_files:
        if original.read_bytes() != (installed / original.name).read_bytes():
            raise RuntimeError("FlashMLA Python source/deployment mismatch.")
    backend = importlib.import_module("flash_mla.npu")
    binary = Path(backend.__file__).resolve()
    if binary.parent != installed.resolve():
        raise RuntimeError("FlashMLA resolved outside its audited deployment.")
    built = root / "build/lib.linux-x86_64-cpython-312/flash_mla" / binary.name
    import hashlib
    if hashlib.sha256(binary.read_bytes()).digest() != hashlib.sha256(built.read_bytes()).digest():
        raise RuntimeError("FlashMLA deployed binary differs from its source-tree build.")
    native_files = [p for base in (root / "csrc/ascend_kernels", root / "csrc/api",
                                  root / "csrc/3rdparty/kerutils/include/kerutils")
                    for p in base.rglob("*") if p.suffix in (".h", ".cpp", ".asc")]
    hashes = source_hashes(Path(__file__), binary, built, root / "csrc/params.h", *python_files, *native_files)
    return backend.sparse_prefill_fwd, hashes


def flashmla_gqa(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    q, k, v, *extra = case.inputs
    p = case.params
    batch, sq, heads, dim = q.shape
    sk, kvheads, vdim = k.shape[1], k.shape[2], v.shape[-1]
    if (any(t is not None for t in extra) or any(t.dtype != torch.bfloat16 for t in (q, k, v))
            or heads % kvheads or dim + vdim > 512 or p.get("softcap") is not None
            or p.get("pos_encoding_mode", "none") != "none"
            or p.get("window_size_left", -1) != -1 or p.get("window_size_right", -1) != -1
            or p.get("out_dtype") not in (None, "bfloat16")):
        raise UnsupportedCase("Original sparse BF16 attention needs unencoded BF16 Q/K/V, no softcap, and K+V width<=512.")
    native, hashes = _native()
    group = heads // kvheads
    chunks = (group + 63) // 64
    padded_heads, padded_seq = chunks * 64, ((sk + 63) // 64) * 64
    scale = p.get("sm_scale")
    scale = dim ** -0.5 if scale is None else scale
    # Shape-derived index metadata only; it never depends on tensor values.
    tokens = torch.arange(padded_seq, device=q.device, dtype=torch.int32).view(1, 1, 1, 1, -1)
    offsets = torch.arange(batch * kvheads, device=q.device, dtype=torch.int32).view(batch, kvheads, 1, 1, 1) * sk
    visible = tokens < sk
    if p.get("is_causal"):
        positions = (torch.arange(sq, device=q.device, dtype=torch.int32) + sk - sq).view(1, 1, sq, 1, 1)
        visible = visible & (tokens <= positions)
    indices = torch.where(visible, offsets + tokens, -1).expand(batch, kvheads, sq, chunks, padded_seq)
    indices = indices.contiguous().reshape(-1, 1, padded_seq)

    def run(q, k, v, *extra):
        # Q'=[Q,0,0], KV'=[K,V,0] gives original scores. Selecting the V
        # segment of the native output gives the original weighted V result.
        grouped = q.permute(0, 2, 1, 3).reshape(batch, kvheads, group, sq, dim).permute(0, 1, 3, 2, 4)
        native_q = F.pad(grouped, (0, 512 - dim, 0, padded_heads - group)).reshape(-1, 64, 512)
        combined = torch.cat((k, v), dim=-1).permute(0, 2, 1, 3).contiguous()
        native_kv = F.pad(combined, (0, 512 - dim - vdim)).reshape(-1, 1, 512)
        output, _, _ = native(native_q, native_kv, indices, scale, 512, None, None)
        selected = output.reshape(batch, kvheads, sq, padded_heads, 512)[..., :group, dim:dim + vdim]
        return selected.permute(0, 2, 1, 3, 4).contiguous().reshape(batch, sq, heads, vdim)

    return Candidate(Implementation(run), "FlashMLA::sparse_prefill_fwd",
                     "Concat separate K/V, zero-pad Q/KV to512, pad heads to64, map complete/causal KV indices, crop V; original scale and native schedule.",
                     hashes, limitations=("Includes per-call packing/padding costs; static shape-derived index metadata is setup.",))


def flashmla_amax(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation

    x, = case.inputs
    if x.dtype != torch.bfloat16:
        raise UnsupportedCase("FlashMLA's original BF16 inputs cannot losslessly represent arbitrary FP16/FP32 values.")
    dims = case.params["dim"]
    dims = (dims,) if isinstance(dims, int) else tuple(dims)
    axes = tuple(d % x.ndim for d in dims)
    remaining = tuple(d for d in range(x.ndim) if d not in axes)
    width = math.prod(x.shape[d] for d in axes)
    rows = x.numel() // width
    output_shape = (tuple(1 if d in axes else n for d, n in enumerate(x.shape))
                    if case.params["keepdim"] else tuple(x.shape[d] for d in remaining))
    native, hashes = _native()
    padded_width = ((width + 63) // 64) * 64
    columns = torch.arange(padded_width, dtype=torch.int32, device=x.device)
    offsets = torch.arange(rows, dtype=torch.int32, device=x.device)[:, None] * width
    indices = torch.where(columns[None, :] < width, offsets + columns, -1).view(rows, 1, padded_width)

    def run(x):
        packed = x.permute(*remaining, *axes).contiguous().reshape(-1, 1, 1)
        kv = F.pad(packed, (0, 511))
        q = torch.zeros((rows, 64, 512), dtype=x.dtype, device=x.device)
        q[:, 0, 0] = 1
        _, maximum, _ = native(q, kv, indices, 1.0, 512, None, None)
        return maximum[:, 0].to(x.dtype).reshape(output_shape)

    return Candidate(Implementation(run), "FlashMLA::sparse_prefill_fwd.max_logits",
                     "Permute reduction axes, basis query QK=x, zero-pad native512-dimensional KV and select exposed unscaled maximum.",
                     hashes, limitations=("KV padding can be large and belongs to every invocation; no kernel-only timing claim.",))
