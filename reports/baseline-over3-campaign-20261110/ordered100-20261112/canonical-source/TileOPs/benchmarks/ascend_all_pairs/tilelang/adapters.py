"""Manifest adapters to unchanged vendor TileLang kernels."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


def bmm(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import source_hashes
    from benchmarks.ascend_gemm_cases import tilelang_gemm_cases
    a, b = case.inputs
    proxy = SimpleNamespace(inputs=(a[0], b[0]), params={"trans_a": False, "trans_b": False})
    native = tilelang_gemm_cases(proxy)
    def run(a, b):
        return torch.stack([native.implementation.run(a[i], b[i]) for i in range(a.shape[0])])
    return replace(native, implementation=Implementation(run),
                   adaptation=native.adaptation + ' Batch slicing and output stacking on every invocation; unchanged original GEMM per batch.',
                   source_sha256=native.source_sha256 | source_hashes(Path(__file__)))


def _gemm_native(m, n, k, dtype, out_dtype=None):
    """Only choose existing dtype templates and pad complete original tiles."""
    import torch
    import torch.nn.functional as F
    from benchmarks.ascend_baselines import LIBRARIES, local_function, source_hashes
    from benchmarks.ascend_tilelang_baselines import _runtime, _runtime_hashes
    tl = _runtime()
    host = 'examples/ascend/example_gemm.py'
    builder = local_function('tilelang', 'examples/ascend', 'example_gemm', 'gemm', host)
    ma = 1024 if m > 1024 else 256
    ka = 128 if dtype == torch.float32 else 256
    mp, np, kp = m + -m % ma, n + -n % 256, k + -k % ka
    dt = str(dtype).split('.')[-1]
    odt = str(out_dtype).split('.')[-1] if out_dtype is not None else ('bfloat16' if dtype == torch.bfloat16 else 'float32')
    if odt == 'float16':
        odt = 'float32'  # Original mixed dual_copy cannot convert FP32 L0C to FP16 UB.
    kernel = tl.compile(builder(mp, kp, np, dtype=dt, out_dtype=odt), target='ascend', out_idx=-1)
    def run(a, b):
        # b is already [N,K]. No data-dependent input is retained by the factory.
        ap = F.pad(a.contiguous(), (0, kp - k, 0, mp - m))
        bp = F.pad(b.contiguous(), (0, kp - k, 0, np - n))
        return kernel(ap, bp)[:m, :n].to(out_dtype or dtype).contiguous()
    hashes = source_hashes(Path(__file__), LIBRARIES / 'tilelang' / host) | _runtime_hashes(tl)
    return run, hashes


def convolution(case):
    """Spatial windows are a strided layout of the input; compute is native GEMM."""
    import torch
    import torch.nn.functional as F
    from math import prod
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase
    x, weight, bias = case.inputs
    p, nd = case.params, x.ndim - 2
    def dims(v):
        return (v,) * nd if isinstance(v, int) else tuple(v)
    stride, dilation, ks = dims(p['stride']), dims(p['dilation']), tuple(weight.shape[2:])
    padding = p['padding']
    if isinstance(padding, str):
        if padding == 'valid':
            pairs = [(0, 0)] * nd
        elif padding == 'same':
            pairs = [(d * (k - 1) // 2, d * (k - 1) - d * (k - 1) // 2) for d, k in zip(dilation, ks)]
        else:
            raise UnsupportedCase('Unknown convolution padding contract.')
    else:
        pairs = [(v, v) for v in dims(padding)]
    pad = tuple(v for pair in reversed(pairs) for v in pair)
    out_spatial = tuple((size + lo + hi - d * (k - 1) - 1) // s + 1
                        for size, (lo, hi), d, k, s in zip(x.shape[2:], pairs, dilation, ks, stride))
    groups = p.get('groups', 1)
    ci, co = x.shape[1] // groups, weight.shape[0] // groups
    m, k = x.shape[0] * prod(out_spatial), ci * prod(ks)
    # Bound temporary patch memory using independent output-row calls only.
    rows = min(m, 4096)
    augmented_k = k + (bias is not None)
    native, hashes = _gemm_native(rows, co, augmented_k, x.dtype)
    tail = None
    if m % rows:
        tail, more = _gemm_native(m % rows, co, augmented_k, x.dtype)
        hashes |= more
    def run(x, weight, bias):
        padded = F.pad(x, pad)
        st = padded.stride()
        view = padded.as_strided((x.shape[0], x.shape[1], *out_spatial, *ks),
                                 (st[0], st[1], *(st[i+2]*stride[i] for i in range(nd)),
                                  *(st[i+2]*dilation[i] for i in range(nd))))
        outputs = []
        for g in range(groups):
            patches = view[:, g*ci:(g+1)*ci].permute(0, *range(2, 2+nd), 1, *range(2+nd, 2+2*nd)).contiguous().reshape(m, k)
            coeff = weight[g*co:(g+1)*co].reshape(co, k)
            if bias is not None:
                patches = torch.cat((patches, torch.ones((m, 1), device=x.device, dtype=x.dtype)), dim=1)
                coeff = torch.cat((coeff, bias[g*co:(g+1)*co, None]), dim=1)
            chunks = []
            for start in range(0, m, rows):
                block = patches[start:start+rows]
                chunks.append((native if block.shape[0] == rows else tail)(block, coeff))
            outputs.append(torch.cat(chunks, dim=0))
        result = torch.cat(outputs, dim=1).reshape(x.shape[0], *out_spatial, weight.shape[0])
        return result.movedim(-1, 1).contiguous()
    return Candidate(Implementation(run), 'tilelang::example_gemm.gemm over strided convolution windows',
                     'Original GEMM math and schedule; per-call input padding, as_strided receptive-field views, group slices, complete-tile padding and output layout. Bias is one appended constant-one coordinate. All packing/copies are timed.', hashes)


def fused_add_rms(case):
    raise RuntimeError("Original-only boundary: separate Add plus RMSNorm composition is retired; original API support remains unresolved.")
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, source_hashes
    from benchmarks.ascend_tilelang_more import tilelang_add
    from benchmarks.ascend_rms_cases import tilelang_rms_cases
    x, residual, weight = case.inputs
    # The manifest rounds the sum to input dtype before normalization.
    add = tilelang_add(SimpleNamespace(inputs=(x.float(), residual.float()), params={'alpha': 1}))
    norm = tilelang_rms_cases(SimpleNamespace(inputs=(x.float(), None if weight is None else weight.float()), params=case.params | {'normalized_shape':case.params.get('normalized_shape',[x.shape[-1]])}))
    def run(x, residual, weight):
        summed = add.implementation.run(x.float(), residual.float()).to(x.dtype)
        normalized = norm.implementation.run(summed.float(), None if weight is None else weight.float())
        return normalized.to(x.dtype), summed.to(x.dtype)
    return Candidate(Implementation(run), add.api + ' + ' + norm.api,
                     'Original FP32 Add and RMSNorm API composition; exact input widening; round the residual to manifest dtype before norm and cast both manifest outputs. No input data is cached.',
                     add.source_sha256 | norm.source_sha256 | source_hashes(Path(__file__)))


def grouped_gemm(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    a, b, sizes, offsets = case.inputs
    ta, tb = case.params['transpose_a'], case.params['transpose_b']
    # Runtime group sizes may change; compile keyed native shapes lazily, never cache operands.
    kernels, hashes = {}, {}
    def run(a, b, sizes, offsets):
        size_list = sizes.cpu().tolist()
        offset_list = offsets.cpu().tolist()
        out = []
        for g, (size, start) in enumerate(zip(size_list, offset_list)):
            if not ta:
                lhs = a[start:start+size]
                rhs = b[g] if tb else b[g].T
            else:
                lhs = a[start:start+size].T
                rhs = b[:, start:start+size] if tb else b[start:start+size].T
            key = (lhs.shape[0], rhs.shape[0], lhs.shape[1])
            if not size:
                out.append(torch.zeros((key[0], key[1]), device=a.device, dtype=a.dtype))
                continue
            if key not in kernels:
                kernels[key], more = _gemm_native(*key, a.dtype)
                hashes.update(more)
            out.append(kernels[key](lhs, rhs))
        return torch.stack(out) if ta else torch.cat(out, dim=0)
    # Populate native code fingerprints before the checker records the Candidate.
    from benchmarks.ascend_baselines import LIBRARIES, source_hashes
    from benchmarks.ascend_tilelang_baselines import _runtime, _runtime_hashes
    hashes.update(source_hashes(Path(__file__), LIBRARIES/'tilelang/examples/ascend/example_gemm.py') | _runtime_hashes(_runtime()))
    return Candidate(Implementation(run), 'tilelang::example_gemm.gemm per input group',
                     'Read current group sizes/offsets on every call; slice/transpose each group and invoke original GEMM. Dynamic shape compilation is cached, input-derived tensors are not; metadata host transfer belongs in timing.',
                     hashes, timing_mode='host_roundtrip')


def sum_mean(case):
    raise RuntimeError("Original-only boundary: constant-projection GEMM reduction is retired; original API support remains unresolved.")
    import torch
    from math import prod
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    x,=case.inputs
    dims=case.params.get('dim')
    dims=tuple(range(x.ndim)) if dims is None else ((dims,) if isinstance(dims,int) else tuple(dims))
    dims=tuple(d%x.ndim for d in dims)
    outer=tuple(d for d in range(x.ndim) if d not in dims)
    order=outer+dims
    m=prod(x.shape[d] for d in outer)
    k=prod(x.shape[d] for d in dims)
    coefficient=1/k if case._call.signature.name=='MeanFwdOp' else 1.
    shape=tuple(1 if d in dims else x.shape[d] for d in range(x.ndim)) if case.params.get('keepdim',False) else tuple(x.shape[d] for d in outer)
    dtype=getattr(torch,case.params['dtype']) if case.params.get('dtype') else x.dtype
    native,hashes=_gemm_native(m,1,k,torch.float32)
    def run(x):
        weights=torch.full((1,k),coefficient,dtype=torch.float32,device=x.device)
        return native(x.permute(order).contiguous().reshape(m,k).float(),weights).reshape(shape).to(dtype)
    return Candidate(Implementation(run),'tilelang::example_gemm.gemm constant projection',
                     'Pack full reduction axes; exact FP32 widening; original FP32 GEMM against constant 1 or 1/N coefficients; restore requested output dtype/keepdim. No external tensor reduction.',hashes)


def moe_grouped(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,LIBRARIES,UnsupportedCase,source_hashes
    from benchmarks.ascend_tilelang_baselines import _runtime,_runtime_hashes
    if case.params.get('activation') is not None:
        raise UnsupportedCase('Original Ascend GEMM has no SiLU-and-multiply epilogue; generic fusedMoE route requires CUDA VF lowering.')
    layout=case._call.params['layout']
    out_dtype=getattr(torch,case.params['out_dtype']) if case.params.get('out_dtype') else case.inputs[0].dtype
    kernels={}
    hashes=source_hashes(Path(__file__),LIBRARIES/'tilelang/examples/ascend/example_gemm.py')|_runtime_hashes(_runtime())
    def run(a,b,metadata):
        meta=metadata.cpu().tolist()
        out=torch.zeros((*a.shape[:-1],b.shape[1]),device=a.device,dtype=out_dtype)
        start=0
        for g in range(b.shape[0]):
            if layout.kind=='masked':
                selection=slice(0,meta[g]);aa=a[g,selection]
            elif layout.metadata_kind=='per_row':
                selection=torch.tensor([i for i,v in enumerate(meta) if v==g],device=a.device,dtype=torch.int64)
                aa=a.index_select(0,selection)
            else:
                start=(start+layout.alignment-1)//layout.alignment*layout.alignment
                selection=slice(start,meta[g]);aa=a[selection];start=meta[g]
            if aa.shape[0]==0:
                continue
            key=aa.shape[0]
            if key not in kernels:
                kernels[key],more=_gemm_native(aa.shape[0],b.shape[1],a.shape[-1],a.dtype,out_dtype)
                hashes.update(more)
            result=kernels[key](aa,b[g])
            if layout.kind=='masked':out[g,selection].copy_(result)
            elif layout.metadata_kind=='per_row':out.index_copy_(0,selection,result)
            else:out[selection].copy_(result)
        return out
    return Candidate(Implementation(run),'tilelang::example_gemm.gemm per expert',
                     'Runtime masked/physical-psum/per-row metadata selects original expert GEMMs; original output dtype template, layout packing and independent output zero padding; all metadata transfers/copies in run.',hashes,timing_mode='host_roundtrip')
