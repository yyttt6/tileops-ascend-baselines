"""Original TileKernels API selections with shape/precision adapters only."""
from pathlib import Path
from benchmarks.ascend_baselines import Candidate, LIBRARIES, UnsupportedCase, local_function, source_hashes


def native(module, name):
    return local_function('TileKernels', '.', 'tile_kernels.' + module, name,
                          'tile_kernels/' + module.replace('.', '/') + '.py')


def result(run, api, adaptation, *sources, limitations=()):
    from benchmarks.api import Implementation
    root = LIBRARIES / 'TileKernels/tile_kernels'
    return Candidate(Implementation(run), 'tile_kernels.' + api, adaptation,
                     source_hashes(Path(__file__), Path(run.__code__.co_filename), root / 'config.py', root / 'quant/common.py',
                                   *(root / s for s in sources)), limitations=limitations)


def silu(case):
    import torch
    fn = native('quant.swiglu_forward_kernel', 'swiglu_forward')
    def run(x):
        rows = x.float().reshape(-1, 1024)
        paired = torch.cat((rows, torch.ones_like(rows)), dim=-1)
        return fn(paired, 'fp32').to(x.dtype).reshape_as(x)
    return result(run, 'quant.swiglu_forward_kernel.swiglu_forward',
                  'SwiGLU with unit companion values selects SiLU; exact widening and output restoration; every packing is inside run.',
                  'quant/swiglu_forward_kernel.py', 'quant/swiglu_forward_asc.py')


def mul(case):
    import torch
    fn = native('mhc.norm_fn_kernel', 'mhc_fn_normw_merge_fwd')
    shape = torch.broadcast_shapes(*(x.shape for x in case.inputs))
    count = __import__('math').prod(shape)
    def pack(x):
        return torch.nn.functional.pad(x.expand(shape).contiguous().float().reshape(-1), (0, -count % 256))
    def run(x, y):
        left, right = pack(x), pack(y)
        out = torch.empty_like(left).reshape(1, -1)
        fn(left.reshape(1, -1), right, out)
        return out.reshape(-1)[:count].to(x.dtype).reshape(shape)
    return result(run, 'mhc.norm_fn_kernel.mhc_fn_normw_merge_fwd',
                  'Broadcast both operands, exact FP32 widening and flatten to one original weight row; original pointwise product; restore shape/dtype.',
                  'mhc/norm_fn_kernel.py', 'mhc/norm_fn_asc.py')


def fused_add_rms(case):
    import torch
    fn = native('quant.norm_forward_kernel', 'norm_forward')
    eps = case.params.get('eps', 1e-6)
    dtype = case.inputs[0].dtype
    def run(x, residual, weight):
        shape = x.shape
        h = shape[-1]
        if dtype == torch.float16:
            # The contract rounds the residual sum to FP16 before RMSNorm.
            # Obtain that sum from the original API, then normalize its exact widening.
            summed = fn(x.float().reshape(-1, h), None, eps,
                        residual=residual.float().reshape(-1, h))[2].to(dtype)
            out = fn(summed.float(), weight.float().reshape(h), eps)[0].to(dtype)
        else:
            out, _, summed = fn(x.contiguous().reshape(-1, h), weight.contiguous().reshape(h), eps,
                                residual=residual.contiguous().reshape(-1, h))
        return out.reshape(shape), summed.reshape(shape)
    return result(run, 'quant.norm_forward_kernel.norm_forward',
                  'Original residual addition and RMSNorm; FP16 uses two original calls with contract residual rounding between them; return both full outputs.',
                  'quant/norm_forward_kernel.py', 'quant/norm_forward_asc.py', 'quant/rstd.py')


def argmax(case):
    import math
    fn = native('moe.topk_gate_kernel', 'topk_gate')
    x = case.inputs[0]
    dim = case.params.get('dim')
    axes = tuple(range(x.ndim)) if dim is None else (dim % x.ndim,)
    kept = tuple(i for i in range(x.ndim) if i not in axes)
    width = math.prod(x.shape[i] for i in axes)
    shape = tuple(1 if i in axes else n for i, n in enumerate(x.shape)) if case.params.get('keepdim') else tuple(x.shape[i] for i in kept)
    def run(x):
        return fn(x.permute(*(kept + axes)).contiguous().reshape(-1, width).float(), 1).reshape(shape)
    return result(run, 'moe.topk_gate_kernel.topk_gate(k=1)',
                  'Permute reduction axis into rows; exact FP32 widening; original stable top-one indices; restore output shape.',
                  'moe/topk_gate_kernel.py', 'moe/topk_gate_asc.py',
                  limitations=('Native API rejects nonfinite scores; current manifest draws finite values.',))


def mhc_post(case):
    import torch
    fn = native('mhc.post_kernel', 'mhc_post_fwd')
    x, post, residual = case.inputs
    n, h = x.shape
    expand = post.shape[-1]
    identity = torch.eye(expand, device=x.device, dtype=torch.float32).expand(1, n, expand, expand).contiguous()
    def run(x, post, residual):
        out = fn(x.contiguous().reshape(1, n, h), residual.contiguous().reshape(1, n, expand, h),
                 post.contiguous().reshape(1, n, expand, 1), identity)
        return out.reshape(n, expand * h)
    return result(run, 'mhc.post_kernel.mhc_post_fwd',
                  'Manifest post-mix is original mHC post with identity residual mix; reshape singleton sequence and full output.',
                  'mhc/post_kernel.py', 'mhc/post_asc.py')


def fp8(case):
    import torch
    import torch.nn.functional as F
    fn = native('quant.per_block_cast_kernel', 'per_block_cast')
    from tile_kernels.config import get_amax_clamp_for_quant
    if get_amax_clamp_for_quant('e4m3', False) != 1e-4:
        raise RuntimeError('Native quantization floor differs from manifest.')
    x = case.inputs[0]
    width = x.shape[-1]
    if width > 1024:
        raise UnsupportedCase('One complete manifest group must fit a native 32x32 block.')
    groups = x.numel() // width
    padded_groups = (groups + 15) // 16 * 16
    def run(x):
        rows = F.pad(x.float().reshape(groups, width), (0, 1024-width, 0, padded_groups-groups))
        packed = rows.reshape(-1, 4, 32, 32).permute(0, 2, 1, 3).contiguous().reshape(-1, 128)
        q, scale = fn(packed, 'e4m3', (32, 32))
        # View raw FP8 bytes for portable permutation/copy on Ascend.
        unpacked = q.view(torch.uint8).reshape(-1, 32, 4, 32).permute(0, 2, 1, 3).contiguous().reshape(padded_groups, 1024)
        output = unpacked[:groups, :width].contiguous().view(torch.float8_e4m3fn).reshape_as(x)
        return scale.reshape(-1)[:groups].reshape(x.shape[:-1]), output
    return result(run, 'quant.per_block_cast_kernel.per_block_cast',
                  'Embed each complete manifest group in a distinct native 32x32 block using zeros; arrange 4 independent blocks across columns; crop zero padding and reorder scale/output. All packing is timed.',
                  'quant/per_block_cast_kernel.py', 'quant/per_block_cast_asc.py')


def rope_variant(case):
    import torch
    from workloads import rope as reference_tables
    fn = native('transform.rope_kernel', 'apply_rotary')
    name = case._call.signature.name
    p = dict(case._call.params)
    table_fn = {'RoPELlama31FwdOp': reference_tables.llama31_frequency_tables,
                'YaRNFwdOp': reference_tables.yarn_frequency_tables,
                'LongRoPEFwdOp': reference_tables.longrope_frequency_tables,
                'RoPENeoxPositionIdsFwdOp': reference_tables.rope_frequency_tables}[name]
    def run(x, position_ids=None):
        if name == 'RoPENeoxPositionIdsFwdOp':
            dim = p.get('rotary_dim') or x.shape[-1]
            cos, sin = table_fn(dim, p['max_position'], base=p['base'], dtype=x.dtype, device=x.device)
            private = x.float().clone()
            fn(private[..., :dim], torch.cat((cos, sin), -1).float(), positions=position_ids)
        else:
            seq = x.shape[0] if x.ndim == 2 else x.shape[1]
            kw = {k: v for k, v in p.items() if k not in ('input_layout', 'rope_layout')}
            if 'rescale_factors' in case._call.specs:
                kw['rescale_factors'] = case.workload.tensors['rescale_factors']
            cos, sin = table_fn(x.shape[-1], seq, dtype=x.dtype, device=x.device, **kw)
            private = x.float().clone()
            packed = private.unsqueeze(1) if x.ndim == 2 else private
            positions = torch.arange(seq, device=x.device, dtype=torch.int32)
            if packed.ndim == 4:
                positions = positions.expand(packed.shape[0], seq).contiguous()
            fn(packed, torch.cat((cos, sin), -1).float(), positions=positions,
               interleaved=p.get('rope_layout', 'neox') == 'interleaved')
        return private.to(x.dtype)
    return result(run, 'transform.rope_kernel.apply_rotary',
                  'Original native rotation with manifest frequency tables generated each invocation; private exactly widened input; explicit positions and partial-rotary slice; restore dtype.',
                  'transform/rope_kernel.py', 'transform/rope_asc.py')
