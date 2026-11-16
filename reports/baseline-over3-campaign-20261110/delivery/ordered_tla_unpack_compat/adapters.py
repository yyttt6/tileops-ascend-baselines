"""Adapters around original tilelang-ascend kernels; no device work at import."""
from __future__ import annotations

from functools import partial
from math import prod

from . import native
from .plans import (COMPARISON, DIRECT, ELEMENTWISE, GENERIC_UNARY, SELECT,
                    TILE_API, UNARY, assess)


def _up(n, block):
    return (n + block - 1) // block * block


def _check(op, case):
    from benchmarks.ascend_baselines import UnsupportedCase
    plan = assess(op, case.params)
    if not plan.supported:
        raise UnsupportedCase(plan.reason)
    return plan


def _output(case, key='output'):
    import torch
    shape, dtype = case.params[key]
    return tuple(shape), getattr(torch, dtype)


def _flat_pack(value, shape, rows, cols, dtype, fill=0):
    import torch.nn.functional as F
    flat = value.expand(shape).contiguous().reshape(-1).to(dtype)
    return F.pad(flat, (0, rows * cols - flat.numel()), value=fill).reshape(rows, cols)


def elementwise(op, case):
    plan = _check(op, case)
    import torch
    shape, out_dtype = _output(case)
    size = prod(shape)
    # Use an original 64x64 tiled specialization, with padding for incomplete tiles.
    comparison=op in COMPARISON or (op=='LogicalNotFwdOp' and plan.dtype!='bool')
    split64=op.startswith('Bitwise') and plan.dtype=='int64'
    native_size=size*2 if split64 else size
    rows, cols = _up((native_size + 63) // 64, 64), 64
    dt = ('int32' if plan.dtype in {'int32','int64'} else 'int16') if (op.startswith('Bitwise') or (op.startswith('Logical') and plan.dtype=='bool')) else (
        'int32' if plan.dtype == 'int32' else 'float32')
    dtype = getattr(torch, dt)
    sources = [TILE_API]
    if op in DIRECT and not comparison:
        name, arity = DIRECT[op]
        if op=='BitwiseNotFwdOp' and plan.dtype=='bool': name='bitwise_xor'
        module = native.load(ELEMENTWISE, name, 'pass_configs')
        extra = []
        if op in {'ClampScalarFwdOp', 'HardtanhFwdOp'}:
            p = case.params
            lo = p.get('min', p.get('min_val', -1.))
            hi = p.get('max', p.get('max_val', 1.))
            extra = [float('inf') if hi is None else hi, -float('inf') if lo is None else lo]
        if op == 'NegFwdOp':
            extra = [-1]
        kernel = native.compile_builder(module, name, rows, cols, 64, 64, *extra,
                                        dtype=dt, out_idx=[-1])
        sources.append(ELEMENTWISE)
    elif op in GENERIC_UNARY:
        name, arity = GENERIC_UNARY[op], 1
        module = native.load(UNARY, 'make_unary_kernel')
        common = native.load('testing/python/base/common.py', 'DEFAULT_PASS_CONFIGS')
        tile_op = getattr(native.runtime().language.tile, name)
        if op == 'RoundFwdOp':
            tile_op = partial(tile_op, count=32 * 64)
        if op == 'LeakyReluFwdOp':
            tile_op = partial(tile_op, scalar_value=case.params.get('negative_slope', .01))
        kernel = native.runtime().compile(
            module.make_unary_kernel(tile_op)(rows, cols, 64, 64, dt), out_idx=[-1],
            target='pto', pass_configs=common.DEFAULT_PASS_CONFIGS)
        sources += [UNARY, 'testing/python/base/common.py']
    elif op == 'TanhFwdOp':
        name, arity = 'tanh', 1
        src = 'examples/activation/tanh.py'
        module = native.load(src, name)
        kernel = native.compile_builder(module, name, rows, cols, 64, 64,
                                        dtype=dt, out_idx=[-1])
        sources.append(src)
    elif comparison:
        name, arity = 'compare', 1 if op in {'IsnanFwdOp','LogicalNotFwdOp'} else 2
        module = native.load(ELEMENTWISE, name, 'pass_configs')
        kernel = native.compile_builder(module, name, rows, cols, 64, 64, COMPARISON.get(op,'EQ'),
                                        dtype=dt, out_dtype='uint8', out_idx=[-1])
        sources.append(ELEMENTWISE)
    else:
        raise ValueError(op)

    def run(*inputs):
        if split64:
            packed=[_flat_pack(v.expand(shape).contiguous().view(torch.int32).reshape(-1),
                               (native_size,),rows,cols,dtype,fill=1) for v in inputs[:arity]]
        else:
            packed = [_flat_pack(v, shape, rows, cols, dtype, fill=1) for v in inputs[:arity]]
        if op=='IsnanFwdOp':
            packed.append(packed[0])
        elif op=='LogicalNotFwdOp' and comparison:
            packed.append(torch.zeros((rows,cols),dtype=dtype,device=inputs[0].device))
        elif op=='LogicalNotFwdOp' or (op=='BitwiseNotFwdOp' and plan.dtype=='bool'):
            packed.append(torch.ones((rows,cols),dtype=dtype,device=inputs[0].device))
        result = kernel(*packed)
        if comparison:
            # Only unpack the native comparison's byte mask. No comparison is
            # performed in torch; the bits themselves come from T.tile.compare.
            shifts = torch.arange(8, dtype=torch.uint8, device=result.device)
            result = ((result.reshape(-1, 1) >> shifts) & 1).reshape(-1)
        if split64:
            return result.reshape(-1)[:native_size].contiguous().view(torch.int64).reshape(shape)
        return result.reshape(-1)[:size].reshape(shape).to(out_dtype)

    return native.candidate(run, f'{sources[-1]}:{name}',
        'Broadcast/contiguous packing, original 64x64 tiles, pad/crop; FP16/BF16 widen '
        'exactly to FP32 and final output casts to manifest dtype. Comparison masks '
        'are unpacked from native result bits. INT64 bitwise views each value as two '
        'INT32 bit lanes and reassembles unchanged bits. All transforms occur inside run.', *sources)


def select(op, case):
    _check(op, case)
    import torch
    src = 'testing/python/language/test_tilelang_ascend_language_select.py'
    shape, out_dtype = _output(case)
    size = prod(shape)
    rows, cols = _up((size + 63) // 64, 64), 64
    module = native.load(src, 'select_kernel_mod2', 'PASS_CONFIGS')
    kernel = native.compile_builder(module, 'select_kernel_mod2', rows, cols, 64, 64,
                                    dtype='float32', out_idx=[-1])

    def run(*inputs):
        if op == 'WhereFwdOp':
            mask, yes, no = inputs
        else:
            value, mask, *other = inputs
            yes = other[0] if other else torch.full((), case.params['value'],
                                                     dtype=value.dtype, device=value.device)
            no = value
        bits = _flat_pack(mask, shape, rows, cols, torch.uint8).reshape(rows, cols//8, 8)
        # Lossless boolean-to-packed-bit layout conversion for native VSEL.
        packed_mask = torch.zeros((rows, cols//8), dtype=torch.uint8, device=mask.device)
        for bit in range(8):
            packed_mask.bitwise_or_(bits[..., bit] << bit)
        a = _flat_pack(yes, shape, rows, cols, torch.float32)
        b = _flat_pack(no, shape, rows, cols, torch.float32)
        return kernel(a, b, packed_mask).reshape(-1)[:size].reshape(shape).to(out_dtype)

    return native.candidate(run, f'{src}:select_kernel_mod2',
        'Lossless mask bit packing, broadcast, FP32 widening, original VSEL, crop/cast; '
        'per-call transformations timed. Native VSEL performs all data selection.', src)


def gated(op, case):
    _check(op, case)
    import torch
    import torch.nn.functional as F
    shape, out_dtype = _output(case)
    rows, hidden = prod(shape[:-1]), shape[-1]
    mr, nr = _up(rows,64), _up(hidden,64)
    name = 'swi_glu' if op == 'SiluAndMulFwdOp' else 'gelu_mul'
    src = f'examples/activation/{name}.py'
    module = native.load(src, name)
    args = (mr, 2*nr, 64, 64, -1) if name == 'swi_glu' else (mr, 2*nr, 64, 64)
    kernel = native.compile_builder(module, name, *args, dtype='float32', out_idx=[-1])

    def run(x):
        x = x.reshape(rows,2*hidden).float()
        a = F.pad(x[:,:hidden], (0,nr-hidden,0,mr-rows))
        b = F.pad(x[:,hidden:], (0,nr-hidden,0,mr-rows))
        return kernel(torch.cat((a,b),-1))[:rows,:hidden].reshape(shape).to(out_dtype)

    return native.candidate(run, f'{src}:{name}',
        'Preserve split halves while padding/cropping; exact FP32 input widening; '
        'original gated activation; manifest output dtype restored per run.',src)


def _axis_layout(shape, dim):
    axes = list(range(len(shape))) if dim is None else ([dim] if isinstance(dim,int) else list(dim))
    axes = sorted({a % len(shape) for a in axes})
    keep = [i for i in range(len(shape)) if i not in axes]
    perm = keep + axes
    return perm, prod(shape[i] for i in keep), prod(shape[i] for i in axes)


def reduction(op, case):
    _check(op, case)
    import torch
    import torch.nn.functional as F
    p = case.params
    in_shape = tuple(p['x'][0])
    out_shape, out_dtype = _output(case)
    perm, rows, cols = _axis_layout(in_shape,p.get('dim',-1))
    if op == 'SoftmaxFwdOp':
        src, name = 'examples/softmax/example_online_softmax.py', 'online_softmax'
        # Padding only rows preserves the original reduction divisor and axis.
        mr = _up(rows,32)
        module = native.load(src,name)
        kernel = native.compile_builder(module,name,mr,cols,32,32,dtype='float32',out_idx=[-1])
        inverse = [perm.index(i) for i in range(len(in_shape))]
        def run(x):
            v = x.permute(perm).contiguous().reshape(rows,cols).float()
            v = F.pad(v,(0,0,0,mr-rows))
            y = kernel(v)[:rows]
            return y.reshape(tuple(in_shape[i] for i in perm)).permute(inverse).contiguous().to(out_dtype)
    else:
        name = {'AmaxFwdOp':'max','AminFwdOp':'min','SumFwdOp':'sum','AllFwdOp':'min','AnyFwdOp':'max'}[op]
        src = ELEMENTWISE
        # Use the original single-V-core builder in complete row strips. This
        # avoids duplicate writers in older reduce_max/reduce_min test builders.
        # The reduction width is unchanged, so no secondary reduction is needed.
        strip = max(1,min(16,16384//cols))
        if cols > 32768:
            from benchmarks.ascend_baselines import UnsupportedCase
            raise UnsupportedCase('Original whole-row UB reduction exceeds the audited 32768 FP32 element bound.')
        module = native.load(src,'reduce_runtime_semantics_kernel','pass_configs')
        kernel = native.compile_builder(module,'reduce_runtime_semantics_kernel',strip,cols,name,-1,
                                        dtype='float32',out_idx=[-1])
        mr = _up(rows,strip)
        def run(x):
            v = x.permute(perm).contiguous().reshape(rows,cols).float()
            v = F.pad(v,(0,0,0,mr-rows))
            results = [kernel(v[i:i+strip].contiguous()) for i in range(0,mr,strip)]
            return torch.cat(results)[:rows].reshape(out_shape).to(out_dtype)
    return native.candidate(run,f'{src}:{name}',
        'Move reduction axes last, flatten, exact FP32 widening; native reduction '
        'on full rows, row-only padding and complete output restoration in run.',src)


def gemm(op, case):
    _check(op,case)
    import torch.nn.functional as F
    p = case.params
    a,b = case.inputs
    ta,tb = p.get('trans_a',False),p.get('trans_b',False)
    ash,bsh = list(a.shape),list(b.shape)
    if ta: ash[-2:]=ash[-2:][::-1]
    if tb: bsh[-2:]=bsh[-2:][::-1]
    m,k,n = ash[-2],ash[-1],bsh[-1]
    if k != bsh[-2]: raise ValueError('incompatible GEMM K dimensions')
    mp,np,kp = _up(m,128),_up(n,256),_up(k,64)
    dt=str(a.dtype).split('.')[-1]
    src,name = ('examples/batch_gemm/batch_gemm.py','batch_matmul') if op == 'BmmFwdOp' else (
        'examples/gemm/example_gemm_pto_developer.py','matmul')
    module=native.load(src,name)
    args=(a.shape[0],mp,np,kp,128,256,64) if op=='BmmFwdOp' else (mp,np,kp,128,256,64)
    kernel=native.compile_builder(module,name,*args,dtype=dt,out_idx=[-1])
    def run(a,b):
        a = (a.transpose(-1,-2) if ta else a).contiguous()
        b = (b.transpose(-1,-2) if tb else b).contiguous()
        a=F.pad(a,(0,kp-k,0,mp-m))
        b=F.pad(b,(0,np-n,0,kp-k))
        return kernel(a,b)[...,:m,:n].contiguous()
    return native.candidate(run,f'{src}:{name}',
        'Original 128x256x64 cube tiles; transposes and zero padding/cropping in run; '
        'native FP16/BF16 inputs and original FP32 accumulation.',src)


def normalization(op,case):
    plan=_check(op,case)
    import torch
    import torch.nn.functional as F
    p=case.params
    shape,dt=_output(case)
    eps=p.get('eps')
    eps=2**-23 if eps is None else eps
    if op=='GroupNormFwdOp' or plan.route=='layer_group':
        src='examples/cann-bench/group_norm/example_group_norm.py'
        module=native.load(src,'group_norm')
        if plan.route=='layer_group':
            hidden=prod(p['normalized_shape']); rows=prod(shape)//hidden
            def run(x,weight=None,bias=None):
                w=torch.ones(hidden,device=x.device) if weight is None else weight.reshape(hidden).float()
                b=torch.zeros(hidden,device=x.device) if bias is None else bias.reshape(hidden).float()
                return module.group_norm(x.reshape(rows,hidden,1).float(),w,b,1,eps).reshape(shape).to(dt)
        else:
            def run(x,weight=None,bias=None):
                channels=x.shape[1]
                w=torch.ones(channels,device=x.device) if weight is None else weight.float()
                b=torch.zeros(channels,device=x.device) if bias is None else bias.float()
                return module.group_norm(x.contiguous().float(),w,b,p['num_groups'],eps).to(dt)
        api='group_norm'
    else:
        hidden=prod(p['normalized_shape']); rows=prod(shape)//hidden
        src='examples/normalization/rms_norm.py' if op=='RMSNormFwdOp' else 'examples/normalization/layer_norm.py'
        api='_rms_norm' if op=='RMSNormFwdOp' else 'layer_norm'
        mr=_up(rows,64)
        module=native.load(src,api)
        kernel=native.compile_builder(module,api,mr,hidden,64,64,eps,dtype='float32',out_idx=[-1])
        def run(x,weight=None,bias=None):
            if weight is not None or bias is not None:
                from benchmarks.ascend_baselines import UnsupportedCase
                raise UnsupportedCase('Unweighted native normalization selected for optional absent affine inputs only.')
            v=F.pad(x.contiguous().reshape(rows,hidden).float(),(0,0,0,mr-rows))
            return kernel(v)[:rows].reshape(shape).to(dt)
    return native.candidate(run,f'{src}:{api}',
        'Original norm implementation; exact FP32 input/affine widening, flatten '
        'normalized dimensions, preserve epsilon; optional neutral affine values and '
        'row padding inside run. LayerNorm uses original GroupNorm G=1, S=1.',src)


def mhc(case):
    _check('MHCPostFwdOp',case)
    import torch
    src='examples/mhc_post/example_mhc_post.py'
    module=native.load(src,'mhc_post')
    def run(x,post,residual):
        n,h=x.shape; hc=post.shape[1]
        identity=torch.eye(hc,dtype=torch.float32,device=x.device).expand(n,hc,hc).contiguous()
        return module.mhc_post(x.contiguous(),residual.reshape(n,hc,h).contiguous(),
                               post.unsqueeze(-1).contiguous(),identity).reshape(n,hc*h)
    return native.candidate(run,f'{src}:mhc_post',
        'Use identity comb_res_mix to retain manifest residual; reshape packed residual '
        'and output; construct identity per call; original FP32 AXPY arithmetic.',src)


def rope(op,case):
    _check(op,case)
    import math
    import torch
    p=case.params
    shape,dt=_output(case)
    hidden=shape[-1]
    rd=p.get('rotary_dim') or hidden
    is_position=op=='RoPENeoxPositionIdsFwdOp'
    seq=shape[0] if len(shape) in (2,3) else shape[1]
    heads=1 if len(shape)==2 else shape[-2]
    layout='interleaved' if p.get('rope_layout')=='interleaved' else 'half'
    src='examples/pos_embedding/RotaryPositionEmbedding/rope_half_interleaved.py'
    module=native.load(src,'rope_kernel','select_block_M')
    bm=module.select_block_M(heads,rd,layout)
    total=prod(shape[:-1]); chunks=(total+bm-1)//bm
    kernel=native.compile_builder(module,'rope_kernel',total,bm,min(chunks,48),chunks,
                                  seq,hidden,rd,heads,layout,dtype='float32',out_idx=[])
    base=p.get('base',10000.)
    # Constructor tensors (LongRoPE extension factors) are never transformed or
    # cached here; their current values are read inside each run.
    factors=case.arguments.get('rescale_factors') if op=='LongRoPEFwdOp' else None

    def run(x,position_ids=None):
        idx=torch.arange(rd//2,dtype=torch.float32,device=x.device)
        divisor=base**(idx/(rd//2))
        freq=1./divisor
        amplitude=1.
        if op=='RoPELlama31FwdOp':
            scale=p.get('scale_factor',8.); original=p.get('original_max_position',8192)
            low=p.get('low_freq_factor',1.); high=p.get('high_freq_factor',4.)
            # Frequency constants are independent of x. Build the table using
            # the manifest frequency API's piecewise law, not a rotation oracle.
            frequency=[]
            for j in range(rd//2):
                f=1./(base**(j/(rd//2))); wave=2*math.pi/f
                if wave < original/high: value=f
                elif wave > original/low: value=f/scale
                else:
                    blend=(original/wave-low)/(high-low)
                    value=(1-blend)*f/scale+blend*f
                frequency.append(value)
            freq=torch.tensor(frequency,dtype=torch.float32,device=x.device)
        elif op=='YaRNFwdOp':
            half=rd//2; orig=p.get('original_max_position',4096)
            def correction(rot):
                return half*math.log(orig/(rot*2*math.pi))/(2*math.log(base))
            lo=max(math.floor(correction(p.get('beta_fast',32.))),0)
            hi=min(math.ceil(correction(p.get('beta_slow',1.))),half-1)
            if lo==hi: hi+=1
            mask=1.-((idx-lo)/(hi-lo)).clamp(0.,1.)
            inter=1./((p.get('scale',16.)*base)**(idx/half))
            freq=inter*(1.-mask)+freq*mask
            amplitude=p.get('attn_factor',1.)
        elif op=='LongRoPEFwdOp':
            if factors is not None: freq=1./(factors.to(device=x.device,dtype=torch.float32)*divisor)
            original=p.get('original_max_position_embeddings',4096)
            scale=p.get('max_position_embeddings',4096)/original
            amplitude=math.sqrt(1+math.log(scale)/math.log(original)) if scale>1 else 1.
        positions=position_ids.to(torch.float32) if is_position else torch.arange(seq,dtype=torch.float32,device=x.device)
        angle=positions[:,None]*freq[None,:]
        # Preserve manifest table rounding before widening it to native FP32.
        cos=(angle.cos()*amplitude).to(dt).float()
        sin=(angle.sin()*amplitude).to(dt).float()
        if layout=='half': cos=torch.cat((cos,cos),-1); sin=torch.cat((sin,sin),-1)
        else: cos=cos.repeat_interleave(2,-1); sin=sin.repeat_interleave(2,-1)
        # Original kernel rotates a trailing slice. The position-id manifest
        # rotates a prefix, so move that prefix to the end and undo afterwards.
        if is_position and rd<hidden:
            value=torch.cat((x[...,rd:],x[...,:rd]),-1).float().contiguous()
        else: value=x.float().contiguous().clone()
        kernel(value.reshape(total,hidden),sin.contiguous(),cos.contiguous())
        if is_position and rd<hidden:
            value=torch.cat((value[...,-rd:],value[...,:hidden-rd]),-1)
        return value.reshape(shape).to(dt)

    return native.candidate(run,f'{src}:rope_kernel',
        'Original in-place rotary kernel on a private FP32 copy; per-call frequency '
        'tables preserve manifest rounding and layout. Prefix rotation is represented '
        'as trailing slice then inverse permutation. No input-dependent cache.',src)


def attention(op,case):
    _check(op,case)
    import dataclasses
    import torch
    import torch.nn.functional as F
    p=case.params
    src='examples/gqa_fwd_varlen/gqa_fwd_varlen.py'
    module=native.load(src,'flashattn','build_attention_mask')
    # Original expert kernel references its own layout intrinsics.
    from tilelang.intrinsics import make_zn_layout, make_nz_layout
    module.make_zn_layout=make_zn_layout
    module.make_nz_layout=make_nz_layout
    qshape=p['q'][0]; kshape=p['k'][0]
    heads=qshape[-2]; kv_heads=kshape[-2]; dim=qshape[-1]
    out_dtype=getattr(torch,p.get('out_dtype') or p['q'][1])
    causal=p.get('is_causal',False)
    kernels={}

    def one(q,k,v):
        # One independently masked logical sequence; packing costs are included.
        b,sq,_,_=q.shape; sk=k.shape[1]
        mq,nk=_up(sq,128),_up(sk,128)
        key=(b,mq,nk)
        if key not in kernels:
            kernels[key]=native.compile_builder(module,'flashattn',b,heads//kv_heads,heads,
                dim,mq,nk,causal,out_idx=[4],workspace_idx=[5,6,7])
        qp=F.pad(q.transpose(1,2).contiguous(),(0,0,0,mq-sq))
        kp=F.pad(k.transpose(1,2).contiguous(),(0,0,0,nk-sk))
        vp=F.pad(v.transpose(1,2).contiguous(),(0,0,0,nk-sk))
        cq=torch.arange(b+1,dtype=torch.int32,device=q.device)*sq
        ck=torch.arange(b+1,dtype=torch.int32,device=q.device)*sk
        mask=module.build_attention_mask(cq,ck,mq,nk,causal,q.device)
        # Windowing is attention-mask representation, supplied to the original
        # Mask input. It never computes QK/PV or softmax outside the target.
        left,right=p.get('window_size_left',-1),p.get('window_size_right',-1)
        qi=torch.arange(mq,device=q.device)[:,None]+sk-sq
        ki=torch.arange(nk,device=q.device)[None,:]
        if left>=0: mask=mask*(ki>=qi-left)
        if right>=0: mask=mask*(ki<=qi+right)
        return kernels[key](qp,kp,vp,mask)[:,:,:sq].transpose(1,2).contiguous().to(out_dtype)

    if op=='GQADenseFwdOp':
        b,sq,_,_=qshape; sk=kshape[1]
        kernels[(b,_up(sq,128),_up(sk,128))]=native.compile_builder(module,'flashattn',b,
            heads//kv_heads,heads,dim,_up(sq,128),_up(sk,128),causal,
            out_idx=[4],workspace_idx=[5,6,7])
        def run(q,k,v,*optional):
            if any(arg is not None for arg in optional):
                raise ValueError('Selected GQA path lacks scale/rope/sink inputs')
            return one(q,k,v)
    else:
        def run(q,k,v,cu_q,cu_k,*optional):
            if any(arg is not None for arg in optional):
                raise ValueError('Selected GQA path lacks scale/rope/sink inputs')
            # Current offsets are read on EVERY call, with no input cache.
            qs,ks=cu_q.tolist(),cu_k.tolist()
            parts=[]
            for a,b,c,d in zip(qs[:-1],qs[1:],ks[:-1],ks[1:]):
                if a==b: continue
                if c==d:
                    raise ValueError('Original GQA kernel cannot normalize an empty key sequence.')
                parts.append(one(q[a:b].unsqueeze(0),k[c:d].unsqueeze(0),v[c:d].unsqueeze(0))[0])
            return torch.cat(parts,dim=0)
    result=native.candidate(run,f'{src}:flashattn',
        'BSHD/BHSD packing, full logical-sequence masks, pad/crop to original 128 tiles; '
        'original FP16 GQA math. Varlen offsets read per call and complete outputs concatenated; '
        'host-roundtrip timing required. Kernel cache keys only metadata.',src,
        'tilelang/intrinsics/__init__.py')
    return dataclasses.replace(result,timing_mode='host_roundtrip')


def factory(op,case):
    if op in DIRECT or op in GENERIC_UNARY or op in COMPARISON or op=='TanhFwdOp':
        return elementwise(op,case)
    if op in SELECT: return select(op,case)
    if op in {'SiluAndMulFwdOp','GeluTanhAndMulFwdOp'}: return gated(op,case)
    if op in {'AmaxFwdOp','AminFwdOp','SumFwdOp','SoftmaxFwdOp','AllFwdOp','AnyFwdOp'}: return reduction(op,case)
    if op in {'GemmFwdOp','BmmFwdOp'}: return gemm(op,case)
    if op in {'RMSNormFwdOp','LayerNormFwdOp','GroupNormFwdOp'}: return normalization(op,case)
    if op=='InstanceNormFwdOp': return instance_norm(case)
    if op=='MoEGroupedGemmFwdOp': return moe_gemm(case)
    if op=='MLADecodeWithKVCacheFwdOp': return mla_decode(case)
    if op in {'GQAPagedFwdOp','MHADecodePagedWithKVCacheFwdOp'}: return paged_attention(op,case)
    if op=='ArgminFwdOp': return argmin(case)
    if op=='GQABwdOp': return attention_backward(case)
    if op in {'Conv1dFwdOp','Conv2dFwdOp','Conv3dFwdOp'}: return convolution(op,case)
    if op in {'MaxPool1dFwdOp','MaxPool2dFwdOp','MaxPool3dFwdOp','AdaptiveMaxPool2dFwdOp'}: return max_pool(op,case)
    if op=='MHCPostFwdOp': return mhc(case)
    if op in {'RoPEFwdOp','RoPENeoxPositionIdsFwdOp','RoPELlama31FwdOp','YaRNFwdOp','LongRoPEFwdOp'}:
        return rope(op,case)
    if op in {'GQADenseFwdOp','GQAVarlenFwdOp'}: return attention(op,case)
    if op.startswith('INT8Dequant'): return dequant(op,case)
    if op=='MoEPostPermuteFwdOp': return moe_post(case)
    if op=='GroupedGemmFwdOp': return grouped_gemm(case)
    if op=='GemmINT8W8A8FwdOp': return int8_gemm(case)
    if op=='CausalConv1dUpdateFwdOp': return causal_decode(case)
    if op=='CausalConv1dFwdOp': return causal_prefill(case)
    raise KeyError(op)


def dequant(op,case):
    _check(op,case)
    import torch
    shape,dt=_output(case,'x')
    size=prod(shape); rows,cols=_up((size+63)//64,64),64
    module=native.load(ELEMENTWISE,'vec_mul','pass_configs')
    kernel=native.compile_builder(module,'vec_mul',rows,cols,64,64,dtype='float32',out_idx=[-1])
    def run(q,scale):
        if op=='INT8DequantPerChannelFwdOp': scale=scale.reshape(-1,1)
        elif op=='INT8DequantPerBlockFwdOp': scale=scale.repeat_interleave(128,-1)[...,:shape[-1]]
        a=_flat_pack(q,shape,rows,cols,torch.float32)
        b=_flat_pack(scale,shape,rows,cols,torch.float32)
        return kernel(a,b).reshape(-1)[:size].reshape(shape).to(dt)
    return native.candidate(run,f'{ELEMENTWISE}:vec_mul',
        'Lossless INT8-to-FP32 conversion and per-tensor/channel/block scale broadcasting; '
        'original native multiplication; restore full manifest output shape/dtype.',ELEMENTWISE)


def moe_post(case):
    _check('MoEPostPermuteFwdOp',case)
    import torch
    p=case.params
    src='examples/moe_token_permute/moe_token_unpermute.py'
    util='examples/moe_token_permute/moe_token_utils.py'
    names=('auto_tile_h','auto_tile_t','is_fp32_dtype','pad_first_dim','pad_last_dim')
    helpers=native.load(util,*names)
    module=native.load(src,'MoeTokenUnpermute')
    for name in names: setattr(module,name,getattr(helpers,name))
    tokens,topk=p['topk_weights'][0]
    hidden=p['expert_output'][0][-1]
    impl=module.MoeTokenUnpermute(tokens,topk,hidden,dtype='float32')
    dtype=getattr(torch,p.get('out_dtype') or p['expert_output'][1])
    def run(expert_output,topk_weights,inverse_indices):
        return impl(expert_output.reshape(tokens*topk,hidden).float().contiguous(),
                    inverse_indices.reshape(-1).to(torch.int32).contiguous(),
                    topk_weights.float().contiguous()).to(dtype)
    return native.candidate(run,f'{src}:MoeTokenUnpermute',
        'Original native FP32 weighted gather/accumulation, tight route-index layout; '
        'lossless input widening and final output cast inside run.',src,util)


def grouped_gemm(case):
    _check('GroupedGemmFwdOp',case)
    import dataclasses
    import torch
    import torch.nn.functional as F
    p=case.params
    src='examples/gemm/example_gemm_pto_developer.py'
    module=native.load(src,'matmul')
    dt=p['a'][1]
    ta,tb=p['transpose_a'],p['transpose_b']
    kernels={}
    def matmul(a,b):
        m,k=a.shape; n=b.shape[-1]
        mp,np,kp=max(128,_up(m,128)),_up(n,256),max(64,_up(k,64))
        key=mp,np,kp
        if key not in kernels:
            kernels[key]=native.compile_builder(module,'matmul',mp,np,kp,128,256,64,
                                                dtype=dt,out_idx=[-1])
        aa=F.pad(a.contiguous(),(0,kp-k,0,mp-m))
        bb=F.pad(b.contiguous(),(0,np-n,0,kp-k))
        return kernels[key](aa,bb)[:m,:n].contiguous()
    def run(a,b,batch_sizes,batch_offsets):
        sizes,offsets=batch_sizes.tolist(),batch_offsets.tolist()
        outputs=[]
        for g,(count,start) in enumerate(zip(sizes,offsets)):
            aa=a[start:start+count]
            if ta:
                bb=b[:,start:start+count].T if tb else b[start:start+count]
                outputs.append(matmul(aa.T,bb))
            else:
                bb=b[g].T if tb else b[g]
                if count:
                    outputs.append(matmul(aa,bb))
        if ta: return torch.stack(outputs)
        if not outputs: return a.new_empty((0,p['output'][0][-1]))
        return torch.cat(outputs,0)
    result=native.candidate(run,f'{src}:matmul',
        'Group slicing/transposes from current sizes and offsets; original GEMM per '
        'logical group with zero pad/crop, including zero-K empty group products. '
        'Only metadata-specialized kernels cached; all input transforms in run.',src)
    return dataclasses.replace(result,timing_mode='host_roundtrip')


def int8_gemm(case):
    _check('GemmINT8W8A8FwdOp',case)
    import torch.nn.functional as F
    p=case.params; m,k=p['a'][0]; n=p['b'][0][0]
    mp,np,kp=_up(m,32),_up(n,32),_up(k,32)
    src='examples/deepseek_v4/int8_gemm.py'
    module=native.load(src,'int8_gemm_kernel_corrected')
    kernel=native.compile_builder(module,'int8_gemm_kernel_corrected',np,kp,
                                  out_dtype=p['out_dtype'],out_idx=[4],workspace_idx=[5])
    def run(a,b,scale_a,scale_b,bias=None):
        if bias is not None:
            raise ValueError('Native INT8 GEMM lacks bias')
        aa=F.pad(a.contiguous(),(0,kp-k,0,mp-m))
        bb=F.pad(b.contiguous(),(0,kp-k,0,np-n))
        sa=F.pad(scale_a.reshape(-1).expand(m).contiguous(),(0,mp-m))
        sb=F.pad(scale_b.reshape(-1).expand(n).contiguous(),(0,np-n))
        return kernel(aa,bb,sa,sb)[:m,:n].contiguous()
    return native.candidate(run,f'{src}:int8_gemm_kernel_corrected',
        'Original INT32 accumulation and two independent FP32 scales; broadcast '
        'scalar scale representations, pad/crop complete 32x32x32 tiles in run.',src)


def causal_decode(case):
    _check('CausalConv1dUpdateFwdOp',case)
    import torch
    import torch.nn.functional as F
    p=case.params; batch,dim=p['x'][0]; padded=_up(dim,2048)
    src='examples/causal_conv1d/causal_conv1d_decode.py'
    module=native.load(src,'_build_decode_kernel_jit')
    kernel=native.compile_builder(module,'_build_decode_kernel_jit',4,padded//2048,
                                  dtype_str=p['x'][1],has_silu=p.get('activation')=='silu',out_idx=[-1])
    def run(x,weight,conv_state,bias=None):
        xx=F.pad(x.contiguous(),(0,padded-dim))
        ww=F.pad(weight.T.contiguous(),(0,padded-dim))
        # Manifest keeps W history positions; native API keeps W-1 positions.
        state=F.pad(conv_state[...,1:].transpose(1,2).contiguous(),(0,padded-dim)).clone()
        idx=torch.arange(batch,device=x.device,dtype=torch.int32)
        flags=torch.ones(batch,device=x.device,dtype=torch.int32)
        bb=torch.zeros(dim,dtype=x.dtype,device=x.device) if bias is None else bias
        bb=F.pad(bb,(0,padded-dim))
        y=kernel(xx,ww,state,idx,idx,flags,bb)[:,:dim].contiguous()
        # Return complete manifest state, taking the updated suffix from the
        # actual native state tensor and the remaining entry from original data.
        result_state=torch.cat((conv_state[...,1:2],state[...,:dim].transpose(1,2)),-1)
        return y,result_state
    return native.candidate(run,f'{src}:_build_decode_kernel_jit',
        'Bypass wrapper FP16-to-BF16 narrowing using original dtype template; '
        'channel padding, transposed private W-1 state, exact full-W state restoration; '
        'native kernel performs convolution, SiLU and state update.',src)


def causal_prefill(case):
    _check('CausalConv1dFwdOp',case)
    import dataclasses
    import torch
    import torch.nn.functional as F
    p=case.params; dim,total=p['x'][0]; batch=p['conv_state'][0][0]
    src='examples/causal_conv1d/causal_conv1d_pto.py'
    module=native.load(src,'_build_kernel','CORE_NUM')
    # Two sequences per call match the original kernel's two cache-index loads.
    # Channel slices bound UB storage; neither the 24-core token schedule nor
    # the source kernel is changed. FP32 widening avoids intermediate BF16 loss.
    kernel=native.compile_builder(module,'_build_kernel',4,module.CORE_NUM,
                                  2,1024,dtype_str='float32',out_idx=[-1])
    def run(x,weight,conv_state,cu_seqlens,bias=None,has_initial_state=None):
        if bias is not None: raise ValueError('Original prefill kernel has no bias input')
        offsets=cu_seqlens.tolist()
        pairs=[]; states=[]
        for b in range(0,batch,2):
            lo,mid,hi=offsets[b:b+3]
            # Very short sequence tasks make this original kernel read history
            # before the first input token; reject at runtime before launching.
            lengths=(mid-lo,hi-mid)
            if any((n+23)//24<3 or 23*((n+23)//24)>=n for n in lengths):
                raise ValueError('Original 24-core prefill requires at least three history-safe tokens per task and no empty trailing task')
            seq=torch.tensor([0,mid-lo,hi-lo],dtype=torch.int32,device=x.device)
            idx=torch.arange(2,dtype=torch.int32,device=x.device)
            flags=(torch.zeros(2,dtype=torch.int32,device=x.device) if has_initial_state is None
                   else has_initial_state[b:b+2].to(torch.int32).contiguous())
            channels=[]; histories=[]
            for start in range(0,dim,1024):
                end=min(start+1024,dim); width=end-start
                xx=F.pad(x[start:end,lo:hi].T.contiguous().float(),(0,1024-width,0,3))
                ww=F.pad(weight[start:end].T.contiguous().float(),(0,1024-width))
                ss=F.pad(conv_state[b:b+2,start:end].transpose(1,2).contiguous().float(),
                         (0,1024-width)).clone()
                yy=kernel(xx,ww,ss,seq,idx,flags)
                channels.append(yy[:hi-lo,:width].T)
                histories.append(ss[:,:,:width].transpose(1,2))
            pairs.append(torch.cat(channels,dim=0))
            states.append(torch.cat(histories,dim=1))
        return torch.cat(pairs,dim=1).to(x.dtype),torch.cat(states,dim=0).to(conv_state.dtype)
    result=native.candidate(run,f'{src}:_build_kernel',
        'Original 24-core prefill schedule; pairs of logical sequences avoid the '
        'source two-cache-index limitation; 1024-wide channel packing, FP32 widening, '
        'three safe tail-read rows and private state. Return complete output and '
        'native updated state. Sequence offsets read on every call.',src)
    return dataclasses.replace(result,timing_mode='host_roundtrip')


def instance_norm(case):
    _check('InstanceNormFwdOp',case)
    import torch
    src='examples/cann-bench/group_norm/example_group_norm.py'
    module=native.load(src,'group_norm')
    def run(x,running_mean=None,running_var=None,weight=None,bias=None):
        if running_mean is not None or running_var is not None:
            raise ValueError('Selected InstanceNorm route excludes running statistics')
        c=x.shape[1]
        w=torch.ones(c,device=x.device) if weight is None else weight.float()
        b=torch.zeros(c,device=x.device) if bias is None else bias.float()
        return module.group_norm(x.contiguous().float(),w,b,c,case.params['eps']).to(x.dtype)
    return native.candidate(run,f'{src}:group_norm',
        'Select original GroupNorm with groups=channels; each group is one instance/channel '
        'spatial slice. Preserve affine values and epsilon; exact FP32 widening per run.',src)


def moe_gemm(case):
    _check('MoEGroupedGemmFwdOp',case)
    import dataclasses
    import torch
    import torch.nn.functional as F
    p=case.params;layout=p['layout'];masked='masked' in layout
    src='examples/gemm/example_gemm_pto_developer.py'
    module=native.load(src,'matmul');kernels={}
    alignment=1 if masked else layout['contiguous'].get('alignment',1)
    def multiply(a,b):
        m,k=a.shape;n=b.shape[0];mp,np,kp=_up(m,128),_up(n,256),_up(k,64)
        key=mp,np,kp
        if key not in kernels:
            kernels[key]=native.compile_builder(module,'matmul',mp,np,kp,128,256,64,
                                                dtype=p['a'][1],out_idx=[-1])
        aa=F.pad(a.contiguous(),(0,kp-k,0,mp-m))
        bb=F.pad(b.T.contiguous(),(0,np-n,0,kp-k))
        return kernels[key](aa,bb)[:m,:n]
    def run(a,b,layout_metadata):
        meta=layout_metadata.tolist();n=b.shape[1]
        out=torch.zeros((*a.shape[:-1],n),dtype=a.dtype,device=a.device)
        start=0
        for g,count in enumerate(meta):
            if masked:
                if count:out[g,:count]=multiply(a[g,:count],b[g])
            else:
                start=_up(start,alignment);end=count
                if end>start:out[start:end]=multiply(a[start:end],b[g])
                start=end
        return out
    result=native.candidate(run,f'{src}:matmul',
        'Original GEMM per expert; masked/physical-prefix layout slicing, zero padding '
        'and complete output scatter. Metadata read each call; no mathematical '
        'postprocessing and no cached input values.',src)
    return dataclasses.replace(result,timing_mode='host_roundtrip')


def mla_decode(case):
    _check('MLADecodeWithKVCacheFwdOp',case)
    import torch
    p=case.params;batch,heads,dim=p['q'][0];tail=p['q_pe'][0][-1]
    length,groups=p['k'][0][1:3]
    src='examples/sparse_flash_attention/example_sparse_flash_attn_dynamic_shape.py'
    module=native.load(src,'sparse_attention_fwd')
    kernel=native.compile_builder(module,'sparse_attention_fwd',heads,dim,tail,length,1,
                                  kv_group=groups,dtype=p['q'][1],out_idx=[3],workspace_idx=[4,5,6,7,8])
    def run(q,q_pe,k,k_pe):
        query=torch.cat((q,q_pe),-1).unsqueeze(1).contiguous()
        key=torch.cat((k,k_pe),-1).contiguous()
        # Selecting every key once expresses dense MLA in the original sparse
        # attention API. No padded or duplicate index participates in softmax.
        indices=torch.arange(length,dtype=torch.int32,device=q.device).reshape(1,1,1,length)
        indices=indices.expand(batch,1,groups,length).contiguous()
        return kernel(query,key,indices)[:,0].contiguous()
    return native.candidate(run,f'{src}:sparse_attention_fwd',
        'Pack latent/rotary Q and K fields; select all keys exactly once through '
        'the original sparse MLA kernel. Native softmax and latent-value projection; '
        'original FP16/BF16 template. Full per-run selection-index creation is timed.',src)


def paged_attention(op,case):
    _check(op,case)
    import dataclasses
    import torch
    from types import SimpleNamespace
    p=case.params;cached={}
    def one(q,k,v):
        key=q.shape,k.shape,v.shape
        if key not in cached:
            packed=dict(p,q=(tuple(q.shape),'float16'),k=(tuple(k.shape),'float16'),
                        v=(tuple(v.shape),'float16'),o=(tuple(q.shape),'float16'))
            proxy=SimpleNamespace(params=packed,arguments={})
            cached[key]=attention('GQADenseFwdOp',proxy).implementation.run
        return cached[key](q,k,v)
    def gather(pages,table,length):
        if length<=0:raise ValueError('Native GQA cannot normalize an empty key sequence')
        page_size=pages.shape[1];count=(length+page_size-1)//page_size
        return pages[table[:count].long()].reshape(-1,*pages.shape[2:])[:length].unsqueeze(0).contiguous()
    if op=='GQAPagedFwdOp':
        def run(q,k_pages,v_pages,page_table,cache_seqlens,cu_seqlens_q,*optional):
            if any(value is not None for value in optional):raise ValueError('Native GQA route excludes scale/rope/sink inputs')
            offsets=cu_seqlens_q.tolist();lengths=cache_seqlens.tolist();parts=[]
            for i,(begin,end) in enumerate(zip(offsets[:-1],offsets[1:])):
                if begin==end:continue
                k=gather(k_pages,page_table[i],lengths[i]);v=gather(v_pages,page_table[i],lengths[i])
                parts.append(one(q[begin:end].unsqueeze(0),k,v)[0])
            return torch.cat(parts,0)
    else:
        def run(q,k,v,real_seqlen_kv,block_table):
            pages_k=k.reshape(-1,p['page_size'],*k.shape[1:])
            pages_v=v.reshape(-1,p['page_size'],*v.shape[1:])
            parts=[]
            for i,length in enumerate(real_seqlen_kv.tolist()):
                kk=gather(pages_k,block_table[i],length);vv=gather(pages_v,block_table[i],length)
                parts.append(one(q[i:i+1],kk,vv))
            return torch.cat(parts,0)
    result=native.candidate(run,'examples/gqa_fwd_varlen/gqa_fwd_varlen.py:flashattn',
        'Gather logical KV pages from current block-table/length metadata each run; '
        'original dense GQA per request with full mask and pad/crop. Layout-only '
        'gathers, all native QK/softmax/PV; compile cache contains shapes only.',
        'examples/gqa_fwd_varlen/gqa_fwd_varlen.py','tilelang/intrinsics/__init__.py')
    return dataclasses.replace(result,timing_mode='host_roundtrip')


def max_pool(op,case):
    _check(op,case)
    import torch
    import torch.nn.functional as F
    p=case.params;out_shape,out_dtype=_output(case)
    module=native.load(ELEMENTWISE,'reduce_runtime_semantics_kernel','pass_configs')
    kernels={}
    def reduce_windows(windows,width):
        cols=_up(width,32);rows=windows.numel()//width
        strip=max(1,min(1024,16384//cols));mr=_up(rows,strip)
        key=strip,cols
        if key not in kernels:
            kernels[key]=native.compile_builder(module,'reduce_runtime_semantics_kernel',strip,cols,
                                                'max',-1,dtype='float32',out_idx=[-1])
        packed=F.pad(windows.contiguous().reshape(rows,width).float(),
                     (0,cols-width,0,mr-rows),value=-float('inf'))
        return torch.cat([kernels[key](packed[i:i+strip].contiguous()) for i in range(0,mr,strip)])[:rows]
    if op=='AdaptiveMaxPool2dFwdOp':
        height,width=p['input'][0][-2:];oh,ow=out_shape[-2:]
        def run(x):
            values=[]
            for y in range(oh):
                y0=y*height//oh;y1=((y+1)*height+oh-1)//oh
                for z in range(ow):
                    z0=z*width//ow;z1=((z+1)*width+ow-1)//ow
                    values.append(reduce_windows(x[...,y0:y1,z0:z1],(y1-y0)*(z1-z0)))
            return torch.stack(values,-1).reshape(out_shape).to(out_dtype)
    else:
        dims=len(out_shape)-2
        def axes(value):return [value]*dims if isinstance(value,int) else list(value)
        kernel=axes(p['kernel_size']);stride=axes(p['stride'] if p['stride'] is not None else p['kernel_size'])
        padding=axes(p['padding']);dilation=axes(p['dilation'])
        extent=[d*(k-1)+1 for k,d in zip(kernel,dilation)]
        padded=[]
        for n,out,k,s,lo in reversed(list(zip(p['input'][0][2:],out_shape[2:],extent,stride,padding))):
            padded.extend((lo,max(lo,(out-1)*s+k-n-lo)))
        def run(x):
            windows=F.pad(x,padded,value=-float('inf'))
            for axis in range(dims):windows=windows.unfold(2+axis,extent[axis],stride[axis])
            # Explicitly crop ceil-mode windows and stride through dilated taps.
            slices=(slice(None),slice(None))+tuple(slice(0,n) for n in out_shape[2:])
            slices+=tuple(slice(None,None,d) for d in dilation)
            windows=windows[slices]
            return reduce_windows(windows,prod(kernel)).reshape(out_shape).to(out_dtype)
    return native.candidate(run,f'{ELEMENTWISE}:reduce_runtime_semantics_kernel',
        'Represent pooling windows as unfolded/sliced rows, including dilation, '
        'negative-infinity padding and ceil-mode crop; native maximum per complete '
        'window. Adaptive bounds are metadata. No torch reduction or input cache.',ELEMENTWISE)


def convolution(op,case):
    _check(op,case)
    import torch
    import torch.nn.functional as F
    p=case.params;out_shape,out_dtype=_output(case)
    ishape=p['input'][0];wshape=p['weight'][0];dims=len(ishape)-2
    groups=p['groups'];ci=ishape[1]//groups;co=wshape[0]//groups
    def axes(value):return [value]*dims if isinstance(value,int) else list(value)
    stride=axes(p['stride']);dilation=axes(p['dilation']);ks=list(wshape[2:])
    extent=[d*(k-1)+1 for k,d in zip(ks,dilation)]
    if p['padding']=='same':
        # PyTorch's SAME convention puts an odd extra position on the right.
        pads=[((e-1)//2,e-1-(e-1)//2) for e in extent]
    elif p['padding']=='valid':pads=[(0,0)]*dims
    else:pads=[(a,a) for a in axes(p['padding'])]
    padding=[n for pair in reversed(pads) for n in pair]
    m=ishape[0]*prod(out_shape[2:]);k=ci*prod(ks);with_bias='bias' in p
    kk=k+int(with_bias);mp,np,kp=_up(m,128),_up(co,256),_up(kk,64)
    src='examples/gemm/example_gemm_pto_developer.py';module=native.load(src,'matmul')
    kernel=native.compile_builder(module,'matmul',mp,np,kp,128,256,64,dtype=p['input'][1],out_idx=[-1])
    def run(x,weight,bias=None):
        if (bias is not None)!=with_bias:raise ValueError('Convolution bias presence differs from manifest')
        windows=F.pad(x,padding)
        for axis in range(dims):windows=windows.unfold(axis+2,extent[axis],stride[axis])
        slices=(slice(None),slice(None))+tuple(slice(0,n) for n in out_shape[2:])
        slices+=tuple(slice(None,None,d) for d in dilation)
        windows=windows[slices]
        order=[0]+list(range(2,dims+2))+[1]+list(range(dims+2,2*dims+2))
        pieces=[]
        for g in range(groups):
            aa=windows[:,g*ci:(g+1)*ci].permute(order).contiguous().reshape(m,k)
            bb=weight[g*co:(g+1)*co].contiguous().reshape(co,k).T.contiguous()
            if with_bias:
                # Homogeneous-coordinate layout: one exact constant input and
                # the original bias values occupy an additional K position.
                aa=torch.cat((aa,torch.ones((m,1),dtype=x.dtype,device=x.device)),1)
                bb=torch.cat((bb,bias[g*co:(g+1)*co].reshape(1,co)),0)
            aa=F.pad(aa,(0,kp-kk,0,mp-m));bb=F.pad(bb,(0,np-co,0,kp-kk))
            pieces.append(kernel(aa,bb)[:m,:co])
        result=torch.cat(pieces,1).reshape(ishape[0],*out_shape[2:],wshape[0])
        return result.permute([0,dims+1]+list(range(1,dims+1))).contiguous().to(out_dtype)
    return native.candidate(run,f'{src}:matmul',
        'Im2col by pad/unfold/dilated views and group slicing; original cube GEMM '
        'computes each complete cross-correlation. Optional bias is an exact '
        'homogeneous-coordinate input (constant one plus original bias K row), '
        'inside the same native dot product. Full output layout restored in run.',src)


def argmin(case):
    _check('ArgminFwdOp',case)
    import torch
    import torch.nn.functional as F
    p=case.params;shape=tuple(p['x'][0]);out_shape,_=_output(case)
    perm,rows,width=_axis_layout(shape,p.get('dim'))
    src='examples/cann-bench/cummin/example_cummin.py'
    module=native.load(src,'_run_vec_kernel','_find_vec_N')
    padded=rows
    while padded%module._find_vec_N(padded,'float32'):
        padded=_up(padded,module._find_vec_N(padded,'float32'))
    def run(x):
        # Native cummin selects the last equal minimum. Reversing the logical
        # axis makes that the first minimum in manifest order.
        data=x.permute(perm).contiguous().reshape(rows,width).T.flip((0,)).float()
        data=F.pad(data,(0,padded-rows),value=float('inf'))
        _,indices=module._run_vec_kernel(data.contiguous(),padded,width,'float32',True)
        reversed_positions=torch.arange(width-1,-1,-1,dtype=torch.int64,device=x.device)
        return reversed_positions[indices[-1,:rows].long()].reshape(out_shape)
    return native.candidate(run,f'{src}:_run_vec_kernel',
        'Reverse reduction-axis layout to map native last-tie cummin to first-tie '
        'argmin; select final native index and invert its coordinate with an index '
        'lookup. Full FP32 scan, column padding and metadata-only coordinate map; '
        'no torch comparison or arg-reduction.',src)


def attention_backward(case):
    _check('GQABwdOp',case)
    import torch
    p=case.params;batch,length,heads,dim=p['q'][0];kv_heads=p['k'][0][-2]
    src='examples/attention_sink/example_gqa_sink_bwd_bhsd/example_gqa_sink_bwd_bhsd.py'
    module=native.load(src,'flashattn_fwd','flashattn_bwd')
    forward=native.compile_builder(module,'flashattn_fwd',batch,heads,length,dim,heads//kv_heads,None,
                                  out_idx=[3,4],workspace_idx=[6,7,8])
    backward=native.compile_builder(module,'flashattn_bwd',batch,heads,length,dim,dim,None,64,64,
                                   heads//kv_heads,out_idx=[])
    def run(q,k,v,o,do,lse):
        qq,kk,vv,dd=[t.transpose(1,2).contiguous() for t in (q,k,v,do)]
        sinks=torch.full((heads,),-float('inf'),dtype=q.dtype,device=q.device)
        # Manifest stores base-2 LSE; the native backward consumes natural-log
        # LSE. Select the original native forward/backward pipeline to generate
        # consistent intermediates, including independent q-zero controls.
        # Its complete recomputation cost belongs to this candidate's run.
        fresh_o,fresh_lse=forward(qq,kk,vv,sinks)
        delta=torch.zeros((batch,heads,length),dtype=torch.float32,device=q.device)
        ds=torch.zeros_like(delta)
        dq=torch.zeros((batch,heads,length,dim),dtype=q.dtype,device=q.device)
        dk=torch.zeros((batch,kv_heads,length,dim),dtype=torch.float32,device=q.device)
        dv=torch.zeros_like(dk)
        backward(qq,kk,vv,dd,fresh_o,fresh_lse,sinks,delta,ds,dq,dk,dv)
        return tuple(t.transpose(1,2).contiguous().to(q.dtype) for t in (dq,dk,dv))
    return native.candidate(run,f'{src}:flashattn_fwd+flashattn_bwd',
        'Original native attention forward/backward pipeline with a neutral -inf '
        'sink; recompute native O/natural-log LSE each invocation instead of '
        'converting manifest base-2 LSE. Full forward and backward cost timed; '
        'all three gradients returned, original inputs preserved.',src)
