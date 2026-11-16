"""Layout-only bindings to unmodified ops-transformer implementations."""
from pathlib import Path

DISPATCH_EVENTS=[]
_gelu_schema=None
_gelu_loaded=False


def bmm(case):
    import torch
    from benchmarks.ascend_graph_call import GraphCall
    from benchmarks.ascend_transformer_graph import _candidate
    from benchmarks.ascend_baselines import UnsupportedCase, source_hashes
    from .hashing import source_hashes
    from dataclasses import replace

    a, b = case.inputs
    if a.dtype not in (torch.float16, torch.bfloat16) or b.dtype != a.dtype:
        raise UnsupportedCase('GroupedMatmulAdd native inputs require matching FP16/BF16.')
    binding = GraphCall('ops-transformer')
    def run(a, b):
        # Every slice, transpose and accumulator initialization belongs to run.
        outputs = []
        for lhs, rhs in zip(a, b, strict=True):
            group = torch.full((1,), lhs.shape[1], device=a.device, dtype=torch.int64)
            out = torch.zeros((1, lhs.shape[0], rhs.shape[1]), device=a.device, dtype=torch.float32)
            binding.call('GroupedMatmulAdd', [lhs.T.contiguous(), rhs.contiguous(), group, out], [out],
                         {'transpose_x': True, 'transpose_weight': False,
                          'group_type': 2, 'group_list_type': 0})
            outputs.append(out[0].to(a.dtype))
        return torch.stack(outputs)
    result = _candidate(binding, run, 'GroupedMatmulAdd', 'gmm/grouped_matmul_add',
                        'One native K-group per independent batch; per-call transpose, zero FP32 accumulator, manifest output cast and stack. No input-dependent caching.')
    return replace(result, source_sha256=result.source_sha256 | source_hashes(Path(__file__)))


def _candidate(binding, run, op, directory, adaptation, host=False):
    from benchmarks.ascend_transformer_graph import _candidate as original
    from benchmarks.ascend_baselines import source_hashes
    from .hashing import source_hashes
    from dataclasses import replace
    result = original(binding, run, op, directory, adaptation)
    return replace(result, source_sha256=result.source_sha256 | source_hashes(Path(__file__),Path(__file__).with_name('native.py'),Path(__file__).with_name('aclnn_bridge.py'),Path(__file__).with_name('hashing.py'),Path(__file__).parents[2]/'ascend_math_more.py',Path(__file__).with_name('graph_layout.py'),Path(__file__).with_name('infer_shim.cpp'),Path('/root/dyq/workspace/.cache/all-pairs/ops_transformer/libtransformer_infer_shim.so')),
                   timing_mode='host_roundtrip' if host else result.timing_mode)


def layernorm(case):
    import math
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    from benchmarks.ascend_graph_call import GraphCall
    hidden = math.prod(case.params['normalized_shape'])
    if hidden > 1024:
        raise UnsupportedCase('NormRopeConcat original headDim limit is 1024; normalization-axis splitting changes the reduction.')
    binding = GraphCall('ops-transformer')
    def run(x, weight, bias):
        inp = x.contiguous().reshape(1, -1, 1, hidden)
        gamma = torch.ones(hidden, dtype=x.dtype, device=x.device) if weight is None else weight.reshape(hidden).contiguous()
        beta = torch.zeros_like(gamma) if bias is None else bias.reshape(hidden).contiguous()
        encoder = torch.empty((1,0,1,hidden),dtype=x.dtype,device=x.device)
        zero,one = torch.zeros_like(gamma),torch.ones_like(gamma)
        rope = torch.empty((0,hidden),dtype=x.dtype,device=x.device)
        outs = [torch.empty((1,1,inp.shape[1],hidden),device=x.device,dtype=x.dtype) for _ in range(3)]
        binding.call('NormRopeConcat',[inp,inp,inp,encoder,encoder,encoder,gamma,beta,gamma,beta,one,zero,one,zero,rope,rope],outs,
                     {'norm_type':2,'norm_added_type':0,'rope_type':0,'concat_order':0,'eps':float(case.params['eps']),'is_training':False})
        return outs[0].reshape_as(x)
    return _candidate(binding,run,'NormRopeConcat','posembedding/norm_rope_concat','Native affine LayerNorm mode 2, no RoPE/encoder; flatten complete normalized dimensions and restore shape.')


def pointwise_conv(case):
    import math
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    from benchmarks.ascend_graph_call import GraphCall
    p=case.params
    x,w,*_=case.inputs
    rank=x.ndim-2
    tup=lambda v: (v,)*rank if isinstance(v,int) else tuple(v)
    if x.dtype not in (torch.float16,torch.bfloat16):
        raise UnsupportedCase('GroupedMatmulAdd requires FP16/BF16 inputs without lossy conversion.')
    binding=GraphCall('ops-transformer')
    def run(x,w,bias=None):
        stride,padding,dilation=tup(p['stride']),tup(p['padding']),tup(p['dilation'])
        padded=torch.nn.functional.pad(x,tuple(v for n in reversed(padding) for v in (n,n)))
        spatial=tuple((s+2*pad-dil*(kernel-1)-1)//step+1 for s,pad,dil,kernel,step in zip(x.shape[2:],padding,dilation,w.shape[2:],stride,strict=True))
        sizes=(x.shape[0],x.shape[1],*spatial,*w.shape[2:])
        strides=(*padded.stride()[:2],*[s*t for s,t in zip(padded.stride()[2:],stride,strict=True)],*[s*t for s,t in zip(padded.stride()[2:],dilation,strict=True)])
        patches=padded.as_strided(sizes,strides)
        results=[];cin=x.shape[1]//p['groups'];cout=w.shape[0]//p['groups']
        for g in range(p['groups']):
            a=patches[:,g*cin:(g+1)*cin].permute(0,*range(2,2+rank),1,*range(2+rank,2+2*rank)).contiguous().reshape(-1,cin*math.prod(w.shape[2:]))
            b=w[g*cout:(g+1)*cout].reshape(cout,-1).T.contiguous();shape=(1,a.shape[0],cout)
            out=torch.zeros(shape,dtype=torch.float32,device=x.device) if bias is None else bias[g*cout:(g+1)*cout].float().view(1,1,-1).expand(shape).contiguous()
            group=torch.full((1,),a.shape[1],dtype=torch.int64,device=x.device)
            binding.call('GroupedMatmulAdd',[a.T.contiguous(),b,group,out],[out],{'transpose_x':True,'transpose_weight':False,'group_type':2,'group_list_type':0})
            results.append(out[0].to(x.dtype).reshape(x.shape[0],*spatial,cout))
        return torch.cat(results,-1).movedim(-1,1).contiguous()
    return _candidate(binding,run,'GroupedMatmulAdd','gmm/grouped_matmul_add','1D/3D convolution via zero padding and strided patch layout, original GEMM per group with FP32 bias accumulator; all packing/cropping/layout copies are timed.')


def _attention(binding,q,k,v,p,mask_override=None):
    import torch
    b,sq,h,d=q.shape;sk=k.shape[1]
    out=torch.empty((b,sq,h,v.shape[-1]),dtype=q.dtype,device=q.device)
    mx=torch.empty((b,h,sq,8),dtype=torch.float32,device=q.device);sm=torch.empty_like(mx)
    empty=torch.empty((0,0,0,0),dtype=q.dtype,device=q.device)
    mask=mask_override
    if mask_override is None and (p.get('is_causal') or p.get('window_size_left',-1)>=0 or p.get('window_size_right',-1)>=0):
        qp=torch.arange(sq,device=q.device)+sk-sq;kp=torch.arange(sk,device=q.device)
        mask=torch.zeros((sq,sk),dtype=torch.bool,device=q.device)
        if p.get('is_causal'):mask |= kp[None,:]>qp[:,None]
        if p.get('window_size_left',-1)>=0:mask |= kp[None,:]<qp[:,None]-p['window_size_left']
        if p.get('window_size_right',-1)>=0:mask |= kp[None,:]>qp[:,None]+p['window_size_right']
    ins=[q.contiguous(),k.contiguous(),v.contiguous(),*([None]*16)];ins[6]=mask
    binding.call('FlashAttentionScore',ins,[mx,sm,empty,out],{'scale_value':float(p.get('sm_scale') or d**-.5),'keep_prob':1.,'pre_tockens':2147483647,'next_tockens':2147483647,'head_num':h,'input_layout':'BSND','inner_precise':0,'sparse_mode':0})
    return out


def segmented_attention(case):
    import torch
    from benchmarks.ascend_graph_call import GraphCall
    from benchmarks.ascend_baselines import UnsupportedCase
    p=case.params;op=case._call.signature.name
    paged=op=='GQAPagedFwdOp'
    extra_start=6 if paged else 5
    if any(t is not None for t in case.inputs[extra_start:]) or p.get('softcap') is not None or p.get('pos_encoding_mode','none')!='none':
        raise UnsupportedCase('FlashAttentionScore path requires no external quantization scales, RoPE tensors, softcap or sink logits.')
    if case.inputs[0].dtype not in (torch.float16,torch.bfloat16):
        raise UnsupportedCase('This native segmented path selects original FP16/BF16 attention.')
    binding=GraphCall('ops-transformer')
    def run(q,k,v,*metadata):
        outputs=[]
        if paged:
            table,lens,cq,*_=metadata
            qb=cq.tolist();lengths=lens.tolist();page=k.shape[1]
            for i,ln in enumerate(lengths):
                pages=table[i,:-(-ln//page)].long()
                kb=k[pages].flatten(0,1)[:ln];vb=v[pages].flatten(0,1)[:ln]
                outputs.append(_attention(binding,q[qb[i]:qb[i+1]][None],kb[None],vb[None],p)[0])
        else:
            cq,ck,*_=metadata;qb=cq.tolist();kb=ck.tolist()
            for i in range(len(qb)-1):
                outputs.append(_attention(binding,q[qb[i]:qb[i+1]][None],k[kb[i]:kb[i+1]][None],v[kb[i]:kb[i+1]][None],p)[0])
        return torch.cat(outputs).to(getattr(torch,p['out_dtype']) if p.get('out_dtype') else q.dtype)
    return _candidate(binding,run,'FlashAttentionScore','attention/flash_attention_score','Per-call sequence metadata and paged gathers; independent native attention per sequence with right-aligned causal/window masks; concatenate complete outputs.',host=True)


def mhc_post(case):
    import torch
    from .native import GraphCall
    binding=GraphCall('ops-transformer')
    def run(x_layer_out,h_post,x_res):
        b,d=x_layer_out.shape;n=h_post.shape[-1]
        out=torch.empty((b,n,d),dtype=x_res.dtype,device=x_res.device)
        binding.call('MhcPost',[x_res.reshape(b,n,d).contiguous(),None,x_layer_out.contiguous(),h_post.contiguous()],[out],{})
        return out.reshape_as(x_res)
    return _candidate(binding,run,'MhcPost','mhc/mhc_post','Ascend950 original optional h_res=None directly selects residual plus post projection; reshape residual axes.')


def softmax(case):
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    from .native import GraphCall
    p=case.params;x=case.inputs[0];dim=p['dim']%x.ndim;width=x.shape[dim]
    if width>1024:raise UnsupportedCase('Native MoeGatingTopKSoftmax k<=1024; all coordinates are required for a pure output permutation.')
    dtype=getattr(torch,p['dtype']) if p.get('dtype') else x.dtype
    if dtype not in (x.dtype,torch.float32):raise UnsupportedCase('Only original dtype or exact input widening to FP32.')
    binding=GraphCall('ops-transformer')
    def run(x):
        moved=x.movedim(dim,-1);packed=moved.to(dtype).contiguous().reshape(-1,width)
        y=torch.empty_like(packed);idx=torch.empty(packed.shape,dtype=torch.int32,device=x.device);rows=torch.empty_like(idx)
        binding.call('MoeGatingTopKSoftmax',[packed,None],[y,idx,rows],{'k':width})
        out=torch.empty_like(y).scatter_(1,idx.long(),y)
        return out.reshape(moved.shape).movedim(-1,dim).contiguous()
    return _candidate(binding,run,'MoeGatingTopKSoftmax','moe/moe_gating_top_k_softmax','Select k=entire reduction width; invert original top-k permutation with indices; no softmax arithmetic in adapter.')


def postpermute(case):
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    from .native import GraphCall
    p=case.params
    if p.get('epilogue') is not None or p.get('out_dtype') not in (None, str(case.inputs[0].dtype).removeprefix('torch.')):
        raise UnsupportedCase('Native routed output requires input dtype; distinct epilogue or output dtype is not bound.')
    binding=GraphCall('ops-transformer')
    def run(expert_output,topk_weights,inverse_indices):
        out=torch.empty((topk_weights.shape[0],expert_output.shape[-1]),dtype=expert_output.dtype,device=expert_output.device)
        binding.call('MoeFinalizeRoutingV2',[expert_output.reshape(-1,expert_output.shape[-1]).contiguous(),inverse_indices.reshape(topk_weights.shape).T.contiguous().flatten().int(),None,None,None,topk_weights.contiguous(),None],[out],{'drop_pad_mode':0,'k':topk_weights.shape[1]})
        return out
    return _candidate(binding,run,'MoeFinalizeRoutingV2','moe/moe_finalize_routing_v2','Transpose manifest token-major inverse indices to original expert-rank-major flattened layout; native weighted sum with explicit k, flatten padded expert rows.')


def cache_write(case):
    from benchmarks.ascend_baselines import UnsupportedCase
    if any(x is not None for x in case.inputs[5:]):
        raise UnsupportedCase('ScatterPaKvCache does not implement the manifest per-cache FP8 scaling.')
    from .native import GraphCall
    binding=GraphCall('ops-transformer')
    def run(k,v,k_pages,v_pages,slot_mapping,k_scale=None,v_scale=None):
        kc,vc=k_pages.clone(),v_pages.clone()
        live=(slot_mapping>=0).nonzero().flatten()
        if live.numel():
            binding.call('ScatterPaKvCache',[k[live].contiguous(),kc,slot_mapping[live].contiguous(),v[live].contiguous(),vc,None,None,None],[kc,vc],{'cache_mode':'Norm','scatter_mode':'None'})
        return kc,vc
    return _candidate(binding,run,'ScatterPaKvCache','attention/scatter_pa_kv_cache','Clone cache inputs each call, original slot-mapping scatter into private outputs; preserve original input caches.')


def recurrent_delta(case):
    import torch
    from .native import GraphCall
    from benchmarks.ascend_baselines import UnsupportedCase
    if case.inputs[0].dtype != torch.bfloat16:
        raise UnsupportedCase('RecurrentGatedDeltaRule native query/key/value/beta dtype is BF16; no lossy cast.')
    binding=GraphCall('ops-transformer')
    def run(q,k,v,beta,state):
        out=torch.empty_like(v);st=state.transpose(-2,-1).contiguous().clone()
        lengths=torch.ones(q.shape[0],dtype=torch.int32,device=q.device)
        ids=torch.arange(q.shape[0],dtype=torch.int32,device=q.device)
        binding.call('RecurrentGatedDeltaRule',[q.contiguous(),k.contiguous(),v.contiguous(),beta.contiguous(),st,lengths,ids,None,None,lengths],[out,st],{'scale_value':1.})
        return out,st.transpose(-2,-1).contiguous()
    return _candidate(binding,run,'RecurrentGatedDeltaRule','attention/recurrent_gated_delta_rule','One token per sequence; null log gates select original ungated Delta rule, scale=1; transpose and clone state per call.')


def fused_topk(case):
    import torch
    from .native import GraphCall
    from benchmarks.ascend_baselines import UnsupportedCase
    p=case.params
    if p['scoring_func']!='sigmoid':
        raise UnsupportedCase('MoeFusedTopk documented activation is sigmoid; softmax variant needs a separate native normalized route.')
    binding=GraphCall('ops-transformer')
    def run(gating_output,correction_bias=None):
        x=gating_output.float().contiguous()
        bias=torch.zeros(x.shape[-1],dtype=torch.float32,device=x.device) if correction_bias is None else correction_bias.float().contiguous()
        shape=(x.shape[0],p['top_k']);weights=torch.empty(shape,dtype=torch.float32,device=x.device);ids=torch.empty(shape,dtype=torch.int32,device=x.device)
        binding.call('MoeFusedTopk',[x,bias,None,None],[weights,ids],{'group_num':p['num_expert_group'],'group_topk':p['topk_group'],'top_n':2 if correction_bias is not None else 1,'top_k':p['top_k'],'activate_type':0,'is_norm':p['renormalize'],'scale':float(p['routed_scaling_factor']),'enable_expert_mapping':False})
        return weights,ids
    return _candidate(binding,run,'MoeFusedTopk','moe/moe_fused_topk','Exact FP32 widening, native sigmoid and group-top-k flags, original renormalization/scaling; top_n follows unbiased max versus biased top2-sum contract.')


def _gemm(binding,a,b,out_dtype):
    import torch
    if not a.shape[0] or not a.shape[1]:
        return torch.zeros((a.shape[0],b.shape[1]),dtype=out_dtype,device=a.device)
    out=torch.zeros((1,a.shape[0],b.shape[1]),dtype=torch.float32,device=a.device)
    group=torch.full((1,),a.shape[1],dtype=torch.int64,device=a.device)
    binding.call('GroupedMatmulAdd',[a.T.contiguous(),b.contiguous(),group,out],[out],{'transpose_x':True,'transpose_weight':False,'group_type':2,'group_list_type':0})
    return out[0].to(out_dtype)


def reduction(case):
    import torch,math
    from benchmarks.ascend_graph_call import GraphCall
    from benchmarks.ascend_baselines import UnsupportedCase
    x=case.inputs[0];p=case.params;mean=case._call.signature.name=='MeanFwdOp'
    if x.dtype not in (torch.float16,torch.bfloat16):raise UnsupportedCase('GroupedMatmulAdd has no exact FP32/integer input template.')
    ds=p['dim'];ds=range(x.ndim) if ds is None else ((ds,) if isinstance(ds,int) else ds)
    dims=sorted(d%x.ndim for d in ds);kept=[i for i in range(x.ndim) if i not in dims]
    width=math.prod(x.shape[d] for d in dims);factor=1/width if mean else 1.
    if mean and width&(width-1):raise UnsupportedCase('Native low-precision GEMM coefficient must represent 1/N exactly; non-power-of-two mean is not lossless.')
    binding=GraphCall('ops-transformer')
    def run(x):
        packed=x.permute(kept+dims).contiguous().reshape(-1,width)
        weight=torch.full((width,1),factor,dtype=x.dtype,device=x.device)
        dtype=getattr(torch,p['dtype']) if p.get('dtype') else x.dtype
        out=_gemm(binding,packed,weight,dtype)
        shape=[1 if i in dims else s for i,s in enumerate(x.shape)] if p['keepdim'] else [x.shape[i] for i in kept]
        return out.reshape(shape)
    return _candidate(binding,run,'GroupedMatmulAdd','gmm/grouped_matmul_add','Pack reduction axes into K; original GEMM with exactly representable unit or 1/N constant column; native FP32 accumulation, restore keepdim/dtype.')


def grouped_gemm(case):
    import torch
    from benchmarks.ascend_graph_call import GraphCall
    from benchmarks.ascend_baselines import UnsupportedCase
    p=case.params;moe=case._call.signature.name=='MoEGroupedGemmFwdOp'
    if case.inputs[0].dtype not in (torch.float16,torch.bfloat16):raise UnsupportedCase('GroupedMatmulAdd selects original FP16/BF16 inputs.')
    if moe and p.get('activation') is not None:raise UnsupportedCase('Native GroupedMatmulAdd has no gated activation epilogue.')
    binding=GraphCall('ops-transformer')
    def run(a,b,metadata,offsets=None):
        sizes=metadata.tolist();out_dtype=getattr(torch,p['out_dtype']) if p.get('out_dtype') else a.dtype
        if not moe:
            outputs=[];start=0
            for i,n in enumerate(sizes):
                end=start+n
                if p['transpose_a']:
                    lhs=a[start:end].T;rhs=b[:,start:end].T if p['transpose_b'] else b[start:end]
                else:
                    lhs=a[start:end];rhs=b[i].T if p['transpose_b'] else b[i]
                outputs.append(_gemm(binding,lhs,rhs,out_dtype));start=end
            return torch.stack(outputs) if p['transpose_a'] else torch.cat(outputs)
        layout=p['layout'];kind=next(iter(layout));layout=layout[kind]
        out=torch.zeros((*a.shape[:-1],b.shape[1]),dtype=out_dtype,device=a.device)
        if kind=='masked':
            for i,n in enumerate(sizes):
                if n:out[i,:n]=_gemm(binding,a[i,:n],b[i].T,out_dtype)
        elif layout['metadata_kind']=='per_row':
            for i in range(b.shape[0]):
                indices=(metadata==i).nonzero().flatten()
                if indices.numel():out[indices]=_gemm(binding,a[indices],b[i].T,out_dtype)
        else:
            start=0
            for i,end in enumerate(sizes):
                align=layout['alignment'];start=-(-start//align)*align
                if end>start:out[start:end]=_gemm(binding,a[start:end],b[i].T,out_dtype)
                start=end
        return out
    return _candidate(binding,run,'GroupedMatmulAdd','gmm/grouped_matmul_add','Per-call original group metadata selects independent slices; call original GEMM for each group, assemble outputs and preserve zero padding.',host=True)


def rope_coefficients(case,x,position_ids=None):
    from workloads import rope as tables
    p=case.params;op=case._call.signature.name
    dim=p.get('rotary_dim') or x.shape[-1]
    if position_ids is not None:
        cos,sin=tables.rope_frequency_tables(dim,p['max_position'],base=p['base'],dtype=x.dtype,device=x.device)
        return cos[position_ids.long()],sin[position_ids.long()]
    fn={'RoPELlama31FwdOp':tables.llama31_frequency_tables,'YaRNFwdOp':tables.yarn_frequency_tables,'LongRoPEFwdOp':tables.longrope_frequency_tables}[op]
    kwargs={k:v for k,v in case._call.params.items() if k not in ('input_layout','rope_layout')}
    if 'rescale_factors' in case._call.specs:
        kwargs['rescale_factors']=case.workload.tensors['rescale_factors']
    return fn(dim,x.shape[0] if x.ndim==2 else x.shape[1],dtype=x.dtype,device=x.device,**kwargs)


def rope_variant(case):
    import torch
    from benchmarks.ascend_graph_call import GraphCall
    from benchmarks.ascend_baselines import source_hashes,WORKSPACE
    from .hashing import source_hashes
    from dataclasses import replace
    binding=GraphCall('ops-transformer')
    def run(x,position_ids=None):
        cos,sin=rope_coefficients(case,x,position_ids)
        dim=case.params.get('rotary_dim') or x.shape[-1]
        shape=(1,x.shape[0],x.shape[1],dim) if position_ids is not None else ((1,x.shape[0],1,dim) if x.ndim==2 else x.shape)
        inp=x[...,:dim].float().contiguous().reshape(shape)
        interleaved=case.params.get('rope_layout')=='interleaved'
        if interleaved:inp=inp.reshape(*inp.shape[:-1],dim//2,2).transpose(-1,-2).contiguous().reshape(shape)
        cs=[torch.cat((t,t),-1).float().reshape(1,t.shape[0],1,dim) for t in (cos,sin)]
        out=torch.empty_like(inp)
        binding.call('RotaryPositionEmbedding',[inp,*cs,None],[out],{'mode':0})
        result=out.reshape(*x.shape[:-1],dim)
        if interleaved:result=result.reshape(*result.shape[:-1],2,dim//2).transpose(-1,-2).contiguous().reshape(*x.shape[:-1],dim)
        result=result.to(x.dtype)
        return result if dim==x.shape[-1] else torch.cat((result,x[...,dim:]),-1)
    result=_candidate(binding,run,'RotaryPositionEmbedding','posembedding/rotary_position_embedding','Manifest frequency-table construction and per-call position gather; original native NeoX rotation using exact FP32 widening, optional even/odd layout packing, partial-width crop/concatenate. Table construction is included in run.',host=case._call.signature.name=='RoPELlama31FwdOp')
    return replace(result,source_sha256=result.source_sha256 | source_hashes(WORKSPACE/'TileOPs/workloads/rope.py'))


def coupling(case):
    import torch
    from benchmarks.ascend_graph_call import GraphCall
    from benchmarks.ascend_baselines import UnsupportedCase
    if case.inputs[0].dtype not in (torch.float16,torch.bfloat16):raise UnsupportedCase('Native coupling GEMM requires FP16/BF16; no FP32 narrowing.')
    binding=GraphCall('ops-transformer');chunk=case.params['chunk_len']
    def run(c,b):
        batch,s,groups,d=c.shape;chunks=s//chunk
        cp=c.reshape(batch,chunks,chunk,groups,d).transpose(2,3).contiguous().reshape(-1,chunk,d)
        bp=b.reshape(batch,chunks,chunk,groups,d).transpose(2,3).contiguous().reshape(-1,chunk,d)
        outputs=[]
        for lhs,rhs in zip(cp,bp,strict=True):
            dense=_gemm(binding,lhs,rhs.T,c.dtype)
            out=torch.zeros_like(dense)
            for row in range(chunk):out[row,:row+1].copy_(dense[row,:row+1])
            outputs.append(out)
        return torch.stack(outputs).reshape(batch,chunks,groups,chunk,chunk)
    return _candidate(binding,run,'GroupedMatmulAdd','gmm/grouped_matmul_add','Chunk/group layout packing, original matrix product, causal row-prefix output cropping into a zero-padded output. Full multiplication and every copy are timed.')


def gqa_backward(case):
    import torch
    from benchmarks.ascend_graph_call import GraphCall as ForwardGraph
    from .native import GraphCall
    from benchmarks.ascend_baselines import source_hashes,LIBRARIES
    from .hashing import source_hashes
    from dataclasses import replace
    forward=ForwardGraph('ops-transformer');binding=GraphCall('ops-transformer')
    def run(q,k,v,o,grad_output,lse):
        b,s,h,d=q.shape
        mask=torch.ones((s,s),dtype=torch.bool,device=q.device).triu(1) if case.params['is_causal'] else None
        maxima=torch.empty((b,h,s,8),dtype=torch.float32,device=q.device);sums=torch.empty_like(maxima)
        unused=torch.empty((0,0,0,0),dtype=q.dtype,device=q.device);out=torch.empty_like(q)
        attrs={'scale_value':float(d**-.5),'keep_prob':1.,'pre_tockens':2147483647,'next_tockens':2147483647,'head_num':h,'input_layout':'BSND','inner_precise':0,'sparse_mode':0}
        fi=[q.contiguous(),k.contiguous(),v.contiguous(),*([None]*16)];fi[6]=mask
        forward.call('FlashAttentionScore',fi,[maxima,sums,unused,out],attrs)
        gi=[q.contiguous(),k.contiguous(),v.contiguous(),grad_output.contiguous(),None,None,None,mask,maxima,sums,None,out,*([None]*15)]
        dq,dk,dv=[torch.empty_like(t) for t in (q,k,v)]
        # Original op_def requires all seven output descriptors, even when
        # optional RoPE/sink inputs are absent and their outputs are empty.
        empty=torch.empty((0,),dtype=q.dtype,device=q.device)
        empty_fp32=torch.empty((0,),dtype=torch.float32,device=q.device)
        binding.call('FlashAttentionScoreGrad',gi,[dq,dk,dv,empty,empty,empty,empty_fp32],attrs|{'pse_type':1})
        return dq,dk,dv
    result=_candidate(binding,run,'FlashAttentionScoreGrad','attention/flash_attention_score_grad','Original forward regenerates native softmax statistics on every invocation, then original backward computes all Q/K/V gradients; includes forward work and mask packing.')
    hashes=source_hashes(*[p for p in (LIBRARIES/'ops-transformer/attention/flash_attention_score').rglob('*') if p.suffix in ('.h','.cpp','.py','.json')])
    def cleanup():binding.cleanup();forward.cleanup()
    return replace(result,cleanup=cleanup,source_sha256=result.source_sha256|hashes)


def chunk_delta(case):
    import torch
    from .native import GraphCall
    from benchmarks.ascend_baselines import UnsupportedCase
    p=case.params;op=case._call.signature.name;chunk=op=='DeltaNetChunkFwdOp';gdn=op=='GDNFwdOp'
    if case.inputs[0].dtype!=torch.bfloat16:raise UnsupportedCase('ChunkGatedDeltaRule original query/key/value/beta are BF16.')
    if any(p.get(k,False) for k in ('use_qk_l2norm_in_kernel','use_gate_in_kernel','use_beta_sigmoid_in_kernel','allow_neg_eigval')):
        raise UnsupportedCase('Original ChunkGatedDeltaRule has no QK L2, raw-gate, beta sigmoid, or negative-eigenvalue transform API; computing these in torch would replace target mathematics.')
    binding=GraphCall('ops-transformer')
    def run(*inputs):
        if chunk:
            q,k,v,beta=inputs;q,k,v=[t.transpose(1,2).contiguous() for t in (q,k,v)];beta=beta.transpose(1,2).contiguous();g=None;initial=None;offsets=None;scale=1.
        elif gdn:
            q,k,v,g,beta,initial,offsets,offsets_cpu,a_log,dt_bias=inputs;scale=p.get('scale') or q.shape[-1]**-.5
        else:
            q,k,v,beta,initial,offsets,offsets_cpu=inputs;g=None;scale=p.get('scale') or q.shape[-1]**-.5
        batch,s,heads,dk=q.shape;hv,dv=v.shape[-2:]
        if offsets is not None:
            bounds=offsets.tolist();lengths=torch.tensor([b-a for a,b in zip(bounds[:-1],bounds[1:],strict=True)],dtype=torch.int32,device=q.device);sequences=len(bounds)-1
        else:
            lengths=torch.full((batch,),s,dtype=torch.int32,device=q.device);sequences=batch
        st=torch.zeros((sequences,hv,dv,dk),dtype=torch.float32,device=q.device) if initial is None else (initial if p.get('state_v_first') else initial.transpose(-2,-1)).contiguous()
        out=torch.empty((batch*s,hv,dv),dtype=v.dtype,device=q.device);final=torch.empty_like(st)
        binding.call('ChunkGatedDeltaRule',[q.reshape(-1,heads,dk).contiguous(),k.reshape(-1,heads,dk).contiguous(),v.reshape(-1,hv,dv).contiguous(),beta.reshape(-1,hv).contiguous(),st,lengths,None if g is None else g.reshape(-1,hv).float().contiguous()],[out,final],{'scale_value':float(scale)})
        out=out.reshape(batch,s,hv,dv)
        if chunk:return out.transpose(1,2).contiguous()
        return out,final if p.get('state_v_first') else final.transpose(-2,-1).contiguous()
    return _candidate(binding,run,'ChunkGatedDeltaRule','attention/chunk_gated_delta_rule','Use original ungated/null-g or preactivated log-g interface; pack token-major inputs and sequence-length metadata, exact state transpose; preserve all final-state outputs.',host=True)


def cache_gather(case):
    import torch
    from .native import GraphCall
    from benchmarks.ascend_baselines import UnsupportedCase
    dst,cache,_,_,_,scale=case.inputs
    if scale is not None or dst.dtype!=cache.dtype:raise UnsupportedCase('GatherPaKvCache copies bits; manifest scaled FP8 dequantization is not supplied by this original API.')
    binding=GraphCall('ops-transformer')
    def run(dst,cache,table,cuseq,starts=None,scale=None):
        out=torch.empty_like(dst);spare=torch.empty_like(dst)
        # Index tensor integer widening is exact and all conversions run per call.
        binding.call('GatherPaKvCache',[cache.contiguous(),cache.contiguous(),table.long().contiguous(),cuseq.long().contiguous(),out,spare,None if starts is None else starts.long().contiguous()],[out,spare],{'cache_mode':'Norm','is_seq_lens_cumsum':True})
        return out
    return _candidate(binding,run,'GatherPaKvCache','attention/gather_pa_kv_cache','Original bitwise gather with supplied cumulative lengths/start offsets; duplicate cache input only to satisfy K/V public API; select the full destination output.')


def causal_conv(case):
    import torch
    from .native import GraphCall
    from benchmarks.ascend_baselines import UnsupportedCase
    if case.inputs[0].dtype not in (torch.float16,torch.bfloat16):
        raise UnsupportedCase('CausalConv1d original templates require FP16/BF16 inputs.')
    binding=GraphCall('ops-transformer','causalpairs_transformer')
    update=case._call.signature.name=='CausalConv1dUpdateFwdOp'
    def run(x,weight,conv_state,*args):
        if update:
            bias=args[0] if args else None;offsets=None;flags=None
            packed=x[:,None,:].contiguous()
        else:
            offsets,bias,has_initial=args
            offsets=offsets.int().contiguous()
            flags=torch.zeros(conv_state.shape[0],dtype=torch.int32,device=x.device) if has_initial is None else has_initial.int().contiguous()
            packed=x.T.contiguous()
        history=conv_state[...,-(weight.shape[-1]-1):] if update else conv_state
        state=history.transpose(1,2).contiguous().clone()
        out=torch.empty_like(packed)
        binding.call('CausalConv1d',[packed,weight.T.contiguous(),state,None if bias is None else bias.contiguous(),offsets,None,flags,None],[state,out],
                     {'activation_mode':case.params['activation'] or 'none','null_block_id':-1})
        final_state=torch.cat((conv_state[...,1:],x[...,None]),-1) if update else state.transpose(1,2).contiguous()
        return (out[:,0,:] if update else out.T.contiguous()),final_state
    return _candidate(binding,run,'CausalConv1d','mamba/causal_conv1d','Original prefill/decode API, token/channel transpose and private state clone per call; original activation flag and complete history output preserved.')


def recurrent_family(case):
    import torch
    from .native import GraphCall
    from benchmarks.ascend_baselines import UnsupportedCase
    p=case.params;op=case._call.signature.name
    if case.inputs[0].dtype!=torch.bfloat16:
        raise UnsupportedCase('RecurrentKda native Q/K/V are BF16; FP16 cannot be narrowed losslessly.')
    if p.get('use_qk_l2norm_in_kernel'):
        raise UnsupportedCase('Original RecurrentKda NormalizeRows omits the manifest QK normalization epsilon=1e-6; no kernel math edit or torch normalization is allowed.')
    binding=GraphCall('ops-transformer','kdapairs_transformer')
    def run(*inputs):
        chunk=op=='DeltaNetChunkFwdOp';decode=op=='DeltaNetRecurrentFwdOp'
        if chunk:
            q,k,v,beta=inputs;q,k,v=[t.transpose(1,2) for t in (q,k,v)];beta=beta.transpose(1,2)
            g=initial=offsets=a_log=dt_bias=None;scale=1.
        elif decode:
            q,k,v,beta,initial=inputs;q,k,v=[t[:,None] for t in (q,k,v)];beta=beta[:,None]
            g=offsets=a_log=dt_bias=None;scale=1.
        elif op=='DeltaNetFwdOp':
            q,k,v,beta,initial,offsets,_=inputs;g=a_log=dt_bias=None;scale=p.get('scale') or q.shape[-1]**-.5
        else:
            q,k,v,g,beta,initial,offsets,_,a_log,dt_bias=inputs;scale=p.get('scale') or q.shape[-1]**-.5
        batch,length,heads,dk=q.shape;hv,dv=v.shape[-2:]
        original_dk,original_dv=dk,dv
        if dk > 128 or dv > 256:
            raise UnsupportedCase('Original RecurrentKda requires K=128 and V=128/256; larger dimensions cannot be cropped losslessly.')
        native_dv=128 if dv<=128 else 256
        if dk!=128 or dv!=native_dv:
            q=torch.nn.functional.pad(q,(0,128-dk));k=torch.nn.functional.pad(k,(0,128-dk))
            v=torch.nn.functional.pad(v,(0,native_dv-dv))
            if g is not None and g.ndim==4:g=torch.nn.functional.pad(g,(0,128-dk))
            if initial is not None:
                widths=(0,128-dk,0,native_dv-dv) if p.get('state_v_first') else (0,native_dv-dv,0,128-dk)
                initial=torch.nn.functional.pad(initial,widths)
            if dt_bias is not None and dt_bias.numel()==hv*dk:
                dt_bias=torch.nn.functional.pad(dt_bias.reshape(hv,dk),(0,128-dk)).flatten()
            dk,dv=128,native_dv
        if offsets is None:spans=[(i,0,length) for i in range(batch)]
        else:
            bounds=offsets.tolist();spans=[(0,a,b) for a,b in zip(bounds[:-1],bounds[1:],strict=True)]
        state_shape=(len(spans),hv,dv,dk) if p.get('state_v_first') else (len(spans),hv,dk,dv)
        states=torch.zeros(state_shape,dtype=torch.float32,device=q.device) if initial is None else initial.float().contiguous().clone()
        output=torch.zeros_like(v)
        # RecurrentKda's native MTP limit is eight. Calling it repeatedly carries
        # the original state between ordered token slices without new arithmetic.
        for sequence,(row,begin,end) in enumerate(spans):
            state=states[sequence:sequence+1].contiguous()
            for start in range(begin,end,8):
                stop=min(start+8,end);qs=q[row:row+1,start:stop].contiguous();ks=k[row:row+1,start:stop].contiguous();vs=v[row:row+1,start:stop].contiguous()
                gate=torch.zeros((1,stop-start,hv,dk),dtype=torch.float32,device=q.device) if g is None else g[row:row+1,start:stop]
                bias=dt_bias
                if gate.ndim==3:
                    gate=gate[...,None].expand(-1,-1,-1,dk)
                    if bias is not None:bias=bias[:,None].expand(-1,dk).reshape(-1)
                out=torch.empty_like(vs);final=torch.empty_like(state)
                binding.call('RecurrentKda',[qs,ks,vs,gate.contiguous(),beta[row:row+1,start:stop].contiguous(),state,None,None,a_log,None if bias is None else bias.contiguous(),None],[out,state,final],
                    {'layout':'BSND','scale':float(scale),'output_final_state':True,'inplace_final_state':False,'use_qk_l2norm_in_kernel':bool(p.get('use_qk_l2norm_in_kernel',False)),
                     'use_gate_in_kernel':bool(p.get('use_gate_in_kernel',False)),'use_beta_sigmoid_in_kernel':bool(p.get('use_beta_sigmoid_in_kernel',False)),
                     'allow_neg_eigval':bool(p.get('allow_neg_eigval',False)),'safe_gate':p.get('lower_bound') is not None,'lower_bound':float(p.get('lower_bound') or -5.),'state_v_first':bool(p.get('state_v_first',False))})
                output[row:row+1,start:stop].copy_(out);state=final
            states[sequence:sequence+1].copy_(state)
        output=output[...,:original_dv].contiguous()
        states=(states[...,:original_dv,:original_dk] if p.get('state_v_first') else states[...,:original_dk,:original_dv]).contiguous()
        if chunk:return output.transpose(1,2).contiguous()
        if decode:return output[:,0],states.to(initial.dtype)
        return output,states
    return _candidate(binding,run,'RecurrentKda','attention/recurrent_kda','Original native QK normalization/raw gate/beta transforms; broadcast scalar GDN gate along K and use zero gates for DeltaNet. Zero-pad K/V/state to native 128/128-or-256 and crop, retaining the original unpadded attention scale. Sequential slices of at most eight tokens carry complete original FP32 state. All packing, state copies and metadata reads are timed.',host=True)


def channel_norm(case):
    import math,torch
    from benchmarks.ascend_baselines import UnsupportedCase
    from benchmarks.ascend_graph_call import GraphCall
    p=case.params;instance=case._call.signature.name=='InstanceNormFwdOp'
    x=case.inputs[0];groups=x.shape[1] if instance else p['num_groups'];spatial=math.prod(x.shape[2:]);width=x.shape[1]//groups*spatial
    if width>1024 or width%2:
        raise UnsupportedCase('NormRopeConcat requires complete normalized width even and <=1024; splitting the reduction changes the contract.')
    if instance and p['use_input_stats'] and case.inputs[1] is not None:
        raise UnsupportedCase('Original NormRopeConcat does not expose the complete running-mean/running-variance output contract.')
    if instance and not p['use_input_stats']:
        raise UnsupportedCase('Original NormRopeConcat computes input statistics and cannot select supplied running mean/variance.')
    binding=GraphCall('ops-transformer')
    def run(x,*rest):
        weight,bias=rest[-2:];values=[];channels=x.shape[1]//groups
        # Select the original FP32 template with lossless input widening for
        # affine normalization; avoid an extra native half rounding stage.
        compute_dtype=torch.float32 if weight is not None or bias is not None else x.dtype
        for g in range(groups):
            inp=x[:,g*channels:(g+1)*channels].to(compute_dtype).contiguous().reshape(1,x.shape[0],1,width)
            gamma=torch.ones(width,dtype=compute_dtype,device=x.device) if weight is None else weight[g*channels:(g+1)*channels,None].to(compute_dtype).expand(-1,spatial).contiguous().flatten()
            beta=torch.zeros_like(gamma) if bias is None else bias[g*channels:(g+1)*channels,None].to(compute_dtype).expand(-1,spatial).contiguous().flatten()
            encoder=torch.empty((1,0,1,width),dtype=compute_dtype,device=x.device);rope=torch.empty((0,width),dtype=compute_dtype,device=x.device)
            one,zero=torch.ones_like(gamma),torch.zeros_like(gamma);out=[torch.empty((1,1,x.shape[0],width),dtype=compute_dtype,device=x.device) for _ in range(3)]
            binding.call('NormRopeConcat',[inp,inp,inp,encoder,encoder,encoder,gamma,beta,gamma,beta,one,zero,one,zero,rope,rope],out,{'norm_type':2,'norm_added_type':0,'rope_type':0,'concat_order':0,'eps':float(p['eps']),'is_training':False})
            values.append(out[0].reshape(x.shape[0],channels,*x.shape[2:]))
        return torch.cat(values,1).to(x.dtype)
    return _candidate(binding,run,'NormRopeConcat','posembedding/norm_rope_concat','Group/instance normalization by complete per-group layer normalization; repeat channel affine parameters over spatial coordinates, original native mean/variance and epsilon, concatenate groups. Affine mode selects the original FP32 template by lossless input/parameter widening with a final output cast, all inside run.')


def gelu_tanh_mul(case):
    import json,torch
    from benchmarks.ascend_baselines import Candidate,UnsupportedCase,WORKSPACE,source_hashes
    from .hashing import source_hashes
    from benchmarks.api import Implementation
    if case.inputs[0].dtype!=torch.float16:
        raise UnsupportedCase('Original BiasGateGelu exposes FP16 pointer arithmetic only.')
    global _gelu_loaded,_gelu_schema
    source=WORKSPACE/'ascend_baselines/ops-transformer/experimental/moe/biasgategelu/BIAS_GATE_GELU.cpp'
    binary=WORKSPACE/'.cache/all-pairs/ops_transformer/gelu_build/liboriginal_bias_gate_gelu.so'
    receipt=WORKSPACE/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_transformer/gelu_build.json'
    hashes=source_hashes(source,binary,receipt,Path(__file__))
    build=json.loads(receipt.read_text())
    if hashes[str(source.relative_to(WORKSPACE))]!=build['source_sha256'] or hashes[str(binary.relative_to(WORKSPACE))]!=build['binary_sha256']:
        raise RuntimeError('Original BiasGateGelu source/binary does not match isolated build receipt.')
    if not _gelu_loaded:
        if hasattr(torch.ops.ascend_ops,'bias_gate_gelu'):raise RuntimeError('Another BiasGateGelu implementation already registered.')
        _gelu_schema=torch.library.Library('ascend_ops','FRAGMENT')
        _gelu_schema.define('bias_gate_gelu(int block_dim, Tensor input, Tensor bias, Tensor(a!) output, int rows, int columns) -> int')
        torch.ops.load_library(str(binary))
        table=torch._C._dispatch_dump_table('ascend_ops::bias_gate_gelu')
        if not any(line.startswith('PrivateUse1:') and str(source) in line for line in table.splitlines()):raise RuntimeError('BiasGateGelu does not resolve to original source.')
        _gelu_loaded=True
    def run(x):
        inp=x.reshape(-1,x.shape[-1]).contiguous();out=torch.empty((inp.shape[0],inp.shape[1]//2),device=x.device,dtype=x.dtype)
        bias=torch.zeros(inp.shape[1],device=x.device,dtype=x.dtype)
        rc=torch.ops.ascend_ops.bias_gate_gelu(min(40,inp.shape[0]),inp,bias,out,*inp.shape)
        if rc:raise RuntimeError('Original BiasGateGelu failed: '+str(rc))
        DISPATCH_EVENTS.append({'op':'ascend_ops::bias_gate_gelu','source':str(source),'binary':str(binary),'dispatch_owner_checked':True,'returned':rc})
        return out.reshape(*x.shape[:-1],x.shape[-1]//2)
    return Candidate(Implementation(run),'ops-transformer::bias_gate_gelu','Original complete fused BiasGateGelu source; FP16 zero bias and row packing. Deployment retarget/header preinclude only; original math and --cce-auto-sync preserved.',hashes)


def silu(case):
    import torch
    from .native import GraphCall
    from benchmarks.ascend_baselines import UnsupportedCase
    if case.inputs[0].dtype not in (torch.float16,torch.bfloat16):raise UnsupportedCase('Original CausalConv1d SiLU templates are FP16/BF16 only.')
    binding=GraphCall('ops-transformer','causalpairs_transformer')
    def run(x):
        flat=x.contiguous().flatten();width=1024;count=flat.numel();flat=torch.nn.functional.pad(flat,(0,(-count)%width)).reshape(-1,width)
        weight=torch.stack((torch.zeros(width,dtype=x.dtype,device=x.device),torch.ones(width,dtype=x.dtype,device=x.device)))
        results=[]
        for start in range(0,flat.shape[0],1024):
            inp=flat[start:start+1024,None,:].contiguous();state=torch.zeros_like(inp);out=torch.empty_like(inp)
            binding.call('CausalConv1d',[inp,weight,state,None,None,None,None,None],[state,out],{'activation_mode':'silu','null_block_id':-1})
            results.append(out.flatten())
        return torch.cat(results)[:count].reshape_as(x)
    return _candidate(binding,run,'CausalConv1d','mamba/causal_conv1d','Select original native SiLU after identity depthwise convolution (zero history and [0,1] weights); independent one-token rows, pad/crop, no activation arithmetic in adapter.')


def quant_gemm(case):
    import torch
    from .native import GraphCall
    from benchmarks.ascend_baselines import UnsupportedCase
    op=case._call.signature.name;p=case.params;batched=op=='BmmFP8FwdOp'
    a,b,sa,sb,*extra=case.inputs
    m=a.shape[-2];n=b.shape[-1] if batched and not p['trans_b'] else b.shape[-2]
    if sa.numel() not in (1,m) or sb.numel() not in (1,n):
        raise UnsupportedCase('This original GroupedMatmul route selects T-T/K-C scale shapes; blockwise scale packing is separately unresolved, never replaced by torch dequantization.')
    if extra and extra[0] is not None and op!='GemmINT8W8A8FwdOp':
        raise UnsupportedCase('Original FP8 GroupedMatmul dynamic quantization mode requires null bias.')
    binding=GraphCall('ops-transformer')
    def run(a,b,sa,sb,bias=None):
        outputs=[]
        pairs=zip(a,b,strict=True) if batched else [(a,b)]
        for lhs,rhs in pairs:
            w=rhs if batched and not p['trans_b'] else rhs.T
            scale_a=sa.reshape(-1).expand(lhs.shape[0]).contiguous() if sa.numel()==1 else sa.flatten().contiguous()
            scale_b=sb.reshape(-1).expand(w.shape[1]).contiguous().reshape(1,-1) if sb.numel()==1 else sb.contiguous().reshape(1,-1)
            groups=torch.full((1,),lhs.shape[0],dtype=torch.int64,device=a.device)
            out=torch.empty((lhs.shape[0],w.shape[1]),dtype=getattr(torch,p['out_dtype']),device=a.device)
            binding.call('GroupedMatmul',[lhs.contiguous(),w.contiguous()[None],None if bias is None else bias.contiguous().reshape(1,-1),scale_b,None,None,None,groups,scale_a],[out],
                {'split_item':3,'transpose_weight':False,'transpose_x':False,'group_type':0,'group_list_type':0,'act_type':0,'tuning_config':(0,)},
                names=['x0','weight0','bias0','scale0','offset0','antiquant_scale0','antiquant_offset0','group_list','per_token_scale','y0'])
            outputs.append(out)
        return torch.stack(outputs) if batched else outputs[0]
    return _candidate(binding,run,'GroupedMatmul','gmm/grouped_matmul','Original Ascend950 FP8/INT8 dynamic quantization, one group, exact tensor/token/channel scale expansion and weight transpose; complete native scaled dot product and optional original bias.')


def decode_attention(case):
    import torch
    from benchmarks.ascend_graph_call import GraphCall
    from benchmarks.ascend_baselines import UnsupportedCase
    if case.inputs[0].dtype not in (torch.float16,torch.bfloat16):raise UnsupportedCase('This original FlashAttentionScore route requires FP16/BF16 inputs.')
    binding=GraphCall('ops-transformer');mla=case._call.signature.name=='MLADecodeWithKVCacheFwdOp'
    def run(*inputs):
        if mla:
            q,qpe,k,kpe=inputs
            # Original arch35 accepts D_qk <=768 and D_v <=D_qk.
            return _attention(binding,torch.cat((q,qpe),-1)[:,None],torch.cat((k,kpe),-1),k,{'sm_scale':(q.shape[-1]+qpe.shape[-1])**-.5})[:,0]
        q,k,v,lens,table=inputs;page=case.params['page_size'];outputs=[]
        keys=k.reshape(-1,page,*k.shape[-2:]);values=v.reshape(-1,page,*v.shape[-2:])
        for i,length in enumerate(lens.tolist()):
            pages=table[i,:-(-length//page)].long()
            outputs.append(_attention(binding,q[i:i+1],keys[pages].flatten(0,1)[:length][None],values[pages].flatten(0,1)[:length][None],case.params))
        return torch.cat(outputs)
    return _candidate(binding,run,'FlashAttentionScore','attention/flash_attention_score','MLA concatenates original non-positional and positional Q/K channels with unmodified value cache; paged MHA gathers logical sequence pages per call. Original arch35 unequal QK/value head dimensions, complete output.',host=not mla)


def dsa(case):
    import torch
    from dataclasses import replace
    from benchmarks.ascend_graph_call import GraphCall
    from benchmarks.ascend_baselines import source_hashes,WORKSPACE,UnsupportedCase
    from .hashing import source_hashes
    if case.inputs[0].dtype not in (torch.float16,torch.bfloat16):raise UnsupportedCase('Original FlashAttentionScore dtype templates require FP16/BF16.')
    binding=GraphCall('ops-transformer');p=case.params
    def run(q,kv,indices):
        # The workload explicitly exposes its selection-mask helper for baseline
        # metadata construction. All attention arithmetic is the native API.
        selected=case.workload.selection_mask(indices)
        batch,s,h,dk=q.shape;g=kv.shape[2];dv=dk-p['dim_tail'];group=h//g;outputs=[]
        for b in range(batch):
            by_head=[]
            for head in range(g):
                by_head.append(_attention(binding,q[b:b+1,:,head*group:(head+1)*group],kv[b:b+1,:,head:head+1],kv[b:b+1,:,head:head+1,:dv],{'sm_scale':p.get('sm_scale') or dk**-.5},~selected[b,head]))
            outputs.append(torch.cat(by_head,2))
        return torch.cat(outputs)
    result=_candidate(binding,run,'FlashAttentionScore','attention/flash_attention_score','Construct original documented sparse/compressed-causal selection mask from current route indices, run original full attention per KV-head group; preserve complete output. Mask construction and group copies are timed.')
    return replace(result,source_sha256=result.source_sha256|source_hashes(WORKSPACE/'TileOPs/workloads/attention/dsa.py'))


_state_passing_schema=None
_state_passing_loaded=False

def state_passing(case):
    import json,torch
    from benchmarks.ascend_baselines import Candidate,UnsupportedCase,WORKSPACE,source_hashes
    from .hashing import source_hashes
    from benchmarks.api import Implementation
    if case.inputs[0].shape[-1]>8192:raise UnsupportedCase('Original native state vector is padded to supported N=128,P=64; larger vectors require a separately checked original tiling.')
    global _state_passing_schema,_state_passing_loaded
    source=WORKSPACE/'ascend_baselines/ops-transformer/experimental/mamba/mamba2_chunk_state_passing/torch_interface.cpp'
    binary=WORKSPACE/'.cache/all-pairs/ops_transformer/state_passing_build/liboriginal_state_passing.so'
    receipt=WORKSPACE/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_transformer/state_passing_build.json'
    hashes=source_hashes(source.parent,binary,receipt,Path(__file__))
    build=json.loads(receipt.read_text())
    if hashes[str(source.relative_to(WORKSPACE))]!=build['source_sha256'] or hashes[str(binary.relative_to(WORKSPACE))]!=build['binary_sha256']:raise RuntimeError('Original state-passing source/binary mismatch.')
    if not _state_passing_loaded:
        if hasattr(torch.ops.npu_ops_transformer_ext,'mambav2_chunk_state_passing'):raise RuntimeError('Another state-passing implementation is already registered.')
        _state_passing_schema=torch.library.Library('npu_ops_transformer_ext','FRAGMENT')
        _state_passing_schema.define('mambav2_chunk_state_passing(Tensor dacs, Tensor init_states, Tensor states, Tensor ct) -> (Tensor, Tensor)')
        torch.ops.load_library(str(binary))
        table=torch._C._dispatch_dump_table('npu_ops_transformer_ext::mambav2_chunk_state_passing')
        if not any(line.startswith('PrivateUse1:') and str(source) in line for line in table.splitlines()):raise RuntimeError('State-passing dispatch is not original source.')
        _state_passing_loaded=True
    def run(states,da,initial):
        b,c,h,d=states.shape
        # Select the native FP32 final-state output after each chunk. The other
        # native output deliberately projects a half-rounded state and cannot
        # represent the manifest's complete FP32 previous-state output.
        state=torch.zeros((b,h,8192),device=states.device,dtype=torch.float32) if initial is None else torch.nn.functional.pad(initial.float(),(0,8192-d)).contiguous()
        packed=torch.nn.functional.pad(states.float(),(0,8192-d)).contiguous()
        ct=torch.zeros((b,1,64,1,128),dtype=torch.float16,device=states.device)
        previous=[]
        for chunk in range(c):
            previous.append(state[...,:d].clone())
            # The unchanged original frontend reads storage().data(), so a
            # contiguous view with a nonzero storage offset must be copied.
            native_da=da[:,:,chunk:chunk+1].transpose(1,2).clone(memory_format=torch.contiguous_format)
            native_chunk=packed[:,chunk:chunk+1].clone(memory_format=torch.contiguous_format)
            _,final=torch.ops.npu_ops_transformer_ext.mambav2_chunk_state_passing(native_da,state,native_chunk,ct)
            state=final.reshape(b,h,8192)
            DISPATCH_EVENTS.append({'op':'npu_ops_transformer_ext::mambav2_chunk_state_passing','source':str(source),'binary':str(binary),'dispatch_owner_checked':True,'returned':0})
        return torch.stack(previous,1),state[...,:d].contiguous()
    return Candidate(Implementation(run),'ops-transformer::mambav2_chunk_state_passing','Original FP32 state recurrence, one original API call per chunk with zero-padded state. Select FP32 final state and carry it to the next original call; collect complete previous states without arithmetic. Native half-projection output is unused. Original source/math/schedule unchanged.',hashes)


def prefill_append(case):
    import torch
    from benchmarks.ascend_graph_call import GraphCall
    from benchmarks.ascend_baselines import UnsupportedCase
    from .native import GraphCall as CacheGraph
    p=case.params
    if p.get('softcap') or p.get('fuse_rope') or case.inputs[3].dtype!=case.inputs[0].dtype:
        raise UnsupportedCase('Original basic append path does not map fused RoPE, softcap or scaled FP8 cache.')
    if case.inputs[0].dtype not in (torch.float16,torch.bfloat16):raise UnsupportedCase('Original FlashAttentionScore half/BF16 input templates.')
    attention=GraphCall('ops-transformer');cache=CacheGraph('ops-transformer')
    def run(q,kn,vn,kp,vp,ks,vs,cu,lens,table):
        bounds=cu.tolist();lengths=lens.tolist();page=p['page_size'];kc=kp.clone();vc=vp.clone();outputs=[]
        for b,(begin,end) in enumerate(zip(bounds[:-1],bounds[1:],strict=True)):
            old=lengths[b];pos=torch.arange(old+end-begin,device=q.device);rows=table[b,pos//page].long()*page+pos%page
            write=rows[old:].contiguous()
            if end>begin:
                cache.call('ScatterPaKvCache',[kn[begin:end].contiguous(),kc.reshape(-1,page,*kc.shape[1:]),write,vn[begin:end].contiguous(),vc.reshape(-1,page,*vc.shape[1:]),None,None,None],[kc.reshape(-1,page,*kc.shape[1:]),vc.reshape(-1,page,*vc.shape[1:])],{'cache_mode':'Norm','scatter_mode':'None'})
                outputs.append(_attention(attention,q[None,begin:end],kc[rows][None],vc[rows][None],p)[0])
        return torch.cat(outputs),kc,vc
    result=_candidate(attention,run,'FlashAttentionScore+ScatterPaKvCache','attention/flash_attention_score','Per-request physical page indices, private cache clones, original native cache append and original attention over gathered complete cache; complete attention/K/V outputs and all input-dependent transforms are timed.',host=True)
    from dataclasses import replace
    from benchmarks.ascend_baselines import source_hashes,WORKSPACE
    from .hashing import source_hashes
    return replace(result,source_sha256=result.source_sha256|source_hashes(WORKSPACE/'ascend_baselines/ops-transformer/attention/scatter_pa_kv_cache',cache.vendor))


def nsa_selected(case):
    import torch
    from benchmarks.ascend_graph_call import GraphCall
    from benchmarks.ascend_baselines import UnsupportedCase
    if case.inputs[0].dtype not in (torch.float16,torch.bfloat16):raise UnsupportedCase('Native half/BF16 FlashAttentionScore templates.')
    binding=GraphCall('ops-transformer');p=case.params
    def run(q,k,v,indices,counts,offsets,token_indices):
        bounds=offsets.tolist();bs=p['block_size'];group=q.shape[1]//k.shape[1];out=torch.zeros((*q.shape[:-1],v.shape[-1]),dtype=q.dtype,device=q.device)
        slot=torch.arange(indices.shape[-1]*bs,device=q.device)
        for begin,end in zip(bounds[:-1],bounds[1:],strict=True):
            length=end-begin
            for head in range(k.shape[1]):
                pos=(indices[begin:end,head,:,None].long()*bs+torch.arange(bs,device=q.device)).flatten(-2)
                valid=(pos>=0)&(pos<length)&(slot[None,:]//bs<counts[begin:end,head,None])
                if p['is_causal']:valid &= pos<=torch.arange(length,device=q.device)[:,None]
                live=valid.any(-1).nonzero().flatten()
                if live.numel()==0:continue
                select=pos[live].clamp(0,length-1)+begin
                qs=q[begin+live,head*group:(head+1)*group][:,None]
                keys=k[select,head][:,:,None];values=v[select,head][:,:,None]
                # One independent native batch per query; repeated selected blocks
                # stay repeated, preserving the original softmax denominator.
                values_out=_attention(binding,qs,keys,values,{'sm_scale':p.get('scale') or q.shape[-1]**-.5},~valid[live,None,None,:])
                out[begin+live,head*group:(head+1)*group]=values_out[:,0]
        return out
    return _candidate(binding,run,'FlashAttentionScore','attention/flash_attention_score','Gather selected block slots per query/KV head, retaining repeated slots; build causal/count/padding masks from metadata and execute original attention with independent query batches. Rows with no selected key retain zero padding. No floating attention arithmetic outside original API.',host=True)


def nsa_compressed(case):
    import ctypes as C
    import torch
    from .native import GraphCall
    from benchmarks.ascend_math_more import _MathCall
    from benchmarks.ascend_baselines import WORKSPACE,source_hashes
    from .hashing import source_hashes
    from dataclasses import replace
    from benchmarks.api import Implementation
    binding=GraphCall('ops-transformer','attentionpairs_transformer')
    binary=binding.vendor/'op_api/lib/libcust_opapi.so'
    metadata=_MathCall('aclnnFlashAttnMetadata',[C.c_void_p]*4+[C.c_int64]*10+[C.c_char_p]*3+[C.c_bool,C.c_void_p],library='ops-transformer',host_library=binary)
    p=case.params
    def run(q,k,v,offsets,chunk_offsets,token_indices):
        bounds=offsets.tolist();chunks=chunk_offsets.tolist();bs=p['bs'];h=q.shape[1];g=k.shape[1];out=torch.zeros((q.shape[0],h,v.shape[-1]),dtype=q.dtype,device=q.device);lse=torch.zeros(q.shape[:2],dtype=q.dtype,device=q.device)
        for seq,(begin,end) in enumerate(zip(bounds[:-1],bounds[1:],strict=True)):
            # Queries sharing the same count of closed blocks share a prefix.
            # Separate unmasked original calls yield both native O and native LSE.
            for count in range(1,(end-begin)//bs+1):
                start=begin+count*bs-1;stop=min(end,begin+(count+1)*bs-1)
                query=q[None,start:stop].contiguous();key=k[None,chunks[seq]:chunks[seq]+count].contiguous();value=v[None,chunks[seq]:chunks[seq]+count].contiguous()
                # Original metadata example formula with device-capacity upper
                # bounds 72 cube/144 vector cores; actual schedule is generated
                # by the original native metadata API, never by this adapter.
                size=(((72+144)*g+1)*16+412+4095)//4096*4096
                meta=torch.empty(size,dtype=torch.int32,device=q.device);rec=metadata.record();desc=metadata.tensor(meta,rec)
                metadata.call([None,None,None,None,1,stop-start,count,h,g,q.shape[-1],v.shape[-1],0,-1,-1,b'BSND',b'BSND',b'BSND',False,desc],rec,q.device)
                DISPATCH_EVENTS.append({'op':'aclnnFlashAttnMetadata','binary':str(binary),'source':str(WORKSPACE/'ascend_baselines/ops-transformer/attention/flash_attn_metadata/op_host/op_api/aclnn_flash_attn_metadata.cpp'),'dispatch_owner_checked':True,'returned':0})
                o=torch.empty((1,stop-start,h,v.shape[-1]),dtype=q.dtype,device=q.device);l=torch.empty((1,h,stop-start),dtype=torch.float32,device=q.device)
                binding.call('FlashAttn',[query,key,value,None,None,None,None,None,None,None,meta],[o,l],{'softmax_scale':float(p.get('scale') or q.shape[-1]**-.5),'mask_mode':0,'win_left':-1,'win_right':-1,'max_seqlen_q':stop-start,'max_seqlen_kv':count,'layout_q':'BSND','layout_kv':'BSND','layout_out':'BSND','return_softmax_lse':1})
                out[start:stop].copy_(o[0]);lse[start:stop].copy_(l[0].T.to(q.dtype))
        return out,lse
    result=_candidate(binding,run,'FlashAttn+aclnnFlashAttnMetadata','attention/flash_attn','Pack queries by identical closed-block prefix; original native metadata and attention APIs provide complete attention and natural-log LSE outputs. Metadata/packing and dtype output casts are timed; no external attention arithmetic.',host=True)
    cleanup_original=result.cleanup
    def cleanup():
        cleanup_original();metadata.cleanup()
    return replace(result,cleanup=cleanup,source_sha256=result.source_sha256|source_hashes(WORKSPACE/'ascend_baselines/ops-transformer/attention/flash_attn_metadata',binary,WORKSPACE/'TileOPs/benchmarks/ascend_math_more.py'))


def mla_cache_write(case):
    import torch
    from .native import GraphCall
    from benchmarks.ascend_graph_call import GraphCall as RopeGraph
    from benchmarks.ascend_baselines import UnsupportedCase,source_hashes,WORKSPACE
    from .hashing import source_hashes
    from dataclasses import replace
    if case.inputs[2].dtype!=case.inputs[0].dtype:raise UnsupportedCase('Original ScatterPaKvCache does not expose scaled FP8 conversion.')
    binding=GraphCall('ops-transformer');rope=RopeGraph('ops-transformer');p=case.params
    def run(latent,pe,cache,slots,scale=None,positions=None,tables=None):
        live=(slots>=0).nonzero().flatten();selected_pe=pe[live].contiguous()
        if p['fuse_rope']:
            dim=pe.shape[-1];half=dim//2;inp=selected_pe.float().reshape(1,-1,1,dim)
            inter=p['rope_layout']=='interleaved'
            if inter:inp=inp.reshape(1,-1,1,half,2).transpose(-1,-2).contiguous().reshape(1,-1,1,dim)
            cos,sin=tables[positions[live].long()].float().chunk(2,-1)
            coeff=[torch.cat((x,x),-1).reshape(1,-1,1,dim).contiguous() for x in (cos,sin)];rotated=torch.empty_like(inp)
            rope.call('RotaryPositionEmbedding',[inp,*coeff,None],[rotated],{'mode':0})
            if inter:rotated=rotated.reshape(1,-1,1,2,half).transpose(-1,-2).contiguous()
            selected_pe=rotated.reshape(-1,dim).to(pe.dtype)
        values=torch.cat((latent[live],selected_pe),-1)[:,None].contiguous();out=cache.clone();keycache=out[:,:,None,:]
        # Required value side is an independent same-layout throwaway copy.
        spare=keycache.clone()
        if live.numel():binding.call('ScatterPaKvCache',[values,keycache,slots[live].contiguous(),values,spare,None,None,None],[keycache,spare],{'cache_mode':'Norm','scatter_mode':'None'})
        return out
    result=_candidate(binding,run,'ScatterPaKvCache+RotaryPositionEmbedding','attention/scatter_pa_kv_cache','Concatenate latent and optional original-native rotated PE, preserve original cos/sin table dtype via lossless FP32 widening and even/odd layout packing, native cache scatter into a private full cache. Entire cache output retained; every selection/copy/rotation is timed.')
    cleanup_original=result.cleanup
    def cleanup():cleanup_original();rope.cleanup()
    return replace(result,cleanup=cleanup,source_sha256=result.source_sha256|source_hashes(WORKSPACE/'ascend_baselines/ops-transformer/posembedding/rotary_position_embedding',rope.vendor))


def mla_varlen(case):
    import ctypes as C
    import torch
    from .native import GraphCall
    from benchmarks.ascend_math_more import _MathCall
    from benchmarks.ascend_baselines import WORKSPACE,source_hashes
    from .hashing import source_hashes
    from dataclasses import replace
    binding=GraphCall('ops-transformer','attentionpairs_transformer');binary=binding.vendor/'op_api/lib/libcust_opapi.so'
    metadata=_MathCall('aclnnFlashAttnMetadata',[C.c_void_p]*4+[C.c_int64]*10+[C.c_char_p]*3+[C.c_bool,C.c_void_p],library='ops-transformer',host_library=binary)
    p=case.params
    def run(q,kn,kpe,v,cu):
        bounds=cu.tolist();h=q.shape[1];results=[];logsum=[];mode=3 if p['is_causal'] else 0
        mask=(torch.arange(2048,device=q.device)[None,:]>torch.arange(2048,device=q.device)[:,None]).to(torch.int8) if mode else None
        for begin,end in zip(bounds[:-1],bounds[1:],strict=True):
            length=end-begin;query=q[None,begin:end].contiguous();key=torch.cat((kn[begin:end],kpe[begin:end,None,:].expand(-1,h,-1)),-1)[None].contiguous();value=v[None,begin:end].contiguous()
            size=(((72+144)*h+1)*16+412+4095)//4096*4096;meta=torch.empty(size,dtype=torch.int32,device=q.device);rec=metadata.record();desc=metadata.tensor(meta,rec)
            metadata.call([None,None,None,None,1,length,length,h,h,q.shape[-1],v.shape[-1],mode,-1,-1,b'BSND',b'BSND',b'BSND',False,desc],rec,q.device)
            DISPATCH_EVENTS.append({'op':'aclnnFlashAttnMetadata','binary':str(binary),'source':str(WORKSPACE/'ascend_baselines/ops-transformer/attention/flash_attn_metadata/op_host/op_api/aclnn_flash_attn_metadata.cpp'),'dispatch_owner_checked':True,'returned':0})
            out=torch.empty((1,length,h,v.shape[-1]),dtype=q.dtype,device=q.device);lse=torch.empty((1,h,length),dtype=torch.float32,device=q.device)
            binding.call('FlashAttn',[query,key,value,None,None,None,None,None,None,mask,meta],[out,lse],{'softmax_scale':float(p.get('sm_scale') or q.shape[-1]**-.5),'mask_mode':mode,'win_left':-1,'win_right':-1,'max_seqlen_q':length,'max_seqlen_kv':length,'layout_q':'BSND','layout_kv':'BSND','layout_out':'BSND','return_softmax_lse':1})
            results.append(out[0]);logsum.append(lse[0].T.contiguous())
        return torch.cat(results),torch.cat(logsum)
    result=_candidate(binding,run,'FlashAttn+aclnnFlashAttnMetadata','attention/flash_attn','Native D=192,Dv=128 MLA prefill, per-sequence packed NoPE/shared-PE concatenation, original causal mask and original metadata API. Preserve full output and natural-log LSE; all runtime metadata and packing are timed.',host=True)
    cleanup_original=result.cleanup
    def cleanup():cleanup_original();metadata.cleanup()
    return replace(result,cleanup=cleanup,source_sha256=result.source_sha256|source_hashes(WORKSPACE/'ascend_baselines/ops-transformer/attention/flash_attn_metadata',binary,WORKSPACE/'TileOPs/benchmarks/ascend_math_more.py'))


def fused_qk_rope(case):
    import torch
    from benchmarks.ascend_graph_call import GraphCall
    from benchmarks.ascend_baselines import source_hashes,WORKSPACE
    from .hashing import source_hashes
    from dataclasses import replace
    binding=GraphCall('ops-transformer');p=case.params
    def run(qkv,qweight,kweight,tables,positions):
        width=qweight.numel();h=p['num_heads'];g=p['num_kv_heads'];result=qkv.clone().reshape(qkv.shape[0],h+2*g,width)
        cos,sin=tables[positions.long()].float().chunk(2,-1);rotary=cos.shape[-1]*2
        coeff=[torch.cat((x,x),-1)[None,:,None,:].contiguous() for x in (cos,sin)]
        for start,count,weight in ((0,h,qweight),(h,g,kweight)):
            value=result[:,start:start+count].float().contiguous()[None];gamma=weight.float().contiguous();zero=torch.zeros_like(gamma);one=torch.ones_like(gamma)
            encoder=torch.empty((1,0,count,width),dtype=torch.float32,device=qkv.device);rope_empty=torch.empty((0,width),dtype=torch.float32,device=qkv.device)
            outputs=[torch.empty((1,count,qkv.shape[0],width),dtype=torch.float32,device=qkv.device) for _ in range(3)]
            binding.call('NormRopeConcat',[value,value,value,encoder,encoder,encoder,gamma,zero,gamma,zero,one,zero,one,zero,rope_empty,rope_empty],outputs,{'norm_type':4,'norm_added_type':0,'rope_type':0,'concat_order':0,'eps':float(p['eps']),'is_training':False})
            normalized=outputs[0].transpose(1,2).contiguous();inp=normalized[...,:rotary].contiguous();inter=p['rope_layout']=='interleaved'
            if inter:inp=inp.reshape(*inp.shape[:-1],rotary//2,2).transpose(-1,-2).contiguous().reshape(inp.shape)
            rotated=torch.empty_like(inp);binding.call('RotaryPositionEmbedding',[inp,*coeff,None],[rotated],{'mode':0})
            if inter:rotated=rotated.reshape(*rotated.shape[:-1],2,rotary//2).transpose(-1,-2).contiguous().reshape(inp.shape)
            normalized[...,:rotary].copy_(rotated);result[:,start:start+count].copy_(normalized[0].to(qkv.dtype))
        return result.reshape_as(qkv)
    result=_candidate(binding,run,'NormRopeConcat+RotaryPositionEmbedding','posembedding/norm_rope_concat','Losslessly widen Q/K and weights to FP32, original affine RMS then original native RoPE with supplied tables, preserving FP32 intermediate contract; partial rotary crop and exact unchanged V columns. All transformations run per invocation.')
    return replace(result,source_sha256=result.source_sha256|source_hashes(WORKSPACE/'ascend_baselines/ops-transformer/posembedding/rotary_position_embedding'))


def topk_select(case):
    import torch
    from .native import GraphCall
    binding=GraphCall('ops-transformer');topk=case.params['topk']
    def run(scores,starts,ends):
        b,s,n,g=scores.shape;rows=scores.movedim(2,-1).contiguous().reshape(-1,n)
        low=starts[:,:,None].expand(b,s,g).reshape(-1);high=ends[:,:,None].expand(b,s,g).reshape(-1);results=[]
        # Bound scratch and preserve FP32 logits. Native softmax is monotone;
        # all top-k selection arithmetic is performed by the original kernel.
        for begin in range(0,rows.shape[0],128):
            end=min(begin+128,rows.shape[0]);pos=torch.arange(n,device=scores.device)[None]
            valid=(pos>=low[begin:end,None])&(pos<high[begin:end,None]);active=rows[begin:end].masked_fill(~valid,-float('inf'));selected=[]
            for k0 in range(0,topk,1024):
                k=min(1024,topk-k0);values=torch.empty((end-begin,k),dtype=torch.float32,device=scores.device);ids=torch.empty_like(values,dtype=torch.int32);rowids=torch.empty_like(ids)
                binding.call('MoeGatingTopKSoftmax',[active,None],[values,ids,rowids],{'k':k})
                idx=ids.long();inside=valid.gather(1,idx)
                selected.append(torch.where(inside,ids,n))
                valid.scatter_(1,idx,False);active.scatter_(1,idx,-float('inf'))
            results.append(torch.cat(selected,-1))
        return torch.cat(results).reshape(b,s,g,topk)
    return _candidate(binding,run,'MoeGatingTopKSoftmax','moe/moe_gating_top_k_softmax','Per-window metadata masking, original monotone-softmax top-k indices in rounds of <=1024, selected-slot exclusion and exact sentinel padding. Original FP32 values and exact manifest selected-value/uniqueness oracle retained; native rounding can cause a numerical failure and is not excused.')


def _fp8_block_product(binding,a,b,sa,sb,dtype):
    import torch
    m=a.shape[0];n=b.shape[0];out=torch.empty((m,n),dtype=dtype,device=a.device)
    if not m:return out
    groups=torch.full((1,),m,dtype=torch.int64,device=a.device)
    binding.call('GroupedMatmul',[a.contiguous(),b.T.contiguous()[None],None,sb.T.contiguous()[None],None,None,None,groups,sa.contiguous()],[out],{'split_item':3,'transpose_weight':False,'transpose_x':False,'group_type':0,'group_list_type':0,'act_type':0,'tuning_config':(0,)},names=['x0','weight0','bias0','scale0','offset0','antiquant_scale0','antiquant_offset0','group_list','per_token_scale','y0'])
    return out


def fp8_block_gemm(case):
    import torch
    from .native import GraphCall
    from benchmarks.ascend_baselines import UnsupportedCase
    moe=case._call.signature.name=='MoEGroupedGemmFP8FwdOp';binding=GraphCall('ops-transformer')
    if not moe:
        a,b,sa,sb=case.inputs;m,k=a.shape;n=b.shape[0]
        if sa.shape!=(m,-(-k//128)) or sb.shape!=(-(-n//128),-(-k//128)):
            raise UnsupportedCase('Original G-B supports activation per row/K128 and weight N128/K128; independent per-weight-row/K128 scales are not this native mode.')
        dtype=getattr(torch,case.params['out_dtype'])
        def run(a,b,sa,sb):return _fp8_block_product(binding,a,b,sa,sb,dtype)
    else:
        if next(iter(case.params['layout']))!='masked':raise UnsupportedCase('Original FP8 route currently binds manifest masked expert layout.')
        dtype=getattr(torch,case._call.specs['output'].dtype)
        def run(a,sa,b,sb,metadata):
            out=torch.zeros((*a.shape[:2],b.shape[1]),dtype=dtype,device=a.device)
            for i,count in enumerate(metadata.tolist()):
                if count:out[i,:count]=_fp8_block_product(binding,a[i,:count],b[i],sa[i,:count],sb[i],dtype)
            return out
    return _candidate(binding,run,'GroupedMatmul','gmm/grouped_matmul','Original Ascend950 G-B FP8 mode, exact M/K128 and N128/K128 scale layouts, scale transpose and singleton native group; no dequantization outside the original kernel. Masked experts use native products for live rows and preserve zero output padding.',host=moe)


def fp8_gemm_dispatch(case):
    a,b,sa,sb,*_=case.inputs
    if sa.numel() not in (1,a.shape[-2]) or sb.numel() not in (1,b.shape[-2]):return fp8_block_gemm(case)
    return quant_gemm(case)


def merge_attention(case):
    import torch
    from .native import GraphCall
    from benchmarks.ascend_baselines import UnsupportedCase
    width=case.inputs[0].shape[-1]
    if not 8<=width<=512 or width%8:raise UnsupportedCase('Native AttentionUpdate output head dimension must be divisible by 8 and <=512.')
    binding=GraphCall('ops-transformer')
    def run(va,sa,vb,sb):
        out=torch.empty_like(va);lse=torch.empty_like(sa)
        binding.call('AttentionUpdate',[sa.contiguous().flatten(),sb.contiguous().flatten(),va.contiguous().reshape(-1,width),vb.contiguous().reshape(-1,width)],[out.reshape(-1,width),lse.flatten()],{'update_type':1,'sp':2},names=['lse0','lse1','go0','go1','output','lse_m'])
        return out,lse
    return _candidate(binding,run,'AttentionUpdate','attention/attention_update','Pack batch/token/head axes; use original two-state natural-log merge with update_type=1 returning complete value and LSE. Current shared manifest oracle specifies natural-log states; no log-unit conversion or other arithmetic is performed.')


_experimental_loaded={}
def _load_experimental(name,schema):
    import json,torch
    from benchmarks.ascend_baselines import WORKSPACE,source_hashes
    from .hashing import source_hashes
    source=WORKSPACE/f'ascend_baselines/ops-transformer/experimental/mamba/mamba2_{name}/torch_interface.cpp'
    binary=WORKSPACE/f'.cache/all-pairs/ops_transformer/{name}_build/liboriginal_{name}.so'
    receipt=WORKSPACE/f'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/ops_transformer/{name}_build.json'
    hashes=source_hashes(source.parent,binary,receipt,Path(__file__));build=json.loads(receipt.read_text());op='mambav2_'+name
    if hashes[str(source.relative_to(WORKSPACE))]!=build['source_sha256'] or hashes[str(binary.relative_to(WORKSPACE))]!=build['binary_sha256']:raise RuntimeError('Original experimental source/binary mismatch.')
    if name not in _experimental_loaded:
        if hasattr(torch.ops.npu_ops_transformer_ext,op):raise RuntimeError('Another implementation is already registered: '+op)
        schema_lib=torch.library.Library('npu_ops_transformer_ext','FRAGMENT');schema_lib.define(schema);torch.ops.load_library(str(binary))
        table=torch._C._dispatch_dump_table('npu_ops_transformer_ext::'+op)
        if not any(line.startswith('PrivateUse1:') and str(source) in line for line in table.splitlines()):raise RuntimeError('Experimental dispatch source mismatch.')
        _experimental_loaded[name]=schema_lib
    def call(*inputs):
        result=getattr(torch.ops.npu_ops_transformer_ext,op)(*inputs)
        DISPATCH_EVENTS.append({'op':'npu_ops_transformer_ext::'+op,'source':str(source),'binary':str(binary),'dispatch_owner_checked':True,'returned':0})
        return result
    return call,hashes


def chunk_scan(case):
    import torch
    from .native import GraphCall
    from benchmarks.ascend_baselines import UnsupportedCase
    from dataclasses import replace
    # The original frontend converts BF16 activations internally to FP16. Keep
    # that original implementation unchanged; do not add adapter narrowing.
    binding=GraphCall('ops-transformer')
    native,hashes=_load_experimental('chunk_scan','mambav2_chunk_scan(Tensor ct, Tensor bt, Tensor xt, Tensor dt, Tensor states, Tensor dacs, Tensor dtout) -> Tensor')
    def run(x,cb,da,c,states,dt):
        b,s,h,p=x.shape;chunks=cb.shape[1];g=cb.shape[2];length=cb.shape[3];n=c.shape[-1]
        if length%64 or p%64:raise UnsupportedCase('Original chunk scan requires complete 64-element cube tiles.')
        # Original scanner derives CB as ct @ bt.T. An identity ct and the
        # supplied CB.T preserve the full independent manifest CB tensor.
        identity=torch.eye(length,dtype=cb.dtype,device=x.device).reshape(1,1,length,1,length).expand(b,chunks,-1,g,-1).contiguous()
        bt=cb.transpose(-1,-2).permute(0,1,3,2,4).contiguous()
        cs=c.reshape(b,chunks,length,g,n).permute(0,1,3,2,4).repeat_interleave(h//g,2).float().contiguous().reshape(-1,length,n)
        weights=states.float().transpose(-1,-2).contiguous().reshape(-1,n,p)
        projection=[]
        for start in range(0,cs.shape[0],64):
            count=min(64,cs.shape[0]-start);lhs=cs[start:start+count].reshape(-1,n).contiguous();rhs=weights[start:start+count].contiguous()
            groups=torch.arange(1,count+1,dtype=torch.int64,device=x.device)*length;out=torch.empty((count*length,p),dtype=torch.float32,device=x.device)
            binding.call('GroupedMatmul',[lhs,rhs,None,None,None,None,None,groups,None],[out],{'split_item':3,'transpose_weight':False,'transpose_x':False,'group_type':0,'group_list_type':0,'act_type':0,'tuning_config':(0,)},names=['x0','weight0','bias0','scale0','offset0','antiquant_scale0','antiquant_offset0','group_list','per_token_scale','y0'])
            projection.append(out.reshape(count,length,p))
        inter=torch.cat(projection).reshape(b,chunks,h,length,p)
        skip=torch.zeros(h,dtype=x.dtype,device=x.device)
        out=native(identity,bt,x.reshape(b,chunks,length,h,p).contiguous(),skip,inter,da.transpose(1,2).contiguous(),dt.transpose(1,2).contiguous())
        return out.reshape_as(x)
    result=_candidate(binding,run,'GroupedMatmul+mambav2_chunk_scan','gmm/grouped_matmul','Original FP32 grouped projection of current C and previous states; identity-factor layout reproduces supplied CB inside original scanner. Zero native skip-D input; original frontend internal dtype conversions and all original math/schedule retained. No torch scan/product arithmetic.',host=False)
    return replace(result,source_sha256=result.source_sha256|hashes)


def w4a16(case):
    import torch
    from .native import GraphCall
    from benchmarks.ascend_baselines import UnsupportedCase,source_hashes,WORKSPACE
    from .hashing import source_hashes
    from workloads.gemm import unrepack_w4a16_weight
    from dataclasses import replace
    if case.params['group_size'] not in (32,64,128,256):raise UnsupportedCase('Original INT4 group sizes are 32/64/128/256.')
    binding=GraphCall('ops-transformer')
    def run(a,packed,scale,zero):
        # Invert the original manifest's storage permutation on every call.
        # XOR bit 3 relabels unsigned q as signed q-8, without dequantization.
        signed_bytes=torch.bitwise_xor(unrepack_w4a16_weight(packed),0x88)
        # The original A16W4 pergroup checker requires non-transposed scale
        # metadata. Transpose the exact four-bit codes into physical [K,N]
        # storage and use transpose_weight=False; no floating dequantization.
        codes=torch.stack((signed_bytes & 15,signed_bytes >> 4),dim=-1).reshape(signed_bytes.shape[0],-1).T.contiguous()
        transposed_bytes=(codes[:,0::2] | (codes[:,1::2] << 4)).contiguous()
        storage=transposed_bytes.view(torch.int32)[None]
        # The zero-point convention is an exact integer metadata conversion:
        # signed_code + (8-z) == unsigned_code-z. Every offset is exact in
        # FP16/BF16, and the original API performs scaling/matmul itself.
        offset_table=torch.tensor([8-i for i in range(16)],dtype=a.dtype,device=a.device)
        offsets=offset_table[zero.long()].T.contiguous()[None];scales=scale.T.contiguous()[None]
        groups=torch.full((1,),a.shape[0],dtype=torch.int64,device=a.device);out=torch.empty((a.shape[0],packed.shape[0]),dtype=a.dtype,device=a.device)
        binding.call('GroupedMatmul',[a.contiguous(),storage,None,None,None,scales,offsets,groups,None],[out],{'split_item':3,'transpose_weight':False,'transpose_x':False,'group_type':0,'group_list_type':0,'act_type':0,'tuning_config':(0,)},names=['x0','weight0','bias0','scale0','offset0','antiquant_scale0','antiquant_offset0','group_list','per_token_scale','y0'])
        return out
    result=_candidate(binding,run,'GroupedMatmul','gmm/grouped_matmul','Per-call inverse manifest INT4 bit permutation, exact unsigned-to-signed nibble relabeling and zero-point metadata lookup; original native pergroup dequantization/GEMM, packed INT32 descriptor holding eight original INT4 codes. No tensor dequantization or product outside native API.')
    return replace(result,source_sha256=result.source_sha256|source_hashes(WORKSPACE/'TileOPs/workloads/gemm.py'))


def _float_product(binding,a,b):
    import torch
    out=torch.empty((a.shape[0],b.shape[1]),dtype=torch.float32,device=a.device);groups=torch.full((1,),a.shape[0],dtype=torch.int64,device=a.device)
    binding.call('GroupedMatmul',[a.contiguous(),b.contiguous()[None],None,None,None,None,None,groups,None],[out],{'split_item':3,'transpose_weight':False,'transpose_x':False,'group_type':0,'group_list_type':0,'act_type':0,'tuning_config':(0,)},names=['x0','weight0','bias0','scale0','offset0','antiquant_scale0','antiquant_offset0','group_list','per_token_scale','y0'])
    return out


def hadamard(case):
    import math,torch
    from .native import GraphCall
    from benchmarks.ascend_baselines import WORKSPACE,source_hashes
    from .hashing import source_hashes
    from workloads.ascend_extended_transform_quant import _HADAMARD_SIGNS
    from dataclasses import replace
    base=case.params['base_order'];n=case.inputs[0].shape[-1]
    order=64 if base==1 and n>=4096 else base;radix=n//order
    binding=GraphCall('ops-transformer')
    def run(x):
        # Only fixed mathematical coefficient tables are constructed here.
        # All input-dependent transform arithmetic is original FP32 GEMM.
        signs=lambda d:[[(-1. if (i&j).bit_count()%2 else 1.) for j in range(d)] for i in range(d)]
        hradix=torch.tensor(signs(radix),dtype=torch.float32,device=x.device)
        first=_float_product(binding,x.float().reshape(-1,radix),hradix)
        source=signs(order) if base==1 else [[1. if sign=='+' else -1. for sign in row] for row in _HADAMARD_SIGNS[base]]
        normalized=[[value/math.sqrt(n) for value in row] for row in source]
        hbase=torch.tensor(normalized,dtype=torch.float32,device=x.device).T.contiguous()
        packed=first.reshape(-1,order,radix).transpose(1,2).contiguous().reshape(-1,order)
        second=_float_product(binding,packed,hbase).reshape(-1,radix,order).transpose(1,2).contiguous()
        return second.reshape_as(x).to(x.dtype)
    result=_candidate(binding,run,'GroupedMatmul','gmm/grouped_matmul','Exact signed Hadamard/Kronecker coefficient convention from original shared workload, two original FP32 matrix products with lossless input widening; normalization is a constant native matrix coefficient. Runtime coefficient construction, packing and output cast are timed; no external Hadamard/transform implementation is called.')
    return replace(result,source_sha256=result.source_sha256|source_hashes(WORKSPACE/'TileOPs/workloads/ascend_extended_transform_quant.py'))
