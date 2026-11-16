"""Read actual sequence metadata each run and invoke original SGL attention."""
from pathlib import Path
import math


def _build(case,paged=False):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,LIBRARIES,UnsupportedCase,local_function,source_hashes
    p=case.params
    if p.get('softcap') not in (None,0):
        raise UnsupportedCase('Original SGL decode attention has no logits softcap.')
    if case.inputs[0].dtype not in (torch.float16,torch.bfloat16):
        raise UnsupportedCase('This original SGL unquantized decode API has no external FP8 Q/K/V scale inputs.')
    root=LIBRARIES/'sgl-kernel-npu'
    host='python/sgl_kernel_npu/sgl_kernel_npu/attention/decode_attention.py'
    fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.attention.decode_attention','decode_gqa',host)
    sinkhost='python/sgl_kernel_npu/sgl_kernel_npu/attention/sinks_attention.py'
    sinkfn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.attention.sinks_attention','attention_sinks_triton',sinkhost)
    prefillfn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.attention.sinks_attention','attention_sinks_prefill_triton',sinkhost)
    ropehost='python/sgl_kernel_npu/sgl_kernel_npu/norm/fused_rope_qk_mqa.py'
    rotate=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.norm.fused_rope_qk_mqa','fused_rope_qk_mqa',ropehost)
    def request(q,k,v,cos,sin,sinks):
        nq,heads,dim=q.shape
        nk,kvheads,vdim=v.shape
        if p.get('pos_encoding_mode','none')!='none':
            rd=p.get('rotary_dim') or dim
            cache=torch.cat((cos,sin),dim=-1).float()
            q,_=rotate(q.float().contiguous(),q.float().contiguous(),cache[nk-nq:nk],rd,p.get('rope_layout','neox')=='neox')
            k,_=rotate(k.float().contiguous(),k.float().contiguous(),cache[:nk],rd,p.get('rope_layout','neox')=='neox')
            q,k=q.to(v.dtype),k.to(v.dtype)
        group=heads//kvheads
        padgroup=1<<(group-1).bit_length()
        scale=p.get('sm_scale')
        scale=dim**-0.5 if scale is None else scale
        if 1<nq<=nk and dim==vdim and p.get('is_causal',True):
            pages=math.ceil(nk/128)
            kk=F.pad(k,(0,0,0,0,0,pages*128-nk)).reshape(pages,128,kvheads,dim)
            vv=F.pad(v,(0,0,0,0,0,pages*128-nk)).reshape(pages,128,kvheads,dim)
            table=torch.arange(pages,device=q.device,dtype=torch.int32).reshape(1,pages)
            lengths=torch.tensor([nk],device=q.device,dtype=torch.int32)
            queries=torch.tensor([nq],device=q.device,dtype=torch.int32)
            ss=torch.full((heads,),float('-inf'),device=q.device,dtype=torch.float32) if sinks is None else sinks
            window=p.get('window_size_left',-1)
            window=-1 if window<0 else window+1
            return prefillfn(q.reshape(nq,-1).contiguous(),kk,vv,ss,queries,table,lengths,scale,window,heads,kvheads).reshape(nq,heads,vdim)
        result=[]
        for i in range(nq):
            pos=nk-nq+i
            begin=max(0,pos-p['window_size_left']) if p.get('window_size_left',-1)>=0 else 0
            end=min(nk,pos+1) if p.get('is_causal',True) else nk
            if p.get('window_size_right',-1)>=0:end=min(end,pos+p['window_size_right']+1)
            if end<=begin:
                result.append(torch.zeros((1,heads,vdim),device=q.device,dtype=q.dtype));continue
            n=end-begin
            pages=math.ceil(n/128)
            kk=F.pad(k[begin:end],(0,0,0,0,0,pages*128-n)).reshape(pages,128,kvheads,dim)
            vv=F.pad(v[begin:end],(0,0,0,0,0,pages*128-n)).reshape(pages,128,kvheads,vdim)
            table=torch.arange(pages,device=q.device,dtype=torch.int32).reshape(1,pages)
            lengths=torch.tensor([n],device=q.device,dtype=torch.int32)
            if sinks is not None:
                if dim!=vdim:raise UnsupportedCase('SGL sink attention requires equal Q/K/V feature width.')
                qq=F.pad(q[i:i+1].reshape(1,kvheads,group,dim),(0,0,0,padgroup-group)).reshape(1,-1).contiguous()
                ss=F.pad(sinks.reshape(kvheads,group),(0,padgroup-group)).reshape(-1).contiguous()
                out=sinkfn(qq,kk,vv,ss,table,lengths,scale,-1,kvheads*padgroup,kvheads)
                result.append(out.reshape(1,kvheads,padgroup,vdim)[:,:,:group].reshape(1,heads,vdim))
            else:
                qq=F.pad(q[i:i+1].reshape(1,kvheads,group,dim),(0,0,0,padgroup-group)).reshape(1,kvheads*padgroup,dim)
                out=torch.empty((1,kvheads*padgroup,vdim),device=q.device,dtype=q.dtype)
                fn(qq,kk,vv,out,lengths,scale,128,table)
                result.append(out.reshape(1,kvheads,padgroup,vdim)[:,:,:group].reshape(1,heads,vdim))
        return torch.cat(result,dim=0) if result else torch.empty((0,heads,vdim),device=q.device,dtype=q.dtype)
    def run_varlen(q,k,v,cuq,cuk,qs=None,ks=None,vs=None,cos=None,sin=None,sinks=None):
        if any(x is not None for x in (qs,ks,vs)):
            raise UnsupportedCase('Unquantized SGL API has no supplied scale parameters; input dequantization is not a layout adaptation.')
        qb,kb=cuq.cpu().tolist(),cuk.cpu().tolist()
        out=[request(q[qb[i]:qb[i+1]],k[kb[i]:kb[i+1]],v[kb[i]:kb[i+1]],cos,sin,sinks) for i in range(len(qb)-1)]
        return torch.cat(out,dim=0)
    def run_paged(q,kp,vp,table,lengths,cuq,qs=None,ks=None,vs=None,cos=None,sin=None,sinks=None):
        if any(x is not None for x in (qs,ks,vs)):
            raise UnsupportedCase('Unquantized SGL API has no supplied FP8 scales.')
        qb,kl=cuq.cpu().tolist(),lengths.cpu().tolist()
        out=[]
        for i,n in enumerate(kl):
            pages=table[i,:math.ceil(n/kp.shape[1])].long()
            k,v=[x[pages].flatten(0,1)[:n] for x in (kp,vp)]
            out.append(request(q[qb[i]:qb[i+1]],k,v,cos,sin,sinks))
        return torch.cat(out,dim=0)
    return Candidate(Implementation(run_paged if paged else run_varlen),'sgl_kernel_npu::decode_gqa / attention_sinks_triton / attention_sinks_prefill_triton + native RoPE',
                     'Read packed boundaries and actual lengths per call; gather paged KV when needed, select original causal prefill API for multi-query equal-feature rows (negative-infinity sink for no sink), otherwise decode selected causal/window intervals. Native optional sinks/rotation; pad pages/independent heads only.',
                     source_hashes(Path(__file__),root/host,root/sinkhost,root/ropehost),timing_mode='host_roundtrip')


def varlen(case):return _build(case,False)
def paged(case):return _build(case,True)


def mla_decode(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,LIBRARIES,local_function,source_hashes
    host='python/sgl_kernel_npu/sgl_kernel_npu/attention/decode_attention.py'
    fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.attention.decode_attention','decode_mla',host)
    q,qp,k,kp=case.inputs
    b,n,h,d=k.shape
    pages=math.ceil(n/128)
    table=torch.arange(b*pages,device=q.device,dtype=torch.int32).reshape(b,pages)
    lengths=torch.full((b,),n,device=q.device,dtype=torch.int32)
    def run(q,qp,k,kp):
        kk=F.pad(k,(0,0,0,0,0,pages*128-n)).contiguous().reshape(b*pages,128,h,d)
        pp=F.pad(kp,(0,0,0,0,0,pages*128-n)).contiguous().reshape(b*pages,128,h,kp.shape[-1])
        out=torch.empty_like(q)
        fn(torch.cat((q,qp),dim=-1),kk,pp,out,lengths,(d+qp.shape[-1])**-0.5,128,table)
        return out
    return Candidate(Implementation(run),'sgl_kernel_npu.attention.decode_attention.decode_mla',
                     'Concatenate supplied latent/positional queries; pad and view supplied independent latent/positional keys as pages; native full MLA and actual lengths, including tail pages.',
                     source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/host))


def mha_paged(case):
    from dataclasses import replace
    from types import SimpleNamespace
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import source_hashes
    q,k,v,lengths,table=case.inputs
    page=case.params['page_size']
    b,sq,h,d=q.shape
    qflat=q.reshape(b*sq,h,d)
    cu=torch.arange(b+1,device=q.device,dtype=torch.int32)*sq
    def pages(x):return x.reshape(-1,page,h,d)
    native=paged(SimpleNamespace(inputs=(qflat,pages(k),pages(v),table,lengths,cu,None,None,None,None,None,None),params=case.params))
    def run(q,k,v,lengths,table):
        cu=torch.arange(q.shape[0]+1,device=q.device,dtype=torch.int32)*q.shape[1]
        return native.implementation.run(q.reshape(-1,h,d),pages(k),pages(v),table,lengths,cu).reshape_as(q)
    return replace(native,implementation=Implementation(run),source_sha256=native.source_sha256|source_hashes(Path(__file__)),
                   adaptation='Dense batch queries become packed rows with explicit offsets; flat physical KV cache becomes page views; original SGL GQA with one query per KV head.')


def dsa_decode(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,LIBRARIES,UnsupportedCase,local_function,source_hashes
    host='python/sgl_kernel_npu/sgl_kernel_npu/attention/decode_attention.py'
    fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.attention.decode_attention','decode_mla',host)
    tail=case.params['dim_tail']
    if tail>64:raise UnsupportedCase('Inspected MLA rotary component is padded to64; wider tail has no validated source route.')
    stride=case.params.get('stride_kv',1)
    def run(q,kv,indices):
        b,sq,h,totaldim=q.shape
        sk,g=kv.shape[1:3];d=totaldim-tail;group=h//g
        start=case.params.get('q_start_index_s')
        start=sk*stride-sq if start is None else start
        out=torch.empty((b,sq,h,d),device=q.device,dtype=q.dtype)
        for batch in range(b):
            for i in range(sq):
                for head in range(g):
                    selected=indices[batch,i,head].long()
                    selected=selected[(selected>=0)&(selected<sk)&(selected*stride+stride-1<=start+i)]
                    if i<stride-1:selected=torch.cat((selected,torch.zeros(1,device=q.device,dtype=torch.long)))
                    selected=selected.unique(sorted=True)
                    n=selected.numel()
                    if not n:
                        # The manifest's empty-selection softmax is NaN.
                        out[batch,i,head*group:(head+1)*group].fill_(float('nan'));continue
                    data=kv[batch,selected,head]
                    pages=math.ceil(n/128)
                    kk=F.pad(data[:,:d],(0,0,0,pages*128-n)).reshape(pages,128,1,d)
                    kr=F.pad(data[:,d:],(0,64-tail,0,pages*128-n)).reshape(pages,128,1,64)
                    qq=F.pad(q[batch,i,head*group:(head+1)*group],(0,64-tail)).unsqueeze(0).contiguous()
                    table=torch.arange(pages,device=q.device,dtype=torch.int32).reshape(1,pages)
                    lengths=torch.tensor([n],device=q.device,dtype=torch.int32)
                    yy=torch.empty((1,group,d),device=q.device,dtype=q.dtype)
                    scale=totaldim**-0.5 if case.params.get('sm_scale') is None else case.params['sm_scale']
                    fn(qq,kk,kr,yy,lengths,scale,128,table)
                    out[batch,i,head*group:(head+1)*group]=yy[0]
        return out
    return Candidate(Implementation(run),'sgl_kernel_npu.attention.decode_attention.decode_mla',
                     'Deduplicate selection metadata and enforce manifest compressed-causal positions per query, gather selected original KV rows, pad selected cache pages and rotary tail with zeros; original MLA performs entire softmax/reduction.',
                     source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/host),timing_mode='host_roundtrip')


def prefill_append(case):
    from dataclasses import replace
    from types import SimpleNamespace
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import UnsupportedCase,LIBRARIES,local_function,source_hashes
    from .cache import paged_cache_write
    q,kn,vn,kp,vp,ks,vs,cu,oldlens,table=case.inputs
    if kp.dtype!=q.dtype or vp.dtype!=q.dtype:
        raise UnsupportedCase('Original store/decode API cannot provide the declared FP8 cache quantization/dequantization scales.')
    page=case.params['page_size'];h=kn.shape[-2];d=kn.shape[-1]
    p={**case.params,'pos_encoding_mode':'none'}
    native=paged(SimpleNamespace(inputs=(q,),params=p))
    host='python/sgl_kernel_npu/sgl_kernel_npu/norm/fused_rope_qk_mqa.py'
    rotate=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.norm.fused_rope_qk_mqa','fused_rope_qk_mqa',host)
    from workloads.rope import rope_frequency_tables
    store=paged_cache_write(SimpleNamespace(inputs=(kn,vn,kp.reshape(-1,page,h,d),vp.reshape(-1,page,h,d),torch.empty(q.shape[0],device=q.device,dtype=torch.int64),None,None),params={}))
    def run(q,kn,vn,kp,vp,ks,vs,cu,oldlens,table):
        bounds,old=cu.cpu().tolist(),oldlens.cpu().tolist()
        pos=torch.cat([torch.arange(n,n+bounds[i+1]-bounds[i],device=q.device) for i,n in enumerate(old)])
        slots=torch.cat([table[i,pos[bounds[i]:bounds[i+1]]//page].long()*page+pos[bounds[i]:bounds[i+1]]%page for i in range(len(old))])
        if case.params.get('fuse_rope'):
            rd=case.params.get('rotary_dim') or d
            maxpos=max(n+bounds[i+1]-bounds[i] for i,n in enumerate(old))
            cos,sin=rope_frequency_tables(rd,maxpos,base=case.params['rope_base'],dtype=q.dtype,device=q.device)
            cache=torch.cat((cos,sin),dim=-1)[pos].float().contiguous()
            q,_=rotate(q.float().contiguous(),q.float().contiguous(),cache,rd,True)
            kn,_=rotate(kn.float().contiguous(),kn.float().contiguous(),cache,rd,True)
            q,kn=q.to(vn.dtype),kn.to(vn.dtype)
        kout,vout=store.implementation.run(kn,vn,kp.reshape(-1,page,h,d),vp.reshape(-1,page,h,d),slots)
        lengths=oldlens+(cu[1:]-cu[:-1]).to(oldlens.dtype)
        output=native.implementation.run(q,kout,vout,table,lengths,cu)
        return output,kout.reshape_as(kp),vout.reshape_as(vp)
    return replace(native,implementation=Implementation(run),api=native.api+' + original prefix-valid KV store + original RoPE',
        adaptation='Use actual sequence metadata to place new keys/values into private original pools with native store; optional native rotate new rows only; native paged attention reads updated lengths. Return attention and both complete caches.',
        source_sha256=native.source_sha256|store.source_sha256|source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/host,Path(__file__).parents[3]/'workloads/rope.py'))


def nsa_selected(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,LIBRARIES,local_function,source_hashes
    host='python/sgl_kernel_npu/sgl_kernel_npu/attention/decode_attention.py'
    fn=local_function('sgl-kernel-npu','python/sgl_kernel_npu','sgl_kernel_npu.attention.decode_attention','decode_gqa',host)
    block=case.params['block_size'];causal=case.params['is_causal']
    def run(q,k,v,blocks,counts,offsets,token_indices):
        bounds=offsets.cpu().tolist();count=counts.cpu().tolist()
        heads=q.shape[1];kvheads=k.shape[1];group=heads//kvheads
        paddedgroup=1<<(group-1).bit_length()
        out=torch.zeros((q.shape[0],heads,v.shape[-1]),device=q.device,dtype=q.dtype)
        scale=case.params.get('scale');scale=k.shape[-1]**-0.5 if scale is None else scale
        for seq,(bos,eos) in enumerate(zip(bounds[:-1],bounds[1:])):
            for pos in range(bos,eos):
                for head in range(kvheads):
                    selected=(blocks[pos,head,:count[pos][head]].long()[:,None]*block+torch.arange(block,device=q.device)).reshape(-1)
                    selected=selected[(selected>=0)&(selected<(pos-bos+1 if causal else eos-bos))]
                    # Keep multiplicity: the manifest selected-slot softmax does
                    # not deduplicate repeated supplied block indices.
                    n=selected.numel()
                    if not n:continue
                    pages=math.ceil(n/128)
                    kk=F.pad(k[bos+selected,head],(0,0,0,pages*128-n)).reshape(pages,128,1,k.shape[-1])
                    vv=F.pad(v[bos+selected,head],(0,0,0,pages*128-n)).reshape(pages,128,1,v.shape[-1])
                    qq=F.pad(q[pos:pos+1,head*group:(head+1)*group],(0,0,0,paddedgroup-group)).contiguous()
                    yy=torch.empty((1,paddedgroup,v.shape[-1]),device=q.device,dtype=q.dtype)
                    table=torch.arange(pages,device=q.device,dtype=torch.int32).reshape(1,pages)
                    length=torch.tensor([n],device=q.device,dtype=torch.int32)
                    fn(qq,kk,vv,yy,length,scale,128,table)
                    out[pos,head*group:(head+1)*group]=yy[0,:group]
        return out
    return Candidate(Implementation(run),'sgl_kernel_npu.attention.decode_attention.decode_gqa',
        'Use current packed sequence offsets, block counts and selected slots to gather per-query original K/V; preserve duplicate multiplicity and causal visibility, pad only native page/head tails; original complete GQA reduction. Empty selected sets produce declared zeros.',
        source_hashes(Path(__file__),LIBRARIES/'sgl-kernel-npu'/host),timing_mode='host_roundtrip',
        limitations=('Per-query/per-KV-head native launches may exceed the runtime budget on long packed cases; no kernel schedule change is used to hide this cost.',))
