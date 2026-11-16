"""Original matrix and convolution APIs, including dimension-lifting adaptations."""
from .common import candidate,new_binding,op_name
from .catalog import OTHER


def bmm(case):
    import torch
    binding=new_binding()
    def run(a,b):
        out=torch.empty((*a.shape[:-2],a.shape[-2],b.shape[-1]),device=a.device,dtype=a.dtype)
        binding.call('BatchMatMulV3',[a.contiguous(),b.contiguous(),None,None],[out],{'adj_x1':False,'adj_x2':False,'enable_hf32':False})
        return out
    return candidate(binding,run,'BatchMatMulV3','Original batched matrix product; per-call contiguous inputs; unchanged output dtype.','matmul/batch_mat_mul_v3')


def convolution(case):
    import torch
    name=op_name(case);api,directory=OTHER[name];binding=new_binding();p=case.params
    n=int(name[4]);target=2 if n==1 else 3
    def tup(value):return (value,)*n if isinstance(value,int) else tuple(value)
    stride=(1,)*(target-n)+tup(p['stride']);pad=(0,)*(target-n)+tup(p['padding']);dilation=(1,)*(target-n)+tup(p['dilation'])
    def run(x,w,bias=None):
        xx=x.contiguous();ww=w.contiguous()
        if n==1:xx=xx.unsqueeze(-2);ww=ww.unsqueeze(-2)
        shape=(xx.shape[0],ww.shape[0],*[(xx.shape[i+2]+2*pad[i]-dilation[i]*(ww.shape[i+2]-1)-1)//stride[i]+1 for i in range(target)])
        out=torch.empty(shape,device=x.device,dtype=x.dtype)
        attrs={'strides':[1,1,*stride],'pads':[v for v in pad for _ in (0,1)],'dilations':[1,1,*dilation],
               'groups':int(p['groups']),'data_format':'NCHW' if target==2 else 'NCDHW','pad_mode':'SPECIFIC','enable_hf32':False}
        fmt=0 if target==2 else 30
        args=[xx,ww,bias,None] if target==2 else [xx,ww,bias,None,None,None]
        binding.call(api,args,[out],attrs,formats=[fmt,fmt,*([2]*(len(args)-2)),fmt])
        return out.squeeze(-2) if n==1 else out
    return candidate(binding,run,api,'Preserve groups, stride, symmetric padding, dilation and bias; 1D lifts a singleton height dimension to Conv2DV2; per-call packing.',directory)


def legacy(case):
    import importlib
    from .catalog import LEGACY
    module,fn,_=LEGACY[op_name(case)]
    return getattr(importlib.import_module(module),fn)(case)


def _mm(binding,a,b,bias=None):
    """a[M,K], b[K,N]; same full-tile NT layout as the verified original MatMulV3 route."""
    import torch
    import torch.nn.functional as F
    m,k=a.shape;n=b.shape[1]
    lhs=F.pad(a.contiguous(),(0,-k%64,0,-m%1024))
    rhs=F.pad(b.T.contiguous(),(0,-k%64,0,-n%256))
    bb=None if bias is None else F.pad(bias.contiguous(),(0,-n%256))
    out=torch.empty((lhs.shape[0],rhs.shape[0]),device=a.device,dtype=a.dtype)
    binding.call('MatMulV3',[lhs,rhs,bb,None],[out],{'transpose_x1':False,'transpose_x2':True})
    return out[:m,:n].contiguous()


def grouped(case):
    import torch
    from dataclasses import replace
    binding=new_binding();p=case.params
    def run(a,b,sizes,offsets):
        sizes=sizes.tolist();pieces=[];start=0
        for i,size in enumerate(sizes):
            end=start+size
            if not p['transpose_a']:
                aa=a[start:end];bb=b[i].T if p['transpose_b'] else b[i]
            else:
                aa=a[start:end].T;bb=b[:,start:end].T if p['transpose_b'] else b[start:end]
            if aa.shape[1]==0:piece=torch.zeros((aa.shape[0],bb.shape[1]),device=a.device,dtype=a.dtype)
            else:piece=_mm(binding,aa,bb)
            pieces.append(piece);start=end
        return torch.stack(pieces) if p['transpose_a'] else torch.cat(pieces)
    result=candidate(binding,run,'MatMulV3 per original logical group','Read group sizes afresh; slice and transpose each original group; native padded NT matrix multiply and concatenate/stack exact group order.','matmul/mat_mul_v3')
    return replace(result,timing_mode='host_roundtrip')


def fp8_gemm(case):
    import torch
    name=op_name(case);binding=new_binding();p=case.params
    def multiply(x,scale):
        out=torch.empty(x.shape,device=x.device,dtype=torch.float32)
        binding.call('ForeachMulList',[x.float().contiguous(),scale.expand(x.shape).contiguous()],[out],{},names=['x10','x20','y0'])
        return out
    def expand(scale,rows,cols):
        if scale.numel()==1:return scale.reshape(1,1).expand(rows,cols)
        if scale.shape[0]!=rows:scale=scale.repeat_interleave(128,dim=0)[:rows]
        return scale.repeat_interleave(128,dim=1)[:,:cols]
    def run(a,b,sa,sb,bias=None):
        if name=='GemmFP8FwdOp':
            aa=multiply(a,expand(sa,*a.shape));bb=multiply(b,expand(sb,*b.shape))
            out=_mm(binding,aa,bb.T,None if bias is None else bias.float())
        else:
            aa=multiply(a,sa);bb=multiply(b,sb)
            if p['trans_b']:bb=bb.transpose(-2,-1)
            out=torch.stack([_mm(binding,x,y) for x,y in zip(aa,bb,strict=True)])
        return out.to(getattr(torch,p['out_dtype']))
    return candidate(binding,run,'ForeachMulList + MatMulV3','Lossless FP8-to-FP32 widening; per-call original vendor multiplication applies expanded original scale grids; original MatMulV3 with exact zero padding, bias and output cast.','foreach/foreach_mul_list','matmul/mat_mul_v3')


def moe_grouped(case):
    import torch
    from dataclasses import replace
    from benchmarks.ascend_baselines import UnsupportedCase
    binding=new_binding();p=case._call.params;layout=p['layout']
    if p.get('activation') not in (None,'silu_and_mul','gelu_and_mul'):
        raise UnsupportedCase('No inspected original gated activation for this manifest attribute.')
    def run(a,b,metadata):
        meta=metadata.tolist();shape=(*a.shape[:-1],b.shape[1]);out=torch.zeros(shape,device=a.device,dtype=torch.float32)
        flat=a.reshape(-1,a.shape[-1]);dst=out.reshape(-1,out.shape[-1]);start=0
        for g in range(b.shape[0]):
            if layout.kind=='masked':
                rows=list(range(g*a.shape[1],g*a.shape[1]+int(meta[g])))
            elif layout.metadata_kind=='per_row':rows=[i for i,value in enumerate(meta) if value==g]
            else:
                start=-(-start//layout.alignment)*layout.alignment;end=int(meta[g]);rows=list(range(start,end));start=end
            if rows:
                ids=torch.tensor(rows,device=a.device,dtype=torch.int64)
                mm=_mm(binding,flat.index_select(0,ids).float(),b[g].float().T)
                dst.index_copy_(0,ids,mm)
        activation=p.get('activation')
        if activation:
            gated=torch.empty((*out.shape[:-1],out.shape[-1]//2),device=a.device,dtype=torch.float32)
            api='SwiGlu' if activation=='silu_and_mul' else 'GeluMul'
            attrs={'dim':-1} if activation=='silu_and_mul' else {'approximate':'tanh' if activation=='gelu_tanh' else 'none'}
            if api=='GeluMul':
                from .gating import gelu_mul
                out=gelu_mul(binding,out,attrs)
            else:binding.call(api,[out],[gated],attrs);out=gated
        dtype=a.dtype if p.get('out_dtype') is None else getattr(torch,p['out_dtype'])
        return out.to(dtype)
    result=candidate(binding,run,'MatMulV3'+(' + SwiGlu' if p.get('activation')=='silu_and_mul' else ' + GeluMul' if p.get('activation') else ''),
                     'Read expert layout metadata afresh, gather exact logical rows, original FP32 MatMul per expert, scatter to declared layout and use original gated epilogue API when requested; all copies and host metadata cost included.','matmul/mat_mul_v3','activation/swi_glu','activation/gelu_mul')
    if p.get('activation')=='gelu_and_mul':
        from pathlib import Path
        from benchmarks.ascend_baselines import source_hashes
        result=replace(result,source_sha256=result.source_sha256|source_hashes(Path(__file__).with_name('gating.py')))
    return replace(result,timing_mode='host_roundtrip')


def w4_gemm(case):
    import ctypes as C
    import torch
    from dataclasses import replace
    from pathlib import Path
    from workloads.gemm import unrepack_w4a16_weight
    from benchmarks.ascend_baselines import source_hashes
    binding=new_binding()
    def run(a,packed,scale,zero):
        raw=unrepack_w4a16_weight(packed)
        n,kp=raw.shape;k=2*kp
        # Bit extraction is the lossless storage-format conversion; all dequant arithmetic stays native.
        q=torch.empty((n,k),device=raw.device,dtype=torch.float32)
        q[:,0::2]=(raw & 15).float();q[:,1::2]=(raw >> 4).float()
        group=k//scale.shape[1];z=zero.float().repeat_interleave(group,dim=1)
        centered=torch.empty_like(q);weight=torch.empty_like(q)
        alpha=torch.tensor(1.0,device=a.device,dtype=torch.float32)
        binding.call('ForeachSubList',[q,z,alpha],[centered],{},names=['x10','x20','alpha','y0'],constants={2:(C.c_float*1)(1.0)})
        binding.call('ForeachMulList',[centered,scale.float().repeat_interleave(group,dim=1)],[weight],{},names=['x10','x20','y0'])
        return _mm(binding,a,weight.to(a.dtype).T)
    result=candidate(binding,run,'ForeachSubList + ForeachMulList + MatMulV3','Invert original INT4 storage permutation each invocation; lossless nibble unpack; native vendor zero-point subtraction and scaling; declared dequantized-weight narrowing before original MatMulV3. No cached input-derived weights.','foreach/foreach_sub_list','foreach/foreach_mul_list','matmul/mat_mul_v3')
    import workloads.gemm
    return replace(result,source_sha256=result.source_sha256|source_hashes(Path(workloads.gemm.__file__)))


def moe_fp8_grouped(case):
    import torch
    from dataclasses import replace
    from benchmarks.api import Implementation
    binding=new_binding();native=moe_grouped(case)
    def scaled(x,s):
        y=torch.empty(x.shape,device=x.device,dtype=torch.float32)
        binding.call('ForeachMulList',[x.float().contiguous(),s.contiguous()],[y],{},names=['x10','x20','y0'])
        return y
    def run(a,sa,b,sb,metadata):
        aa=scaled(a,sa.repeat_interleave(128,dim=-1))
        bb=scaled(b,sb.repeat_interleave(128,dim=1).repeat_interleave(128,dim=2))
        return native.implementation.run(aa,bb,metadata).to(torch.bfloat16)
    def cleanup():
        binding.cleanup();native.cleanup()
    result=candidate(binding,run,'ForeachMulList + MatMulV3','Exact FP8 widening and original scale-grid expansion per call; original vendor multiply and independent original MatMul per expert; masked layout metadata reread, all declared rows preserved.','foreach/foreach_mul_list','matmul/mat_mul_v3')
    return replace(result,implementation=Implementation(run),cleanup=cleanup,timing_mode='host_roundtrip',source_sha256=result.source_sha256|native.source_sha256)


def int8_gemm(case):
    import torch
    binding=new_binding();dtype=getattr(torch,case.params['out_dtype'])
    def run(a,b,scale_a,scale_b,bias=None):
        out=torch.empty((a.shape[0],b.shape[0]),device=a.device,dtype=dtype)
        binding.call('QuantBatchMatmulV3',[a.contiguous(),b.contiguous(),scale_b.contiguous().reshape(-1),None,
                     None if bias is None else bias.contiguous(),scale_a.contiguous().reshape(-1)],[out],
                     {'dtype':binding.codes[dtype],'transpose_x1':False,'transpose_x2':True,'group_size':0})
        return out
    return candidate(binding,run,'QuantBatchMatmulV3','Original INT8 dot product with independent per-row/per-column scales and original optional postscale bias; flatten only scale views.','matmul/quant_batch_matmul_v3')


def causal_convolution(case):
    if case.params.get('activation') == 'silu':
        raise RuntimeError('Retired Conv2DV2 plus Swish composition; original non-SiLU route remains available.')
    import torch
    from dataclasses import replace
    binding=new_binding();p=case.params;name=op_name(case)
    def conv(xx,weight,bias):
        # One spatial dimension lifted to NCHW, C depthwise groups.
        x4=xx.float().contiguous().unsqueeze(-2);w4=weight.float().contiguous().reshape(weight.shape[0],1,1,weight.shape[-1])
        shape=(xx.shape[0],xx.shape[1],1,xx.shape[-1]-weight.shape[-1]+1)
        y=torch.empty(shape,device=xx.device,dtype=torch.float32)
        binding.call('Conv2DV2',[x4,w4,None if bias is None else bias.float().contiguous(),None],[y],
                     {'strides':[1,1,1,1],'pads':[0,0,0,0],'dilations':[1,1,1,1],
                      'groups':xx.shape[1],'data_format':'NCHW','pad_mode':'SPECIFIC','enable_hf32':False},
                     formats=[0,0,2,2,0])
        y=y.squeeze(-2)
        if p.get('activation')=='silu':
            z=torch.empty_like(y);binding.call('Swish',[y],[z],{'scale':1.0});y=z
        return y
    def run(x,weight,state,*args):
        if name=='CausalConv1dUpdateFwdOp':
            bias=args[0] if args else None
            shifted=torch.cat((state[...,1:],x.unsqueeze(-1)),dim=-1)
            y=conv(shifted,weight,bias).squeeze(-1)
            return y.to(x.dtype),shifted
        offsets=args[0].tolist();bias=args[1] if len(args)>1 else None
        flags=args[2].tolist() if len(args)>2 and args[2] is not None else None
        pieces=[];states=[];width=weight.shape[-1]
        for i,(begin,end) in enumerate(zip(offsets[:-1],offsets[1:])):
            history=state[i] if flags is not None and flags[i] else torch.zeros_like(state[i])
            tokens=torch.cat((history,x[:,begin:end]),dim=-1)
            pieces.append(conv(tokens.unsqueeze(0),weight,bias).squeeze(0))
            states.append(tokens[:,-(width-1):] if width>1 else tokens[:,:0])
        return torch.cat(pieces,dim=-1).to(x.dtype),torch.stack(states)
    result=candidate(binding,run,'Conv2DV2'+(' + Swish' if p.get('activation')=='silu' else ''),
                     'Read ragged offsets and state flags afresh; concatenate copied causal history, run original depthwise convolution and optional Swish, return full copied updated history. FP32 widening follows manifest reference; no torch convolution/math fallback.',
                     'conv/conv2d_v2','activation/swish')
    return replace(result,timing_mode='host_roundtrip' if name=='CausalConv1dFwdOp' else 'device_async')
