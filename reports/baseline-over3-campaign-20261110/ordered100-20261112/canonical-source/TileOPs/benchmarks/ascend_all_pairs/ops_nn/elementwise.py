"""Thin activation and one-element Foreach-list adapters."""
from .catalog import UNARY, FOREACH
from .common import candidate, new_binding, op_name


def activation(case):
    import torch
    name=op_name(case); api,directory=UNARY[name]; p=case.params
    binding=new_binding()
    attrs={}
    if name=='GeluFwdOp':attrs={'approximate':p.get('approximate','none')}
    elif name in ('GeluAndMulFwdOp','GeluTanhAndMulFwdOp'):
        attrs={'approximate':'tanh' if name=='GeluTanhAndMulFwdOp' else 'none'}
    elif name=='SiluAndMulFwdOp':attrs={'dim':-1}
    elif name=='SiluFwdOp':attrs={'scale':1.0}
    elif name=='LeakyReluFwdOp':attrs={'negative_slope':float(p['negative_slope'])}
    elif name=='EluFwdOp':attrs={'alpha':float(p['alpha']),'scale':1.0,'input_scale':1.0}
    elif name=='SoftplusFwdOp':attrs={'beta':float(p['beta']),'threshold':float(p['threshold'])}

    def run(x,*extra):
        if api=='GeluMul':
            from .gating import gelu_mul
            return gelu_mul(binding,x,attrs)
        shape=(*x.shape[:-1],x.shape[-1]//2) if 'AndMul' in name else x.shape
        out=torch.empty(shape,device=x.device,dtype=x.dtype)
        binding.call(api,[x.contiguous(),*(t.contiguous() for t in extra)],[out],attrs)
        return out
    result=candidate(binding,run,api,'Original activation attributes, per-call contiguous inputs, unchanged manifest dtype and complete output. GeluMul pairs matching gate-half chunks into width<=1024 and batches<=200000, then restores the complete output.',directory)
    if api=='GeluMul':
        from dataclasses import replace
        from pathlib import Path
        from benchmarks.ascend_baselines import source_hashes
        result=replace(result,source_sha256=result.source_sha256|source_hashes(Path(__file__).with_name('gating.py')))
    return result


def bitwise_not(case):
    import ctypes as C
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    if case.inputs[0].dtype not in (torch.int32,torch.int64):
        raise UnsupportedCase('Current complement route covers exact 32-bit lanes of INT32/INT64 inputs.')
    binding=new_binding()
    def run(x):
        lanes=x.contiguous().view(torch.int32)
        minus_one=torch.full_like(lanes,-1);out=torch.empty_like(lanes)
        alpha=torch.tensor(1,device=x.device,dtype=torch.int32)
        binding.call('ForeachSubList',[minus_one,lanes,alpha],[out],{},
                     names=['x10','x20','alpha','y0'],constants={2:(C.c_int32*1)(1)})
        return out.view(x.dtype).reshape(x.shape)
    return candidate(binding,run,'ForeachSubList','Use original exact INT32 subtraction -1-x, whose result is always in INT32 range. INT64 is a lossless two-INT32-lane view, complemented independently and reassembled; no Torch bitwise arithmetic.','foreach/foreach_sub_list')


def foreach(case):
    import ctypes as C
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    name=op_name(case);api,directory=FOREACH[name];p=case.params
    if name=='DivFwdOp' and p.get('rounding_mode') is not None:
        raise UnsupportedCase('ForeachDivList implements true division, not floor/trunc division.')
    if any(isinstance(x,torch.Tensor) and x.dtype==torch.int64 for x in case.inputs):
        raise UnsupportedCase('Original Foreach API lacks int64; narrowing arbitrary int64 values is not lossless.')
    binding=new_binding();binding.codes.update({torch.int16:6,torch.float64:11})
    def run(*inputs):
        tensors=list(inputs)
        shape=torch.broadcast_shapes(*(t.shape for t in tensors))
        dt=getattr(torch,p.get('output',[None,str(tensors[0].dtype).split('.')[-1]])[1])
        tensors=[t.expand(shape).contiguous() for t in tensors]
        constants={}; names=[]
        if 'RoundOff' in api:
            modes={'RoundFwdOp':1,'FloorFwdOp':2,'CeilFwdOp':3,'TruncFwdOp':5}
            value=modes[name]
            tensors.append(torch.tensor(value,dtype=torch.int8,device=tensors[0].device))
            constants[1]=(C.c_int8*1)(value);names=['x0','roundMode','y0']
        elif name in ('SubFwdOp','LerpScalarFwdOp'):
            value=float(p.get('alpha',1)) if name=='SubFwdOp' else float(p['weight'])
            dtype=(torch.float16 if name=='SubFwdOp' and tensors[0].dtype==torch.float16 else
                   torch.float32 if tensors[0].dtype.is_floating_point else torch.int32)
            tensors.append(torch.tensor(value,dtype=dtype,device=tensors[0].device))
            if dtype==torch.float16:
                import struct
                constants[2]=(C.c_uint16*1)(int.from_bytes(struct.pack('<e',value),'little'))
            else:constants[2]=((C.c_float if dtype==torch.float32 else C.c_int32)*1)(value if dtype==torch.float32 else int(value))
            names=['x10','x20','alpha' if name=='SubFwdOp' else 'weight','y0']
        elif len(tensors)==1:names=['x0','y0']
        elif len(tensors)==2:names=['x10','x20','y0']
        else:names=['x10','x20','weight0','y0']
        out=torch.empty(shape,device=tensors[0].device,dtype=dt)
        binding.call(api,tensors,[out],{},names=names,constants=constants)
        return out
    return candidate(binding,run,api,'One tensor in each original dynamic list; materialize broadcast/strides each call; pass original scalar controls as constant tensor inputs.',directory,'foreach/foreach_utils')


def softmax(case):
    import torch
    from .catalog import OTHER
    api,directory=OTHER[op_name(case)];binding=new_binding();p=case.params
    def run(x):
        dtype=getattr(torch,p['dtype']) if p.get('dtype') else x.dtype
        x=x.to(dtype).contiguous();out=torch.empty_like(x)
        binding.call(api,[x],[out],{'axes':[int(p['dim'])]})
        return out
    return candidate(binding,run,api,'Original reduction axis and manifest accumulation/output dtype, per-call packing.',directory)


def dequant(case):
    import torch
    name=op_name(case);binding=new_binding()
    def run(q,scale):
        m,k=q.shape
        if 'PerTensor' in name:full=scale.expand(m,k)
        elif 'PerChannel' in name:full=scale.reshape(m,1).expand(m,k)
        else:full=scale.repeat_interleave(128,dim=1)[:,:k]
        out=torch.empty((m,k),device=q.device,dtype=torch.float32)
        binding.call('ForeachMulList',[q.to(torch.float32),full.contiguous()],[out],{},names=['x10','x20','y0'])
        return out.to(getattr(torch,case.params['out_dtype']))
    return candidate(binding,run,'ForeachMulList','Exact INT8-to-FP32 widening and scale broadcast/repetition; original vendor multiply; cast to declared output dtype inside run.','foreach/foreach_mul_list','foreach/foreach_utils')


def clamp(case):
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    binding=new_binding();p=case.params;name=op_name(case)
    def run(x,*bounds):
        if name=='HardtanhFwdOp':lower,upper=p['min_val'],p['max_val']
        elif name=='ClampScalarFwdOp':lower,upper=p.get('min'),p.get('max')
        else:lower,upper=bounds
        current=x.contiguous()
        for value,api in ((lower,'ForeachMaximumList'),(upper,'ForeachMinimumList')):
            if value is None:continue
            bound=value if isinstance(value,torch.Tensor) else torch.full((),value,device=x.device,dtype=x.dtype)
            shape=torch.broadcast_shapes(current.shape,bound.shape)
            out=torch.empty(shape,device=x.device,dtype=x.dtype)
            binding.call(api,[current.expand(shape).contiguous(),bound.expand(shape).contiguous()],[out],{},names=['x10','x20','y0'])
            current=out
        return current
    return candidate(binding,run,'ForeachMaximumList + ForeachMinimumList','Original max/min API pipeline implements both clamp bounds; every input broadcast and intermediate is charged per invocation.','foreach/foreach_maximum_list','foreach/foreach_minimum_list')


def masked_fill(case):
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    name=op_name(case);binding=new_binding()
    binding.codes.update({torch.int16:6,torch.float64:11})
    def run(x,mask,*values):
        shape=torch.broadcast_shapes(x.shape,mask.shape)
        xx=x.expand(shape).contiguous();mm=mask.expand(shape).contiguous()
        if values:
            if values[0].numel()!=1:raise UnsupportedCase('Original masked-fill contract requires a scalar value tensor.')
            updates=values[0].to(x.dtype).expand(xx.numel()).contiguous()
        else:updates=torch.full((xx.numel(),),case.params['value'],dtype=x.dtype,device=x.device)
        out=torch.empty_like(xx)
        binding.call('MaskedScatter',[xx,mm,updates],[out],{})
        return out
    return candidate(binding,run,'MaskedScatter','Original masked scatter with a repeated scalar update stream gives exact masked fill; all broadcasts and updates are materialized on each call.','index/masked_scatter')


def boolean_logic(case):
    import ctypes as C
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    name=op_name(case)
    if name.startswith('Bitwise') and any(t.dtype!=torch.bool for t in case.inputs):
        raise UnsupportedCase('This original arithmetic specialization is exact for boolean 0/1 inputs only; arbitrary numeric truth/bitwise conversion is not a lossless dtype mapping.')
    binding=new_binding()
    from .truth import truth01
    numeric=any(t.dtype!=torch.bool for t in case.inputs)
    api='ForeachMulList' if 'And' in name else 'ForeachMaximumList' if 'Or' in name else 'ForeachSubList'
    def run(*xs):
        shape=torch.broadcast_shapes(*(x.shape for x in xs));xx=[truth01(binding,x).expand(shape).contiguous() for x in xs]
        out=torch.empty(shape,device=xs[0].device,dtype=torch.float32)
        if 'Not' in name:xx=[torch.ones_like(xx[0]),xx[0]]
        if api=='ForeachSubList':
            scalar=torch.tensor(1.0,device=xs[0].device,dtype=torch.float32)
            binding.call(api,[*xx,scalar],[out],{},names=['x10','x20','alpha','y0'],constants={2:(C.c_float*1)(1.0)})
            if 'Xor' in name:
                absolute=torch.empty_like(out);binding.call('ForeachAbs',[out],[absolute],{},names=['x0','y0']);out=absolute
        else:binding.call(api,xx,[out],{},names=['x10','x20','y0'])
        return out.to(torch.bool)
    return candidate(binding,run,api+(' + ForeachAbs' if 'Xor' in name else '')+(' + LpNormV2' if numeric else ''),
                     'Original LpNormV2 p=0 singleton reduction converts numeric truth, including NaNs; lossless BOOL widening; native multiply/max/subtract/abs; exact 0/1-to-BOOL result cast and per-call broadcast.',
                     'foreach/foreach_mul_list','foreach/foreach_maximum_list','foreach/foreach_sub_list','foreach/foreach_abs','norm/lp_norm_v2')


def where(case):
    import torch
    binding=new_binding()
    def run(condition,x,y):
        shape=torch.broadcast_shapes(condition.shape,x.shape,y.shape)
        # A two-column layout makes the BOOL selection an exact native gather index.
        choices=torch.stack((y.expand(shape),x.expand(shape)),dim=-1).contiguous().reshape(-1,2)
        index=condition.expand(shape).contiguous().to(torch.int32).reshape(-1,1)
        out=torch.empty((index.shape[0],1),device=x.device,dtype=x.dtype)
        binding.call('GatherElements',[choices,index],[out],{'dim':1})
        return out.reshape(shape)
    return candidate(binding,run,'GatherElements','Stack the two broadcast inputs as two columns; lossless BOOL-to-index conversion; original own GatherElements performs every selection. No torch.where/indexed-select output computation.','index/gather_elements')
