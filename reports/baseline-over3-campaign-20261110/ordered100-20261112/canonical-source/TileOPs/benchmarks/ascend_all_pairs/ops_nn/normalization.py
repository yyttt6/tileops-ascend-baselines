"""Manifest normalization adapters using original ops-nn APIs."""
from math import prod
from .catalog import NORM
from .common import candidate,new_binding,op_name


def normalization(case):
    import ctypes as C
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    name=op_name(case);p=case.params;api,directory=NORM[name];binding=new_binding()
    eps=p.get('eps');eps=float(torch.finfo(torch.float32).eps if eps is None else eps)
    instance_running=name=='InstanceNormFwdOp' and case.inputs[1] is not None
    instance_training=instance_running and p['use_input_stats']
    if instance_running:api,directory='BatchNormV3','norm/batch_norm_v3'
    def run(x,*inputs):
        if name=='VectorNormFwdOp':
            dtype=getattr(torch,p['dtype']) if p.get('dtype') else x.dtype
            xx=x.to(dtype).contiguous();out=torch.empty(tuple(p['output'][0]),device=x.device,dtype=dtype)
            dim=p.get('dim');axes=list(range(x.ndim)) if dim is None else [dim] if isinstance(dim,int) else list(dim)
            binding.call(api,[xx],[out],{'p':float(p['ord']),'axes':axes,'keepdim':bool(p['keepdim']),'epsilon':0.0})
            return out
        if name in ('GroupNormFwdOp','InstanceNormFwdOp') and not instance_running:
            if name=='GroupNormFwdOp':weight,bias=inputs;groups=int(p['num_groups'])
            else:_,_,weight,bias=inputs;groups=x.shape[1]
            weight=torch.ones(x.shape[1],device=x.device,dtype=x.dtype) if weight is None else weight.contiguous()
            bias=torch.zeros(x.shape[1],device=x.device,dtype=x.dtype) if bias is None else bias.contiguous()
            out=torch.empty_like(x);mean=torch.empty((x.shape[0],groups),device=x.device,dtype=weight.dtype);rstd=torch.empty_like(mean)
            binding.call(api,[x.contiguous(),weight,bias],[out,mean,rstd],{'num_groups':groups,'data_format':'NCHW','eps':eps,'is_training':True})
            return out
        if name=='BatchNormFwdOp' or instance_running:
            rm,rv,weight,bias=inputs;c=x.shape[1]
            dtype=torch.float32
            weight=torch.ones(c,device=x.device,dtype=dtype) if weight is None else weight.to(dtype)
            bias=torch.zeros(c,device=x.device,dtype=dtype) if bias is None else bias.to(dtype)
            mean=torch.zeros(c,device=x.device,dtype=dtype) if rm is None else rm.to(dtype).clone()
            var=torch.ones(c,device=x.device,dtype=dtype) if rv is None else rv.to(dtype).clone()
            if instance_training:
                weight=weight.repeat(x.shape[0]);bias=bias.repeat(x.shape[0]);mean=mean.repeat(x.shape[0]);var=var.repeat(x.shape[0])
                xx=x.float().contiguous().reshape(1,x.shape[0]*c,1,-1)
            else:xx=x.float().contiguous().reshape(x.shape[0],c,1,-1)
            out=torch.empty_like(xx)
            save_mean=torch.empty_like(mean);save_rstd=torch.empty_like(mean)
            binding.call(api,[xx,weight,bias,mean,var],[out,mean,var,save_mean,save_rstd],
                         {'epsilon':eps,'momentum':float(p['momentum']),'is_training':bool(p['use_input_stats'] if instance_running else p['training'])},
                         formats=[0,2,2,2,2,0,2,2,2,2])
            if instance_training:
                updated=[]
                for value in (mean,var):
                    packed=value.reshape(x.shape[0],c).T.contiguous().reshape(c,1,1,x.shape[0])
                    avg=torch.empty((c,1,1,1),device=x.device,dtype=torch.float32)
                    binding.call('AdaptiveAvgPool2d',[packed],[avg],{'output_size':[1,1]});updated.append(avg.reshape(c))
                mean,var=updated
            return out.reshape(x.shape).to(x.dtype), None if rm is None else mean, None if rv is None else var
        if name=='BatchNormBwdOp':
            grad=x;x,weight,mean,rstd=inputs
            xx=x.contiguous().reshape(x.shape[0],x.shape[1],1,-1);gg=grad.contiguous().reshape(xx.shape)
            dx=torch.empty_like(xx);dw=torch.empty_like(weight);db=torch.empty_like(weight)
            binding.call(api,[gg,xx,weight.contiguous(),torch.zeros_like(mean),torch.ones_like(mean),mean.contiguous(),rstd.contiguous()],[dx,dw,db],{'is_training':True,'epsilon':eps},formats=[0,0,2,2,2,2,2,0,2,2])
            return dx.reshape(x.shape),dw,db
        hidden=prod(p.get('normalized_shape') or [x.shape[-1]]);rows=x.numel()//hidden
        xx=x.contiguous().reshape(rows,hidden);out=torch.empty_like(xx)
        stats=torch.empty((rows,1),device=x.device,dtype=torch.float32)
        if name=='LayerNormFwdOp':
            weight,bias=inputs;shape=torch.tensor([hidden],device=x.device,dtype=torch.int32)
            # Exact widening matches the manifest's explicit FP32 normalization and affine arithmetic.
            xx=xx.float();out=torch.empty_like(xx)
            weight=None if weight is None else weight.float().contiguous().reshape(hidden)
            bias=None if bias is None else bias.float().contiguous().reshape(hidden)
            binding.call(api,[xx,shape,weight,bias],[out,stats,torch.empty_like(stats)],{'epsilon':eps},constants={1:(C.c_int32*1)(hidden)})
            return out.reshape(x.shape).to(x.dtype)
        residual,*rest=inputs
        rr=residual.contiguous().reshape(rows,hidden).clone();xx=xx.clone()
        if name=='FusedAddRMSNormFwdOp':
            weight=rest[0];weight=torch.ones(hidden,device=x.device,dtype=x.dtype) if weight is None else weight.contiguous().reshape(hidden)
            binding.call(api,[xx,rr,weight],[xx,stats,rr],{'epsilon':eps})
            normalized=torch.empty_like(rr,dtype=torch.float32)
            binding.call('RmsNorm',[rr.float(),weight.float()],[normalized,torch.empty_like(stats)],{'epsilon':eps})
            return normalized.to(x.dtype).reshape(x.shape),rr.reshape(x.shape)
        weight,bias=rest
        weight=torch.ones(hidden,device=x.device,dtype=x.dtype) if weight is None else weight.contiguous().reshape(hidden)
        bias=torch.zeros(hidden,device=x.device,dtype=x.dtype) if bias is None else bias.contiguous().reshape(hidden)
        summed=torch.empty_like(xx)
        binding.call(api,[xx,rr,weight,bias,None],[out,stats,torch.empty_like(stats),summed],{'epsilon':eps,'additional_output':True})
        shape=torch.tensor([hidden],device=x.device,dtype=torch.int32)
        normalized=torch.empty_like(summed,dtype=torch.float32)
        binding.call('LayerNormV4',[summed.float(),shape,weight.float(),bias.float()],
                     [normalized,torch.empty_like(stats),torch.empty_like(stats)],{'epsilon':eps},constants={1:(C.c_int32*1)(hidden)})
        return normalized.to(x.dtype).reshape(x.shape),summed.reshape(x.shape)
    extra_api={'FusedAddRMSNormFwdOp':' + RmsNorm','FusedAddLayerNormFwdOp':' + LayerNormV4'}.get(name,'')
    extra_dirs={'FusedAddRMSNormFwdOp':('norm/rms_norm',),'FusedAddLayerNormFwdOp':('norm/layer_norm_v4',)}.get(name,())
    if instance_training:extra_api=' + AdaptiveAvgPool2d';extra_dirs=('pooling/adaptive_avg_pool2d',)
    return candidate(binding,run,api+extra_api,'Preserve complete normalization axes, epsilon and every manifest output; initialize optional identity affine parameters; private copies for mutable native aliases; FP32 widening only where exact. Fused variants consume the native sum output after manifest dtype rounding in a second native normalization.',directory,*extra_dirs)


def count_nonzero(case):
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    binding=new_binding();p=case.params
    if case.inputs[0].dtype==torch.complex128:raise UnsupportedCase('Original p=0 has no FP64 lane; no exact complex128 representation specialization is implemented.')
    from .truth import truth01
    def run(x):
        dim=p.get('dim');axes=list(range(x.ndim)) if dim is None else [dim] if isinstance(dim,int) else list(dim)
        count=1
        for d in axes:count*=x.shape[d]
        if count>2**24:raise UnsupportedCase('FP32 p=0 count no longer guarantees exact integer output above 2^24.')
        out=torch.empty(tuple(p['output'][0]),device=x.device,dtype=torch.float32)
        xx=x.float().contiguous() if x.dtype in (torch.float16,torch.bfloat16,torch.float32,torch.bool) else truth01(binding,x)
        binding.call('LpNormV2',[xx],[out],{'p':0.0,'axes':axes,'keepdim':False,'epsilon':0.0})
        return out.to(torch.int64)
    return candidate(binding,run,'LpNormV2(p=0)','Exact FP32 widening, or original p=0 applied to all integer bytes / both complex64 components, and exactly representable integer count (<=2^24); original p=0 reduction; lossless integer output conversion.','norm/lp_norm_v2')


def variance(case):
    import torch
    from math import prod
    from benchmarks.ascend_baselines import UnsupportedCase
    name=op_name(case);p=case.params;binding=new_binding()
    def run(x):
        dims=p.get('dim');dims=tuple(range(x.ndim)) if dims is None else (dims,) if isinstance(dims,int) else tuple(dims)
        dims=tuple(d%x.ndim for d in dims);outer=tuple(d for d in range(x.ndim) if d not in dims)
        width=prod(x.shape[d] for d in dims);rows=x.numel()//width
        correction=int(p.get('correction',1))
        if width<=correction:raise UnsupportedCase('Nonpositive variance divisor requires nonfinite special semantics absent from this original auxiliary-output route.')
        xx=x.permute(*outer,*dims).contiguous().reshape(rows,width).float()
        mean=torch.empty((rows,1),device=x.device,dtype=torch.float32);var=torch.empty_like(mean)
        binding.call('LayerNorm',[xx,torch.ones(width,device=x.device),torch.zeros(width,device=x.device)],
                     [torch.empty_like(xx),mean,var],{'begin_norm_axis':1,'begin_params_axis':1,'epsilon':1e-5})
        if correction:
            factor=torch.full(var.shape,width/(width-correction),device=x.device,dtype=torch.float32)
            vv=torch.empty_like(var)
            binding.call('ForeachMulList',[var,factor],[vv],{},names=['x10','x20','y0']);var=vv
        if name=='StdFwdOp':
            std=torch.empty_like(var);binding.call('ForeachSqrt',[var],[std],{},names=['x0','y0']);var=std
        if name=='VarMeanFwdOp':
            shape=tuple(p['var'][0]);return var.reshape(shape).to(x.dtype),mean.reshape(shape).to(x.dtype)
        return var.reshape(tuple(p['output'][0])).to(x.dtype)
    return candidate(binding,run,'LayerNorm' + (' + ForeachMulList' if int(p.get('correction',1)) else '') + (' + ForeachSqrt' if name=='StdFwdOp' else ''),
                     'Select original population mean/variance auxiliary outputs after complete-axis packing and exact FP32 widening; original vendor multiply applies manifest correction; native sqrt for standard deviation; restore shape/dtype.','norm/layer_norm','foreach/foreach_mul_list','foreach/foreach_sqrt')


def ada_layer_norm(case):
    import ctypes as C
    import torch
    binding=new_binding();eps=float(case.params['eps'])
    def run(x,scale,shift,*gate):
        n=x.shape[-1];rows=x.numel()//n
        xx=x.float().contiguous().reshape(rows,n);normed=torch.empty_like(xx)
        stats=torch.empty((rows,1),device=x.device,dtype=torch.float32)
        shape=torch.tensor([n],device=x.device,dtype=torch.int32)
        binding.call('LayerNormV4',[xx,shape,None,None],[normed,stats,torch.empty_like(stats)],
                     {'epsilon':eps},constants={1:(C.c_int32*1)(n)})
        normed=normed.reshape(x.shape);scaled=torch.empty_like(normed);shifted=torch.empty_like(normed)
        binding.call('ForeachMulList',[normed,scale.float().expand_as(normed).contiguous()],[scaled],{},names=['x10','x20','y0'])
        alpha=torch.tensor(-1.0,device=x.device,dtype=torch.float32)
        binding.call('ForeachSubList',[scaled,shift.float().expand_as(scaled).contiguous(),alpha],[shifted],{},
                     names=['x10','x20','alpha','y0'],constants={2:(C.c_float*1)(-1.0)})
        if gate:
            out=torch.empty_like(shifted)
            binding.call('ForeachMulList',[shifted,gate[0].float().expand_as(shifted).contiguous()],[out],{},names=['x10','x20','y0'])
        else:out=shifted
        return out.to(x.dtype)
    return candidate(binding,run,'LayerNormV4 + ForeachMulList + ForeachSubList',
                     'Original API composition exactly preserves manifest scale-only modulation and optional gate. Exact FP32 widening, native normalization/multiply/subtract(alpha=-1) and final manifest cast, charged per call.',
                     'norm/layer_norm_v4','foreach/foreach_mul_list','foreach/foreach_sub_list')
