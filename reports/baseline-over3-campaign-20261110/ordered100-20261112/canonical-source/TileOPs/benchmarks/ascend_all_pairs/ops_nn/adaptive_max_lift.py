"""Use the original 3D API with a singleton depth for exact 2D adaptive max."""
from .common import candidate,new_binding,op_name


def factory(case):
    import torch
    from benchmarks.ascend_baselines import UnsupportedCase
    p=case.params;name=op_name(case);binding=new_binding();indices_requested='Indices' in name
    if case.inputs[0].ndim not in (3,4):raise UnsupportedCase('Adaptive 2D lift requires CHW or NCHW input.')
    if case.inputs[0].shape[-2]*case.inputs[0].shape[-1]>=2**31:
        raise UnsupportedCase('This inspected original 3D specialization uses exact INT32 flattened indices.')
    output_key='output' if 'output' in p else 'values';out_shape=tuple(p[output_key][0])
    def run(x):
        base=(1,*x.shape) if x.ndim==3 else tuple(x.shape)
        xx=x.contiguous().reshape(base[0],base[1],1,base[2],base[3])
        size=p['output_size'];size=(size,size) if isinstance(size,int) or size is None else tuple(size)
        height,width=(x.shape[-2+i] if v is None else int(v) for i,v in enumerate(size))
        values=torch.empty((base[0],base[1],1,height,width),device=x.device,dtype=x.dtype)
        indices=torch.empty_like(values,dtype=torch.int32)
        binding.call('AdaptiveMaxPool3d',[xx],[values,indices],{'output_size':[1,height,width],'indices_dtype':3})
        values=values.reshape(out_shape)
        return (values,indices.to(torch.int64).reshape(out_shape)) if indices_requested else values
    return candidate(binding,run,'AdaptiveMaxPool3d',
                     'Select original adaptive 3D max with singleton depth and output depth one. Exact 2D bin boundaries and flattened H*W indices are preserved; restore optional batch and complete INT64 index output. This avoids the separately recorded original AdaptiveMaxPool2d SIMT stack fault without kernel or schedule edits.',
                     'pooling/adaptive_max_pool3d')
