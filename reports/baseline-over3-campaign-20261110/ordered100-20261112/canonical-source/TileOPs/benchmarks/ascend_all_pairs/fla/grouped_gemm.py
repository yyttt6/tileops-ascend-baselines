"""Original FLA window-dot GEMMs with current per-group metadata."""
from pathlib import Path
from types import SimpleNamespace


def grouped_gemm(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,source_hashes
    from .providers import _triton_module
    from .spatial import convolution
    _,source=_triton_module('causal_conv1d')
    p=case.params
    cube=None
    hashes=source_hashes(Path(__file__),Path(__file__).with_name('spatial.py'),source)
    if case.inputs[0].dtype==torch.bfloat16:
        from .cube_matmul import make_matmul
        cube,cube_hashes=make_matmul()
        hashes.update(cube_hashes)

    def dot(left,right):
        # right is already transposed into [output channels, reduction width].
        if left.shape[1]==0:
            return torch.zeros((left.shape[0],right.shape[0]),dtype=left.dtype,device=left.device)
        if cube is not None:
            return cube(left,right.t().contiguous())
        a=left.unsqueeze(-1);w=right.unsqueeze(-1)
        view=SimpleNamespace(inputs=(a,w),params=dict(stride=1,padding=0,dilation=1,groups=1))
        return convolution(view).implementation.run(a,w)[...,0]

    def run(a,b,batch_sizes,batch_offsets):
        sizes=batch_sizes.cpu().tolist();offsets=batch_offsets.cpu().tolist()
        if not p['transpose_a']:
            columns=b.shape[1] if p['transpose_b'] else b.shape[2]
            result=torch.empty((a.shape[0],columns),device=a.device,dtype=a.dtype)
            for group,(start,size) in enumerate(zip(offsets,sizes,strict=True)):
                if size:
                    weight=b[group] if p['transpose_b'] else b[group].t().contiguous()
                    result[start:start+size]=dot(a[start:start+size],weight)
            return result
        results=[]
        for start,size in zip(offsets,sizes,strict=True):
            left=a[start:start+size].t().contiguous()
            right=b[:,start:start+size] if p['transpose_b'] else b[start:start+size].t().contiguous()
            results.append(dot(left,right))
        return torch.stack(results)

    return Candidate(Implementation(run),'FLA ChunkFwdH final-state matrix products' if cube is not None else 'FLA causal_conv1d_fwd_impl_old(grouped matrix window dots)',
        'Read current sizes and offsets inside run; slice each group and select transpose layout, retaining empty groups. BF16 products use original ChunkFwdH FP32 final-state accumulation; other dtypes use original FLA window-dot/residual composition. Private complete output and all metadata/packing included in host-roundtrip timing.',hashes)
