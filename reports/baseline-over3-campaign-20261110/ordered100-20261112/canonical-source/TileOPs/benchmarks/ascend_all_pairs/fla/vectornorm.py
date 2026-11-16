"""L2 vector norm from the original FLA L2 reciprocal-norm outputs."""
from math import prod
from pathlib import Path


def vectornorm(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate, UnsupportedCase, source_hashes
    from .providers import _triton_module
    order=case.params.get('ord',2)
    if order not in (1,2):
        raise UnsupportedCase('Original L2 and scalar-L2 plus sum routes implement ord=2 and ord=1. No original maximum reduction API is matched for infinity.')
    norm,source=_triton_module('l2norm')
    p=case.params
    hashes=source_hashes(Path(__file__),source,source.with_name('utils.py'))
    if order==1:
        from .absolute import absolute
        magnitude=absolute(case)
        scan,scan_source=_triton_module('cumsum')
        hashes.update(magnitude.source_sha256|source_hashes(scan_source))

    def norm_values(data):
        # The second native call returns 1/abs(1/L2(data)) = L2(data).
        # Its normalized-value output is unused; zero rows yield inf then zero.
        if data.shape[-1]<=65536//data.element_size():
            _,inverse=norm.l2norm_fwd(data,eps=0.0,output_dtype=torch.float32)
        else:
            lengths=[norm_values(data[:,start:start+8192].contiguous()) for start in range(0,data.shape[-1],8192)]
            _,inverse=norm.l2norm_fwd(torch.stack(lengths,dim=1),eps=0.0,output_dtype=torch.float32)
        _,length=norm.l2norm_fwd(inverse.reshape(-1,1),eps=0.0,output_dtype=torch.float32)
        return length

    def run(x):
        dims=p.get('dim')
        dims=tuple(range(x.ndim)) if dims is None else ((dims,) if isinstance(dims,int) else tuple(dims))
        dims=tuple(d%x.ndim for d in dims)
        left=tuple(d for d in range(x.ndim) if d not in dims)
        width=prod(x.shape[d] for d in dims)
        data=x.permute(left+dims).contiguous().reshape(-1,width)
        shape=tuple(1 if d in dims else x.shape[d] for d in range(x.ndim)) if p.get('keepdim',False) else tuple(x.shape[d] for d in left)
        dtype=getattr(torch,p['dtype']) if p.get('dtype') else x.dtype
        if order==1:
            import torch.nn.functional as F
            sums=magnitude.implementation.run(data.float())
            rows,parts=sums.shape
            while parts>16384:
                chunks=(parts+4095)//4096
                packed=F.pad(sums,(0,chunks*4096-parts)).reshape(-1,4096,1)
                sums=scan.chunk_local_cumsum(packed,chunk_size=4096,output_dtype=torch.float32)[:,-1,0].reshape(rows,chunks)
                parts=chunks
            extent=1<<(parts-1).bit_length()
            packed=F.pad(sums,(0,extent-parts)).reshape(-1,extent,1)
            value=scan.chunk_local_cumsum(packed,chunk_size=extent,output_dtype=torch.float32)[:,parts-1,0]
        else:value=norm_values(data)
        return value.to(dtype).reshape(shape)

    return Candidate(Implementation(run),'FLA l2norm_fwd(rstd) composed with scalar l2norm_fwd(rstd)',
        'Permute/flatten requested reduction dimensions; obtain original reciprocal L2, then original scalar reciprocal L2 to recover the norm. L1 uses the same original scalar magnitude followed by original cumulative sum. Long rows compose bounded original calls. Zero/inf rows preserved by reciprocal composition. No torch arithmetic on values or kernel/schedule edits.',hashes)
