"""Nonzero semantics through original unsigned masks and native bit operations."""
from pathlib import Path
from .merge_native import _builder


def _convert():
    import torch
    import torch.nn.functional as F
    from benchmarks.ascend_baselines import source_hashes
    tl,builder,hashes=_builder()
    kernels={op:tl.compile(builder(op,'int32'),target='ascend') for op in ('vand','vor','vxor')}
    hashes.update(source_hashes(Path(__file__)))
    def truth(x):
        flat=x.contiguous().flatten();n=flat.numel()
        # Complex values are true if either component is nonzero. Bit views
        # preserve FP64 components and avoid any complex-to-real narrowing.
        if flat.dtype in (torch.complex64,torch.complex128,torch.float64,torch.int64):
            count=flat.element_size()//4
            words=flat.view(torch.int32).reshape(-1,count)
            columns=[words[:,j].contiguous() for j in range(count)]
            clear={0,1} if flat.dtype==torch.complex64 else {1,3} if flat.dtype==torch.complex128 else {1} if flat.dtype==torch.float64 else set()
        elif flat.dtype.is_floating_point:
            columns=[flat.float().view(torch.int32)];clear={0}
        else:columns=[flat.to(torch.int32)];clear=set()
        columns=[F.pad(a,(0,-n%64)) for a in columns]
        zero=torch.zeros(64,device=x.device,dtype=torch.int32);one=torch.ones_like(zero);mask=one.view(torch.uint32)
        signless=torch.full_like(zero,0x7fffffff);acc=torch.empty_like(zero);temporary=torch.empty_like(zero);out=torch.empty_like(columns[0])
        for start in range(0,out.numel(),64):
            bits=columns[0][start:start+64]
            if 0 in clear:
                kernels['vand'](bits,signless,zero,mask,acc);bits=acc
            for j,column in enumerate(columns[1:],1):
                other=column[start:start+64]
                if j in clear:
                    kernels['vand'](other,signless,zero,mask,temporary);other=temporary
                kernels['vor'](bits,other,zero,mask,acc);bits=acc
            kernels['vxor'](one,zero,zero,bits.view(torch.uint32),out[start:start+64])
        return out[:n].reshape(x.shape)
    return truth,kernels,hashes


def logical(case):
    import torch
    import torch.nn.functional as F
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    convert,kernels,hashes=_convert();name=case._call.signature.name
    op='vxor' if name=='LogicalNotFwdOp' else 'vand' if name=='LogicalAndFwdOp' else 'vor'
    def run(x,y=None):
        shape=tuple(x.shape) if y is None else tuple(torch.broadcast_shapes(x.shape,y.shape))
        a=convert(x).expand(shape).contiguous().flatten()
        b=torch.ones_like(a) if y is None else convert(y).expand(shape).contiguous().flatten()
        n=a.numel();a=F.pad(a,(0,-n%64));b=F.pad(b,(0,-n%64));out=torch.empty_like(a)
        zero=torch.zeros(64,device=x.device,dtype=torch.int32);mask=torch.ones_like(zero).view(torch.uint32)
        for start in range(0,a.numel(),64):kernels[op](a[start:start+64],b[start:start+64],zero,mask,out[start:start+64])
        return out[:n].to(torch.bool).reshape(shape)
    return Candidate(Implementation(run),'tilelang::merging_kernel native nonzero + '+op,
                     'Original unsigned mask comparison performs nonzero conversion. Original AND removes float sign bits so both signed zeros are false; original OR combines integer limbs and both complex components; FP64 bits are never narrowed. Native 0/1 bit operations perform logic; only final exact 0/1-to-bool cast is external.',hashes)


def extrema(case):
    """Retired: test-kernel truth conversion plus a custom reduction tree."""
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase(
        "TileLang All/Any composed truth-reduction route retired by user: "
        "no restored complete original API is registered for this operation.")


def count_nonzero(case):
    """Retired: test-kernel truth conversion plus a custom reduction tree."""
    from benchmarks.ascend_baselines import UnsupportedCase
    raise UnsupportedCase(
        "TileLang CountNonzero composed truth-reduction route retired by user: "
        "no restored complete original API is registered for this operation.")
