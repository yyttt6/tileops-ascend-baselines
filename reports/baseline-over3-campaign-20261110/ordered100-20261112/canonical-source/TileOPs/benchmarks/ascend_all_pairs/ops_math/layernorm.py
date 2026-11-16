"""Use the complete original fast-launch LayerNorm for equal row-normalization semantics."""
from pathlib import Path
import hashlib,json
from math import prod
from .specs import ROOT,SOURCE,CACHE


def make(case,op):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,UnsupportedCase,source_hashes
    source=SOURCE/'examples/fast_kernel_launch_example/csrc/layer_norm/dav-2201/layer_norm.asc'
    binary=CACHE/'layernorm/libops_math_layernorm.so'; receipt=CACHE/'layernorm/receipt.json'
    if not binary.exists():raise RuntimeError('Original LayerNorm example has not linked for dav-3510.')
    r=json.loads(receipt.read_text())
    if r['source_sha256']!=hashlib.sha256(source.read_bytes()).hexdigest() or r.get('binary_sha256')!=hashlib.sha256(binary.read_bytes()).hexdigest():raise RuntimeError('Original LayerNorm source/binary receipt mismatch.')
    torch.ops.load_library(str(binary)); native=torch.ops.allpairs_math_example.layer_norm
    p=case.params;eps=p.get('eps');eps=torch.finfo(torch.float32).eps if eps is None else eps
    if op=='InstanceNormFwdOp' and (not p.get('use_input_stats',True) or any(v is not None for v in case.inputs[1:3])):raise UnsupportedCase('LayerNorm cannot satisfy the complete InstanceNorm tracked/eval running-statistics output contract.')
    if op=='BatchNormFwdOp' and (not p.get('training',True) or case.inputs[1] is not None):raise UnsupportedCase('LayerNorm does not return updated running statistics; only training without running-statistics tensors is equivalent.')
    def affine(v,k,default,device):
        return torch.full((k,),default,device=device,dtype=torch.float32) if v is None else v.float().contiguous().reshape(k)
    def row(x,w,b):
        k=x.shape[-1]
        return native(x.float().contiguous().reshape(-1,k),affine(w,k,1.,x.device),affine(b,k,0.,x.device),eps)
    def run(*args):
        x=args[0];shape=x.shape
        if op=='LayerNormFwdOp':
            w,b=args[1:];k=prod(p.get('normalized_shape',(shape[-1],)))
            out=row(x.reshape(-1,k),w,b).reshape(shape)
        elif op=='AdaLayerNormFwdOp':
            _,scale,shift=args;h=shape[-1];xr=x.reshape(-1,h);sr=scale.expand(shape).reshape(-1,h);br=shift.expand(shape).reshape(-1,h)
            out=torch.cat([row(xr[i:i+1],sr[i],br[i]) for i in range(xr.shape[0])],dim=0).reshape(shape)
        elif op=='GroupNormFwdOp':
            _,w,b=args;n,c=shape[:2];g=p['num_groups'];s=prod(shape[2:]);cg=c//g;k=cg*s;xx=x.reshape(n,g,k)
            if w is None and b is None:out=row(xx.reshape(-1,k),None,None).reshape(shape)
            else:
                values=[]
                for j in range(g):
                    ww=None if w is None else w[j*cg:(j+1)*cg,None].expand(cg,s).contiguous().reshape(k)
                    bb=None if b is None else b[j*cg:(j+1)*cg,None].expand(cg,s).contiguous().reshape(k)
                    values.append(row(xx[:,j],ww,bb))
                out=torch.stack(values,dim=1).reshape(shape)
        elif op in ('InstanceNormFwdOp','BatchNormFwdOp'):
            _,rm,rv,w,b=args;n,c=shape[:2];s=prod(shape[2:]);xx=x.reshape(n,c,s)
            if op=='BatchNormFwdOp':
                xx=xx.transpose(0,1).contiguous().reshape(c,n*s);k=n*s
                if w is None and b is None:out=row(xx,None,None)
                else:
                    out=torch.cat([row(xx[j:j+1],None if w is None else w[j:j+1].expand(k),None if b is None else b[j:j+1].expand(k)) for j in range(c)],dim=0)
                out=out.reshape(c,n,s).transpose(0,1).reshape(shape)
                return out.to(x.dtype),None,None
            if w is None and b is None:out=row(xx.reshape(n*c,s),None,None).reshape(shape)
            else:
                out=torch.stack([row(xx[:,j],None if w is None else w[j:j+1].expand(s),None if b is None else b[j:j+1].expand(s)) for j in range(c)],dim=1).reshape(shape)
        else:raise UnsupportedCase(op)
        return out.to(x.dtype)
    return Candidate(Implementation(run),'ops-math::examples/fast_kernel_launch_example.layer_norm (original complete source)',
      'Row layout and per-group/per-channel original API calls; original affine and epsilon; FP16/BF16 inputs widen exactly to FP32, output cast to manifest dtype; all packing within run.',
      source_hashes(Path(__file__),source,binary,receipt),limitations=('Host-only compilation is not validation; require native layer_norm_fp32 launch and full manifest receipt.',))
