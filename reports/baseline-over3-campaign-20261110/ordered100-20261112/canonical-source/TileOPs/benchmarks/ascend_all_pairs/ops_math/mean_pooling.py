"""Original ReduceMean over packed original chunks; sequence boundaries stay intact."""
from types import SimpleNamespace
from pathlib import Path
from .specs import SOURCE

def make(case):
    import torch
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate,source_hashes
    from .graph import make as graph_make, deployment_hashes
    p=case.params; chunk=p['chunk_size']; children={}
    def reduce(x):
        key=(tuple(x.shape),x.dtype)
        if key not in children:
            shape=list(x.shape);shape.pop(2)
            children[key]=graph_make(SimpleNamespace(params={'output':(shape,str(x.dtype).split('.')[-1]),'dim':2,'keepdim':False}), 'MeanFwdOp')
        return children[key].implementation.run(x)
    def run(x,offsets=None,indices=None):
        b,t,h,d=x.shape
        if offsets is None and t%chunk==0:
            return reduce(x.contiguous().reshape(b,t//chunk,chunk,h,d))
        if offsets is None: slices=[(s,min(s+chunk,t)) for s in range(0,t,chunk)]
        else:
            bounds=offsets.tolist()
            slices=[(bounds[seq]+i*chunk,min(bounds[seq]+(i+1)*chunk,bounds[seq+1])) for seq,i in indices.tolist()]
        groups={}
        for i,(s,e) in enumerate(slices):groups.setdefault(e-s,[]).append((i,s,e))
        out=torch.empty((b,len(slices),h,d),dtype=x.dtype,device=x.device)
        for length,parts in groups.items():
            packed=torch.stack([x[:,s:e] for _,s,e in parts],dim=1)
            value=reduce(packed)
            target=torch.tensor([i for i,_,_ in parts],device=x.device,dtype=torch.int64)
            out.index_copy_(1,target,value)
        return out
    def cleanup():
        for child in children.values():child.cleanup()
    original=SOURCE/'math/reduce_mean'
    sources=[f for f in original.rglob('*') if f.is_file() and f.suffix in ('.cpp','.h') and 'tests' not in f.parts]
    return Candidate(Implementation(run),'ops-math::ReduceMean(original chunk packing)',
      'Pack equal-length original chunks, reduce their sequence axis, restore original chunk ordering. Runtime offsets/indices are read on every call; only shape-keyed native executables cached.',
      source_hashes(Path(__file__),Path(__file__).with_name('graph.py'),*sources) | deployment_hashes(),timing_mode='host_roundtrip',cleanup=cleanup)
