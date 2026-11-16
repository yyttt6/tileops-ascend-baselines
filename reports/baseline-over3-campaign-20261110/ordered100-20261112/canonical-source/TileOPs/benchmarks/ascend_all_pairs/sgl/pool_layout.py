"""CPU-only frozen pooling window tables, prepared before device reservations."""
from collections import defaultdict
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import pickle

ROOT=Path(__file__).resolve().parents[4]
CACHE=ROOT/'.cache/all-pairs/sgl/pool_layouts'


def key(name,shape,output_shape,p):
    fields=('kernel_size','stride','padding','dilation','count_include_pad','divisor_override')
    return hashlib.sha256(json.dumps([name,list(shape),list(output_shape),{k:p.get(k) for k in fields}],sort_keys=True).encode()).hexdigest()


def windows(name,shape,output_shape,p):
    maximum='Max' in name
    adaptive='Adaptive' in name
    ndim=2 if adaptive else int(name.split('Pool')[1][0])
    spatial=tuple(shape[-ndim:]);outsp=tuple(output_shape[-ndim:])
    def tup(v):return (v,)*ndim if isinstance(v,int) else tuple(v)
    if not adaptive:
        kernel=tup(p['kernel_size']);stride=tup(p.get('stride') or p['kernel_size'])
        padding=tup(p.get('padding',0));dilation=tup(p.get('dilation',1))
    groups=defaultdict(list)
    strides=[math.prod(spatial[d+1:]) for d in range(ndim)]
    for oi,coord in enumerate(product(*(range(s) for s in outsp))):
        if adaptive:
            axes=[range(c*n//m,((c+1)*n+m-1)//m) for c,n,m in zip(coord,spatial,outsp)]
        else:
            axes=[range(c*s-pad,c*s-pad+d*(k-1)+1,d) for c,s,pad,d,k in zip(coord,stride,padding,dilation,kernel)]
        positions=[]
        for loc in product(*axes):
            valid=all(0<=v<n for v,n in zip(loc,spatial))
            if valid:positions.append(sum(v*s for v,s in zip(loc,strides)))
            elif maximum:positions.append(-1)
            elif p.get('count_include_pad',True) and all(-pad<=v<n+pad for v,n,pad in zip(loc,spatial,padding)):
                positions.append(-1)
        if not positions:raise ValueError('Empty pooling window')
        divisor=p.get('divisor_override') or len(positions)
        groups[(len(positions),divisor)].append((oi,positions))
    return dict(groups)


def load(name,shape,output_shape,p):
    path=CACHE/(key(name,shape,output_shape,p)+'.pkl')
    if path.exists():
        with path.open('rb') as stream:return pickle.load(stream)
    raise RuntimeError('Shape-only pooling tables must be prepared on CPU before reserving the NPU: '+str(path))


def main():
    import time
    inventory=json.loads((ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/sgl/inventory_snapshot_v2.json').read_text())
    CACHE.mkdir(parents=True,exist_ok=True)
    prepared=[]
    for case in inventory['cases']:
        name=case['op']
        if 'Pool' not in name or name=='MeanPoolingFwdOp':continue
        tensors=case['tensors']
        if 'input' not in tensors or 'output' not in tensors:continue
        shape=tensors['input'][0];out=tensors['output'][0];params=case['params']
        path=CACHE/(key(name,shape,out,params)+'.pkl')
        if path.exists():continue
        started=time.monotonic()
        data=windows(name,shape,out,params)
        with path.open('xb') as stream:pickle.dump(data,stream,protocol=5)
        prepared.append({'op':name,'case_id':case['case_id'],'table':str(path.relative_to(ROOT)),'groups':len(data),'windows':sum(len(v) for v in data.values()),'bytes':path.stat().st_size,'seconds':time.monotonic()-started,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (CACHE/'preparation.json').write_text(json.dumps(prepared,indent=2))
    print(json.dumps({'prepared_tables':len(prepared),'torch_imported':False,'total_seconds':sum(r['seconds'] for r in prepared)}))


if __name__=='__main__':main()
