"""Pure-CPU pooling launch metadata, including padding-slot and batch indices."""
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys

ROOT=Path(__file__).resolve().parents[4]
CACHE=ROOT/'.cache/all-pairs/sgl/pool_layouts'


def load(name,shape,output_shape,params):
    from .pool_layout import key
    path=CACHE/('launch_'+key(name,shape,output_shape,params)+'.pkl')
    if not path.exists():
        raise RuntimeError('Pooling launch metadata must be prepared outside the NPU reservation: '+str(path))
    with path.open('rb') as stream:return pickle.load(stream),path


def main():
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(ROOT/'TileOPs'))
    from benchmarks.ascend_all_pairs.sgl.pool_layout import load as windows,key
    inventory=json.loads((ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/sgl/inventory_snapshot_v2.json').read_text())
    records=[]
    for case in inventory['cases']:
        name=case['op'];ts=case['tensors']
        if 'Pool' not in name or name=='MeanPoolingFwdOp' or 'input' not in ts:continue
        shape=ts['input'][0];output=ts['output'][0];params=case['params']
        path=CACHE/('launch_'+key(name,shape,output,params)+'.pkl')
        if path.exists():continue
        ndim=2 if 'Adaptive' in name else int(name.split('Pool')[1][0])
        volume=math.prod(shape[-ndim:])
        parts=[]
        for (width,divisor),items in windows(name,shape,output,params).items():
            for start in range(0,len(items),1024):
                part=items[start:start+1024]
                positions=[r for _,r in part]
                gathers=[[v if v>=0 else volume for v in r] for r in positions]
                parts.append((positions,gathers,[i for i,_ in part],width,divisor))
        with path.open('xb') as f:pickle.dump(parts,f,protocol=5)
        records.append({'op':name,'case_id':case['case_id'],'file':str(path.relative_to(ROOT)),'parts':len(parts),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    report=ROOT/'.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/sgl/continuation_20261021T1415/pool_launch_metadata_preparation.json'
    with report.open('x') as f:json.dump({'torch_imported':'torch' in sys.modules,'tables':records},f,indent=2)
    print(json.dumps({'tables':len(records),'torch_imported':'torch' in sys.modules}))


if __name__=='__main__':main()
