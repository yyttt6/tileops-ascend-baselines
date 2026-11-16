"""Read-only manifest metadata resolution, without importing TileOPs workloads."""
from functools import lru_cache
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


@lru_cache(None)
def inventory():
    return json.loads((HERE/'inventory_snapshot.json').read_text())


@lru_cache(None)
def manifest(op):
    import yaml
    path=next(row['manifest'] for row in inventory()['operators'] if row['op']==op)
    return yaml.safe_load((ROOT/path).read_text())[op]


def parameters(row):
    p=dict(row.get('tensors',{}),**row['params'])
    opaque=[k for k,v in p.items() if isinstance(v,str) and v.startswith('<tileops.manifest.values.ADTValue ')]
    if not opaque:return p,{}
    spec=manifest(row['op']);matches=[]
    defaults={k:v.get('default') for k,v in spec['signature'].get('params',{}).items()}
    for workload in spec['workloads']:
        label=workload.get('label','')
        if not label or not (row['case_id']==label or row['case_id'].startswith(label+'-')):continue
        if any(k not in opaque and k in row['params'] and row['params'][k]!=workload.get(k,default)
               for k,default in defaults.items()):continue
        matches.append(workload)
    resolved={}
    for key in opaque:
        values={json.dumps(w[key],sort_keys=True) for w in matches if key in w}
        if len(values)==1:
            p[key]=json.loads(next(iter(values)));resolved[key]=p[key]
    return p,resolved
