"""Regenerate exact frozen input from original descriptor, without unrelated providers.

No registry is disabled or changed. This worker uses an explicitly pinned API,
so its unrelated ops-nn import-time legacy registration is not an input generator.
Only a bit-identical original payload/descriptor is accepted.
"""
import os,sys,importlib.abc
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
class NoNPU(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0] in ('torch_npu','acl'):
            raise ModuleNotFoundError('CPU input preparation forbids '+name,name=name)
sys.meta_path.insert(0,NoNPU())
from pathlib import Path
import json,pickle,hashlib,base64,datetime,shutil
P=Path(__file__).resolve().parent;T=P/'workspace/TileOPs'
sys.path[:0]=[str(P),str(T),str(T/'src')]
sys.path.append('/opt/tileops-ci/runtime/lib/python3.12/site-packages')
import torch
from workloads.device import set_run_device
from workloads.numerics import describe
set_run_device('cpu');torch.set_num_threads(4)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cert=json.loads((P/'GOLDEN42_CERTIFICATES.json').read_text())
jobs=json.loads(Path(sys.argv[1]).read_text())
sources={str(f):sha(f) for folder in [T/'benchmarks',T/'workloads',T/'src/tileops/manifest'] for f in folder.rglob('*.py')}
sources[str(P/'reference_policy.py')]=sha(P/'reference_policy.py');sources[str(P/'TASKS.json')]=sha(P/'TASKS.json')
for job in jobs:
    i=job['ordinal'];m=job['metadata'];proof=next(r for r in cert['rows'] if r['ordinal']==i)
    assert all(m[k]==proof[k] for k in ('key','params','payload_sha256','descriptor_sha256'))
    d=P/'prepared-rollout'/str(i)
    if (d/'prepare.json').exists():
        assert sha(d/'case.pkl')==m['descriptor_sha256'] and sha(d/'payload.pt')==m['payload_sha256']
        print('EXACT_INPUT_ALREADY_PREPARED',i,flush=True);continue
    raw=base64.b64decode(job['descriptor'])
    assert hashlib.sha256(raw).hexdigest()==m['descriptor_sha256']
    case=pickle.loads(raw);torch.manual_seed(m['input_seed']);args=case.inputs
    assert json.loads(json.dumps(case.params))==m['params'] and describe(case.verification)==m['verification']
    storage=Path('/workspace/user_data/baseline-over3-20261110/ordered-inputs')/str(i)
    storage.mkdir(parents=True,exist_ok=True)
    assert min(shutil.disk_usage('/').free,shutil.disk_usage(storage).free)>=20_000_000_000
    target=storage/'payload.pt';assert not target.exists(),'Preserve incomplete prior input, do not overwrite'
    torch.save({'inputs':args},target)
    assert sha(target)==m['payload_sha256'],'Frozen payload bits changed: never submit'
    d.mkdir(parents=True,exist_ok=True);(d/'case.pkl').write_bytes(raw);(d/'payload.pt').symlink_to(target)
    new=dict(m,sources=sources,payload_storage=str(target))
    (d/'prepare.json').write_text(json.dumps(new,indent=2))
    (d/'FROZEN_DESCRIPTOR_REGENERATION.json').write_text(json.dumps({'checked_at_remote':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ordinal':i,'CPU_only':True,'original_generator_and_seed':True,'exact_payload_and_descriptor_sha256_verified':True,'source_preparation_sha256':sha(__file__),'no_legacy_registry_source_modified':True},indent=2))
    print('EXACT_FROZEN_INPUT_REGENERATED',i,flush=True)
