"""Six exact server1 GDN contracts, verified with original FP64 CPU reference."""
import hashlib, importlib.util, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]/'reports/baseline-single-agent-server1-20261027/fp64_fix'
PINS={'sgl_adapter.py': '4154c11fd85ae41eca7878bd9e054124c5c85353c8bf86625e8f4a23048aeabf', 'SGL_VERIFIED_CONTRACTS.json': 'fe858d951e281b8a2a3c51bc0aeb0932ec5561f270c06f08b19f85fbdfa365ea'}
for name,digest in PINS.items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Verified GDN binding changed: '+name)
CONTRACTS=json.loads((ROOT/'SGL_VERIFIED_CONTRACTS.json').read_text())
def install(providers):
    previous=providers.get('GDNFwdOp')
    def exact(case):
        if case.id not in CONTRACTS:
            return previous(case)
        plain=lambda v:json.loads(json.dumps(v))
        c=CONTRACTS[case.id]
        if plain(case.params)!=c['parameters'] or plain(case._call.tensors)!=c['declared_tensors']:
            raise ValueError('Verified GDN exact contract changed')
        guard=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
        if guard.get('phase')!='running' or guard.get('child',{}).get('pid')!=os.getpgrp():
            raise RuntimeError('Full foreground NPU guard required')
        for p in ['/opt/tileops-ci/runtime/lib/python3.12/site-packages',str(ROOT.parents[2]/'.cache/all-pairs/remote_native_tla/python-deps')]:
            if p not in sys.path:sys.path.append(p)
        spec=importlib.util.spec_from_file_location('_single_server1_sgl_exact',ROOT/'sgl_adapter.py')
        adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
        return adapter.build(case)
    exact.verified_exact_case_ids=tuple(CONTRACTS)
    providers['GDNFwdOp']=exact
