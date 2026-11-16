"""One exact server1 KDA contract, verified with original FP64 CPU reference."""
import hashlib, importlib.util, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]/'reports/baseline-single-agent-server1-20261027/fp64_fix'
PINS={'fla_provider.py': 'aae5e3e41186cedd2e743596deb5ababa57a0eb44d952612cb47c1201c7e120a', 'FLA_VERIFIED_CONTRACTS.json': '043ce24ce3111d1a5d37baed30b53b93193539fd25a2f82b0e87327d8ef3c2c6'}
for name,digest in PINS.items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Verified KDA binding changed: '+name)
CONTRACTS=json.loads((ROOT/'FLA_VERIFIED_CONTRACTS.json').read_text())
def install(providers):
    previous=providers.get('KDAFwdOp')
    def exact(case):
        if case.id not in CONTRACTS:
            return previous(case)
        plain=lambda v:json.loads(json.dumps(v))
        c=CONTRACTS[case.id]
        if plain(case.params)!=c['parameters'] or plain(case._call.tensors)!=c['declared_tensors']:
            raise ValueError('Verified KDA exact contract changed')
        guard=json.loads(Path('/run/lock/ascend-npu/device-0.lock').read_text())
        if guard.get('phase')!='running' or guard.get('child',{}).get('pid')!=os.getpgrp():
            raise RuntimeError('Full foreground NPU guard required')
        for p in ['/opt/tileops-ci/runtime/lib/python3.12/site-packages',str(ROOT.parents[2]/'.cache/all-pairs/remote_native_tla/python-deps')]:
            if p not in sys.path:sys.path.append(p)
        package=ROOT/'fla_package/fla_npu'
        vendor=package/'opp/vendors/fla_npu_transformer'
        if str(vendor) not in os.environ.get('ASCEND_CUSTOM_OPP_PATH','').split(':'):
            raise RuntimeError('Set the verified FLA ASCEND_CUSTOM_OPP_PATH before Torch initialization')
        os.environ['ASCEND_CLASSIC_FLA_PACKAGE']=str(package)
        if str(package.parent) not in sys.path:sys.path.insert(0,str(package.parent))
        spec=importlib.util.spec_from_file_location('_single_server1_fla_exact',ROOT/'fla_provider.py')
        adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
        return adapter.chunk_kda(case)
    exact.verified_exact_case_ids=tuple(CONTRACTS)
    providers['KDAFwdOp']=exact
