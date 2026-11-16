"""One exact server1 KDA contract, verified with original FP64 CPU reference."""
import hashlib, importlib.util, json, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]/'reports/baseline-single-agent-server1-20261027/fp64_fix'
PINS={'vllm_private/omission_admission.json': 'bdaf5371e0e83079d6d59a72340a3511957e00f4a58093fcbefd29ed9688b444', 'vllm_private/linear_attention.py': '5c08af7bb977562fadbb9114224c3290df72fda42077464b100c389209a03fbb', 'vllm_private/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'vllm_private/contracts.py': 'f385156f48ed5ba2e7c0ff07df1a5eb63f7d9ef3c474b2e22f18957a92bc488e', 'vllm_private/omission_contracts.py': 'aa329db2ce3c24360dddc9604337c53118a5791cced7cbc31f02fa1858448a48', 'vllm_private/runtime.py': '2cfba170f191bea2a57cf99266d0b110244df7523209722d4b7acd9ad34fd0f2', 'VLLM_VERIFIED_CONTRACTS.json': '043ce24ce3111d1a5d37baed30b53b93193539fd25a2f82b0e87327d8ef3c2c6'}
for name,digest in PINS.items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Verified KDA binding changed: '+name)
CONTRACTS=json.loads((ROOT/'VLLM_VERIFIED_CONTRACTS.json').read_text())
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
        name='_single_server1_vllm_verified'
        if name not in sys.modules:
            spec=importlib.util.spec_from_file_location(name,ROOT/'vllm_private/__init__.py',submodule_search_locations=[str(ROOT/'vllm_private')])
            package=importlib.util.module_from_spec(spec);sys.modules[name]=package;spec.loader.exec_module(package)
        adapter=__import__(name+'.linear_attention',fromlist=['linear_attention'])
        return adapter.linear_attention(case)
    exact.verified_exact_case_ids=tuple(CONTRACTS)
    providers['KDAFwdOp']=exact
