"""Exact keys verified locally; immutable tested bindings, no fallback replacement."""
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
import hashlib
ROOT = Path('/root/dyq/workspace/reports/baseline-integration-700-20261026/nn_remote_remaining6')
_KEYS = {
    ('FusedAddRMSNormFwdOp', 'llama-8b-decode-bfloat16'): 'case0_inplace',
    ('BatchNormFwdOp', 'large-spatial-float16'): 'case4',
}
def make(case):
    key = (case._call.signature.name, case.id)
    if key not in _KEYS:
        from benchmarks.ascend_baselines import UnsupportedCase
        raise UnsupportedCase('No verified exact key in nn_remote_remaining6')
    path = ROOT / _KEYS[key] / 'adapter.py'
    import json
    receipt = json.loads((ROOT / 'REGISTRATION_BINDINGS.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt[_KEYS[key]]['sha256']:
        raise RuntimeError('Tested binding hash changed')
    spec = spec_from_file_location('nn_remote_remaining6_' + _KEYS[key], path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make(case)
