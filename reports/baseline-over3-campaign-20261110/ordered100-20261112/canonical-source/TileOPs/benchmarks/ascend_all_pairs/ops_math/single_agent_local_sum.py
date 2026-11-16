"""Single-agent original ReduceSum registration; CPU-only import."""
import hashlib, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4] / "reports/baseline-single-agent-local-20261027"
PINS = {'sum_registration.py': '7e734589c4161fba2a2bae8bf9b9e724c8f3bf5db617a41a9408718dc9342953', 'SUM_VERIFIED_CASE.json': '47e8dd800d0c9dbccbaaf1a9a284b969c4d9c7382efd727eafd6aa75a38aa5ec', 'SUM_ISOLATED_PREFLIGHT03.json': '18cd9071243300ebe277964c9bbe78497bcebbf3d6ad894364eaf2f7e91c952e'}
for name, digest in PINS.items():
    if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError("Verified ReduceSum registration changed: " + name)
spec = importlib.util.spec_from_file_location("_single_agent_sum_registration", ROOT / "sum_registration.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
install = module.install
