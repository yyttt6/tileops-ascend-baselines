"""Single-agent local FP8 full-API registration; CPU-only import."""
import hashlib, importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4] / "reports/baseline-single-agent-local-20261027"
PINS = {'fp8_registration.py': '1dac82bdd47776f8bda0075fac810abe1bf8b530885e4b7722d8b8fa48305a72', 'FP8_VERIFIED_CASES.json': '608b7fa66521afc3e609fbb3946ea81a8c02ef2eca0b25c12f3d1d83bac32668', 'FP8_RUNTIME_PREFLIGHT.json': 'c9fe2749da37acdb51a0d4d9413957bf55b8fc23a2a99a03c5e3c62518b977d7'}
for name, digest in PINS.items():
    if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError("Verified FP8 registration changed: " + name)
spec = importlib.util.spec_from_file_location("_single_agent_fp8_registration", ROOT / "fp8_registration.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
install = module.install
