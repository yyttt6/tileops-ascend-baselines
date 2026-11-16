"""Verified complete TileLang Bmm API, two exact BF16 contracts on server1."""
import hashlib
import importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4] / 'reports/baseline-single-agent-server1-20261027'
PINS = {'bmm_registration.py': '7189d6dae52e45a99c5bbbca48fa659cf62a6a3c8753195d155acd492825f812', 'bmm_adapter.py': '4966da25007c4e585663d4afb573e9374b1f3e433539100aa15682dc65e983bb', 'BMM_CONTRACTS.json': '93bd5265222b6f02898a967c48b6c21c6ed960b760822a47bac830f8c041ba26', 'BMM_SOURCE_PINS.json': 'e0067fab3feb06818ef4bd88364b148f60a6ee0e1c0d929c9e2e2095a75c3123', 'RUNTIME_PINS.json': 'c4cecf4a14ee388d40cfa6e43fab21c1c63aea88362b96a9ed80a07c82d8a6ad'}
for name, digest in PINS.items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('Verified server1 Bmm registration changed: '+name)
spec=importlib.util.spec_from_file_location('_server1_bmm_registration', ROOT/'bmm_registration.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
install=module.install
