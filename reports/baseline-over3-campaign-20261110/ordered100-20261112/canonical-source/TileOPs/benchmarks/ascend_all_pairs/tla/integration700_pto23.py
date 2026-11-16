import hashlib,importlib.util
from pathlib import Path
_root=Path(__file__).resolve().parents[4]
_path=_root/'reports/baseline-integration-700-20261026/threehost_pto_registration23/registration_parent.py'
if hashlib.sha256(_path.read_bytes()).hexdigest()!='c3a29a8c68172513f1d5a63dab4ec135111a3f7f14d87a93edfc5ff2a2e6e73b':raise RuntimeError('PTO23 pinned registration changed')
_spec=importlib.util.spec_from_file_location('_baseline700_pto23_pinned',_path)
_module=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_module)
install=_module.install
