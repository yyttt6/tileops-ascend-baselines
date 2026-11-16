import hashlib,importlib.util
from pathlib import Path
_root=Path(__file__).resolve().parents[4]
_path=_root/'reports/baseline-integration-700-20261026/threehost_linear_registration3/registration.py'
if hashlib.sha256(_path.read_bytes()).hexdigest()!='188db47a290c50e48238a876240c34a74b8df70521a747462d9e4b6136bb3137':raise RuntimeError('Linear3 pinned registration changed')
_spec=importlib.util.spec_from_file_location('_baseline700_linear3_pinned',_path)
_module=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_module)
install=_module.install
