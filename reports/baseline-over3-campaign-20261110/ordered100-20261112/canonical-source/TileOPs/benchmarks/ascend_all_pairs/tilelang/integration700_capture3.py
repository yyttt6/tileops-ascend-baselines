import hashlib,importlib.util
from pathlib import Path
_root=Path(__file__).resolve().parents[4]
_path=_root/'reports/baseline-integration-700-20261026/threehost_tilelang3_complete_capture/registration_parent.py'
if hashlib.sha256(_path.read_bytes()).hexdigest()!='ea11f0317b07b7ef7fcf4bf3416b74648bb83f139ba5e121f6caf1c1bebc561e':raise RuntimeError('TileLang3 pinned registration changed')
_spec=importlib.util.spec_from_file_location('_baseline700_tilelang3_pinned',_path)
_module=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_module)
install=_module.install
