"""Pinned original DeltaNet8 registry closure; no device initialization at import."""
import hashlib,importlib.util
from pathlib import Path
_root=Path(__file__).resolve().parents[4]
_path=_root/'reports/baseline-integration-700-20261026/threehost_linear_registration8/registration.py'
if hashlib.sha256(_path.read_bytes()).hexdigest()!='15d56bacb56ff1fe2fe5bd1399ae0db73d033a06bd94577e98c720489445f87d':raise RuntimeError('Linear8 registration closure changed')
_spec=importlib.util.spec_from_file_location('_baseline700_linear8_pinned',_path)
_module=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
install=_module.install
