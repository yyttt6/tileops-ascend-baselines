"""Load the original CPU metadata dependency without its unrelated TVM runtime.

The unchanged NSA oracle imports tilelang only for cdiv in sequence_metadata.
Its installed implementation lives in tilelang/math/__init__.py and has no
imports. Execute that exact file during the CPU oracle call; retain CANN's TVM
namespace and restore every Python module entry afterwards.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
from pathlib import Path
import sys


MATH_SOURCE = Path('/root/dyq/workspace/.venv/lib/python3.12/site-packages/tilelang/math/__init__.py')


@contextmanager
def original_tilelang_math():
    old = sys.modules.get('tilelang')
    if old is not None and hasattr(old, 'cdiv'):
        yield
        return
    spec = importlib.util.spec_from_file_location('tilelang', MATH_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules['tilelang'] = module
    try:
        yield
    finally:
        if old is None:
            sys.modules.pop('tilelang', None)
        else:
            sys.modules['tilelang'] = old


def source_evidence():
    return {'file': str(MATH_SOURCE), 'sha256': hashlib.sha256(MATH_SOURCE.read_bytes()).hexdigest(),
            'api': 'Original tilelang.math.cdiv, imported from its installed source file unchanged',
            'scope': 'CPU reference integer sequence metadata only; candidate implementation unaffected'}
