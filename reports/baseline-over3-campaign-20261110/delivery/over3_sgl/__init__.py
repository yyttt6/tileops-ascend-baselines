"""Exact-key, source-checked SGL routes. Call factory inside the host NPU guard.

The returned callable includes every measured broadcast, layout and allocation.
No fallback is provided for unreviewed keys. This package does not own a lock.
"""
import hashlib
import importlib
import importlib.machinery
import json
from pathlib import Path
import sys
import types

_ROOT = Path(__file__).resolve().parent

def factory(library, op, case, params):
    routes = json.loads((_ROOT / 'routes.json').read_text())
    matches = [r for r in routes if r['key'] == [library, op, case]]
    if len(matches) != 1:
        raise KeyError((library, op, case))
    manifest = json.loads((_ROOT / 'vendor_sha256.json').read_text())
    for rel, expected in manifest.items():
        if hashlib.sha256((_ROOT / 'vendor' / rel).read_bytes()).hexdigest() != expected:
            raise RuntimeError('SGL source identity changed: ' + rel)
    expected_path = str(_ROOT / 'vendor/sgl_kernel_npu')
    pkg = sys.modules.get('sgl_kernel_npu')
    if pkg is None:
        pkg = types.ModuleType('sgl_kernel_npu')
        pkg.__path__ = [expected_path]
        pkg.__package__ = 'sgl_kernel_npu'
        pkg.__spec__ = importlib.machinery.ModuleSpec('sgl_kernel_npu', loader=None, is_package=True)
        pkg.__spec__.submodule_search_locations = pkg.__path__
        sys.modules['sgl_kernel_npu'] = pkg
    elif list(getattr(pkg, '__path__', [])) != [expected_path]:
        raise RuntimeError('Use a fresh process: another SGL package is already loaded')
    route = matches[0]
    module = importlib.import_module('.' + route['implementation'], __name__)
    return module.factory(op, params, route['selected_variant'] == 'sgl_fixed')
