"""Extract original entry points while keeping every native function body unchanged."""
import ast
import copy
import hashlib
import importlib.util
from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[4]
VENDOR = WORKSPACE / 'ascend_baselines/tilelang'
CACHE = WORKSPACE / '.cache/all-pairs/tilelang/sources'


def load(relative, names, *, jit=None, extract=None):
    from benchmarks.ascend_tilelang_baselines import _runtime, _runtime_hashes
    from benchmarks.ascend_baselines import source_hashes
    tl = _runtime()
    path = VENDOR / relative
    tree = ast.parse(path.read_text())
    selected = []
    for name in names:
        original = next(n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
        node = copy.deepcopy(original)
        if extract and name in extract:
            # Only strip host-side compile/test code after the original PrimFunc.
            # No statement of that PrimFunc or its enclosing setup is changed.
            prim = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == extract[name])
            node.body = node.body[:node.body.index(prim)+1] + [ast.Return(ast.Name(id=prim.name, ctx=ast.Load()))]
        if jit and name in jit:
            node.decorator_list = ast.parse('@' + jit[name] + '\ndef placeholder(): pass').body[0].decorator_list
        else:
            node.decorator_list = [] if extract and name in extract else node.decorator_list
        selected.append(node)
    imports = ast.parse('import torch\nimport tilelang\nimport tilelang.language as T\nimport math\n').body
    adapted = ast.unparse(ast.fix_missing_locations(ast.Module(body=imports+selected, type_ignores=[]))) + '\n'
    key = hashlib.sha256(adapted.encode()).hexdigest()
    CACHE.mkdir(parents=True, exist_ok=True)
    generated = CACHE / (key + '.py')
    if generated.exists():
        assert generated.read_text() == adapted
    else:
        generated.write_text(adapted)
    spec = importlib.util.spec_from_file_location('_allpairs_tilelang_' + key, generated)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, source_hashes(path, generated, Path(__file__)) | _runtime_hashes(tl)
