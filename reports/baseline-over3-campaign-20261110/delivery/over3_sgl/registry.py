"""Explicit exact-case hook for the SGL provider registry; no publication side effects."""
import hashlib
import json
from pathlib import Path
from . import factory


def install(providers, workspace_root):
    """Return a copied provider mapping; retain original providers for other cases.

    The package must be deployed inside workspace_root. Importing/installing this
    hook performs CPU metadata work only. Providers must be called under npu-guard.
    """
    root = Path(__file__).resolve().parent
    workspace_root = Path(workspace_root).resolve()
    if not root.is_relative_to(workspace_root):
        raise ValueError('Deploy the package inside the benchmark workspace')
    routes = json.loads((root / 'routes.json').read_text())
    keys = {tuple(r['key']) for r in routes}
    result = dict(providers)

    def wrap(op, original):
        def provider(case):
            key = ('sgl-kernel-npu', op, case.id)
            if key not in keys:
                return original(case)
            from benchmarks.api import Implementation
            from benchmarks.ascend_baselines import Candidate
            run = factory(*key, case.params)
            sources = {
                str(path.relative_to(workspace_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in root.rglob('*')
                if path.is_file() and path.suffix in ('.py', '.json')
            }
            return Candidate(
                Implementation(run),
                'SGL original complete API / reviewed exact-case adapter',
                'Measured full layout/copy/allocation boundary preserved; original vendor kernels unchanged.',
                sources,
            )
        return provider

    for op in {key[1] for key in keys}:
        if op not in providers:
            raise KeyError('Missing original SGL provider: ' + op)
        result[op] = wrap(op, providers[op])
    return result
