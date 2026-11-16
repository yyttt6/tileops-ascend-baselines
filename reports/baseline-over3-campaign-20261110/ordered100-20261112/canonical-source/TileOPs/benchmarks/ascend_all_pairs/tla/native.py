"""Load selected, unmodified vendor definitions without example demo side effects.

This module and its AST helpers are CPU-only. Runtime loading is deliberately
inside factory calls and must only be used by a separately authorized device run.
"""
from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
import sys
import symtable
import types

from . import deployment

WORKSPACE = Path(__file__).resolve().parents[4]
VENDOR = WORKSPACE / 'ascend_baselines/tilelang-ascend'
HERE = Path(__file__).resolve().parent
CACHE = WORKSPACE / '.cache/all-pairs/tla'
_prepared = False
_deployment = None
_runtime_configured_id = None


def prepare_deployment():
    """Lazy process-local build isolation; no device/compiler imports."""
    global _prepared
    if _prepared:
        return
    import os
    import tempfile
    env = deployment.deployment_environment(VENDOR, os.environ)
    for name in ('TVM_IMPORT_PYTHON_PATH', 'TVM_LIBRARY_PATH'):
        os.environ[name] = env[name]
    for directory in ('kernel', 'tmp'):
        (CACHE / directory).mkdir(parents=True, exist_ok=True)
    os.environ['TILELANG_CACHE_DIR'] = str(CACHE / 'kernel')
    os.environ['TMPDIR'] = str(CACHE / 'tmp')
    tempfile.tempdir = str(CACHE / 'tmp')
    sys.dont_write_bytecode = True
    if str(VENDOR) not in sys.path:
        sys.path.insert(0, str(VENDOR))
    _prepared = True


def selected_tree(relative, names):
    path = VENDOR / relative
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    symbols = symtable.symtable(source, str(path), 'exec')
    tables = {(t.get_name(), t.get_lineno()): t for t in symbols.get_children()}
    definitions = {}
    top_nodes = []
    for node in tree.body:
        # Deployment-only optional-import try blocks: the successful branch is
        # selected after runtime() has established that this compiler is present.
        # Exception fallback definitions are never used as a substitute runtime.
        if isinstance(node, ast.Try):
            top_nodes.extend(n for n in node.body if not isinstance(n, (ast.Import, ast.ImportFrom)))
        else:
            top_nodes.append(node)
    for node in top_nodes:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node
    wanted = set(names)
    pending = list(names)
    while pending:
        name = pending.pop()
        node = definitions.get(name)
        if node is None:
            continue
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            refs = set()
            stack = [tables[(node.name, node.lineno)]]
            while stack:
                table = stack.pop()
                refs.update(s.get_name() for s in table.get_symbols()
                            if s.is_global() and s.is_referenced())
                stack.extend(table.get_children())
            signature_nodes = list(node.decorator_list)
            if isinstance(node, ast.FunctionDef):
                signature_nodes += [node.args]
                if node.returns is not None:
                    signature_nodes += [node.returns]
            refs.update(r.id for part in signature_nodes for r in ast.walk(part)
                        if isinstance(r, ast.Name) and isinstance(r.ctx, ast.Load))
        else:
            refs = {r.id for r in ast.walk(node)
                    if isinstance(r, ast.Name) and isinstance(r.ctx, ast.Load)}
        for ref in refs:
            if ref in definitions and ref not in wanted:
                wanted.add(ref)
                pending.append(ref)
    missing = set(names) - definitions.keys()
    if missing:
        raise ValueError(f'{relative}: missing definitions {sorted(missing)}')
    selected = {id(definitions[name]) for name in wanted if name in definitions}
    nodes = [n for n in top_nodes if id(n) in selected]
    return ast.Module(body=nodes, type_ignores=[]), path


def definition_records(relative, names):
    tree, path = selected_tree(relative, names)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return [dict(file=str(path.relative_to(WORKSPACE)), line=n.lineno,
                 end_line=n.end_lineno, symbol=getattr(n, 'name', None), sha256=sha)
            for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]


def runtime():
    # Importing any compiler/runtime is forbidden during this task's CPU audit.
    global _runtime_configured_id, _deployment
    prepare_deployment()
    if _deployment is None:
        report = deployment.probe_deployment(VENDOR, CACHE)
        if report['blockers']:
            raise RuntimeError(report['reason'])
        _deployment = report
    import tilelang
    expected = (VENDOR / 'tilelang/__init__.py').resolve()
    if Path(tilelang.__file__).resolve() != expected:
        raise RuntimeError(f'tilelang-ascend requires its own runtime at {expected}')
    if _runtime_configured_id != id(tilelang):
        if hasattr(tilelang, 'set_cache_dir'):
            tilelang.set_cache_dir(str(CACHE / 'kernel'))
        _runtime_configured_id = id(tilelang)
    return tilelang


def load(relative, *names):
    import logging
    import typing
    import torch
    tl = runtime()
    tree, path = selected_tree(relative, names)
    key = '_allpairs_tla_' + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    module = types.ModuleType(key)
    module.__file__ = str(path)
    module.__dict__.update(tilelang=tl, tl=tl, T=tl.language, torch=torch,
                           math=math, Literal=typing.Literal, Tuple=typing.Tuple,
                           logger=logging.getLogger(key))
    # Only the inspected functions and referenced constants are executed.
    # Imports, pytest decorators, CLI parsing, cache clearing and demos are absent.
    sys.modules[key] = module
    exec(compile(tree, str(path), 'exec', dont_inherit=True), module.__dict__)
    return module


def compile_builder(module, name, *args, out_idx=None, workspace_idx=None, **kwargs):
    tl = runtime()
    builder = getattr(module, name)
    raw = getattr(builder, '__wrapped__', builder)
    jit_impl = getattr(builder, '__jit_impl__', None)
    config = (jit_impl.pass_configs if jit_impl is not None else
              getattr(module, 'pass_configs', getattr(module, 'PASS_CONFIGS', {})))
    if jit_impl is not None:
        if out_idx is None:
            out_idx = jit_impl.out_idx
        if workspace_idx is None:
            workspace_idx = jit_impl.workspace_idx
    compile_kwargs = dict(target='pto', pass_configs=config)
    if out_idx is not None:
        compile_kwargs['out_idx'] = out_idx
    if workspace_idx is not None:
        compile_kwargs['workspace_idx'] = workspace_idx
    return tl.compile(raw(*args, **kwargs), **compile_kwargs)


def candidate(run, api, adaptation, *sources):
    from benchmarks.api import Implementation
    from benchmarks.ascend_baselines import Candidate
    from benchmarks.ascend_all_pairs.jit_evidence import tilelang_run
    paths = [HERE / '__init__.py', HERE / 'native.py', HERE / 'adapters.py', HERE / 'plans.py',
             HERE / 'deployment.py']
    paths.extend(VENDOR / src for src in sources)
    paths.extend(VENDOR / src for src in ('tilelang/__init__.py', 'tilelang/jit/__init__.py',
                                         'tilelang/jit/adapter/cython/adapter.py', 'tilelang/libinfo.py',
                                         'tilelang/env.py', '3rdparty/tvm/python/tvm/_ffi/libinfo.py',
                                         'tilelang/jit/adapter/cython/cython_wrapper.pyx'))
    hashes = {str(p.relative_to(WORKSPACE)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in paths}
    if _deployment is not None:
        records = list(_deployment['selected_libraries'].values())
        wrapper = _deployment['cython_cache']
        if wrapper['exists']:
            records.append(wrapper)
        for record in records:
            path = Path(record['file'])
            if deployment.sha256(path) != record['sha256']:
                raise RuntimeError(f'Deployment artifact changed after preparation: {path}')
            key = str(path.relative_to(WORKSPACE)) if path.is_relative_to(WORKSPACE) else str(path)
            hashes[key] = record['sha256']
    return Candidate(Implementation(tilelang_run(run, hashes, CACHE)), 'tilelang-ascend::' + api, adaptation, hashes,
                     limitations=('CPU static audit only; no runtime or numerical validation implied.',))
