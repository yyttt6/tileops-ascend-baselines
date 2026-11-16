"""CPU-only deployment inspection and import-time Cython cache placement.

No compiler/device package is imported here. ELF headers and hashes establish
file identity only; they are not dlopen, ABI, kernel, or numerical evidence.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.abc
import importlib.machinery
import os
from pathlib import Path
import platform
import sys

ADAPTER_MODULE = 'tilelang.jit.adapter.cython.adapter'
ADAPTER_SOURCE = 'tilelang/jit/adapter/cython/adapter.py'
ADAPTER_SHA256 = '762cf19b607e2e81337cc4626a4e9dabbf916bed0e7da1bd5eaad10084c0c09f'
LIBRARIES = ('libtilelang.so', 'libtilelang_module.so', 'libtvm.so', 'libtvm_runtime.so')
ELF_MACHINES = {'x86_64': 62, 'amd64': 62, 'aarch64': 183, 'arm64': 183}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def cache_root(cache, environment):
    return Path(environment.get('TLA_CYTHON_CACHE_DIR') or Path(cache) / 'cython').resolve()


def redirected_tree(source, directory):
    """Hash gate precedes parsing; replace exactly one function body, never its signature."""
    source = Path(source).resolve()
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != ADAPTER_SHA256:
        raise RuntimeError(f'TileLang Cython adapter source hash changed: {source}')
    tree = ast.parse(data, filename=str(source))
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == 'get_cache_dir']
    if len(functions) != 1:
        raise RuntimeError('Expected exactly one top-level get_cache_dir function')
    # Embed the prepared path, so later environment changes cannot redirect it.
    functions[0].body = ast.parse(
        f'cache_dir = Path({str(Path(directory).resolve())!r}) / '
        'f"py{sys.version_info.major}{sys.version_info.minor}"\n'
        'cache_dir.mkdir(parents=True, exist_ok=True)\n'
        'return cache_dir\n').body
    return ast.fix_missing_locations(tree)


class CacheLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, fullname, source, directory):
        super().__init__(fullname, str(source))
        self.directory = directory

    def get_code(self, fullname):
        if fullname != ADAPTER_MODULE:
            raise RuntimeError('Unexpected intercepted module')
        # Never reuse/write a vendor pyc; recheck the source at execution time.
        return compile(redirected_tree(self.path, self.directory), self.path,
                       'exec', dont_inherit=True)


class CacheFinder(importlib.abc.MetaPathFinder):
    def __init__(self, source, directory):
        self.source = Path(source).resolve()
        self.directory = Path(directory).resolve()

    def find_spec(self, fullname, path=None, target=None):
        if fullname != ADAPTER_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin or Path(spec.origin).resolve() != self.source:
            raise RuntimeError('Refusing Cython adapter from another TileLang checkout')
        spec.loader = CacheLoader(fullname, self.source, self.directory)
        return spec


def install_cache_hook(vendor, directory):
    """Call before the first TileLang/TVM import; an already loaded runtime is too late."""
    if Path(directory).resolve().is_relative_to(Path(vendor).resolve()):
        raise RuntimeError('Cython cache must be outside the vendor source tree')
    loaded = [name for name in sys.modules
              if name.split('.')[0] in ('tilelang', 'tvm', 'cython_wrapper')]
    if loaded:
        raise RuntimeError('TileLang/TVM/Cython already imported; use a fresh process: ' + ', '.join(loaded))
    source = Path(vendor) / ADAPTER_SOURCE
    redirected_tree(source, directory)  # fail before any import or vendor execution
    finder = CacheFinder(source, directory)
    sys.meta_path.insert(0, finder)
    return finder


def deployment_environment(vendor, environment):
    """Pin TVM Python identity; support vendor build/tvm and explicit private lib paths."""
    vendor = Path(vendor).resolve()
    env = dict(environment)
    expected = vendor / '3rdparty/tvm/python'
    if env.get('TVM_IMPORT_PYTHON_PATH') and Path(env['TVM_IMPORT_PYTHON_PATH']).resolve() != expected:
        raise RuntimeError('TVM_IMPORT_PYTHON_PATH points outside the selected vendor')
    env['TVM_IMPORT_PYTHON_PATH'] = str(expected)
    if not env.get('TVM_LIBRARY_PATH'):
        env['TVM_LIBRARY_PATH'] = os.pathsep.join(str(p) for p in
                                               (vendor / 'build/tvm', vendor / 'tilelang/lib'))
    return env


def library_roots(vendor, environment):
    """Mirror the selected vendor libinfo search order after deployment_environment()."""
    vendor = Path(vendor).resolve()
    env = deployment_environment(vendor, environment)
    def split(key, tvm=False, strip=True):
        if not env.get(key):
            return []
        paths = env[key].split(os.pathsep)
        if strip:
            paths = [p.strip() for p in paths]
        # TVM realpath('') means cwd; TileLang filters empty directories first.
        return [Path(p) for p in paths if p or tvm]
    tl = [vendor / p for p in ('tilelang', 'build', 'build/Release', 'tilelang/lib')]
    # TileLang treats TILELANG_LIBRARY_PATH as ONE directory, unlike TVM's list.
    if env.get('TILELANG_LIBRARY_PATH'):
        tl.append(Path(env['TILELANG_LIBRARY_PATH']))
    if env.get('CONDA_PREFIX'):
        tl.append(Path(env['CONDA_PREFIX']) / 'lib')
    tl += split('LD_LIBRARY_PATH')
    tvm = (split('TVM_LIBRARY_PATH', tvm=True, strip=False) +
           split('LD_LIBRARY_PATH', tvm=True) + split('PATH', tvm=True))
    tvm += [vendor / p for p in ('3rdparty/tvm/python/tvm', '3rdparty/tvm/build',
                                '3rdparty/tvm/build/Release', '3rdparty/tvm/lib', '3rdparty')]
    tvm_home = Path(env.get('TVM_HOME') or vendor / '3rdparty/tvm')
    tvm += [tvm_home / 'web/dist/wasm', tvm_home / 'web/dist']
    return {name: list(dict.fromkeys(p.resolve() for p in (tl if name.startswith('libtilelang') else tvm)))
            for name in LIBRARIES}


def inspect_elf(path, machine):
    """Read bytes only; never use ctypes, ldd, or a package import to inspect a library."""
    path = Path(path)
    result = {'file': str(path), 'exists': path.is_file()}
    if not result['exists']:
        return result
    try:
        with path.open('rb') as stream:
            header = stream.read(64)
        result.update(sha256=sha256(path), bytes=path.stat().st_size)
        valid = len(header) == 64 and header[:7] == b'\x7fELF\x02\x01\x01'
        elf_type = int.from_bytes(header[16:18], 'little') if valid else None
        elf_machine = int.from_bytes(header[18:20], 'little') if valid else None
        result.update(elf64_little_endian=valid, elf_type=elf_type, elf_machine=elf_machine,
                      architecture_compatible=bool(valid and elf_type == 3 and
                      elf_machine == ELF_MACHINES.get(machine.lower())))
    except OSError as exc:
        result.update(error=str(exc), architecture_compatible=False)
    return result


def probe_deployment(vendor, cache, environment=None, machine=None):
    """Report the FIRST existing loader candidate, including a wrong-architecture shadow."""
    vendor = Path(vendor).resolve()
    env = dict(os.environ if environment is None else environment)
    machine = machine or platform.machine()
    probes, selected, blockers = [], {}, []
    roots = library_roots(vendor, env)
    for name, directories in roots.items():
        for directory in directories:
            record = inspect_elf(directory / name, machine)
            probes.append(record)
            if record['exists'] and name not in selected:
                selected[name] = record
        if name not in selected:
            blockers.append(f'Missing library: {name}')
        else:
            record = selected[name]
            path = Path(record['file'])
            explicit = []
            if name.startswith('libtilelang') and env.get('TILELANG_LIBRARY_PATH'):
                explicit = [Path(env['TILELANG_LIBRARY_PATH']).resolve()]
            elif name.startswith('libtvm'):
                explicit = [Path(p).resolve() for p in env.get('TVM_LIBRARY_PATH', '').split(os.pathsep) if p]
            record['selection_origin'] = ('vendor_tree' if path.is_relative_to(vendor) else
                                          'explicit_private_path' if path.parent in explicit else
                                          'ambient_unreviewed_path')
            if record['selection_origin'] == 'ambient_unreviewed_path':
                blockers.append(f'Library outside vendor or explicitly configured private paths: {name}')
            if not record.get('architecture_compatible'):
                blockers.append(f'First loader candidate is not a compatible ELF64 DSO: {name}')
    adapter = vendor / ADAPTER_SOURCE
    adapter_hash = sha256(adapter) if adapter.is_file() else None
    if adapter_hash != ADAPTER_SHA256:
        blockers.append('Cython adapter source hash mismatch; review required')
    directory = cache_root(cache, env) / f'py{sys.version_info.major}{sys.version_info.minor}'
    if directory.is_relative_to(vendor):
        blockers.append('Cython cache must be outside the vendor source tree')
    cython = inspect_elf(directory / 'cython_wrapper.so', machine)
    pyx = vendor / 'tilelang/jit/adapter/cython/cython_wrapper.pyx'
    md5 = directory / 'md5.txt'  # vendor legacy name; contents are SHA256, not MD5
    cython['pyx_sha256'] = sha256(pyx) if pyx.is_file() else None
    cython['cache_source_matches'] = (md5.read_text().strip() == cython['pyx_sha256']
                                      if md5.is_file() else False)
    if cython['exists'] and not cython.get('architecture_compatible'):
        blockers.append('Existing Cython cache has incompatible architecture')
    return {
        'status': 'deployment_blocked' if blockers else 'unverified',
        'reason': '; '.join(blockers) if blockers else
                  'Compatible ELF files observed; runtime, ABI and device behavior remain unverified.',
        'machine': machine, 'runtime_probes': probes, 'selected_libraries': selected,
        'missing_libraries': [name for name in LIBRARIES if name not in selected],
        'cache_adapter': {'file': str(adapter), 'sha256': adapter_hash,
                          'expected_sha256': ADAPTER_SHA256},
        'cython_cache': cython, 'blockers': blockers,
        'binary_evidence': 'observed_file_hash_and_elf_header_only_not_source_build_or_load_proof',
        'runtime_verified': False, 'device_runs': 0,
        'device_validation': 'not_attempted_requires_authorized_host_guard',
        'historical_local_verified_credited': 0,
    }
