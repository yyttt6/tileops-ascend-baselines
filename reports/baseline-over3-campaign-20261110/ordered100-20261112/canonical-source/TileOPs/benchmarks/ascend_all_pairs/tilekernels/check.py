#!/usr/bin/env python3
"""Reuse unmodified full manifest checker and record original vendor call attribution."""
from pathlib import Path
import sys, os, json, hashlib
ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT.parent
CACHE = WORKSPACE / ".cache/all-pairs/tilekernels"
os.environ["TILELANG_CACHE_DIR"] = str(CACHE / "tilelang")
os.environ["TMPDIR"] = str(CACHE / "tmp")
os.environ["TRITON_CACHE_DIR"] = str(CACHE / "triton")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(WORKSPACE)]
from scripts import ascend_classic_check as checker
checker.require_guard()
original_verify = None
oracle_diagnostic = None
if 'MaxPool' in sys.argv[sys.argv.index('--op') + 1]:
    oracle_path = WORKSPACE / '.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/tilekernels/original_pooling_oracle_diagnostic.json'
    oracle_diagnostic = json.loads(oracle_path.read_text())
    assert oracle_diagnostic['status'] == 'original_reference_device_failure'
    assert oracle_diagnostic['candidate_native_calls'] == 0
    assert oracle_diagnostic['cpu_reference_same_function_passed']
    assert hashlib.sha256((WORKSPACE/oracle_diagnostic['reference_source']).read_bytes()).hexdigest() == oracle_diagnostic['reference_source_sha256']
    from workloads import numerics
    original_verify = numerics.verify
    def verify_original_pool_reference_on_cpu(reference, inputs, **kwargs):
        import torch
        return original_verify(checker.cpu_reference(torch, reference, inputs), inputs, **kwargs)
    numerics.verify = verify_original_pool_reference_on_cpu
from benchmarks.ascend_all_pairs.tilekernels import PROVIDERS, DIAGNOSTIC_PROVIDERS
from benchmarks.ops import bench_ascend_classics
bench_ascend_classics.candidate_factories = lambda case: {"TileKernels": lambda: ({**DIAGNOSTIC_PROVIDERS, **PROVIDERS})[sys.argv[sys.argv.index("--op") + 1]](case)}
called = {}
launches = {}
vendor = str(WORKSPACE / "ascend_baselines/TileKernels/tile_kernels")
def trace(frame, event, arg):
    if event == 'call' and frame.f_code.co_name == '__call__' and frame.f_code.co_filename.endswith('/tilelang/jit/kernel.py'):
        obj = frame.f_locals.get('self')
        adapter = getattr(obj, 'adapter', None)
        src = getattr(adapter, 'kernel_global_source', '') or ''
        cached_lib = getattr(adapter, 'libpath', None)
        if not src and cached_lib:
            cached_source = Path(cached_lib).with_name('device_kernel.asc')
            if cached_source.is_file(): src = cached_source.read_text()
        target = str(getattr(adapter, 'target', ''))
        caller = frame.f_back
        vendor_stack = []
        while caller is not None:
            if caller.f_code.co_filename.startswith(vendor):
                vendor_stack.append({'file': str(Path(caller.f_code.co_filename).relative_to(WORKSPACE)),
                                     'line': caller.f_lineno, 'function': caller.f_code.co_name})
            caller = caller.f_back
        name = str(getattr(obj, 'prim_func', None).attrs.get('global_symbol')) if getattr(obj, 'prim_func', None) is not None else 'cached'
        key = (name, hashlib.sha256(str(src).encode()).hexdigest())
        if key not in launches:
            lib = getattr(adapter, 'libpath', None)
            launches[key] = {'name': name, 'target': target, 'generated_source_sha256': key[1],
                             'vendor_stack': vendor_stack, 'calls': 0, 'binary': str(lib),
                             'binary_sha256': hashlib.sha256(Path(lib).read_bytes()).hexdigest() if lib and Path(lib).is_file() else None}
        launches[key]['calls'] += 1
    if event == "call" and frame.f_code.co_filename.startswith(vendor):
        path = str(Path(frame.f_code.co_filename).relative_to(WORKSPACE))
        key = (path, frame.f_code.co_firstlineno, frame.f_code.co_name)
        called[key] = called.get(key, 0) + 1
sys.setprofile(trace)
try:
    rc = checker.main()
finally:
    sys.setprofile(None)
    if original_verify is not None: numerics.verify = original_verify
    path = Path(sys.argv[sys.argv.index("--output") + 1])
    if path.exists():
        row = json.loads(path.read_text())
        row["native_trace_schema"] = 2
        row["provider_module"] = "benchmarks.ascend_all_pairs.tilekernels"
        row["provider_sha256"] = {str(p.relative_to(WORKSPACE)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in Path(__file__).parent.glob("*.py")}
        row["vendor_python_calls"] = [{"file": p, "line": line, "function": name, "calls": n}
                                      for (p, line, name), n in sorted(called.items())]
        row["native_kernel_launches"] = list(launches.values())
        row["input_preservation_requested"] = True
        row["compiler_cache"] = str(CACHE / "tilelang")
        if oracle_diagnostic is not None:
            row['reference_device'] = 'cpu'
            row['independent_cpu_reference'] = True
            row['reference_engine'] = 'Same original manifest reference via unchanged checker.cpu_reference helper'
            row['reference_placement_adaptation'] = 'Original NPU reference independently faults before any vendor call; execute identical original reference on CPU, retain original exact comparison and input preservation.'
            row['oracle_failure_diagnostic'] = str(oracle_path.relative_to(WORKSPACE))
            row['oracle_failure_diagnostic_sha256'] = hashlib.sha256(oracle_path.read_bytes()).hexdigest()
            row['reference_source_sha256'] = oracle_diagnostic['reference_source_sha256']
        path.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n")
sys.exit(rc)
