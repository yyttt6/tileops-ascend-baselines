"""Use the original checker unchanged, adding isolated providers and launch evidence."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT.parent)]
spec = importlib.util.spec_from_file_location("_fla_original_checker", ROOT / "scripts/ascend_classic_check.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def main():
    checker.require_guard()
    from benchmarks.ops import bench_ascend_classics as registry
    from benchmarks.ascend_all_pairs.fla import PROVIDERS
    from benchmarks.ascend_baselines import source_hashes
    import torch
    import ctypes as C
    class DlInfo(C.Structure):
        _fields_=[("fname",C.c_char_p),("base",C.c_void_p),("sname",C.c_char_p),("saddr",C.c_void_p)]
    dynamic=C.CDLL(None)
    dynamic.dladdr.argtypes=[C.c_void_p,C.POINTER(DlInfo)]
    dynamic.dladdr.restype=C.c_int
    events = []
    evidence_path = Path(os.environ.get('FLA_CONTINUATION_EVIDENCE', str(ROOT.parent / '.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/fla')))
    observed = {}
    continuation = 'FLA_CONTINUATION_EVIDENCE' in os.environ
    if continuation:
        from benchmarks.ascend_all_pairs.fla.observation import describe, triton_return
    active_candidate = False
    reference_cpu = False
    oracle_sources = {}
    original = registry.candidate_factories
    from workloads import numerics
    original_verify=numerics.verify

    def verify(reference,inputs,**kwargs):
        nonlocal reference_cpu
        name=sys.argv[sys.argv.index('--op')+1]
        if name in ('KDAFwdOp','DeltaNetChunkFwdOp','DeltaNetChunkBwdOp') or (name=='GDNFwdOp' and inputs[0].shape[1]>1):
            # Preserve the exact original FP64 oracle, whose long-sequence
            # matmul is unavailable on NPU. Reuse the unchanged checker's CPU
            # execution helper; the candidate and all input checks stay on NPU.
            reference=checker.cpu_reference(torch,reference,inputs)
            reference_cpu=True
        if continuation:
            original_reference=reference
            def reference(*args):
                result=original_reference(*args)
                observed['reference_outputs']=describe(result)
                return result
        return original_verify(reference,inputs,**kwargs)

    numerics.verify=verify

    def factories(case):
        name = sys.argv[sys.argv.index("--op") + 1]
        if name not in PROVIDERS:
            return original(case)
        def build():
            from dataclasses import replace
            import inspect
            workload_file=inspect.getsourcefile(type(case.workload))
            if workload_file:
                path=Path(workload_file).resolve()
                if path.is_relative_to(ROOT):oracle_sources.update(source_hashes(path))
                if path.name in ('deltanet.py','gla.py'):
                    oracle_sources.update(source_hashes(ROOT/'workloads/linear_attention/chunk_cpu.py',ROOT/'workloads/linear_attention/input_values.py'))
            candidate=PROVIDERS[name](case)
            if continuation:
                observed.update(inputs=describe(case.inputs), params=case.params,
                                input_generation='Unchanged original case.inputs under set_run_device(npu), after original seed_case')
            implementation=candidate.implementation
            def run(*args,**kwargs):
                nonlocal active_candidate
                previous=active_candidate
                active_candidate=True
                try:
                    result=implementation.run(*args,**kwargs)
                    if continuation and 'candidate_outputs' not in observed:
                        observed['candidate_outputs']=describe(result)
                    return result
                finally:
                    active_candidate=previous
            return replace(candidate,implementation=replace(implementation,run=run))
        return {"flash-linear-attention-npu":build}

    registry.candidate_factories = factories
    # A Python profiler records actual original JIT dispatch/native ACLNN enqueue,
    # without substituting or changing any implementation.
    def trace(frame, event, arg):
        if not active_candidate:
            return trace
        if continuation and event=='return' and frame.f_code.co_name=='run' and frame.f_code.co_filename.endswith('/triton/runtime/jit.py'):
            record=triton_return(frame,arg,evidence_path)
            if record and record not in events: events.append(record)
        if event=="return" and frame.f_code.co_name=="call" and frame.f_code.co_filename.endswith("/fla_npu/ops/ascendc/_runtime.py"):
            launch=frame.f_locals.get("launch")
            symbol=frame.f_locals.get("name")
            if launch is not None:
                info=DlInfo()
                if not dynamic.dladdr(C.cast(launch,C.c_void_p),C.byref(info)) or not info.fname:
                    raise RuntimeError("Cannot attribute actual FLA launch function")
                owner=str(Path(info.fname.decode()).resolve())
                if "/fla_npu/opp/" not in owner:
                    raise RuntimeError("Actual FLA launch resolved to a dependency: "+owner)
                record={"native_symbol":symbol,"actual_launch_owner":owner,"runtime_return_observed":True,"execution_scope":"candidate"}
                if continuation:
                    record.update(owner_sha256=hashlib.sha256(Path(owner).read_bytes()).hexdigest(),
                                  launch_address=C.cast(launch,C.c_void_p).value,
                                  provenance_limit='Actual host launch function resolved by dladdr; device binary selection is deployment lineage')
                if record not in events:events.append(record)
        if event == "call":
            filename = frame.f_code.co_filename
            name = frame.f_code.co_name
            if ("flash-linear-attention-npu" in filename or "/fla_npu/" in filename or
                    filename.endswith("triton/runtime/jit.py")):
                if name in ("run", "__call__", "_run", "_run_op", "_execute", "_call_aclnn", "call_aclnn") or name.startswith(("chunk_local_cumsum", "npu_")):
                    record = {"file": filename, "line": frame.f_code.co_firstlineno, "function": name,"execution_scope":"candidate"}
                    for key in ("op_name", "name"):
                        value = frame.f_locals.get(key)
                        if isinstance(value, str) and value.startswith("aclnn"):
                            record["native_symbol"] = value
                    obj = frame.f_locals.get("self")
                    fn = getattr(obj, "fn", None)
                    if fn is not None and hasattr(fn, "__code__"):
                        record["jit_source"] = fn.__code__.co_filename
                        record["jit_function"] = fn.__name__
                    if record not in events:
                        events.append(record)
        return trace

    sys.setprofile(trace)
    try:
        rc = checker.main()
    finally:
        sys.setprofile(None)
        numerics.verify=original_verify
        registry.candidate_factories=original
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    row = json.loads(output.read_text())
    if continuation:
        row['original_case_observation']=observed
    row['deployment_mode']=os.environ.get('FLA_ALLPAIRS_MODE')
    row['deployment_package']=os.environ.get('ASCEND_CLASSIC_FLA_PACKAGE')
    row["call_attribution"] = events
    row["attribution_scope"] = "candidate_only"
    row['oracle_source_sha256']=oracle_sources
    if reference_cpu:
        row.update(reference_device='cpu',independent_cpu_reference=True,
                   reference_engine='Unmodified manifest '+row['op']+' FP64 oracle, via original checker.cpu_reference',
                   reference_adaptation='execution_device_only_for_original_FP64_contract')
        module='kda.py' if row['op']=='KDAFwdOp' else ('deltanet.py' if row['op'].startswith('DeltaNetChunk') else 'gdn.py')
        row['reference_source_sha256']=source_hashes(ROOT/'workloads/linear_attention'/module,ROOT/'scripts/ascend_classic_check.py')
        if row['op'].startswith('DeltaNetChunk'):
            row['reference_source_sha256'].update(source_hashes(ROOT/'workloads/linear_attention/chunk_cpu.py'))
            row['reference_adaptation']='execution_device_and_original_cpu_helper_thread_limit_only_for_existing_CPU_FP64_contract'
    row["isolated_provider_sha256"] = source_hashes(*(Path(__file__).with_name(name) for name in ('check.py','__init__.py','providers.py','environment.py','launch_case.py','batch.py','observation.py')))
    row["input_preservation"]={"requested":True,"passed":True if row.get("status")=="verified" else None,
                               "mechanism":"Original workloads.numerics.Request(preserve_inputs=True) in unchanged checker"}
    snapshots=evidence_path/'adapter_snapshots'
    snapshots.mkdir(exist_ok=True)
    row["adapter_snapshots"]={}
    hashes=row.get("source_sha256",{})|row["isolated_provider_sha256"]
    for name,sha in hashes.items():
        if name.startswith("TileOPs/benchmarks/ascend_all_pairs/fla/") and name.endswith(".py"):
            path=ROOT.parent/name
            contents=path.read_bytes()
            if hashlib.sha256(contents).hexdigest()!=sha:
                raise RuntimeError("Provider changed while checking: "+name)
            target=snapshots/(path.stem+"-"+sha+".py")
            if not target.exists():target.write_bytes(contents)
            row["adapter_snapshots"][name]=str(target.relative_to(ROOT.parent))
    oracle_snapshots=snapshots.with_name('oracle_snapshots')
    oracle_snapshots.mkdir(exist_ok=True)
    row['oracle_snapshots']={}
    for name,sha in oracle_sources.items():
        path=ROOT.parent/name
        contents=path.read_bytes()
        if hashlib.sha256(contents).hexdigest()!=sha:
            raise RuntimeError('Original workload source changed during check: '+name)
        target=oracle_snapshots/(path.stem+'-'+sha+'.py')
        if not target.exists():target.write_bytes(contents)
        row['oracle_snapshots'][name]=str(target.relative_to(ROOT.parent))
    annotated=output.with_suffix(output.suffix+'.annotating.tmp')
    annotated.write_text(json.dumps(row, indent=2, default=str) + "\n")
    annotated.replace(output)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
