"""Short guarded batches using the shared complete checker and native tracing."""
import argparse
from dataclasses import replace
import dis
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

WORKSPACE = Path(__file__).resolve().parents[4]
ROOT = WORKSPACE / 'TileOPs'
EVIDENCE = WORKSPACE / '.agent_memory/THREADS/tileops-progress-report-20261019/ALL_PAIRS/sgl_original_input_recovery'
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(WORKSPACE)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--seconds', type=float, default=60)
    args = p.parse_args()
    started = time.monotonic()
    cache = WORKSPACE / '.cache/all-pairs/sgl'
    for key, sub in [('TMPDIR','tmp'),('XDG_CACHE_HOME','xdg'),('TORCHINDUCTOR_CACHE_DIR','inductor'),('ASCEND_PROCESS_LOG_PATH','ascend-logs')]:
        folder = cache / sub
        folder.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(folder)
    spec = importlib.util.spec_from_file_location('shared_pair_checker', ROOT / 'scripts/ascend_pair_check.py')
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    checker.require_guard()
    from benchmarks.ascend_all_pairs import registry
    from benchmarks.ascend_all_pairs.sgl import PROVIDERS
    import torch
    import benchmarks.api as api
    from benchmarks.ascend_all_pairs.sgl.recovery_observation import Observation
    original_observe = checker.observe_call
    from torch.utils._python_dispatch import TorchDispatchMode
    result_dir = EVIDENCE / 'receipts' / args.run
    result_dir.mkdir(parents=True, exist_ok=False)
    rc = 0
    for item in json.loads(args.queue.read_text()):
        op, case_id = item['op'], item['case_id']
        candidate_calls = {'entered': 0, 'returned': 0}
        observation = Observation(torch, checker.tensor_metadata, result_dir / f'{op}__{case_id}')
        def observe(run_torch, run, records):
            observed = original_observe(run_torch, run, records)
            def recording(*values, **kwargs):
                inputs = observation.snapshot(values)
                result = observed(*values, **kwargs)
                observation.calls.append({'callable':getattr(run, '__qualname__', repr(run)),
                                          'inputs':inputs, 'outputs':observation.snapshot(result)})
                return result
            return recording
        checker.observe_call = observe
        called, native_calls, launches, ownership = {}, {}, [], []
        inside_candidate = False
        def build(case):
            nonlocal inside_candidate
            native = PROVIDERS[op](case)
            original = native.implementation.run
            def run(*inputs, **kwargs):
                nonlocal inside_candidate
                candidate_calls['entered'] += 1
                inside_candidate = True
                try:
                    result = original(*inputs, **kwargs)
                    candidate_calls['returned'] += 1
                    return result
                finally:
                    inside_candidate = False
            return replace(native, implementation=replace(native.implementation, run=run))
        registry.candidate_factory = lambda case, library: lambda: build(case)
        class CaptureNative(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                name = str(func)
                owned = name.startswith('allpairs_sgl_unidex.') or name.startswith('npu.') and any(tag in name for tag in ('sgemm','bgmv','sgmv','chunk_kda','chunk_gated_delta','recurrent_gated_delta','causal_conv','sparse_flash_attention'))
                result = func(*args, **(kwargs or {}))
                if inside_candidate and owned:
                    native_calls[name] = native_calls.get(name, 0) + 1
                return result
        def trace(frame, event, value):
            # Observe the existing cached_property return; never assign Case
            # inputs/reference or generate an extra random input.
            if event == 'return' and frame.f_code is api.Case.inputs.func.__code__:
                observed_case = frame.f_locals['self']
                observation.factory_returns.append({
                    'case_id':observed_case.id, 'factory_file':frame.f_code.co_filename,
                    'factory_line':frame.f_code.co_firstlineno,
                    'declared_cpu_inputs':[name for name in observed_case._call.signature.inputs
                                           if getattr(observed_case._call.specs.get(name),'cpu',False)],
                    'values':observation.snapshot(value)})
            if (event == 'return' and frame.f_code.co_name == '_owned_native'
                    and frame.f_code.co_filename.endswith('/benchmarks/ascend_baselines.py')
                    and value is not None):
                binary_path = frame.f_locals['path']
                info = frame.f_locals.get('info')
                ownership.append({'path':str(binary_path),
                                  'sha256':hashlib.sha256(binary_path.read_bytes()).hexdigest(),
                                  'symbols':list(frame.f_locals['symbols']),
                                  'library_handle':str(value._handle),
                                  'returned_after_nm_and_dladdr_checks':True,
                                  'last_dladdr_path':info.fname.decode() if info and info.fname else None,
                                  'last_symbol_address':str(info.saddr) if info else None})
            if not inside_candidate:
                return
            filename = frame.f_code.co_filename
            if event == 'call' and '/ascend_baselines/sgl-kernel-npu/' in filename:
                key = (filename, frame.f_code.co_firstlineno, frame.f_code.co_name)
                called[key] = called.get(key, 0) + 1
            # JITFunction.run returns a compiled kernel after the launch call.
            # Exclude warmup, failed launches, and reference-side kernels.
            if event == 'return' and filename.endswith('/triton/runtime/jit.py') and frame.f_code.co_name == 'run' and value is not None and not frame.f_locals.get('warmup', True):
                fn = getattr(frame.f_locals.get('self'), 'fn', None)
                source = getattr(getattr(fn, '__code__', None), 'co_filename', '')
                if '/ascend_baselines/sgl-kernel-npu/' in source:
                    launches.append({'source':source, 'function':getattr(fn,'__name__',''), 'kernel':str(getattr(value,'name','')), 'hash':str(getattr(value,'hash','')), 'grid':list(frame.f_locals.get('grid',())) if isinstance(frame.f_locals.get('grid'),(tuple,list)) else str(frame.f_locals.get('grid')), 'returned_after_launch':True})
                    launches[-1].update(observation.binary(value))
            # Several original SGL hosts warm up once, cache CompiledKernel,
            # and invoke its runner directly. A successful RETURN instruction
            # distinguishes that native launch from an exception unwind, whose
            # profile event also carries value=None.
            if (event == 'return' and filename.endswith('/triton/compiler/compiler.py')
                    and frame.f_code.co_name == 'runner'
                    and dis.opname[frame.f_code.co_code[frame.f_lasti]] in ('RETURN_VALUE','RETURN_CONST')):
                kernel = frame.f_locals.get('self')
                jit = getattr(getattr(kernel,'src',None),'fn',None)
                fn = getattr(jit,'fn',None)
                source = getattr(getattr(fn,'__code__',None),'co_filename','')
                if '/ascend_baselines/sgl-kernel-npu/' in source:
                    launches.append({'source':source,'function':getattr(fn,'__name__',''),
                                     'kernel':str(getattr(kernel,'name','')),'hash':str(getattr(kernel,'hash','')),
                                     'grid':list(frame.f_locals.get('grid',())),
                                     'mechanism':'compiled_kernel_runner','returned_after_launch':True})
                    launches[-1].update(observation.binary(kernel))
        output = result_dir / f'{op}__{case_id}.json'
        print(json.dumps({'phase':'case_start', 'op':op, 'case_id':case_id}), flush=True)
        sys.argv = ['ascend_pair_check.py','--op',op,'--case-id',case_id,'--library','sgl-kernel-npu','--output',str(output)]
        sys.setprofile(trace)
        try:
            with CaptureNative():
                current_rc = checker.main()
        finally:
            sys.setprofile(None)
        row = json.loads(output.read_text())
        row.update(input_generation_device='original_case_factory_on_npu',
                   original_factory_returns=observation.factory_returns,
                   full_tensor_observations=observation.calls,
                   cpu_prepared_cache_consumed=False)
        observation.flush()
        row.update(provider_module='benchmarks.ascend_all_pairs.sgl', runtime_original_calls=[{'file':f,'line':line,'function':name,'calls':n} for (f,line,name),n in sorted(called.items())], runtime_native_calls=native_calls, native_kernel_launches=launches, candidate_calls=candidate_calls, native_trace_schema=2, input_preservation_requested=True)
        row['native_trace_source_sha256'] = {str(path.relative_to(WORKSPACE)):hashlib.sha256(path.read_bytes()).hexdigest()
                                           for path in (Path(__file__), Path(__file__).with_name('recovery_observation.py'))}
        row['native_trace_schema'] = 3
        row['original_native_ownership_checks'] = ownership
        row['binary_evidence_limit'] = 'Returned original CompiledKernel bytes and initialized module/function handles; no independent ACL hardware instruction trace. Native torch dispatcher calls remain dispatcher-level attribution.'
        row['attribution_complete'] = candidate_calls['returned'] > 0 and bool(native_calls or launches) and any('/ascend_baselines/sgl-kernel-npu/' in '/' + f for f in row.get('source_sha256',{}))
        row['original_input_factory_observed'] = len(observation.factory_returns) == 1 and observation.factory_returns[0]['case_id'] == case_id
        if op == 'AdaptiveMaxPool2dFwdOp':
            diagnostic = EVIDENCE.parent / 'sgl/continuation_20261021T1415/original_pooling_oracle_diagnostic.json'
            row['oracle_failure_diagnostic'] = str(diagnostic.relative_to(WORKSPACE))
            row['oracle_failure_diagnostic_sha256'] = hashlib.sha256(diagnostic.read_bytes()).hexdigest()
        if row.get('status') == 'verified' and not row['attribution_complete']:
            row.update(status='attribution_failed', error='No successfully returned original SGL native launch captured')
            current_rc = 1
        if row.get('status') == 'verified' and not row['original_input_factory_observed']:
            row.update(status='input_origin_unproven', error='Expected one observed original Case.inputs factory return')
            current_rc = 1
        output.write_text(json.dumps(row, indent=2, ensure_ascii=False) + '\n')
        rc = max(rc, current_rc)
        if current_rc == 78 or row.get('sync_error'):
            return 78
        torch.npu.synchronize()
        gc.collect()
        torch.npu.empty_cache()
        if time.monotonic() - started >= args.seconds:
            print(json.dumps({'phase':'batch_budget_completed','seconds':time.monotonic()-started}), flush=True)
            break
    torch.npu.synchronize()
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
